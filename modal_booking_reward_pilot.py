"""Opt-in matched linear/logarithmic pilot on the expanded booking verifier."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import modal

from modal_booking_study import (artifacts, claims, cpu_image as base_cpu_image,
    gpu_image as base_gpu_image, begin, finish, load_policy, generate_samples,
    parameter_hash, require_deadline)
from verifier_rl import booking_reward_pilot as study
from verifier_rl import supervised_execution as supervised, supervisor_controls
from verifier_rl.evaluation_journal import persist, ReconciliationRequired
from verifier_rl.panel_execution import PanelBackend, pack_result, request_for, require_evidence, unpack_result
from verifier_rl.suites import canonical_json, digest

app = modal.App("verifier-rl-booking-reward-shape")
cpu_image = (base_cpu_image.add_local_file("modal_booking_study.py", "/root/modal_booking_study.py")
             .add_local_file("modal_booking_reward_pilot.py", "/root/modal_booking_reward_pilot.py"))
gpu_image = (base_gpu_image.add_local_file("modal_booking_study.py", "/root/modal_booking_study.py")
             .add_local_file("modal_booking_reward_pilot.py", "/root/modal_booking_reward_pilot.py"))


def check_snapshot(snapshot, root):
    required = {"verifier_rl/booking_reward_pilot.py", "verifier_rl/booking_verifier_v2.py",
                "modal_booking_reward_pilot.py", "modal_booking_study.py"}
    if not required.issubset(snapshot):
        raise ValueError("incomplete source snapshot")
    for name, source in snapshot.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py" or (root / path).read_text() != source:
            raise ValueError("frozen code differs: " + name)


async def execute_group(samples, role, setup, directory, deadline, backend, store, plan=None, *, retry_policy=None):
    """Durable attempts; bounded pre-candidate retries; no repeat of bad answers."""
    queue, stopped = asyncio.Queue(), asyncio.Event()
    records = {s["sample_id"]: {} for s in samples}
    retries = []
    run_id = plan["run_id"] if plan is not None else study.RUN_ID
    if retry_policy not in (None, supervised.STARTUP_RETRY_VERSION):
        raise ValueError("unknown startup retry policy")
    check_execution = supervised.require_evidence if retry_policy is not None or (plan is not None and study.is_warm(plan)) else require_evidence
    for sample in samples:
        if not sample["extraction_status"].startswith("rejected_"):
            for case in study.cases_for(role):
                queue.put_nowait((sample, case))
    async def attempt(sample, case, number):
        require_deadline(deadline)
        key = f"{run_id}/execution/{sample['sample_id']}/{case.input_hash}/{number}"
        target = directory / "inputs" / sample["sample_id"] / case.input_hash
        intent = {"sample_id": sample["sample_id"], "source_hash": digest(sample["source"]),
                  "input_hash": case.input_hash, "attempt": number, "deadline": deadline}
        if not await store.put.aio(key + "/intent", intent, skip_if_exists=True):
            raise ReconciliationRequired("input already claimed: " + key)
        persist(target, {f"intent-{number}": intent})
        result = pack_result(await backend.execute(request_for(sample["source"], case)))
        if not await store.put.aio(key + "/result", result, skip_if_exists=True):
            raise ReconciliationRequired("input result already journaled: " + key)
        persist(target, {f"attempt-{number}": result})
        return result
    async def worker():
        while not stopped.is_set():
            try:
                sample, case = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                selected = first = await attempt(sample, case, 1)
                request = request_for(sample["source"], case)
                retryable = (supervised.startup_retry_allowed(unpack_result(first), request, setup["sandbox_image_id"],
                              study.BOOKING, policy_version=retry_policy) if retry_policy is not None else
                             study.retry_allowed(unpack_result(first), request, setup["sandbox_image_id"], plan))
                if not stopped.is_set() and retryable:
                    for slot in range(study.MAX_STARTUP_RETRIES):
                        reservation = {"sample_id": sample["sample_id"], "input_hash": case.input_hash, "first": first}
                        if await store.put.aio(run_id + f"/startup-slot/{slot}", reservation, skip_if_exists=True):
                            persist(directory / "startup_slots", {str(slot): reservation})
                            if retry_policy is not None:
                                await asyncio.sleep(1)
                            selected = await attempt(sample, case, 2)
                            receipt = dict(reservation, slot=slot, replacement=selected)
                            retries.append(receipt)
                            persist(directory / "startup_retries", {str(slot): receipt})
                            break
                # Even failed/ambiguous selected records are durable before raising.
                records[sample["sample_id"]][case.input_hash] = selected
                persist(directory / "inputs" / sample["sample_id"] / case.input_hash, {"selected": selected})
                check_execution(unpack_result(selected), study.BOOKING, setup["sandbox_image_id"],
                                 source=sample["source"], case=case)
                n = sum(len(row) for row in records.values())
                if n % 100 == 0:
                    print("INPUT PROGRESS", directory.name, n, "remaining", queue.qsize(), flush=True)
            except Exception as exc:
                stopped.set()
                raise ReconciliationRequired(f"{sample['sample_id']} input {case.input_hash}: {type(exc).__name__}: {exc}") from exc
    results = await asyncio.gather(*(worker() for _ in range(min(8, queue.qsize()))), return_exceptions=True)
    failures = [value for value in results if isinstance(value, BaseException)]
    if failures:
        raise failures[0]
    return records, retries


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=study.GRADE_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def grade_batch(key, samples, role, plan, setup, deadline):
    study.validate_batch(key, role, samples, plan)
    directory = Path("/artifacts") / plan["run_id"] / "grading" / key
    intent = {"key": key, "role": role, "samples": samples, "plan_hash": digest(canonical_json(plan)),
              "setup": setup, "deadline": deadline}
    saved = begin(directory, "grade/" + key, intent, deadline, run_id=plan["run_id"])
    if saved is not None:
        return saved
    backend_type = supervised.SupervisedPanelBackend if study.is_warm(plan) else PanelBackend
    backend = backend_type(setup["app_name"], setup["sandbox_image_id"], study.BOOKING, creation_interval_seconds=.26)
    try:
        records, retries = asyncio.run(execute_group(samples, role, setup, directory, deadline, backend, claims, plan))
        raw = {"key": key, "role": role, "samples": samples, "records": records, "retries": retries,
               "plan_hash": digest(canonical_json(plan))}
        persist(directory, {"raw": raw})
        artifacts.commit()
        reports = [study.grade(s, records[s["sample_id"]], role, setup["sandbox_image_id"], plan) for s in samples]
        study.verify_retries(raw, setup["sandbox_image_id"], plan)
        print("GRADED", key, [[(k, v["passed"], v["total"]) for k, v in r["reports"].items()] for r in reports], flush=True)
        return finish(directory, dict(raw, reports=reports))
    except Exception as exc:
        persist(directory, {f"stopped-{time.time_ns()}": {"type": type(exc).__name__, "detail": str(exc)[:2000]}})
        artifacts.commit()
        raise


@app.function(image=gpu_image, gpu="L40S", cpu=(2, 2), memory=(32768, 32768),
              timeout=study.GPU_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def train_arm(arm, plan, setup, first_rollouts, deadline):
    import gc
    import importlib.metadata
    import random
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    study.validate_plan(plan)
    if arm not in study.ARMS:
        raise ValueError("unknown reward arm")
    directory = Path("/artifacts") / plan["run_id"] / "arms" / arm
    intent = {"arm": arm, "plan_hash": digest(canonical_json(plan)), "deadline": deadline}
    saved = begin(directory, "train/" + arm, intent, deadline, run_id=plan["run_id"])
    if saved is not None:
        return saved
    if json.loads((directory.parents[1] / "preflight.json").read_text())["passed"] is not True:
        raise ValueError("live preflight has not passed")
    set_seed(plan["training_seed"])
    if study.is_warm(plan):
        supervisor_controls.validate_controls(json.loads((directory.parents[1] / "supervisor_controls.json").read_text()), setup["sandbox_image_id"])
        checkpoint = Path(plan["initial_checkpoint"])
        receipt = json.loads((checkpoint.parent / "checkpoints/12.json").read_text())
        if receipt != {"path": str(checkpoint), "parameter_hash": plan["initial_parameter_hash"]}:
            raise ValueError("saved step-12 checkpoint receipt differs")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=False, local_files_only=True,
            use_safetensors=True, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
        if (digest(tokenizer.chat_template) != plan["chat_template_hash"]
                or sum(p.numel() for p in model.parameters()) != plan["parameter_count"]
                or parameter_hash(model) != plan["initial_parameter_hash"]):
            raise ValueError("warm-start weights/tokenizer identity differs")
        persist(directory, {"warm_start": {"checkpoint": str(checkpoint), "receipt": receipt,
            "optimizer": "fresh", "additional_steps": study.STEPS, "exact_resume": False}})
    else:
        model, tokenizer = load_policy(plan)
    if torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3:
        raise ValueError("insufficient full-parameter GPU memory")
    persist(directory, {"runtime": {"gpu": torch.cuda.get_device_name(), "gpu_image_id": gpu_image.object_id,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "weights_dtype": "float32", "autocast_dtype": "bfloat16"}})
    evaluations, hashes, calls, initial_raw = {}, {"0": plan["initial_parameter_hash"]}, [], None
    def evaluate(policy, step):
        py, np_state = random.getstate(), np.random.get_state()
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        was_training, cache = policy.training, policy.config.use_cache
        before = parameter_hash(policy)
        try:
            policy_name = arm if step else "baseline"
            samples = generate_samples(policy, tokenizer, plan,
                [(f"eval-{policy_name}-{seed}", seed) for seed in plan["evaluation_seeds"]], directory / f"evaluation-{step:02d}", deadline)
            if parameter_hash(policy) != before:
                raise ValueError("probe generation changed weights")
            evaluations[str(step)], hashes[str(step)] = samples, before
        finally:
            policy.train(was_training)
            policy.config.use_cache = cache
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state_all(cuda)
    if arm == "linear":
        evaluate(model, 0)
    def execution_reward(completions, completion_ids=None, **kwargs):
        nonlocal initial_raw
        require_deadline(deadline)
        index = len(calls)
        if index >= study.STEPS or len(completions) != study.GROUP or completion_ids is None:
            raise ValueError("unexpected rollout schedule")
        texts = [v[0]["content"] if isinstance(v, list) else v for v in completions]
        if index == 0:
            initial_raw = texts
            if first_rollouts is not None and texts != first_rollouts:
                raise ValueError("matched first rollout groups differ")
        key = f"train-{arm}-{index:02d}"
        samples = [study.old.sample_from_text(text, sid=f"{key}-{j}", plan=plan, tokens=len(completion_ids[j]),
                    eos=bool(completion_ids[j] and completion_ids[j][-1] == tokenizer.eos_token_id))
                   for j, text in enumerate(texts)]
        persist(directory / "rollouts", {key: samples})
        artifacts.commit()
        graded = grade_batch.remote(key, samples, "training", plan, setup,
                                    min(deadline, time.time() + study.GRADE_SECONDS - 60))
        artifacts.reload()
        reports = [study.grade(s, graded["records"][s["sample_id"]], "training", setup["sandbox_image_id"], plan) for s in samples]
        study.verify_retries(graded, setup["sandbox_image_id"], plan)
        values = [study.reward(r, arm) for r in reports]
        persist(directory / "rewards", {key: {"rewards": values, "reports": reports}})
        artifacts.commit()
        calls.append(values)
        print("REWARD", arm, "step", index + 1, "case_passes", [r["reports"]["training"]["passed"] for r in reports],
              "rewards", values, flush=True)
        return values
    class Checkpoints(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            require_deadline(deadline)
            if not study.is_warm(plan) and state.global_step in (12, 24):
                path = directory / f"checkpoint-{state.global_step:02d}"
                trainer.save_model(str(path))
                tokenizer.save_pretrained(path)
                hashes[str(state.global_step)] = parameter_hash(model)
                persist(directory / "checkpoints", {str(state.global_step): {"path": str(path), "parameter_hash": hashes[str(state.global_step)]}})
                artifacts.commit()
            return control
        def on_save(self, args, state, control, **kwargs):
            if study.is_warm(plan):
                path = directory / "trainer" / f"checkpoint-{state.global_step}"
                files = checkpoint_inventory(path, state.global_step)
                hashes[str(state.global_step)] = parameter_hash(model)
                persist(directory / "checkpoints", {str(state.global_step): {"path": str(path),
                    "parameter_hash": hashes[str(state.global_step)], "files": files,
                    "full_trainer_state_saved": True, "exact_resume_tested": False}})
                artifacts.commit()
                print("FULL CHECKPOINT SAVED", arm, state.global_step, flush=True)
            return control
    set_seed(plan["training_seed"])
    model.config.use_cache = False
    args = GRPOConfig(**study.trainer_kwargs(plan, directory / "trainer"))
    persist(directory, {"trainer_config": json.loads(args.to_json_string())})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0, betas=(.9, .999), eps=1e-8, foreach=False)
    trainer = GRPOTrainer(model=model, args=args, reward_funcs=execution_reward,
        train_dataset=Dataset.from_list([{"prompt": [{"role": "user", "content": plan["prompt"]}]}] * study.STEPS),
        processing_class=tokenizer, callbacks=[Checkpoints()], optimizers=(optimizer, None))
    persist(directory, {"generation_config": trainer.generation_config.to_dict()})
    artifacts.commit()
    model.train()
    trainer.train()
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("nonfinite weights")
    after = parameter_hash(model)
    evaluate(model, 24)
    metrics = {"global_step": trainer.state.global_step, "rewards": calls, "log_history": trainer.state.log_history,
        "before_parameter_hash": plan["initial_parameter_hash"], "after_parameter_hash": after, "finite_parameters": True,
        "generated_rollout_tokens": sum(s["tokens"] for path in (directory / "rollouts").glob("*.json")
                                        for s in json.loads(path.read_text()))}
    persist(directory, {"before_reload": metrics})
    artifacts.commit()
    del trainer, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    final_checkpoint = directory / "trainer/checkpoint-24" if study.is_warm(plan) else directory / "checkpoint-24"
    reloaded = AutoModelForCausalLM.from_pretrained(final_checkpoint, trust_remote_code=False,
                    dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    metrics["checkpoint_reload_verified"] = parameter_hash(reloaded) == after
    evidence = study.training_evidence(metrics, plan)
    print("TRAINING COMPLETE", arm, evidence, flush=True)
    return finish(directory, {"arm": arm, "metrics": metrics, "evidence": evidence,
                              "evaluations": evaluations, "checkpoint_hashes": hashes, "first_rollouts": initial_raw})


def checkpoint_inventory(path, step):
    if step not in (12, 24):
        raise ValueError("checkpoint outside frozen schedule")
    files = {p.name: p.stat().st_size for p in path.iterdir() if p.is_file()}
    required = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "config.json", "tokenizer.json"}
    if not required.issubset(files) or any(files[name] <= 0 for name in required):
        raise ValueError("incomplete resumable checkpoint files")
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("checkpoint model weights missing")
    if json.loads((path / "trainer_state.json").read_text())["global_step"] != step:
        raise ValueError("checkpoint trainer step differs")
    return files


async def live_supervisor_controls(plan, setup, directory, deadline):
    source_path = Path("/artifacts") / study.RUN_ID / "arms/linear/rollouts/train-linear-14.json"
    samples = json.loads(source_path.read_text())
    source = next(s["source"] for s in samples if s["sample_id"] == supervisor_controls.FAILED_SAMPLE)
    specifications = supervisor_controls.controls(source)
    if len(specifications) != plan["supervisor_controls"]:
        raise ValueError("control count differs from frozen resource bounds")
    document = {"saved_source": source, "records": {}}
    backend = supervised.SupervisedPanelBackend(setup["app_name"], setup["sandbox_image_id"], study.BOOKING,
                                               creation_interval_seconds=.26)
    for name, (candidate, _, _, _) in specifications.items():
        require_deadline(deadline)
        intent = {"source_hash": digest(candidate), "input_hash": supervisor_controls.FAILED_INPUT}
        if not await claims.put.aio(plan["run_id"] + "/supervisor/" + name, intent, skip_if_exists=True):
            raise ReconciliationRequired("runner control already claimed")
        persist(directory / "supervisor" / name, {"intent": intent})
        result = await backend.execute(request_for(candidate, supervisor_controls.case_for_control()))
        document["records"][name] = pack_result(result)
        persist(directory / "supervisor" / name, {"result": pack_result(result)})
        await artifacts.commit.aio()
        print("SUPERVISOR CONTROL", name, result.status.value, result.detail, flush=True)
    supervisor_controls.validate_controls(document, setup["sandbox_image_id"])
    persist(directory, {"supervisor_controls": document})
    await artifacts.commit.aio()
    print("SUPERVISOR CONTROLS PASSED", len(document["records"]), flush=True)


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=study.SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_pilot(plan, setup, samples, budget, snapshot, deadline):
    # The launcher file is Modal's entrypoint, already installed at this path.
    check_snapshot(snapshot, Path("/root"))
    study.validate_plan(plan)
    require_deadline(deadline)
    if not claims.put(plan["run_id"] + "/controller", {"plan": plan, "deadline": deadline}, skip_if_exists=True):
        raise ReconciliationRequired("controller already claimed; do not repeat the run")
    directory = Path("/artifacts") / plan["run_id"]
    artifacts.reload()
    persist(directory, {"plan": plan, "setup": setup, "preflight_samples": samples,
                        "budget": budget, "source_snapshot": snapshot, "deadline": deadline})
    artifacts.commit()
    stage = "live_preflight"
    try:
        owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
        if list(modal.Sandbox.list(app_id=owner.app_id)):
            raise ReconciliationRequired("candidate sandboxes active before pilot")
        if study.is_warm(plan):
            stage = "supervisor_controls"
            asyncio.run(live_supervisor_controls(plan, setup, directory, deadline))
        stage = "live_preflight"
        graded = grade_batch.remote("preflight", samples, "preflight", plan, setup,
                                    min(deadline, time.time() + study.GRADE_SECONDS - 60))
        checked = study.validate_preflight(samples, graded["reports"])
        artifacts.reload()
        persist(directory, {"preflight": checked})
        artifacts.commit()
        print("LIVE PREFLIGHT PASSED", checked, flush=True)
        arms = {}
        for arm in study.ARMS:
            stage = "training_" + arm
            print("STARTING MATCHED ARM", arm, flush=True)
            arms[arm] = train_arm.remote(arm, plan, setup, arms["linear"]["first_rollouts"] if arms else None,
                                        min(deadline, time.time() + study.GPU_SECONDS - 60))
        stage = "independent_evaluation"
        populations = {"baseline": arms["linear"]["evaluations"]["0"],
                       **{a: r["evaluations"]["24"] for a, r in arms.items()}}
        for policy, samples in populations.items():
            reports = []
            for index in range(4):
                graded = grade_batch.remote(f"eval-{policy}-{index:02d}", samples[4 * index:4 * index + 4],
                    "evaluation", plan, setup, min(deadline, time.time() + study.GRADE_SECONDS - 60))
                reports.extend(graded["reports"])
            artifacts.reload()
            summary = study.policy_summary(samples, reports, plan)
            persist(directory / "evaluation_summaries", {policy: summary})
            artifacts.commit()
            print("POLICY AUDIT", policy, summary, flush=True)
        stage = "offline_verification"
        artifacts.reload()
        result = study.verify_run(directory)
        persist(directory, {"result": result})
        artifacts.commit()
        print("MATCHED PILOT COMPLETE", result, flush=True)
        return result
    except Exception as exc:
        persist(directory, {f"stopped-{time.time_ns()}": {"stage": stage, "type": type(exc).__name__, "detail": str(exc)[:2000]}})
        artifacts.commit()
        raise


def read_modal(*args):
    return json.loads(subprocess.run([sys.executable, "-m", "modal", *args, "--json"],
                                    check=True, capture_output=True, text=True, timeout=45).stdout)


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=240, retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def diagnose_runner(allow_cloud: bool = False, wall: bool = False):
    """Two fixed CPU-only runner probes; never used for rewards or training."""
    import ast
    if not allow_cloud:
        raise ValueError("--allow-cloud required")
    run_id = "runner-wall-diagnostic-20260929-v1" if wall else "runner-resource-diagnostic-20260929-v1"
    directory = Path("/artifacts") / run_id
    deadline = time.time() + 210
    intent = {"explicit_poll_yield": [False, True]} if wall else {"cpu_hard_limits": [3, 2]}
    saved = begin(directory, "diagnostic", intent, deadline, run_id=run_id)
    if saved is not None:
        return saved
    setup = json.loads((Path("/artifacts") / study.RUN_ID / "setup.json").read_text())
    async def probe():
        results = {}
        for hard in ((False, True) if wall else (3, 2)):
            require_deadline(deadline)
            child = ast.literal_eval(ast.parse(supervised.supervised_runner(study.BOOKING)).body[0].value)
            child = child.replace('(resource.RLIMIT_CPU, limits["cpu_seconds"]),',
                                  f'(resource.RLIMIT_CPU, (limits["cpu_seconds"], {hard})),')
            marker = 'resource.setrlimit(name, value if isinstance(value, tuple) else (value, value))'
            child = child.replace('resource.setrlimit(name, (value, value))', marker)
            if child.count(marker) != 1:
                raise ValueError("diagnostic bootstrap changed")
            child = child.replace(marker, 'print("SETTING_LIMIT", name, resource.getrlimit(name), value, '
                                  'file=sys.stderr, flush=True)\n        ' + marker)
            backend = supervised.SupervisedPanelBackend(setup["app_name"], setup["sandbox_image_id"], study.BOOKING,
                                                        creation_interval_seconds=.26)
            backend.runner = (f"CHILD_RUNNER = {child!r}\nOUTPUT_LIMIT = 16384\n"
                              f"EXECUTION_VERSION = {supervised.VERSION!r}\n" + supervised.PARENT)
            candidate = "def required_capacity(bookings): return 1"
            if wall:
                candidate = "import time\ntime.sleep(10)\n" + candidate
                backend.runner = supervised.supervised_runner(study.BOOKING)
                backend.runner = backend.runner.replace("        time.sleep(.05)\n", "")
                backend.runner = backend.runner.replace("import selectors\n", "import selectors\nimport resource\n")
                backend.runner = backend.runner.replace("def kill_group():", "loops, last_second = 0, -1\n\ndef kill_group():")
                backend.runner = backend.runner.replace("elapsed = time.monotonic() - started_at", '''elapsed = time.monotonic() - started_at
        loops += 1
        if int(elapsed) > last_second:
            last_second = int(elapsed)
            print("POLL", elapsed, loops, resource.getrusage(resource.RUSAGE_SELF), file=sys.stderr, flush=True)''')
                if hard:
                    backend.runner = backend.runner.replace("while not waited or selector.get_map():\n",
                                                            "while not waited or selector.get_map():\n        time.sleep(.01)\n")
                # Preserve the parent's diagnostic stderr, not only the child's.
                value = await PanelBackend.execute(backend, request_for(candidate, supervisor_controls.case_for_control()))
            else:
                value = await backend.execute(request_for(candidate, supervisor_controls.case_for_control()))
            results[str(hard)] = pack_result(value)
            persist(directory, {f"hard-{hard}": results[str(hard)]})
            await artifacts.commit.aio()
            print("RUNNER DIAGNOSTIC", "wall" if wall else "limits", hard, value.status.value, value.detail,
                  value.metadata.get("stderr_preview"), flush=True)
        return results
    return finish(directory, asyncio.run(probe()))


@app.local_entrypoint()
def launch(allow_cloud: bool = False, warm_start: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required")
    plan = study.make_plan(Path.cwd(), warm_start=warm_start)
    target = Path("runs") / plan["run_id"]
    if (target / "launch_intent.json").exists():
        raise ReconciliationRequired("launch already recorded; inspect saved IDs instead")
    apps = read_modal("app", "list")
    if any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("other Modal tasks active")
    prior_dir = Path("runs/qwen-booking-eval-recovery-20260928-v1/completed-remote/qwen-booking-eval-recovery-20260928-v1")
    documents = json.loads((prior_dir / "source.json").read_text())["documents"]
    original = next(s for s in documents["arms/structured/result.json"]["evaluations"]["12"] if s["sample_id"] == study.SAVED_ID)
    setup = documents["setup.json"]
    samples = study.preflight_samples(original, plan)
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("candidate sandboxes active")
    rates, billing = read_modal("billing", "rates"), read_modal("billing", "summary")
    budget_path = Path("runs") / study.FAILED_WARM_RUN_ID / "budget.json" if warm_start else prior_dir / "budget.json"
    prior = json.loads(budget_path.read_text())["cumulative_reservation_usd"]
    budget = study.budget_quote(plan, rates, billing, prior)
    snapshot = {p.as_posix(): p.read_text() for p in sorted(Path("verifier_rl").glob("*.py"))
                + [Path("modal_booking_study.py"), Path("modal_booking_reward_pilot.py")]}
    check_snapshot(snapshot, Path.cwd())
    deadline = time.time() + study.SECONDS - 60
    persist(target, {"plan": plan, "setup": setup, "preflight_samples": samples, "budget": budget,
        "source_snapshot": snapshot, "protocol": Path("docs/booking_reward_warmstart_protocol.txt" if warm_start else
                                                     "docs/booking_reward_pilot_protocol.txt").read_text(),
        "launch_intent": {"plan_hash": digest(canonical_json(plan)), "deadline": deadline, "observed_apps": apps}})
    call = run_pilot.spawn(plan, setup, samples, budget, snapshot, deadline)
    persist(target, {"launch": {"call_id": call.object_id, "created_utc": datetime.now(timezone.utc).isoformat()}})
    print("MATCHED REWARD PILOT CALL", call.object_id, "ADDITIONAL RESOURCE RESERVATION", budget["additional_reservation_usd"], flush=True)
    persist(target, {"result": call.get()})
