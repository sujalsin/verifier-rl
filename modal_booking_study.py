"""Versioned booking pilots: original gated study and approved two-arm follow-up."""

import asyncio
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time

import modal

from verifier_rl import booking_study as study
from verifier_rl import booking_two_arm as two
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.evaluation_recovery import check_frozen_sources
from verifier_rl.panel_execution import PanelBackend, pack_result, request_for, require_evidence, unpack_result
from verifier_rl.panel_screen import execute_cases
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import BOOKING
from verifier_rl.verifier_quality import compare_output, noise_draw

app = modal.App("verifier-rl-booking-verifiers")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
claims = modal.Dict.from_name("verifier-rl-evaluation-claims", create_if_missing=False)
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
             .add_local_python_source("verifier_rl"))
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("modal==1.5.5", "torch==2.8.0", "transformers==4.57.1",
                          "trl==0.28.0", "datasets==3.5.1", "accelerate==1.12.0")
             .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
             .add_local_python_source("verifier_rl"))


def root_directory(plan=None):
    return Path("/artifacts") / (plan["run_id"] if plan is not None else study.RUN_ID)


def validate_experiment(plan):
    if plan.get("version") == two.VERSION:
        two.validate_plan(plan)
    else:
        study.validate_plan(plan)


def require_deadline(deadline):
    if time.time() >= deadline:
        raise ReconciliationRequired("absolute study deadline reached; no new work")


def begin(directory, key, intent, deadline, *, run_id=study.RUN_ID):
    require_deadline(deadline)
    artifacts.reload()
    if (directory / "result.json").exists():
        persist(directory, {"intent": intent})
        return json.loads((directory / "result.json").read_text())
    if (directory / "intent.json").exists():
        raise ReconciliationRequired("unfinished work preserved, not automatically repeated: " + key)
    persist(directory, {"intent": intent})
    artifacts.commit()
    if not claims.put(run_id + "/work/" + key, digest(canonical_json(intent)), skip_if_exists=True):
        raise ReconciliationRequired("work already claimed: " + key)
    return None


def finish(directory, result):
    persist(directory, {"result": result})
    artifacts.commit()
    return result


def validate_batch(key, samples, role, plan=None):
    expected = (two.validate_batch(key, role) if plan is not None and plan.get("version") == two.VERSION
                else study.batch_ids(key, role))
    if [sample["sample_id"] for sample in samples] != expected:
        raise ValueError("unexpected batch/sample identities")
    if role != "training" and any(sample["seed"] != int(sample["sample_id"].rsplit("-", 1)[1]) for sample in samples):
        raise ValueError("generation seed differs from declared identity")


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=study.GRADING_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def grade_batch(key, samples, role, plan, setup, deadline):
    validate_experiment(plan)
    validate_batch(key, samples, role, plan)
    for sample in samples:
        study.validate_sample(sample, plan)
    directory = root_directory(plan) / "grading" / key
    intent = {"key": key, "samples": samples, "role": role, "setup": setup,
              "plan_hash": digest(canonical_json(plan)), "deadline": deadline, "retry": False}
    saved = begin(directory, "grade/" + key, intent, deadline, run_id=plan["run_id"])
    if saved is not None:
        return saved
    backend = PanelBackend(setup["app_name"], setup["sandbox_image_id"], BOOKING,
                           creation_interval_seconds=.26)
    async def record(sample, case, packed):
        value = {"sample_id": sample["sample_id"], "input_hash": case.input_hash, "execution": packed}
        raw_key = plan["run_id"] + "/raw/" + sample["sample_id"] + "/" + case.input_hash
        if not await claims.put.aio(raw_key, value, skip_if_exists=True):
            raise ReconciliationRequired("raw result already journaled")
        persist(directory / "inputs" / sample["sample_id"], {case.input_hash: value})
    if plan["version"] == two.VERSION:
        async def reserve_startup(first):
            require_deadline(deadline)
            # Atomic slots bound retries across ALL grading calls, not per batch.
            for slot in range(two.MAX_STARTUP_RETRIES):
                intent = {"slot": slot, "batch": key, "first": first}
                if await claims.put.aio(plan["run_id"] + f"/startup-retry/slot/{slot}", intent, skip_if_exists=True):
                    persist(directory / "startup_retry_intents", {str(slot): intent})
                    return slot
            return None
        async def journal_retry(receipt):
            if not await claims.put.aio(plan["run_id"] + f"/startup-retry/result/{receipt['slot']}", receipt,
                                        skip_if_exists=True):
                raise ReconciliationRequired("startup retry already journaled")
            persist(directory / "startup_retries", {str(receipt["slot"]): receipt})
        backend = two.StartupRetryBackend(backend, setup["sandbox_image_id"], reserve_startup, journal_retry)
    records = asyncio.run(study.execute_group(samples, role, backend, setup["sandbox_image_id"], deadline, record))
    # Preserve all evidence even if the validation below refuses a score.
    raw = {"key": key, "samples": samples, "role": role, "records": records,
           "plan_hash": digest(canonical_json(plan))}
    if plan["version"] == two.VERSION:
        raw["startup_retries"] = backend.retries
    persist(directory, {"raw": raw})
    artifacts.commit()
    reports = [study.grade(sample, records[sample["sample_id"]], role, setup["sandbox_image_id"], plan)
               for sample in samples]
    if plan["version"] == two.VERSION:
        two.verify_retries(raw, setup["sandbox_image_id"])
    result = finish(directory, dict(raw, reports=reports))
    print("GRADED", key, [(r["reference"], r["structured"], r.get("audit_passed")) for r in reports], flush=True)
    return result


def parameter_hash(model):
    import hashlib
    value = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value.update(name.encode())
        value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def load_policy(plan):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan["model_id"], revision=plan["revision"], trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if digest(tokenizer.chat_template) != plan["chat_template_hash"]:
        raise ValueError("chat template identity differs")
    model = AutoModelForCausalLM.from_pretrained(plan["model_id"], revision=plan["revision"],
                trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    if (sum(p.numel() for p in model.parameters()) != plan["parameter_count"]
            or parameter_hash(model) != plan["initial_parameter_hash"]):
        raise ValueError("initial model identity differs")
    return model, tokenizer


def generate_samples(model, tokenizer, plan, identities, directory, deadline):
    import torch
    from transformers import set_seed
    model.eval()
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": plan["prompt"]}],
                                              tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")
    samples = []
    for sid, seed in identities:
        require_deadline(deadline)
        persist(directory / "sample-intents", {sid: {"seed": seed, "sample_id": sid}})
        artifacts.commit()
        set_seed(seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = model.generate(**inputs, max_new_tokens=512, do_sample=True,
                temperature=.8, top_p=.95, top_k=0, repetition_penalty=plan["repetition_penalty"],
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
        completion = tokens[0, inputs["input_ids"].shape[1]:]
        sample = study.sample_from_text(tokenizer.decode(completion, skip_special_tokens=True),
            sid=sid, plan=plan, seed=seed, tokens=len(completion),
            eos=bool(len(completion) and completion[-1].item() == tokenizer.eos_token_id))
        study.validate_sample(sample, plan)
        persist(directory / "samples", {sid: sample})
        artifacts.commit()
        samples.append(sample)
        print("GENERATED", sid, "tokens", len(completion), flush=True)
    return samples


@app.function(image=gpu_image, gpu="L40S", cpu=(2, 2), memory=(32768, 32768),
              timeout=study.INITIAL_GPU_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def initial_generation(plan, deadline):
    study.validate_plan(plan)
    directory = root_directory() / "initial_gpu"
    saved = begin(directory, "initial_gpu", {"plan": plan, "deadline": deadline}, deadline)
    if saved is not None:
        return saved
    model, tokenizer = load_policy(plan)
    samples = generate_samples(model, tokenizer, plan,
        [(f"calibration-{seed}", seed) for seed in study.CALIBRATION_SEEDS], directory, deadline)
    if parameter_hash(model) != plan["initial_parameter_hash"]:
        raise ValueError("generation changed parameters")
    return finish(directory, {"samples": samples, "parameters_unchanged": True,
                              "parameter_hash": plan["initial_parameter_hash"], "gpu_image_id": gpu_image.object_id})


@app.function(image=gpu_image, gpu="L40S", cpu=(2, 2), memory=(32768, 32768),
              timeout=study.GPU_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def train_arm(condition, plan, calibration, setup, first_rollouts, deadline):
    import gc
    import importlib.metadata
    import random
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    validate_experiment(plan)
    if plan["version"] == two.VERSION:
        two.require_training(plan, calibration, condition)
    elif condition not in study.CONDITIONS or calibration["ready_for_review"] is not True:
        raise ValueError("approved condition and passed calibration required")
    directory = root_directory(plan) / "arms" / condition
    event_ids = [f"train-{condition}-{i:02d}-{j}" for i in range(study.STEPS) for j in range(study.GROUP)]
    intent = {"plan": plan, "condition": condition, "calibration_hash": digest(canonical_json(calibration)),
              "deadline": deadline, "events": {key: str(noise_draw(study.NOISE_SEED, key)) for key in event_ids}}
    saved = begin(directory, "train/" + condition, intent, deadline, run_id=plan["run_id"])
    if saved is not None:
        return saved
    if json.loads((root_directory(plan) / "calibration.json").read_text()) != calibration:
        raise ValueError("calibration differs from frozen record")
    set_seed(study.TRAIN_SEED)
    model, tokenizer = load_policy(plan)
    runtime = {"gpu": torch.cuda.get_device_name(), "gpu_image_id": gpu_image.object_id,
               "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
               "parameters_dtype": "float32", "autocast_dtype": "bfloat16"}
    if torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3:
        raise ValueError("full-parameter memory allowance unavailable")
    persist(directory, {"runtime": runtime})
    artifacts.commit()
    evaluations, checkpoint_hashes = {}, {}

    def evaluate_checkpoint(policy, step):
        # Preserve ALL training RNG streams; diagnostic generations must not
        # perturb the next on-policy group or differ in timing across arms.
        py_state, np_state = random.getstate(), np.random.get_state()
        cpu_state, cuda_states = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        was_training, use_cache = policy.training, policy.config.use_cache
        before = parameter_hash(policy)
        try:
            samples = generate_samples(policy, tokenizer, plan,
                [(f"eval-{condition if step else 'baseline'}-{step:02d}-{seed}", seed)
                 for seed in study.EVALUATION_SEEDS], directory / f"evaluation-{step:02d}", deadline)
            if parameter_hash(policy) != before:
                raise ValueError("evaluation generation changed parameters")
            evaluations[str(step)] = samples
            checkpoint_hashes[str(step)] = before
        finally:
            policy.train(was_training)
            policy.config.use_cache = use_cache
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_states)

    if condition == "reference":
        evaluate_checkpoint(model, 0)
    calls, rollout_keys, initial_raw = [], [], None
    def execution_reward(completions, completion_ids=None, **kwargs):
        nonlocal initial_raw
        require_deadline(deadline)
        index = len(calls)
        if index >= study.STEPS or len(completions) != study.GROUP or completion_ids is None:
            raise ValueError("unexpected GRPO generation schedule")
        raw = [value[0]["content"] if isinstance(value, list) else value for value in completions]
        if index == 0:
            initial_raw = raw
            if first_rollouts is not None and first_rollouts != raw:
                raise ValueError("first on-policy group differs between initial policies")
        key = f"train-{condition}-{index:02d}"
        group = [study.sample_from_text(text, sid=f"{key}-{j}", plan=plan,
                    tokens=len(completion_ids[j]), eos=bool(completion_ids[j] and completion_ids[j][-1] == tokenizer.eos_token_id))
                 for j, text in enumerate(raw)]
        persist(directory / "rollouts", {key: group})
        artifacts.commit()
        graded = grade_batch.remote(key, group, "training", plan, setup, min(deadline, time.time() + study.GRADING_SECONDS - 60))
        artifacts.reload()
        reports = [study.grade(sample, graded["records"][sample["sample_id"]], "training", setup["sandbox_image_id"], plan)
                   for sample in group]
        if plan["version"] == two.VERSION:
            two.verify_retries(graded, setup["sandbox_image_id"])
        values = [study.reward(condition, report, calibration["promotion_probability"], sample["sample_id"])
                  for sample, report in zip(group, reports)]
        persist(directory / "rewards", {key: {"rewards": values,
            "reports": reports, "draws": {sample["sample_id"]: intent["events"][sample["sample_id"]] for sample in group}}})
        artifacts.commit()
        calls.append(values)
        rollout_keys.append(key)
        print("REWARD", condition, "step", index + 1, values, flush=True)
        return values

    class Checkpoints(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            require_deadline(deadline)
            if state.global_step in study.CHECKPOINTS:
                checkpoint = directory / f"checkpoint-{state.global_step:02d}"
                trainer.save_model(str(checkpoint))
                tokenizer.save_pretrained(checkpoint)
                artifacts.commit()
                evaluate_checkpoint(model, state.global_step)
                persist(directory / "checkpoints", {str(state.global_step): {
                    "parameter_hash": checkpoint_hashes[str(state.global_step)], "path": str(checkpoint)}})
                artifacts.commit()
            return control

    set_seed(study.TRAIN_SEED)
    model.config.use_cache = False
    args = GRPOConfig(**study.trainer_kwargs(directory / "trainer"))
    persist(directory, {"trainer_config": json.loads(args.to_json_string())})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0,
                                  betas=(.9, .999), eps=1e-8, foreach=False)
    trainer = GRPOTrainer(model=model, args=args, reward_funcs=execution_reward,
        train_dataset=Dataset.from_list([{"prompt": [{"role": "user", "content": plan["prompt"]}]}] * study.STEPS),
        processing_class=tokenizer, callbacks=[Checkpoints()], optimizers=(optimizer, None))
    persist(directory, {"rollout_generation_config": trainer.generation_config.to_dict()})
    artifacts.commit()
    model.train()
    trainer.train()
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("nonfinite weights")
    after = parameter_hash(model)
    metrics = {"global_step": trainer.state.global_step, "rewards": calls,
               "log_history": trainer.state.log_history, "before_parameter_hash": study.PARAMETER_HASH,
               "after_parameter_hash": after, "finite_parameters": True,
               "generated_rollout_tokens": sum(json.loads((directory / "rollouts" / f"{key}.json").read_text())[j]["tokens"]
                                               for key in rollout_keys for j in range(study.GROUP))}
    persist(directory, {"before_reload": metrics})
    artifacts.commit()
    del trainer, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(directory / "checkpoint-24", trust_remote_code=False,
                dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    metrics["checkpoint_reload_verified"] = parameter_hash(reloaded) == after
    evidence = study.training_evidence(metrics)
    result = {"condition": condition, "metrics": metrics, "evidence": evidence,
              "evaluations": evaluations, "checkpoint_hashes": checkpoint_hashes,
              "first_rollouts": initial_raw, "rollout_keys": rollout_keys}
    print("TRAINING COMPLETE", condition, evidence, flush=True)
    return finish(directory, result)


def live_controls(plan, setup, deadline):
    directory = root_directory(plan) / "controls"
    selected = study.suites()["training"]
    cases = (selected[8], selected[2], selected[12])  # Empty, overlap, touching endpoints.
    backend = PanelBackend(setup["app_name"], setup["sandbox_image_id"], BOOKING, creation_interval_seconds=.26)
    saved = {}
    for name, source in study.controls().items():
        saved[name] = begin(directory / name, "control/" + name,
                            {"source": source, "inputs": [c.input_hash for c in cases]}, deadline, run_id=plan["run_id"])
    async def run():
        controls = {}
        for name, source in study.controls().items():
            target = directory / name
            if saved[name] is not None:
                controls[name] = saved[name]
                continue
            async def record(case, result):
                value = pack_result(result)
                if not await claims.put.aio(plan["run_id"] + "/control/" + name + "/" + case.input_hash, value, skip_if_exists=True):
                    raise ReconciliationRequired("control input already recorded")
                persist(target / "inputs", {case.input_hash: value})
            records = await execute_cases(source, cases, backend, setup["sandbox_image_id"], deadline, record)
            controls[name] = {"source": source, "records": {key: pack_result(value) for key, value in records.items()}}
            persist(target, {"result": controls[name]})
        return controls
    result = asyncio.run(run())
    artifacts.commit()
    study.validate_controls(result, setup["sandbox_image_id"])
    return result


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=study.CONTROLLER_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_study(plan, setup, budget, snapshot, deadline, continuation=None):
    study.validate_plan(plan)
    check_frozen_sources(snapshot, Path("/root"))
    directory = root_directory()
    control_directory = directory
    if continuation is not None:
        if continuation.get("id") != study.CONTINUATION:
            raise ValueError("only the explicit startup reconciliation is allowed")
        artifacts.reload()
        persist(directory, {"plan": plan, "setup": setup})
        failed_continuation = directory / "continuation-001"
        failures = [json.loads(p.read_text()) for p in directory.glob("stopped-*.json")]
        if (not (failed_continuation / "repair_intent.json").exists()
                or (failed_continuation / "repair.json").exists()
                or not any(f.get("type") == "NameError" and f.get("detail") == "name 'request_for' is not defined"
                           for f in failures)):
            raise ReconciliationRequired("first continuation must be a confirmed pre-submission NameError")
        raw = json.loads((directory / "grading/calibration-00/raw.json").read_text())
        study.startup_repair_target(raw, plan, setup["sandbox_image_id"])
        if (digest(canonical_json(raw)) != continuation["original_raw_hash"]
                or (directory / "arms").exists() or (directory / "calibration.json").exists()
                or {p.name for p in (directory / "grading").iterdir()} != {"calibration-00"}
                or (directory / "grading/calibration-00/result.json").exists()):
            raise ReconciliationRequired("remote work differs from reconciled pre-training stop")
        control_directory = directory / study.CONTINUATION
    intent = {"plan_hash": digest(canonical_json(plan)), "deadline": deadline}
    saved = begin(control_directory, "controller" + ("/" + study.CONTINUATION if continuation else ""), intent, deadline)
    if saved is not None:
        return saved
    persist(control_directory, {"plan": plan, "setup": setup, "budget": budget, "source_snapshot": snapshot})
    if continuation is not None:
        persist(control_directory, {"reconciliation": continuation})
    artifacts.commit()
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("active evaluation sandboxes before launch")
    stage = "controls"
    try:
        controls = (json.loads((directory / "conformance.json").read_text()) if continuation else
                    live_controls(plan, setup, deadline))
        study.validate_controls(controls, setup["sandbox_image_id"])
        persist(directory, {"conformance": controls})
        artifacts.commit()
        print("All 12 booking controls validated", flush=True)
        stage = "calibration_generation"
        if continuation is None:
            initial = initial_generation.remote(plan, min(deadline, time.time() + study.INITIAL_GPU_SECONDS - 60))
        else:
            initial = json.loads((directory / "initial_gpu/result.json").read_text())
            if (digest(canonical_json(initial)) != continuation["initial_result_hash"]
                    or initial["parameters_unchanged"] is not True
                    or initial["parameter_hash"] != study.PARAMETER_HASH):
                raise ValueError("saved initial generations changed")
            sample, case = study.startup_repair_target(raw, plan, setup["sandbox_image_id"])
            persist(control_directory, {"repair_intent": {"sample": sample, "input_hash": case.input_hash,
                    "original_raw_hash": study.PREFLIGHT_RAW_HASH, "max_replacements": 1}})
            artifacts.commit()
            if not claims.put(study.RUN_ID + "/" + study.CONTINUATION + "/repair", True, skip_if_exists=True):
                raise ReconciliationRequired("startup replacement already claimed")
            backend = PanelBackend(setup["app_name"], setup["sandbox_image_id"], BOOKING,
                                   creation_interval_seconds=.26)
            replacement = pack_result(asyncio.run(backend.execute(request_for(sample["source"], case))))
            persist(control_directory, {"repair": replacement})
            artifacts.commit()
            repaired = study.repair_startup_batch(raw, replacement, plan, setup["sandbox_image_id"])
            persist(directory / "grading/calibration-00", {"result": repaired})
            artifacts.commit()
            print("Reused all 32 generations and 187 input records; one preflight-only replacement completed", flush=True)
        artifacts.reload()
        batches, reports = [], []
        stage = "calibration_grading"
        for index in range(8):
            group = initial["samples"][index * 4:(index + 1) * 4]
            key = f"calibration-{index:02d}"
            result = (repaired if continuation is not None and index == 0 else
                      grade_batch.remote(key, group, "calibration", plan, setup,
                                         min(deadline, time.time() + study.GRADING_SECONDS - 60)))
            batches.append(key)
            reports.extend(result["reports"])
        artifacts.reload()
        calibration = study.calibration(initial["samples"], reports, plan)
        persist(directory, {"calibration": calibration})
        artifacts.commit()
        print("CALIBRATION", {k: v for k, v in calibration.items() if k != "rows"}, flush=True)
        if not calibration["ready_for_review"]:
            return finish(directory, {"status": "calibration_inconclusive", "calibration": calibration,
                                       "training_started": False, "automatic_expansion": False})
        arms, first = {}, None
        for condition in study.CONDITIONS:
            stage = "training_" + condition
            print("STARTING ARM", condition, flush=True)
            arms[condition] = train_arm.remote(condition, plan, calibration, setup, first,
                                               min(deadline, time.time() + study.GPU_SECONDS - 60))
            first = arms[condition]["first_rollouts"]
        # No independent evaluation scoring until all optimizer/checkpoint work finishes.
        stage = "independent_evaluation"
        populations = {"baseline-00": arms["reference"]["evaluations"]["0"]}
        for condition, arm in arms.items():
            for step in study.CHECKPOINTS:
                populations[f"{condition}-{step:02d}"] = arm["evaluations"][str(step)]
        summaries = {}
        for name, samples in populations.items():
            reports = []
            for index in range(4):
                key = f"evaluation-{name}-{index:02d}"
                result = grade_batch.remote(key, samples[index * 4:(index + 1) * 4], "evaluation", plan, setup,
                                             min(deadline, time.time() + study.GRADING_SECONDS - 60))
                reports.extend(result["reports"])
            summaries[name] = study.evaluation_summary(samples, reports, calibration["promotion_probability"])
            artifacts.reload()
            persist(directory / "evaluation_summaries", {name: summaries[name]})
            artifacts.commit()
            print("EVALUATED POLICY", name, summaries[name], flush=True)
        result = {"status": "completed", "version": study.VERSION, "calibration": calibration,
                  "policies": summaries, "training": {name: value["evidence"] for name, value in arms.items()},
                  "limitations": plan["limitations"]}
        print("STUDY COMPLETE", result["policies"], flush=True)
        return finish(directory, result)
    except Exception as exc:
        artifacts.reload()
        persist(directory, {f"stopped-{time.time_ns()}": {"stage": stage, "type": type(exc).__name__,
                "detail": str(exc)[:1000], "automatic_retry": False}})
        artifacts.commit()
        raise


@app.local_entrypoint()
def launch(allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for the authorized four-stage pilot")
    target = Path("runs") / study.RUN_ID
    if target.exists():
        raise ReconciliationRequired("study already prepared: inspect saved IDs, do not relaunch")
    plan = study.make_plan(Path.cwd())
    study.validate_plan(plan)
    setup = json.loads(Path("runs/modal-setup-20260925/setup.json").read_text())
    def read_modal(*args):
        return json.loads(subprocess.run([sys.executable, "-m", "modal", *args, "--json"],
                           check=True, capture_output=True, text=True, timeout=45).stdout)
    apps = read_modal("app", "list")
    if any(app["state"] != "stopped" and int(app["tasks"]) for app in apps):
        raise ReconciliationRequired("other app tasks active before study")
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("evaluation sandboxes already active")
    billing, rates = read_modal("billing", "summary"), read_modal("billing", "rates")
    previous = json.loads(Path("runs/qwen-panel-screen-20260928-v1/continuation-001/budget.json").read_text())
    budget = study.budget_quote(plan, rates, billing, previous["cumulative_reserve_usd"])
    snapshot = {p.as_posix(): p.read_text() for p in sorted(Path("verifier_rl").glob("*.py"))
                + [Path("modal_booking_study.py")]}
    deadline = time.time() + study.CONTROLLER_SECONDS - 60
    directory = persist(target, {"plan": plan, "setup": setup, "budget": budget, "source_snapshot": snapshot,
        "deadline": deadline, "protocol": Path("docs/booking_verifier_protocol.txt").read_text()})
    call = run_study.spawn(plan, setup, budget, snapshot, deadline)
    persist(directory, {"launch": {"call_id": call.object_id, "created_utc": datetime.now(timezone.utc).isoformat()}})
    print("BOOKING STUDY CALL", call.object_id, "MAX CUMULATIVE RESERVATION USD", budget["cumulative_reservation_usd"], flush=True)
    result = call.get()
    persist(directory, {"result": result})


@app.local_entrypoint()
def continue_calibration(allow_cloud: bool = False):
    """Resume only the identified, cleaned-up pre-candidate startup failure."""
    from decimal import Decimal
    if not allow_cloud:
        raise ValueError("--allow-cloud required for the reconciled startup continuation")
    target = Path("runs") / study.RUN_ID / study.CONTINUATION
    if target.exists():
        raise ReconciliationRequired("continuation already prepared; do not relaunch")
    old = Path("runs") / study.RUN_ID / "stopped-remote" / study.RUN_ID
    def read(name):
        return json.loads((old / (name + ".json")).read_text())
    plan, setup, original_budget = read("plan"), read("setup"), read("budget")
    study.validate_plan(plan)
    raw, initial = read("grading/calibration-00/raw"), read("initial_gpu/result")
    sample, case = study.startup_repair_target(raw, plan, setup["sandbox_image_id"])
    study.validate_controls(read("conformance"), setup["sandbox_image_id"])
    previous_sources = read("source_snapshot")
    allowed_changes = {"modal_booking_study.py", "verifier_rl/booking_study.py"}
    for path, source in previous_sources.items():
        if path not in allowed_changes and Path(path).read_text() != source:
            raise ValueError("unrelated frozen source changed: " + path)
    def read_modal(*args):
        return json.loads(subprocess.run([sys.executable, "-m", "modal", *args, "--json"],
                           check=True, capture_output=True, text=True, timeout=45).stdout)
    apps = read_modal("app", "list")
    stopped = next(a for a in apps if a["app_id"] == "ap-k1SG5h95cRPLEmcdM9vlbQ")
    if stopped["state"] != "stopped" or any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("original study or another app still has active tasks")
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("evaluation sandboxes are not quiescent")
    old_id = raw["records"][sample["sample_id"]][case.input_hash]["metadata"]["sandbox_id"]
    terminal = modal.Sandbox.from_id(old_id).poll()
    if terminal is None:
        raise ReconciliationRequired("old preflight sandbox is still running")
    deadline = json.loads((Path("runs") / study.RUN_ID / "deadline.json").read_text())
    require_deadline(deadline)  # Keep the ORIGINAL absolute study deadline.
    reconciliation = {"id": study.CONTINUATION, "original_app": stopped,
        "original_raw_hash": digest(canonical_json(raw)), "initial_result_hash": digest(canonical_json(initial)),
        "sandbox_id": old_id, "sandbox_terminal_returncode": terminal, "active_sandbox_ids": [],
        "observed_utc": datetime.now(timezone.utc).isoformat(), "pre_candidate_replacements": 1,
        "candidate_outcome_retries": 0, "additional_generation": 0, "original_deadline_preserved": True}
    rates, billing = read_modal("billing", "rates"), read_modal("billing", "summary")
    extra = (Decimal(120) * (Decimal(str(rates["cpu_hour_cost_sandbox"]))
             + Decimal(str(rates["mem_gib_hour_cost_sandbox"])) / 4)
             + Decimal(3 * study.CONTROLLER_SECONDS) * (Decimal(str(rates["cpu_hour_cost"]))
             + 2 * Decimal(str(rates["mem_gib_hour_cost"])))) / 3600
    previous_hold = json.loads((target.parent / "continuation-001/budget.json").read_text())["cumulative_reservation_usd"]
    budget = {"original": original_budget, "failed_continuation_reservation_held_usd": previous_hold,
        "rates": rates, "billing_before": billing,
        "additional_startup_and_controller_reserve_usd": str(extra),
        "cumulative_reservation_usd": str(max(Decimal(previous_hold),
                                            Decimal(str(billing["metered_cost"]))) + extra),
        "is_invoice": False, "scope": "one pre-candidate replacement; original remaining work and deadline unchanged"}
    snapshot = {p.as_posix(): p.read_text() for p in sorted(Path("verifier_rl").glob("*.py"))
                + [Path("modal_booking_study.py")]}
    persist(target, {"reconciliation": reconciliation, "budget": budget, "source_snapshot": snapshot,
                     "deadline": deadline, "protocol": Path("docs/booking_verifier_protocol.txt").read_text()})
    call = run_study.spawn(plan, setup, budget, snapshot, deadline, reconciliation)
    persist(target, {"launch": {"call_id": call.object_id, "created_utc": datetime.now(timezone.utc).isoformat()}})
    print("BOOKING STARTUP CONTINUATION", call.object_id, "CUMULATIVE RESERVATION USD",
          budget["cumulative_reservation_usd"], flush=True)
    persist(target, {"result": call.get()})


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=study.CONTROLLER_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_two_arm(plan, setup, budget, snapshot, deadline):
    two.validate_plan(plan)
    check_frozen_sources(snapshot, Path("/root"))
    directory = root_directory(plan)
    saved = begin(directory, "two-arm-controller", {"plan": plan, "deadline": deadline}, deadline, run_id=plan["run_id"])
    if saved is not None:
        return saved
    stage = "parent_verification"
    try:
        parent_directory = Path("/artifacts") / study.RUN_ID
        parent = study.verify_run(parent_directory)
        if (json.loads((parent_directory / "plan.json").read_text()) != plan["parent_plan"]
                or json.loads((parent_directory / "setup.json").read_text()) != setup):
            raise ValueError("source plan or execution image changed")
        calibration = parent["calibration"]
        two.validate_plan(plan, calibration)
        persist(directory, {"plan": plan, "setup": setup, "budget": budget, "source_snapshot": snapshot,
            "parent_verification": parent, "calibration": calibration, "training_gate": two.training_gate(calibration)})
        artifacts.commit()
        owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
        if list(modal.Sandbox.list(app_id=owner.app_id)):
            raise ReconciliationRequired("evaluation sandboxes active before two-arm launch")
        stage = "controls"
        controls = live_controls(plan, setup, deadline)
        persist(directory, {"conformance": controls})
        artifacts.commit()
        print("TWO-ARM GATE PASSED; old calibration remains q=0; 12 live controls validated", flush=True)
        arms, first = {}, None
        for condition in two.CONDITIONS:
            stage = "training_" + condition
            print("STARTING TWO-ARM TRAINING", condition, flush=True)
            arms[condition] = train_arm.remote(condition, plan, calibration, setup, first,
                                               min(deadline, time.time() + study.GPU_SECONDS - 60))
            first = arms[condition]["first_rollouts"]
        stage = "independent_evaluation"
        populations = {"baseline-00": arms["reference"]["evaluations"]["0"]}
        populations.update({f"{condition}-{step:02d}": arm["evaluations"][str(step)]
                            for condition, arm in arms.items() for step in study.CHECKPOINTS})
        summaries = {}
        for name, samples in populations.items():
            reports = []
            for index in range(4):
                key = f"evaluation-{name}-{index:02d}"
                result = grade_batch.remote(key, samples[index * 4:(index + 1) * 4], "evaluation", plan, setup,
                                             min(deadline, time.time() + study.GRADING_SECONDS - 60))
                reports.extend(result["reports"])
            summaries[name] = two.summary(samples, reports)
            artifacts.reload()
            persist(directory / "evaluation_summaries", {name: summaries[name]})
            artifacts.commit()
            print("EVALUATED TWO-ARM POLICY", name, summaries[name], flush=True)
        result = {"status": "completed", "version": two.VERSION, "calibration": calibration,
                  "policies": summaries, "training": {name: value["evidence"] for name, value in arms.items()},
                  "random_arm_enabled": False, "limitations": plan["limitations"]}
        print("TWO-ARM STUDY COMPLETE", result["policies"], flush=True)
        return finish(directory, result)
    except Exception as exc:
        artifacts.reload()
        persist(directory, {f"stopped-{time.time_ns()}": {"stage": stage, "type": type(exc).__name__,
                "detail": str(exc)[:1000], "automatic_controller_retry": False}})
        artifacts.commit()
        raise


@app.local_entrypoint()
def launch_two_arm(allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for the approved strict-versus-weak GRPO comparison")
    target = Path("runs") / two.RUN_ID
    if target.exists():
        raise ReconciliationRequired("two-arm study already prepared; inspect saved IDs instead of relaunching")
    parent_directory = Path("runs") / study.RUN_ID / "completed-remote" / study.RUN_ID
    parent = study.verify_run(parent_directory)
    parent_plan = json.loads((parent_directory / "plan.json").read_text())
    plan = two.make_plan(parent_plan, parent["calibration"])
    setup = json.loads((parent_directory / "setup.json").read_text())
    def read_modal(*args):
        return json.loads(subprocess.run([sys.executable, "-m", "modal", *args, "--json"],
                           check=True, capture_output=True, text=True, timeout=45).stdout)
    apps = read_modal("app", "list")
    if any(int(app["tasks"]) for app in apps):
        raise ReconciliationRequired("other Modal tasks are active")
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("evaluation sandboxes are active")
    rates, billing = read_modal("billing", "rates"), read_modal("billing", "summary")
    previous = json.loads((parent_directory / study.CONTINUATION / "budget.json").read_text())
    budget = two.budget_quote(plan, rates, billing, previous["cumulative_reservation_usd"])
    snapshot = {p.as_posix(): p.read_text() for p in sorted(Path("verifier_rl").glob("*.py"))
                + [Path("modal_booking_study.py")]}
    deadline = time.time() + study.CONTROLLER_SECONDS - 60
    persist(target, {"plan": plan, "setup": setup, "budget": budget, "source_snapshot": snapshot,
        "deadline": deadline, "parent_verification": parent,
        "protocol": Path("docs/booking_two_arm_protocol.txt").read_text()})
    call = run_two_arm.spawn(plan, setup, budget, snapshot, deadline)
    persist(target, {"launch": {"call_id": call.object_id, "created_utc": datetime.now(timezone.utc).isoformat()}})
    print("TWO-ARM STUDY CALL", call.object_id, "MAX CUMULATIVE RESERVATION USD", budget["cumulative_reservation_usd"], flush=True)
    persist(target, {"result": call.get()})
