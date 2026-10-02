"""Opt-in full-state control and matched original-Qwen verifier training."""

import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import modal

from modal_booking_study import artifacts, claims, cpu_image as base_cpu, load_policy, parameter_hash, require_deadline
from modal_booking_baseline_comparison import check_snapshot, live_controls, read_modal
from verifier_rl import booking_baseline_comparison as study, booking_matched_training as release
from verifier_rl import grpo_recovery as recovery
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import canonical_json, digest

app = modal.App("verifier-rl-booking-matched-training")


def with_launchers(image):
    for name in ("modal_booking_study.py", "modal_booking_baseline_comparison.py", "modal_booking_matched_training.py"):
        image = image.add_local_file(name, "/root/" + name)
    return image


cpu_image = with_launchers(base_cpu)
# Build settings precede local source mounts. The workspace setting is present
# before Torch/CUDA initializes, including on a replacement GPU worker.
gpu_image = with_launchers(modal.Image.debian_slim(python_version="3.12")
    .pip_install("modal==1.5.5", "torch==2.8.0", "transformers==4.57.1",
                 "trl==0.28.0", "datasets==3.5.1", "accelerate==1.12.0")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
          "CUBLAS_WORKSPACE_CONFIG": recovery.CUDA_WORKSPACE})
    .add_local_python_source("verifier_rl"))
read_json = recovery.read_json


def claim(key, maximum=1, *, namespace=release.RUN_ID):
    for slot in range(maximum):
        if claims.put(namespace + "/starts/" + key + f"/{slot}", {"utc": datetime.now(timezone.utc).isoformat()}, skip_if_exists=True):
            return slot
    raise ReconciliationRequired("bounded invocation limit reached: " + key)


def runtime():
    import torch
    return {"gpu": torch.cuda.get_device_name(), "gpu_image_id": gpu_image.object_id,
            "determinism": recovery.deterministic_training(),
            "packages": {p: importlib.metadata.version(p) for p in recovery.PACKAGES}}


def build_trainer(directory, plan, reward, *, control=False, after_checkpoint=None):
    import gc
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig
    experiment = plan["experiment"]
    binding = {"release_hash": digest(canonical_json(plan)), "directory": str(directory), "control": control}
    binding_hash = digest(canonical_json(binding))
    recovery.deterministic_training()
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(experiment["training_seed"])
    with ProgressLog(directory.name, label="MODEL") as progress:
        progress.stage("load_original_policy")
        model, tokenizer = load_policy(experiment)
    model.config.use_cache = False
    kwargs = study.trainer_kwargs(directory / "trainer")
    # HF's default embeds the wall-clock time and breaks immutable resume checks.
    kwargs["logging_dir"] = str(directory / "logs")
    if control:
        kwargs["max_steps"] = release.CONTROL_STEPS
    args = GRPOConfig(**kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0, betas=(.9, .999), eps=1e-8, foreach=False)
    trainer = recovery.trainer_class()(model=model, args=args, reward_funcs=reward,
        train_dataset=Dataset.from_list([{"prompt": [{"role": "user", "content": experiment["prompt"]}]}]*study.STEPS),
        processing_class=tokenizer, optimizers=(optimizer, None), journal=directory / "journal",
        binding_hash=binding_hash, initial_parameter_hash=experiment["initial_parameter_hash"],
        parameter_hash=parameter_hash, commit=artifacts.commit, after_checkpoint=after_checkpoint)
    persist(directory, {"binding": binding, "trainer_config": json.loads(args.to_json_string()),
                        "generation_config": trainer.generation_config.to_dict()})
    artifacts.commit()
    checkpoint = recovery.latest_checkpoint(directory / "trainer", binding_hash)
    return trainer, tokenizer, checkpoint


class ControlledInterruption(Exception):
    pass


def control_result(directory, trainer, interrupted):
    steps = trainer.state.global_step
    boundaries = {str(i): read_json(directory / f"journal/boundaries/{i}.json") for i in range(1, steps+1)}
    groups = {}
    for path in sorted((directory / "journal").glob("group-*/generation.json")):
        group = int(path.parent.name.split("-")[-1])
        saved = read_json(path)
        reward = read_json(path.parent / "reward.json")
        groups[str(group)] = {"tokens_hash": saved["tokens_hash"], "reward_hash": digest(canonical_json(reward)),
                              "after_rng_hash": digest(canonical_json(saved["after_rng"]))}
    return {"steps": steps, "boundaries": boundaries, "groups": groups,
            "interruption_observed": interrupted, "replayed_groups": trainer.replayed_groups,
            "nonzero_gradient_steps": sum(row.get("grad_norm", 0) > 0 for row in trainer.state.log_history)}


@app.function(image=gpu_image, gpu="L40S", cpu=(2,2), memory=(32768,32768),
              timeout=release.CONTROL_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_control_part(part, plan, snapshot, deadline):
    release.validate_plan(plan)
    check_snapshot(snapshot, Path("/root"))
    require_deadline(deadline)
    if part not in plan["control_parts"]:
        raise ValueError("unknown control part")
    artifacts.reload()
    root = Path("/artifacts") / plan["control_id"]
    if (root / f"{part}.json").exists():
        return read_json(root / f"{part}.json")
    claim("control/" + part, namespace=plan["control_id"])
    directory = root / ("uninterrupted" if part == "uninterrupted" else "interrupted")
    if part == "resume" and not (root / "interrupt.json").exists():
        raise ValueError("controlled interruption required before resume")
    persist(root, {"plan": plan, "source_snapshot": snapshot})
    persist(root / "runtimes", {part: runtime()})
    def reward(completions, completion_ids=None, trainer_state=None, **kwargs):
        require_deadline(deadline)
        index = trainer_state.global_step
        if len(completions) != 4 or completion_ids is None or index >= 3:
            raise ValueError("control rollout schedule differs")
        target = directory / f"journal/group-{index:02d}"
        generation = read_json(target / "generation.json")
        if generation["output"][1] != completion_ids:
            raise ValueError("reward token identity differs")
        record = {"completion_ids": completion_ids, "completions": completions, "rewards": plan["control_reward"]}
        persist(target, {"reward": record})
        artifacts.commit()
        if part == "interrupt" and index == 1:
            raise ControlledInterruption("pending second group saved before optimizer update")
        return record["rewards"]

    trainer, tokenizer, checkpoint = build_trainer(directory, plan, reward, control=True)
    if (part == "resume") != (checkpoint is not None):
        raise ValueError("unexpected control checkpoint")
    interrupted = False
    try:
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    except ControlledInterruption:
        interrupted = True
    result = control_result(directory, trainer, interrupted)
    persist(root, {part: result})
    artifacts.commit()
    print("RECOVERY CONTROL PART COMPLETE", part, "steps", result["steps"], "reused", result["replayed_groups"], flush=True)
    return result


@app.function(image=cpu_image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=3*release.CONTROL_SECONDS+300, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_control(plan, snapshot, deadline):
    release.validate_plan(plan)
    check_snapshot(snapshot, Path("/root"))
    claim("control-controller", namespace=plan["control_id"])
    parts = {}
    for part in plan["control_parts"]:
        print("RECOVERY CONTROL STARTING", part, flush=True)
        parts[part] = run_control_part.remote(part, plan, snapshot, deadline)
    result = release.validate_control(parts, plan)
    artifacts.reload()
    persist(Path("/artifacts") / plan["control_id"], {"result": result})
    artifacts.commit()
    print("LIVE FULL-STATE CONTROL PASSED", result, flush=True)
    return result


@app.local_entrypoint()
def control(allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required")
    target = Path("runs") / release.CONTROL_ID
    if (target / "launch_intent.json").exists():
        raise ReconciliationRequired("control already launched; inspect it instead of repeating")
    apps = read_modal("app", "list")
    if any(int(row["tasks"]) for row in apps):
        raise ReconciliationRequired("other Modal jobs active")
    plan = release.make_plan(Path.cwd())
    names = sorted(Path("verifier_rl").glob("*.py")) + [Path(n) for n in (
        "modal_booking_study.py", "modal_booking_baseline_comparison.py", "modal_booking_matched_training.py")]
    snapshot = {p.as_posix(): p.read_text() for p in names}
    deadline = time.time() + 3*release.CONTROL_SECONDS
    budget = release.quote(plan, read_modal("billing", "rates"), read_modal("billing", "summary"))
    persist(target, {"plan": plan, "source_snapshot": snapshot, "budget": budget,
                    "launch_intent": {"deadline": deadline, "apps": apps}})
    call = run_control.spawn(plan, snapshot, deadline)
    persist(target, {"launch": {"call_id": call.object_id, "created_utc": datetime.now(timezone.utc).isoformat()}})
    print("RECOVERY CONTROL CALL", call.object_id, "THREE GPU CALLS MAX; NO RESEARCH TRAINING YET", flush=True)
    result = call.get()
    persist(target, {"result": result})
    print("CONTROL RESULT", result, flush=True)


def require_live_control(plan, snapshot, root=Path("/artifacts")):
    """Recompute the gate, and bind it to the tested recovery/trainer code."""
    import ast
    directory = root / plan["control_id"]
    if read_json(directory / "plan.json") != plan:
        raise ValueError("control used different release settings")
    old_sources = read_json(directory / "source_snapshot.json")
    if old_sources["verifier_rl/grpo_recovery.py"] != snapshot["verifier_rl/grpo_recovery.py"]:
        raise ValueError("recovery implementation differs from live control")
    def function(source, name):
        return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name))
    launcher = "modal_booking_matched_training.py"
    if function(old_sources[launcher], "build_trainer") != function(snapshot[launcher], "build_trainer"):
        raise ValueError("trainer construction differs from live control")
    for name in ("modal_booking_study.py", "verifier_rl/booking_baseline_comparison.py"):
        if old_sources[name] != snapshot[name]:
            raise ValueError("model/scoring dependency differs from live control: " + name)
    parts = {part: read_json(directory / f"{part}.json") for part in plan["control_parts"]}
    checked = release.validate_control(parts, plan)
    if read_json(directory / "result.json") != checked:
        raise ValueError("control report changed")
    return checked


@app.function(image=cpu_image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=release.GRADE_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def grade_batch(key, samples, role, plan, setup, deadline):
    from verifier_rl import supervised_execution as supervised
    from verifier_rl.durable_grading import execute_batch
    release.validate_plan(plan)
    study.validate_batch(key, samples, role, plan["experiment"])
    require_deadline(deadline)
    artifacts.reload()
    directory = Path("/artifacts") / plan["run_id"] / "grading" / key
    if (directory / "result.json").exists():
        saved = read_json(directory / "result.json")
        if saved["checked"] != study.verify_raw(saved["raw"], samples, key, role, plan["experiment"], setup["sandbox_image_id"]):
            raise ValueError("saved grading report changed")
        return saved
    claim("grading/" + key, plan["max_grading_starts_per_batch"])
    backend = supervised.SupervisedPanelBackend(setup["app_name"], setup["sandbox_image_id"], study.BOOKING,
                                                creation_interval_seconds=plan["experiment"]["creation_interval_seconds"])
    try:
        entries, progress = asyncio.run(execute_batch(samples, study.cases_for(role), directory, key,
            release.execution_plan(plan), setup["sandbox_image_id"], deadline, backend, claims, artifacts.commit.aio))
        raw = {"key": key, "role": role, "samples": samples, "entries": entries,
               "plan_hash": digest(canonical_json(plan["experiment"]))}
        persist(directory, {"raw": raw})
        artifacts.commit()
        checked = study.verify_raw(raw, samples, key, role, plan["experiment"], setup["sandbox_image_id"])
        result = {"raw": raw, "checked": checked}
        persist(directory, {"result": result, "execution_progress": progress})
        artifacts.commit()
        print("MATCHED GRADED", key, [(r["reference"]["passed_bounds"], r["endpoint_omission"]["passed_bounds"],
            r.get("audit", {}).get("passed_bounds")) for r in checked["rows"]], flush=True)
        return result
    except Exception as exc:
        persist(directory, {f"interrupted-{time.time_ns()}": {"type": type(exc).__name__, "detail": str(exc)[:2000]}})
        artifacts.commit()
        raise


def evaluation_at_boundary(trainer, tokenizer, arm, step, directory, plan, deadline):
    """Fixed-seed draws at declared weights; preserve training RNG and mode."""
    import torch
    from transformers import set_seed
    experiment = plan["experiment"]
    target = directory / f"evaluation-{step:02d}"
    parameter_identity = trainer.boundary_hash
    if (target / "result.json").exists():
        result = read_json(target / "result.json")
        if result["parameter_hash"] != parameter_identity:
            raise ValueError("saved evaluation checkpoint differs")
        return result
    rng, was_training, cache = recovery.capture_rng(), trainer.model.training, trainer.model.config.use_cache
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    samples = []
    try:
        # Keep the baseline's inference flags; this does not relax the exact
        # training/recovery control or promise cross-container bitwise sampling.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        trainer.model.eval()
        with ProgressLog(f"{arm}-{step}", label="EVALUATION_GENERATION") as progress:
            progress.stage("save_fixed_checkpoint")
            snapshot = directory / f"model-{step:02d}"
            receipt = {"parameter_hash": parameter_identity, "step": step}
            if (snapshot / "receipt.json").exists():
                if read_json(snapshot / "receipt.json") != receipt:
                    raise ValueError("snapshot identity changed")
            else:
                trainer.save_model(str(snapshot))
                tokenizer.save_pretrained(snapshot)
                persist(snapshot, {"receipt": receipt})
                artifacts.commit()
            formatted = tokenizer.apply_chat_template([{"role": "user", "content": experiment["prompt"]}],
                                                       tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")
            progress.stage("generate_fixed_samples", total=32, unit="programs")
            for sid, seed in study.identities(arm, step):
                require_deadline(deadline)
                progress.item(sid)
                sample_path = target / "samples" / f"{sid}.json"
                intent = {"sample_id": sid, "seed": seed, "parameter_hash": parameter_identity}
                if sample_path.exists():
                    if read_json(target / "intents" / f"{sid}.json") != intent:
                        raise ValueError("evaluation intent changed")
                    sample = read_json(sample_path)
                else:
                    if (target / "intents" / f"{sid}.json").exists():
                        raise ReconciliationRequired("evaluation intent without saved sample; do not resample")
                    persist(target / "intents", {sid: intent})
                    artifacts.commit()
                    set_seed(seed)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        tokens = trainer.model.generate(**inputs, max_new_tokens=512, do_sample=True,
                            temperature=experiment["temperature"], top_p=experiment["top_p"], top_k=experiment["top_k"],
                            repetition_penalty=experiment["repetition_penalty"], pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id, use_cache=True)
                    completion = tokens[0, inputs["input_ids"].shape[1]:]
                    sample = study.original.sample_from_text(tokenizer.decode(completion, skip_special_tokens=True),
                        sid=sid, seed=seed, plan=experiment, tokens=len(completion),
                        eos=bool(len(completion) and completion[-1].item() == tokenizer.eos_token_id))
                    persist(target / "samples", {sid: sample})
                    artifacts.commit()
                samples.append(sample)
                progress.advance()
                print("MATCHED EVALUATION GENERATED", sid, "tokens", sample["tokens"], flush=True)
            if parameter_hash(trainer.model) != parameter_identity:
                raise ValueError("evaluation changed weights")
            result = {"parameter_hash": parameter_identity, "samples": samples}
            persist(target, {"result": result})
            progress.stage("commit_evaluation_samples")
            artifacts.commit()
            return result
    finally:
        trainer.model.train(was_training)
        trainer.model.config.use_cache = cache
        recovery.restore_rng(rng)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic = cudnn_deterministic


@app.function(image=gpu_image, gpu="L40S", cpu=(2,2), memory=(32768,32768),
              timeout=release.TRAIN_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def train_arm(arm, plan, setup, snapshot, first_tokens, deadline):
    import torch
    from transformers import TrainerCallback
    release.validate_plan(plan)
    check_snapshot(snapshot, Path("/root"))
    require_deadline(deadline)
    if arm not in study.ARMS:
        raise ValueError("unknown training arm")
    artifacts.reload()
    require_live_control(plan, snapshot)
    root = Path("/artifacts") / plan["run_id"]
    if read_json(root / "preflight.json")["passed"] is not True:
        raise ValueError("live verifier controls not passed")
    directory = root / "arms" / arm
    if (directory / "result.json").exists():
        return release.validate_arm(read_json(directory / "result.json"), plan)
    slot = claim("training/" + arm, plan["max_train_starts_per_arm"])
    persist(directory / "runtimes", {str(slot): runtime()})
    experiment, holder = plan["experiment"], {}

    def reward(completions, completion_ids=None, trainer_state=None, **kwargs):
        require_deadline(deadline)
        index = trainer_state.global_step
        if not 0 <= index < 24 or len(completions) != 4 or completion_ids is None:
            raise ValueError("unexpected rollout schedule")
        target = directory / f"journal/group-{index:02d}"
        generated = read_json(target / "generation.json")
        if generated["output"][1] != completion_ids:
            raise ValueError("reward does not belong to saved rollout tokens")
        if index == 0 and first_tokens is not None and completion_ids != first_tokens:
            raise ValueError("matched first rollout groups differ")
        key = f"train-{arm}-{index:02d}"
        texts = [v[0]["content"] if isinstance(v, list) else v for v in completions]
        samples = [study.original.sample_from_text(text, sid=f"{key}-{j}", plan=experiment,
            tokens=len(completion_ids[j]), eos=bool(completion_ids[j] and completion_ids[j][-1] == holder["tokenizer"].eos_token_id))
            for j, text in enumerate(texts)]
        persist(target, {"samples": samples})
        artifacts.commit()
        with ProgressLog(f"{arm}-{index+1}", label="TRAINING_REWARD") as progress:
            progress.stage("sandbox_grading", total=4*96, unit="input_slots")
            graded = grade_batch.remote(key, samples, "training", plan, setup, deadline)
            artifacts.reload()
            progress.stage("validate_training_rewards")
            values = study.rewards_from_raw(graded["raw"], samples, key, arm, experiment, setup["sandbox_image_id"])
        record = {"completion_ids": completion_ids, "samples": samples, "rewards": values,
                  "raw_hash": digest(canonical_json(graded["raw"]))}
        persist(target, {"reward": record})
        artifacts.commit()
        print("MATCHED REWARD", arm, "update", index+1, "rewards", values, flush=True)
        return values

    def after_checkpoint(trainer, path, receipt):
        if receipt["step"] in (12,24):
            evaluation_at_boundary(trainer, holder["tokenizer"], arm, receipt["step"], directory, plan, deadline)

    trainer, tokenizer, checkpoint = build_trainer(directory, plan, reward, after_checkpoint=after_checkpoint)
    holder["tokenizer"] = tokenizer

    class ResumeEvaluation(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step in (12,24):
                evaluation_at_boundary(trainer, tokenizer, arm, state.global_step, directory, plan, deadline)
            return control

    trainer.add_callback(ResumeEvaluation())
    print("MATCHED TRAINING START", arm, "checkpoint", str(checkpoint) if checkpoint else "original/fresh optimizer", flush=True)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    if trainer.state.global_step != 24 or not all(torch.isfinite(p).all().item() for p in trainer.model.parameters()):
        raise ValueError("training did not finish 24 finite updates")
    final_hash = parameter_hash(trainer.model)
    trainer._load_from_checkpoint(str(directory / "trainer/checkpoint-24"))
    evaluations = {str(step): read_json(directory / f"evaluation-{step:02d}/result.json") for step in (12,24)}
    boundaries = {str(step): read_json(directory / f"journal/boundaries/{step}.json") for step in range(1,25)}
    result = {"arm": arm, "evaluations": evaluations, "boundaries": boundaries,
              "first_tokens": read_json(directory / "journal/group-00/generation.json")["output"][1],
              "metrics": {"global_step": trainer.state.global_step, "log_history": trainer.state.log_history,
                  "before_parameter_hash": experiment["initial_parameter_hash"], "after_parameter_hash": final_hash,
                  "parameters_changed": final_hash != experiment["initial_parameter_hash"], "finite_parameters": True,
                  "final_reload_verified": parameter_hash(trainer.model) == final_hash,
                  "nonzero_gradient_steps": sum(r.get("grad_norm", 0) > 0 for r in trainer.state.log_history),
                  "generated_training_tokens": sum(len(ids) for i in range(24) for ids in
                      read_json(directory / f"journal/group-{i:02d}/generation.json")["output"][1])}}
    release.validate_arm(result, plan)
    persist(directory, {"result": result})
    artifacts.commit()
    print("MATCHED TRAINING COMPLETE", arm, result["metrics"], flush=True)
    return result


def verify_policy_journals(directory, plan, arm, result, progress):
    """Link returned arm summaries back to individual checkpoint/rollout records."""
    target = directory / "arms" / arm
    binding = {"release_hash": digest(canonical_json(plan)), "directory": str(target), "control": False}
    if read_json(target / "binding.json") != binding:
        raise ValueError("training identity changed")
    binding_hash = digest(canonical_json(binding))
    previous = plan["experiment"]["initial_parameter_hash"]
    progress.stage("verify_training_journals", total=24, unit="updates", arm=arm)
    for index in range(24):
        progress.item(f"{arm}/group-{index:02d}")
        group = target / f"journal/group-{index:02d}"
        generated = read_json(group / "generation.json")
        intent = read_json(group / "intent.json")
        consumed = read_json(group / "reward.json")
        boundary = read_json(target / f"journal/boundaries/{index+1}.json")
        if (generated["binding"] != intent or intent["binding_hash"] != binding_hash
                or intent["version"] != recovery.VERSION or intent["step"] != index
                or intent["parameter_hash"] != previous
                or generated["tokens_hash"] != digest(canonical_json(generated["output"]))
                or consumed["completion_ids"] != generated["output"][1]
                or read_json(group / "samples.json") != consumed["samples"]
                or boundary != result["boundaries"][str(index+1)]
                or boundary["binding_hash"] != binding_hash or boundary["step"] != index+1
                or boundary["trl_step"] != 4*(index+1)):
            raise ValueError("training rollout/checkpoint journal changed")
        if index == 0 and result["first_tokens"] != generated["output"][1]:
            raise ValueError("first rollout identity changed")
        previous = boundary["parameter_hash"]
        progress.advance()
    progress.stage("verify_evaluation_journals", total=64, unit="programs", arm=arm)
    for step in (12,24):
        evaluation = result["evaluations"][str(step)]
        path = target / f"evaluation-{step:02d}"
        if (read_json(path / "result.json") != evaluation or
                read_json(target / f"model-{step:02d}/receipt.json") != {
                    "step": step, "parameter_hash": evaluation["parameter_hash"]}):
            raise ValueError("fixed evaluation checkpoint journal changed")
        for sample in evaluation["samples"]:
            sid = sample["sample_id"]
            progress.item(sid)
            if (read_json(path / "samples" / f"{sid}.json") != sample or
                    read_json(path / "intents" / f"{sid}.json") != {
                        "sample_id": sid, "seed": sample["seed"], "parameter_hash": evaluation["parameter_hash"]}):
                raise ValueError("evaluation generation journal changed")
            progress.advance()


def verify_comparison(directory, plan, setup, arms, baseline, progress):
    from verifier_rl import supervisor_controls
    experiment = plan["experiment"]
    ids = supervisor_controls.validate_controls(read_json(directory / "supervisor_controls.json"), setup["sandbox_image_id"])
    starts, slots, policies = len(ids), [], {}
    batches = [("controls", "training", study.controls(experiment), None)]
    for arm in study.ARMS:
        release.validate_arm(arms[arm], plan)
        verify_policy_journals(directory, plan, arm, arms[arm], progress)
        for index in range(24):
            samples = read_json(directory / f"arms/{arm}/journal/group-{index:02d}/samples.json")
            batches.append((f"train-{arm}-{index:02d}", "training", samples, arm))
        for step in (12,24):
            name = f"{arm}-{step}"
            policies[name] = []
            samples = arms[arm]["evaluations"][str(step)]["samples"]
            for i in range(8):
                batches.append((f"eval-{arm}-{step:02d}-{i:02d}", "evaluation", samples[4*i:4*i+4], name))
    if arms["reference"]["first_tokens"] != arms["endpoint_omission"]["first_tokens"]:
        raise ValueError("matched first rollout groups differ")
    for index, (key, role, samples, owner) in enumerate(batches, 1):
        position = {"batch": key, "batch_index": index, "batches_total": len(batches)}
        progress.stage("recompute_batch_scores", **position)
        raw = read_json(directory / f"grading/{key}/raw.json")
        checked = study.verify_raw(raw, samples, key, role, experiment, setup["sandbox_image_id"])
        if read_json(directory / f"grading/{key}/result.json") != {"raw": raw, "checked": checked}:
            raise ValueError("saved batch differs from replay")
        ids.extend(checked["sandbox_ids"])
        starts += checked["submitted_attempts"]
        if key == "controls":
            if study.validate_controls(checked) != read_json(directory / "preflight.json"):
                raise ValueError("preflight evidence changed")
        elif role == "training":
            step_index = int(key.rsplit("-",1)[1])
            consumed = read_json(directory / f"arms/{owner}/journal/group-{step_index:02d}/reward.json")
            rewards = study.rewards_from_raw(raw, samples, key, owner, experiment, setup["sandbox_image_id"])
            if consumed["rewards"] != rewards or consumed["samples"] != samples or consumed["raw_hash"] != digest(canonical_json(raw)):
                raise ValueError("consumed training rewards differ from execution")
        else:
            policies[owner].extend(checked["rows"])
        total = sum(len(entry) for inputs in raw["entries"].values() for entry in inputs.values())
        progress.stage("verify_input_journals", total=total, unit="files", **position)
        for sid, inputs in raw["entries"].items():
            for h, entry in inputs.items():
                for name, value in entry.items():
                    relative = f"grading/{key}/inputs/{sid}/{h}/{name}.json"
                    progress.item(relative)
                    if read_json(directory / relative) != value:
                        raise ValueError("individual execution journal differs from batch")
                    progress.advance()
                if "intent-2" in entry:
                    slot = entry["intent-2"]["retry_slot"]
                    if type(slot) is not int or not 0 <= slot < experiment["max_startup_retries"]:
                        raise ValueError("invalid startup replacement slot")
                    slots.append(slot)
    if len(ids) != len(set(ids)) or len(slots) != len(set(slots)) or starts > plan["max_sandbox_starts"]:
        raise ValueError("sandbox reuse or resource envelope exceeded")
    progress.stage("aggregate_comparison")
    summary = release.comparison_summary(baseline, policies)
    amendment_path = directory / "amendments/cleanup-001/amendment.json"
    execution_amendment = read_json(amendment_path) if amendment_path.exists() else None
    return {"version": release.VERSION, "plan_hash": digest(canonical_json(plan)),
            "status": "completed_with_uncertainty" if any(s["unknown_input_outcomes"] for s in summary["policies"].values()) else "completed",
            "summary": summary, "programs": policies, "submitted_sandbox_attempts": starts,
            "startup_replacements": len(slots), "new_optimizer_updates": 48,
            "baseline_sha256": plan["baseline_sha256"], "control_run_id": plan["control_id"],
            "execution_amendment": execution_amendment}


@app.function(image=cpu_image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=release.CONTROLLER_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_comparison(plan, setup, budget, snapshot, deadline, amendment=None):
    release.validate_plan(plan)
    check_snapshot(snapshot, Path("/root"))
    require_deadline(deadline)
    artifacts.reload()
    directory = Path("/artifacts") / plan["run_id"]
    control_evidence = require_live_control(plan, snapshot)
    baseline_path = Path("/artifacts") / plan["baseline_run_id"] / "result.json"
    baseline = release.validate_baseline(baseline_path.read_bytes(), read_json(baseline_path), plan)
    if (directory / "result.json").exists():
        return read_json(directory / "result.json")
    if amendment is not None:
        from verifier_rl import booking_cleanup_recovery as cleanup
        if (cleanup.validate_amendment(read_json(directory / "source_snapshot.json"), snapshot) != amendment
                or read_json(directory / "deadline.json") != deadline
                or read_json(directory / "amendments" / cleanup.AMENDMENT / "prepared.json")["passed"] is not True):
            raise ValueError("cleanup recovery was not prepared against the original deadline/source")
    claim("controller", plan["max_controller_starts"])
    documents = {"plan": plan, "setup": setup, "budget": budget,
                 "deadline": deadline, "control": control_evidence, "baseline": baseline}
    if amendment is None:
        documents["source_snapshot"] = snapshot
    else:
        persist(directory / "amendments" / cleanup.AMENDMENT, {"amendment": amendment, "source_snapshot": snapshot})
    persist(directory, documents)
    artifacts.commit()
    stage = "live_controls"
    try:
        owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
        if list(modal.Sandbox.list(app_id=owner.app_id)):
            raise ReconciliationRequired("candidate sandboxes active before comparison")
        asyncio.run(live_controls(directory, release.execution_plan(plan), setup, deadline))
        graded = grade_batch.remote("controls", study.controls(plan["experiment"]), "training", plan, setup, deadline)
        artifacts.reload()
        preflight = study.validate_controls(graded["checked"])
        persist(directory, {"preflight": preflight})
        artifacts.commit()
        arms = {}
        for arm in study.ARMS:
            stage = "training_" + arm
            first = arms["reference"]["first_tokens"] if arms else None
            arms[arm] = train_arm.remote(arm, plan, setup, snapshot, first, deadline)
        stage = "independent_evaluation"
        for arm in study.ARMS:
            for step in (12,24):
                samples = arms[arm]["evaluations"][str(step)]["samples"]
                rows = []
                for i in range(8):
                    require_deadline(deadline)
                    graded = grade_batch.remote(f"eval-{arm}-{step:02d}-{i:02d}", samples[4*i:4*i+4],
                                                "evaluation", plan, setup, deadline)
                    rows.extend(graded["checked"]["rows"])
                measured = study.policy_summary(rows, arm, step)
                artifacts.reload()
                persist(directory / "measurements", {f"{arm}-{step}": measured})
                artifacts.commit()
                print("POLICY MEASURED; FINAL JOURNAL REPLAY PENDING", arm, step, measured, flush=True)
        stage = "finalization"
        with ProgressLog(plan["run_id"]) as progress:
            progress.stage("reload_artifacts")
            artifacts.reload()
            result = verify_comparison(directory, plan, setup, arms, baseline, progress)
            progress.stage("write_report")
            persist(directory, {"result": result})
            progress.stage("commit_artifacts")
            artifacts.commit()
        print("MATCHED COMPARISON COMPLETE", result["summary"], flush=True)
        return result
    except Exception as exc:
        persist(directory, {f"interrupted-{time.time_ns()}": {"stage": stage, "type": type(exc).__name__, "detail": str(exc)[:2000]}})
        artifacts.commit()
        raise


@app.function(image=cpu_image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=release.GRADE_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def prepare_cleanup_resume(plan, setup, snapshot, amendment, deadline):
    """Resolve only two reviewed PRE-candidate attempts before spending GPU time."""
    from verifier_rl import booking_cleanup_recovery as cleanup
    from verifier_rl.booking_screen_recovery import assess_record
    from verifier_rl.sandbox_lifecycle import confirm_terminal
    release.validate_plan(plan)
    check_snapshot(snapshot, Path("/root"))
    require_deadline(deadline)
    artifacts.reload()
    root = Path("/artifacts") / plan["run_id"]
    target = root / "amendments" / cleanup.AMENDMENT
    if (plan["run_id"] != cleanup.RUN_ID or read_json(root / "plan.json") != plan
            or read_json(root / "setup.json") != setup or read_json(root / "deadline.json") != deadline
            or cleanup.validate_amendment(read_json(root / "source_snapshot.json"), snapshot) != amendment):
        raise ValueError("cleanup recovery source/configuration/deadline changed")
    require_live_control(plan, snapshot)
    if (target / "prepared.json").exists():
        return read_json(target / "prepared.json")
    claim("cleanup-preparation", 1)
    persist(target, {"amendment": amendment, "source_snapshot": snapshot})
    artifacts.commit()
    arm = root / "arms/reference"
    binding = read_json(arm / "binding.json")
    checkpoint = recovery.latest_checkpoint(arm / "trainer", digest(canonical_json(binding)))
    if checkpoint is None or checkpoint.name != "checkpoint-19":
        raise ValueError("reviewed complete checkpoint 19 required")
    receipt = read_json(checkpoint / "recovery.json")
    group = arm / "journal/group-19"
    generated, samples = read_json(group / "generation.json"), read_json(group / "samples.json")
    if (generated["binding"]["parameter_hash"] != receipt["parameter_hash"]
            or generated["binding"]["step"] != 19
            or generated["tokens_hash"] != digest(canonical_json(generated["output"]))
            or (group / "reward.json").exists()):
        raise ValueError("pending unconsumed rollout differs")
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("active candidate sandboxes; reconcile before recovery")
    cases = {c.input_hash: c for c in study.cases_for("training")}
    by_id = {s["sample_id"]: s for s in samples}
    reconciliations = []
    for record_hash, (sid, h, sandbox_id) in cleanup.REVIEWED_FAILURES.items():
        require_deadline(deadline)
        path = root / "grading" / cleanup.FAILED_KEY / "inputs" / sid / h
        first = read_json(path / "attempt-1.json")
        if digest(canonical_json(first)) != record_hash:
            raise ValueError("reviewed failure record changed")
        assess_record(by_id[sid], cases[h], first, setup["sandbox_image_id"])
        terminal = asyncio.run(confirm_terminal(modal.Sandbox.from_id(sandbox_id)))
        reconciliation = cleanup.make_reconciliation(first, terminal, by_id[sid], cases[h])
        key = plan["run_id"] + f"/durable/{cleanup.FAILED_KEY}/{sid}/{h}/cleanup-reconciliation"
        if not claims.put(key, reconciliation, skip_if_exists=True):
            reconciliation = claims.get(key)
            cleanup.validate_reconciliation(first, reconciliation, by_id[sid], cases[h])
        persist(path, {"cleanup-reconciliation": reconciliation})
        artifacts.commit()
        reconciliations.append(reconciliation)
        print("CLEANUP RECONCILED", sid, sandbox_id, terminal["method"], flush=True)
    # Grade exactly the already-generated programs, reusing all completed input
    # records. If anything remains unknown, do not even start a training GPU.
    graded = grade_batch.remote(cleanup.FAILED_KEY, samples, "training", plan, setup, deadline)
    rewards = study.rewards_from_raw(graded["raw"], samples, cleanup.FAILED_KEY, "reference",
                                   plan["experiment"], setup["sandbox_image_id"])
    result = {"passed": True, "checkpoint": 19, "parameter_hash": receipt["parameter_hash"],
              "pending_tokens_hash": generated["tokens_hash"], "reconciliations": reconciliations,
              "pending_rewards": rewards, "new_model_samples": 0, "optimizer_updates": 0}
    artifacts.reload()
    persist(target, {"prepared": result})
    artifacts.commit()
    print("CLEANUP RECOVERY PREPARED; READY TO RESUME UPDATE 20", result, flush=True)
    return result


@app.local_entrypoint()
def resume(allow_cloud: bool = False):
    """One authorized continuation, within original call/retry/deadline bounds."""
    from verifier_rl import booking_cleanup_recovery as cleanup
    if not allow_cloud:
        raise ValueError("--allow-cloud required")
    directory = Path("runs") / release.RUN_ID
    target = directory / "amendments" / cleanup.AMENDMENT
    if (target / "launch_intent.json").exists():
        raise ReconciliationRequired("cleanup recovery already launched; inspect, do not repeat")
    plan, setup, budget = (read_json(directory / f"{n}.json") for n in ("plan", "setup", "budget"))
    deadline = read_json(directory / "launch_intent.json")["deadline"]
    require_deadline(deadline)
    apps = read_modal("app", "list")
    if any(int(row["tasks"]) for row in apps):
        raise ReconciliationRequired("other Modal jobs active")
    names = sorted(Path("verifier_rl").glob("*.py")) + [Path(n) for n in (
        "modal_booking_study.py", "modal_booking_baseline_comparison.py", "modal_booking_matched_training.py")]
    snapshot = {p.as_posix(): p.read_text() for p in names}
    amendment = cleanup.validate_amendment(read_json(directory / "source_snapshot.json"), snapshot)
    persist(target, {"amendment": amendment, "source_snapshot": snapshot,
                    "launch_intent": {"deadline": deadline, "apps": apps}})
    prepared = prepare_cleanup_resume.remote(plan, setup, snapshot, amendment, deadline)
    persist(target, {"prepared": prepared})
    call = run_comparison.spawn(plan, setup, budget, snapshot, deadline, amendment)
    persist(target, {"launch": {"call_id": call.object_id, "created_utc": datetime.now(timezone.utc).isoformat()}})
    print("MATCHED RECOVERY CALL", call.object_id, "RESUME 19/24; SAME PENDING TOKENS; NO NEW BASELINE", flush=True)
    result = call.get()
    persist(directory, {"result": result})
    print("MATCHED COMPARISON RESULT", result["summary"], flush=True)


@app.local_entrypoint()
def launch(allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required")
    plan = release.make_plan(Path.cwd())
    target = Path("runs") / release.RUN_ID
    if (target / "launch_intent.json").exists():
        raise ReconciliationRequired("already launched; inspect existing run, do not repeat")
    if read_json(Path("runs") / release.CONTROL_ID / "result.json").get("passed") is not True:
        raise ValueError("live recovery control must pass first")
    apps = read_modal("app", "list")
    if any(int(row["tasks"]) for row in apps):
        raise ReconciliationRequired("other Modal jobs active")
    baseline_path = Path("runs") / study.RUN_ID / "result.json"
    baseline = release.validate_baseline(baseline_path.read_bytes(), read_json(baseline_path), plan)
    setup = read_json(Path("runs") / study.RUN_ID / "setup.json")
    names = sorted(Path("verifier_rl").glob("*.py")) + [Path(n) for n in (
        "modal_booking_study.py", "modal_booking_baseline_comparison.py", "modal_booking_matched_training.py")]
    snapshot = {p.as_posix(): p.read_text() for p in names}
    budget = release.quote(plan, read_modal("billing", "rates"), read_modal("billing", "summary"))
    deadline = time.time() + release.CONTROLLER_SECONDS - 120
    persist(target, {"plan": plan, "setup": setup, "budget": budget, "source_snapshot": snapshot,
                    "baseline": baseline, "protocol": Path("docs/booking_matched_training_release.txt").read_text(),
                    "launch_intent": {"deadline": deadline, "apps": apps}})
    call = run_comparison.spawn(plan, setup, budget, snapshot, deadline)
    persist(target, {"launch": {"call_id": call.object_id, "created_utc": datetime.now(timezone.utc).isoformat()}})
    print("MATCHED COMPARISON CALL", call.object_id, "24 UPDATES PER ARM; NO NEW BASELINE", flush=True)
    print("WORST-CASE COMPUTE ENVELOPE USD", budget["max_compute_envelope_usd"], "NOT EXPECTED SPEND/INVOICE", flush=True)
    result = call.get()
    persist(target, {"result": result})
    print("MATCHED COMPARISON RESULT", result["summary"], flush=True)
