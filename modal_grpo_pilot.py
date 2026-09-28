"""Four-update real-reward 1.5B pilot; candidates execute only in CPU sandboxes."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import modal

from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.grpo_pilot import (evaluation_suites, pilot_plan, require_report, rollout_rewards,
                                    summarize_pilot, trainer_kwargs, training_evidence)
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.model_trial import evaluate_submission, validate_run_id, validate_submission
from verifier_rl.reward_v2 import behavior_suite, validate_behavior_reward
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import canonical_json, digest
from verifier_rl.training import require_control_evidence
from verifier_rl import reward_shaping, evaluation_recovery

app = modal.App("verifier-rl-real-reward-grpo-pilot")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
             .add_local_python_source("verifier_rl"))
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("modal==1.5.5", "torch==2.8.0", "transformers==4.57.1",
                          "trl==0.28.0", "datasets==3.5.1", "accelerate==1.12.0")
             .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
             .add_local_python_source("verifier_rl"))


def persist_equal(path, value):
    """Reuse committed identical controller evidence; never overwrite a result."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("conflicting persisted pilot evidence")
    else:
        write_private(path, canonical_json(value))


@app.function(image=cpu_image, cpu=(1, 1), memory=(1024, 1024), timeout=300,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def grade_one(run_id: str, sample: dict, setup: dict, audit: bool, study_id: str = ""):
    validate_run_id(run_id)
    validate_submission(sample)
    arm, seed = sample["arm"], sample["seed"]
    evaluation_seeds = reward_shaping.SEEDS if study_id else range(8000, 8008)
    if type(seed) is not int or (audit and (arm not in ("before", "after") or seed not in evaluation_seeds)):
        raise ValueError("unexpected evaluation identity")
    if not audit and (arm != "training" or not 9000 <= seed <= 9015):
        raise ValueError("unexpected training identity")
    suites = (reward_shaping.selected_suites(audit) if study_id
              else evaluation_suites() if audit else (behavior_suite(),))
    artifacts.reload()
    directory = Path(f"/artifacts/{run_id}/grading/{arm}-{seed}")
    result_path = directory / "result.json"
    intent = {"sample": sample, "setup": setup, "audit": audit,
              "suite_hashes": [s.fingerprint for s in suites]}
    study = None
    candidate_id = f"{run_id}-{arm}-{seed}"
    if study_id:
        validate_run_id(study_id)
        if run_id not in {study_id + "-baseline", study_id + "-partial", study_id + "-completion-bonus"}:
            raise ValueError("candidate is outside the study")
        if (run_id.endswith("-baseline")) != (audit and arm == "before"):
            raise ValueError("baseline must be scored exactly once, outside training")
        study = Path(f"/artifacts/{study_id}")
        common_plan = json.loads((study / "plan.json").read_text())
        intent["study_plan_hash"] = digest(canonical_json(common_plan))
    if result_path.exists():
        if study_id:
            saved_intent = json.loads((directory / "intent.json").read_text())
            reservation = saved_intent["spending_reservation"]
            if reservation["candidate_id"] != candidate_id:
                raise ValueError("completed reservation identity mismatch")
            intent["spending_reservation"] = reservation
        persist_equal(directory / "intent.json", intent)
        report = json.loads(result_path.read_text())
        if study_id:
            reward_shaping.require_report(sample, report, audit, setup["sandbox_image_id"])
            receipt = reward_shaping.execution_receipt(candidate_id, report)
            persist_equal(study / "budget-receipts" / f"{candidate_id}.json", receipt)
            artifacts.commit()
        else:
            require_report(sample, report, suites, setup["sandbox_image_id"])
        return report
    # An interrupted intent without a result is not permission to repeat work.
    if study_id:
        reward_shaping.require_deadline(common_plan)
        budget = json.loads((study / "budget.json").read_text())
        receipts = {p.stem: json.loads(p.read_text()) for p in (study / "budget-receipts").glob("*.json")}
        count = 0 if sample["extraction_status"].startswith("rejected_") else len({c.input_hash for s in suites for c in s.cases})
        intent["spending_reservation"] = reward_shaping.reserve_batch(budget, receipts, candidate_id, count)
    create_run_directory(str(directory))
    write_private(directory / "intent.json", canonical_json(intent))
    artifacts.commit()
    backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=.26)
    report = asyncio.run(evaluate_submission(sample, suites, backend, concurrency=8))
    report.update(arm=arm, seed=seed)
    write_private(result_path, canonical_json(report))
    artifacts.commit()
    if study_id:
        reward_shaping.require_report(sample, report, audit, setup["sandbox_image_id"])
        persist_equal(study / "budget-receipts" / f"{candidate_id}.json",
                      reward_shaping.execution_receipt(candidate_id, report))
        artifacts.commit()
    else:
        require_report(sample, report, suites, setup["sandbox_image_id"])
    print("Graded", arm, seed, [(s["suite"], s["passed_count"], s["total"]) for s in report["suites"]], flush=True)
    return report


@app.function(image=gpu_image, gpu="L40S", cpu=(2, 2), memory=(32768, 32768),
              timeout=1200, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def train_and_generate(run_id: str, plan: dict, setup: dict, provenance: dict):
    import gc
    import hashlib
    import importlib.metadata
    import time
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from verifier_rl.baseline import inspect_completion

    validate_run_id(run_id)
    shaping = plan.get("version") == reward_shaping.VERSION
    if shaping:
        reward_shaping.require_deadline(plan)
    started = time.monotonic()
    artifacts.reload()
    directory = create_run_directory(f"/artifacts/{run_id}/gpu")
    for name, value in provenance.items():
        persist_equal(directory.parent / f"{name}.json", value)
    save_suites(directory.parent, reward_shaping.selected_suites(True) if shaping else evaluation_suites())
    write_private(directory / "intent.json", canonical_json({"plan": plan, "setup": setup}))
    artifacts.commit()  # A restart must not silently repeat training.

    def parameter_hash(policy):
        h = hashlib.sha256()
        for name, parameter in policy.named_parameters():
            h.update(name.encode())
            h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    try:
        runtime = {"gpu": torch.cuda.get_device_name(), "gpu_image_id": gpu_image.object_id,
                   "total_cuda_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
        if shaping:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if runtime["total_cuda_bytes"] < 40 * 1024**3:
            raise ValueError("GPU lacks the frozen full-parameter memory allowance")
        write_private(directory / "runtime.json", canonical_json(runtime))
        artifacts.commit()
        set_seed(plan["training_seed"])
        tokenizer = AutoTokenizer.from_pretrained(plan["model_id"], revision=plan["revision"], trust_remote_code=False)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if digest(tokenizer.chat_template) != plan["chat_template_hash"]:
            raise ValueError("chat template identity changed")
        model = AutoModelForCausalLM.from_pretrained(plan["model_id"], revision=plan["revision"],
                    trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
        count = sum(p.numel() for p in model.parameters())
        before_hash = parameter_hash(model)
        if count != plan["parameter_count"] or before_hash != plan["initial_parameter_hash"]:
            raise ValueError("initial policy differs from frozen untouched 1.5B checkpoint")
        write_private(directory / "memory_plan.json", canonical_json({
            "parameters": count, "fp32_parameters_gradients_adam_bytes": count * 16,
            "excludes_activations_logits_cuda_workspace": True, "optimizer_foreach": False,
            "reference_model": False, "gradient_checkpointing": True}))
        messages = [{"role": "user", "content": plan["prompt"]}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")
        samples = []

        def generate(policy, arm):
            policy.eval()
            for seed in plan["seeds"]:
                if shaping:
                    reward_shaping.require_deadline(plan)
                set_seed(seed)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = policy.generate(**inputs, max_new_tokens=512, do_sample=True,
                        temperature=.8, top_p=.95, top_k=0, repetition_penalty=plan["repetition_penalty"],
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
                completion = tokens[0, inputs["input_ids"].shape[1]:]
                sample = inspect_completion(tokenizer.decode(completion, skip_special_tokens=True))
                sample.update(arm=arm, seed=seed, prompt_hash=plan["prompt_hash"], tokens=len(completion),
                              hit_token_cap=len(completion) == 512,
                              ended_with_eos=bool(len(completion) and completion[-1].item() == tokenizer.eos_token_id))
                samples.append(sample)
                write_private(directory / f"{arm}-{seed}.json", canonical_json(sample))
                artifacts.commit()
                print("Generated", arm, seed, "tokens", len(completion), flush=True)

        generate(model, "before")
        if shaping and "expected_before_signature" in provenance:
            if reward_shaping.before_signature(samples) != provenance["expected_before_signature"]:
                raise ValueError("initial-policy generations differ between arms; stop before second training")
        if parameter_hash(model) != before_hash:
            raise RuntimeError("baseline generation changed weights")
        calls, rollout_records = [], []

        def execution_reward(completions, completion_ids=None, **kwargs):
            if shaping:
                reward_shaping.require_deadline(plan)
            index = len(calls)
            if index >= 4 or len(completions) != 4:
                raise RuntimeError("rollout budget exceeded")
            raw = [c[0]["content"] if isinstance(c, list) else c for c in completions]
            if shaping and index == 0 and "expected_first_rollouts" in provenance:
                if raw != provenance["expected_first_rollouts"]:
                    raise ValueError("initial training rollouts differ between arms; stop before second-arm update")
            group = []
            for position, text in enumerate(raw):
                sample = inspect_completion(text)
                sample.update(arm="training", seed=9000 + 4 * index + position,
                              prompt_hash=plan["prompt_hash"], seed_is_rollout_id=True,
                              training_seed=plan["training_seed"])
                if completion_ids is not None:
                    ids = completion_ids[position]
                    sample.update(tokens=len(ids), hit_token_cap=len(ids) == 512,
                                  ended_with_eos=bool(len(ids) and ids[-1] == tokenizer.eos_token_id))
                group.append(sample)
            write_private(directory / f"rollouts-{index}.json", canonical_json(group))
            artifacts.commit()
            reports = [grade_one.remote(run_id, sample, setup, False,
                                       plan["study_run_id"] if shaping else "") for sample in group]
            artifacts.reload()
            rewards = (reward_shaping.rollout_rewards(group, reports, setup["sandbox_image_id"], plan)
                       if shaping else rollout_rewards(group, reports, setup["sandbox_image_id"]))
            write_private(directory / f"reward-{index}.json", canonical_json({"rewards": rewards, "reports": reports}))
            calls.append(rewards)
            rollout_records.append({"samples": group, "reports": reports, "rewards": rewards})
            artifacts.commit()
            print("Rewards", plan.get("reward_formula", "v2_partial"), "step", index + 1, rewards, flush=True)
            return rewards

        class StopWithoutSignal(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if calls and len(set(calls[-1])) == 1:
                    control.should_training_stop = True
                return control

        set_seed(plan["training_seed"])
        model.config.use_cache = False
        args = GRPOConfig(**trainer_kwargs(directory / "trainer"))
        write_private(directory / "trainer_config.json", args.to_json_string())
        # Explicit foreach=False avoids a full extra parameter-sized temporary.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0,
                                      betas=(.9, .999), eps=1e-8, foreach=False)
        trainer = GRPOTrainer(model=model, args=args, reward_funcs=execution_reward,
            train_dataset=Dataset.from_list([{"prompt": messages}] * 4), processing_class=tokenizer,
            callbacks=[StopWithoutSignal()], optimizers=(optimizer, None))
        write_private(directory / "rollout_generation_config.json", canonical_json(trainer.generation_config.to_dict()))
        artifacts.commit()
        model.train()
        trainer.train()
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError("nonfinite trained weights")
        after_hash = parameter_hash(model)
        metrics = {"global_step": trainer.state.global_step, "rewards": calls,
                   "log_history": trainer.state.log_history,
                   "before_parameter_hash": before_hash, "after_parameter_hash": after_hash,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "finite_parameters": True, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
        model.config.use_cache = True
        checkpoint = directory / "checkpoint"
        trainer.save_model(str(checkpoint))
        tokenizer.save_pretrained(checkpoint)
        write_private(directory / "training-before-reload.json", canonical_json(metrics))
        artifacts.commit()
        del trainer, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=False,
                     dtype=torch.float32, attn_implementation="sdpa").to("cuda")
        metrics["checkpoint_reload_hash_matches"] = parameter_hash(reloaded) == after_hash
        evidence = training_evidence(metrics)
        write_private(directory / "training.json", canonical_json(metrics))
        artifacts.commit()
        generate(reloaded, "after")
        if parameter_hash(reloaded) != after_hash:
            raise RuntimeError("post-training generation changed weights")
        result = {"run_id": run_id, "plan": plan, "metrics": metrics, "training_evidence": evidence,
                  "samples": samples, "rollout_records": rollout_records, "runtime": runtime,
                  "checkpoint": f"{run_id}/gpu/checkpoint", "gpu_function_seconds": time.monotonic() - started}
        write_private(directory / "generation.json", canonical_json(result))
        artifacts.commit()
        return result
    except Exception as exc:
        write_private(directory / "error.json", canonical_json({"type": type(exc).__name__,
                      "elapsed_seconds": time.monotonic() - started}))
        artifacts.commit()
        raise


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), timeout=3600,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def audit_completed(run_id: str, generation: dict, setup: dict):
    validate_run_id(run_id)
    training_evidence(generation["metrics"])
    artifacts.reload()
    directory = Path(f"/artifacts/{run_id}")
    persist_equal(directory / "generation.json", generation)
    artifacts.commit()
    reports = []
    # No audit calls occur until training, reload and both generation arms finish.
    for sample in generation["samples"]:
        reports.append(grade_one.remote(run_id, sample, setup, True))
    artifacts.reload()
    summary = summarize_pilot(generation, reports, setup["sandbox_image_id"])
    result = {"reports": reports, "summary": summary, "diagnostics": summary["diagnostics"]}
    for name, data in result.items():
        persist_equal(directory / f"{name}.json", data)
    artifacts.commit()
    return result


@app.local_entrypoint()
def main(setup_file: str, cpu_report: str, optimizer_report: str, allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for one bounded L40S job and CPU audit")
    setup = json.loads(Path(setup_file).read_text())
    conformance = json.loads(Path(cpu_report).read_text())
    control = json.loads(Path(optimizer_report).read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    for condition in ("zero", "mixed"):
        require_control_evidence(control["conditions"][condition], condition)
    validation = validate_behavior_reward()
    plan = pilot_plan(Path("task_001_expiring_cache.txt").read_text())
    run_id = "qwen-grpo-partial-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    paths = sorted(Path("verifier_rl").glob("*.py")) + [Path(__file__), Path("grpo_1_5b_protocol.txt")]
    snapshot = {p.as_posix(): p.read_text() for p in paths}
    provenance = {"plan": plan, "setup": setup, "conformance": conformance,
                  "optimizer-control": control, "reward-validation": validation, "source_snapshot": snapshot}
    for name, data in provenance.items():
        write_private(directory / f"{name}.json", canonical_json(data))
    save_suites(directory, evaluation_suites())
    call = train_and_generate.spawn(run_id, plan, setup, provenance)
    write_private(directory / "training-launch.json", canonical_json({"run_id": run_id, "call_id": call.object_id}))
    print("Real-reward GRPO pilot:", run_id, flush=True)
    generation = call.get()
    write_private(directory / "generation.json", canonical_json(generation))
    print("Training evidence:", generation["training_evidence"], flush=True)
    call = audit_completed.spawn(run_id, generation, setup)
    write_private(directory / "audit-launch.json", canonical_json({"run_id": run_id, "call_id": call.object_id}))
    result = call.get()
    for name, data in result.items():
        write_private(directory / f"{name}.json", canonical_json(data))
    print("Completed:", directory / "summary.json", flush=True)


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048),
              timeout=reward_shaping.CONTROLLER_SECONDS, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def run_shaping(study_id: str, plan: dict, setup: dict, conformance: dict,
                control: dict, snapshot: dict, budget: dict, validation: dict):
    """One durable parent owns both training arms and their post-training audit."""
    validate_run_id(study_id)
    reward_shaping.require_deadline(plan)
    require_current_conformance(conformance, setup["sandbox_image_id"])
    for condition in ("zero", "mixed"):
        require_control_evidence(control["conditions"][condition], condition)
    if reward_shaping.budget_envelope(plan, budget["billing_before"], budget["rates"]) != budget:
        raise ValueError("budget envelope mismatch")
    artifacts.reload()
    directory = create_run_directory(f"/artifacts/{study_id}")
    (directory / "budget-receipts").mkdir(mode=0o700)
    provenance = {"plan": plan, "setup": setup, "conformance": conformance,
        "optimizer-control": control, "source_snapshot": snapshot, "budget": budget,
        "validation": validation}
    for name, data in provenance.items():
        write_private(directory / f"{name}.json", canonical_json(data))
    save_suites(directory, reward_shaping.selected_suites(True))
    artifacts.commit()
    generations = {}
    stage = "initialization"
    try:
        for formula in reward_shaping.FORMULAS:
            stage = f"training:{formula}"
            reward_shaping.require_deadline(plan)
            selected_plan = reward_shaping.arm_plan(plan, formula, study_id)
            arm_id = study_id + "-" + formula.replace("_", "-")
            arm_provenance = {**provenance, "plan": selected_plan}
            if generations:
                arm_provenance["expected_before_signature"] = reward_shaping.before_signature(
                    generations["partial"]["samples"])
                arm_provenance["expected_first_rollouts"] = [s["raw"] for s in
                    generations["partial"]["rollout_records"][0]["samples"]]
            print("Starting matched arm:", formula, flush=True)
            generation = train_and_generate.remote(arm_id, selected_plan, setup, arm_provenance)
            training_evidence(generation["metrics"])
            generations[formula] = generation
            artifacts.reload()
            write_private(directory / f"generation-{formula}.json", canonical_json(generation))
            artifacts.commit()
            print("Training complete:", formula, generation["training_evidence"], flush=True)
        if reward_shaping.before_signature(generations["partial"]["samples"]) != reward_shaping.before_signature(
                generations["completion_bonus"]["samples"]):
            raise ValueError("initial generation repeatability check failed")
        write_private(directory / "generations.json", canonical_json(generations))
        artifacts.commit()
        # No independent correctness execution occurs before BOTH training arms finish.
        populations = {"baseline": [s for s in generations["partial"]["samples"] if s["arm"] == "before"],
            **{f: [s for s in generations[f]["samples"] if s["arm"] == "after"] for f in reward_shaping.FORMULAS}}
        reports = {}
        for policy, samples in populations.items():
            stage = f"evaluation:{policy}"
            reports[policy] = []
            owner = study_id + "-" + policy.replace("_", "-")
            for sample in samples:
                reward_shaping.require_deadline(plan)
                report = grade_one.remote(owner, sample, setup, True, study_id)
                reports[policy].append(report)
            artifacts.reload()
            write_private(directory / f"reports-{policy}.json", canonical_json(reports[policy]))
            artifacts.commit()
        stage = "verification"
        summary = reward_shaping.comparison_summary(study_id, plan, generations, reports, setup["sandbox_image_id"])
        artifacts.reload()
        receipts = {p.stem: json.loads(p.read_text()) for p in (directory / "budget-receipts").glob("*.json")}
        if sum(r["executions"] for r in receipts.values()) != summary["recorded_sandbox_executions"]:
            raise ValueError("budget receipts disagree with complete execution evidence")
        reservations = {}
        for owner, arm, seed, report in reward_shaping.study_execution_records(study_id, generations, reports):
            intent_path = Path(f"/artifacts/{owner}/grading/{arm}-{seed}/intent.json")
            reservations[f"{owner}-{arm}-{seed}"] = json.loads(intent_path.read_text())["spending_reservation"]
        reward_shaping.verify_budget_records(study_id, generations, reports, budget, receipts, reservations)
        result = {"generations": generations, "reports": reports, "summary": summary,
                  "budget-receipts": receipts, "spending-reservations": reservations}
        for name, value in result.items():
            persist_equal(directory / f"{name}.json", value)
        artifacts.commit()
        print("Matched comparison complete:", summary["policies"], flush=True)
        return result
    except Exception as exc:
        artifacts.reload()
        persist_equal(directory / "stopped.json", {"stage": stage, "error_type": type(exc).__name__,
                      "detail": str(exc)[:1000], "automatic_retry": False})
        artifacts.commit()
        raise


@app.local_entrypoint()
def shaping(allow_cloud: bool = False):
    """Launch only the user-approved two-arm v3 reward-formula pilot."""
    if not allow_cloud:
        raise ValueError("--allow-cloud required: two bounded L40S jobs and one shared CPU audit")
    from verifier_rl.measurement_v3_verification import verify_documents
    from verifier_rl.reward_review import review_controls
    previous = Path("runs") / reward_shaping.SOURCE_RUN
    measurement = Path("runs") / reward_shaping.MEASUREMENT_RUN
    required = ("generation", "prior_reports", "reports", "setup", "conformance", "source_snapshot",
                "budget", "plan", "summary")
    validated_measurement = verify_documents(
        {name: json.loads((measurement / f"{name}.json").read_text()) for name in required}, Path.cwd())
    controls = review_controls()
    setup = json.loads((previous / "setup.json").read_text())
    conformance = json.loads((previous / "conformance.json").read_text())
    optimizer = json.loads((previous / "optimizer-control.json").read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    for condition in ("zero", "mixed"):
        require_control_evidence(optimizer["conditions"][condition], condition)
    plan = reward_shaping.study_plan(Path("task_001_expiring_cache.txt").read_text())
    def read_billing(kind):
        result = subprocess.run([sys.executable, "-m", "modal", "billing", kind, "--json"],
                                check=True, capture_output=True, text=True, timeout=60)
        return json.loads(result.stdout)
    budget = reward_shaping.budget_envelope(plan, read_billing("summary"), read_billing("rates"))
    plan["submission_deadline_epoch"] = time.time() + reward_shaping.CONTROLLER_SECONDS
    study_id = "qwen-shaping-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{study_id}")
    paths = sorted(Path("verifier_rl").glob("*.py")) + [Path(__file__), Path("reward_shaping_protocol.txt")]
    snapshot = {p.as_posix(): p.read_text() for p in paths}
    validation = {"paired_measurement": validated_measurement, "authored_controls": controls}
    inputs = {"plan": plan, "setup": setup, "conformance": conformance, "optimizer-control": optimizer,
              "source_snapshot": snapshot, "budget": budget, "validation": validation}
    for name, value in inputs.items():
        write_private(directory / f"{name}.json", canonical_json(value))
    save_suites(directory, reward_shaping.selected_suites(True))
    call = run_shaping.spawn(study_id, plan, setup, conformance, optimizer, snapshot, budget, validation)
    write_private(directory / "launch.json", canonical_json({"run_id": study_id, "call_id": call.object_id}))
    print("Matched reward-formula study:", study_id, "call:", call.object_id, flush=True)
    print("Fixed resource/lag reserve:", budget["fixed_reserved_usd"],
          "; every CPU batch is additionally reserved before launch against the $10 total limit", flush=True)
    result = call.get()
    for name, value in result.items():
        persist_equal(directory / f"{name}.json", value)
    print("Completed:", directory / "summary.json", flush=True)


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048),
              timeout=evaluation_recovery.CONTROLLER_SECONDS, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def recover_evaluation(recovery_id: str, inputs: dict, budget: dict, deadline: float, snapshot: dict):
    """One warm CPU controller; no call to a model or GPU function is possible here."""
    validate_run_id(recovery_id)
    rows = evaluation_recovery.validate_inputs(inputs)
    if evaluation_recovery.recovery_budget(inputs, budget["billing_before"], budget["rates"]) != budget:
        raise ValueError("recovery spending envelope changed")
    reward_shaping.require_deadline({"submission_deadline_epoch": deadline})
    require_current_conformance(inputs["conformance"], inputs["setup"]["sandbox_image_id"])
    artifacts.reload()
    directory = create_run_directory(f"/artifacts/{recovery_id}")
    for name, value in {"inputs": inputs, "budget": budget, "source_snapshot": snapshot,
                        "deadline": deadline}.items():
        write_private(directory / f"{name}.json", canonical_json(value))
    save_suites(directory, reward_shaping.selected_suites(True))
    artifacts.commit()
    reports = dict(inputs["cached"])
    receipts, reservations, attempts, reconciliations = {}, {}, {}, {}
    replacements_used = 0
    current = "initialization"
    try:
        for key, policy, sample in rows:
            current = key
            if key in reports:
                print("Reused saved evaluation:", key, flush=True)
                continue
            reward_shaping.require_deadline({"submission_deadline_epoch": deadline})
            for attempt_index in (1, 2):
                reward_shaping.require_deadline({"submission_deadline_epoch": deadline})
                attempt_key = f"{key}-attempt-{attempt_index}"
                count = 0 if sample["extraction_status"].startswith("rejected_") else 432
                reservation = reward_shaping.reserve_batch(budget, receipts, attempt_key, count)
                item = create_run_directory(str(directory / attempt_key))
                write_private(item / "intent.json", canonical_json({"sample": sample,
                    "policy": policy, "reservation": reservation, "attempt_index": attempt_index}))
                artifacts.commit()
                print("Evaluating saved program:", attempt_key, "reserved total USD", reservation["total_reserved_usd"], flush=True)
                backend = ModalBackend(inputs["setup"]["app_name"], inputs["setup"]["sandbox_image_id"],
                                       creation_interval_seconds=.26)
                async def evaluate():
                    return await asyncio.wait_for(evaluate_submission(sample, reward_shaping.selected_suites(True),
                                                   backend, concurrency=8), timeout=300)
                report = asyncio.run(evaluate())
                report.update(arm=sample["arm"], seed=sample["seed"])
                write_private(item / "result.json", canonical_json(report))
                artifacts.commit()  # Failure evidence is durable before any validation/replacement.
                attempts[attempt_key], reservations[attempt_key] = report, reservation
                if any(s["all_passed"] is None for s in report["suites"]):
                    if attempt_index == 2 or replacements_used >= evaluation_recovery.MAX_EXTRA_ATTEMPTS:
                        raise ValueError("bounded infrastructure replacement budget exhausted")
                    from verifier_rl.measurement_v3 import unique_outcomes
                    pending_ids = {a["metadata"].get("sandbox_id") for o in unique_outcomes(report).values()
                        for a in o["attempts"] if a["status"] == "infrastructure_error"
                        and a["metadata"].get("cleanup") != "terminated"}
                    async def reconcile():
                        results = {}
                        async def one(sid):
                            if not sid:
                                return
                            sandbox = await modal.Sandbox.from_id.aio(sid)
                            code = await sandbox.poll.aio()
                            if code is None:
                                await sandbox.terminate.aio(wait=False)
                                while code is None:
                                    await asyncio.sleep(2)
                                    code = await sandbox.poll.aio()
                            results[sid] = code
                        if pending_ids:
                            await asyncio.wait_for(asyncio.gather(*(one(sid) for sid in pending_ids)), timeout=45)
                        return results
                    resolved = asyncio.run(reconcile())
                    write_private(item / "reconciliation.json", canonical_json(resolved))
                    artifacts.commit()
                    if not evaluation_recovery.retry_eligible(sample, report, inputs["setup"]["sandbox_image_id"], resolved):
                        raise ValueError("infrastructure failure is not safe for a bounded replacement")
                    reconciliations[attempt_key] = resolved
                    receipt = evaluation_recovery.failed_receipt(attempt_key, report)
                    write_private(item / "receipt.json", canonical_json(receipt))
                    receipts[attempt_key] = receipt
                    replacements_used += 1
                    artifacts.commit()
                    print("Recorded infrastructure failure; one replacement authorized:", key, flush=True)
                    continue
                reward_shaping.require_report(sample, report, True, inputs["setup"]["sandbox_image_id"])
                receipt = reward_shaping.execution_receipt(attempt_key, report)
                write_private(item / "receipt.json", canonical_json(receipt))
                receipts[attempt_key], reports[key] = receipt, report
                artifacts.commit()
                break
            print("Completed evaluation", len(reports), "/24:", key,
                  [(s["suite"], s["passed_count"], s["total"]) for s in report["suites"]], flush=True)
        current = "verification"
        summary = evaluation_recovery.verify_recovery(inputs, reports, budget, receipts, reservations, attempts, reconciliations)
        result = {"reports": reports, "receipts": receipts, "reservations": reservations,
                  "attempts": attempts, "reconciliations": reconciliations, "summary": summary}
        for name, value in result.items():
            write_private(directory / f"{name}.json", canonical_json(value))
        artifacts.commit()
        print("Recovered comparison complete:", summary["policies"], flush=True)
        return result
    except Exception as exc:
        write_private(directory / "stopped.json", canonical_json({"stage": current,
            "error_type": type(exc).__name__, "detail": str(exc)[:1000],
            "complete_evaluation_records": len(reports), "automatic_retry": False,
            "checkpoints_and_completed_results_preserved": True}))
        artifacts.commit()
        raise


@app.local_entrypoint()
def recover(allow_cloud: bool = False):
    """One opt-in CPU-only recovery; never relaunch either GRPO arm."""
    if not allow_cloud:
        raise ValueError("--allow-cloud required: bounded CPU grading, no new model generations")
    previous = Path("runs") / evaluation_recovery.SOURCE_RUN
    def read(name):
        return json.loads((previous / f"{name}.json").read_text())
    inputs = {"plan": read("plan"), "setup": read("setup"), "conformance": read("conformance"),
        "generations": {f: read("generation-" + f) for f in reward_shaping.FORMULAS},
        "cached": {f"baseline-{seed}": read(f"baseline-before-{seed}") for seed in range(10000, 10005)},
        "failed_report": read("failed-before-10005-result"),
        "failed_intent": read("failed-before-10005-intent"), "old_budget": read("budget"),
        "original_stop": read("remote-stopped")}
    checked = evaluation_recovery.check_frozen_sources(read("source_snapshot"), Path.cwd())
    evaluation_recovery.validate_inputs(inputs)
    def billing(kind):
        result = subprocess.run([sys.executable, "-m", "modal", "billing", kind, "--json"],
                                check=True, capture_output=True, text=True, timeout=45)
        return json.loads(result.stdout)
    budget = evaluation_recovery.recovery_budget(inputs, billing("summary"), billing("rates"))
    # A late recovery must not silently outlive the earmarked quarantine cover.
    failed_at = datetime.fromisoformat(inputs["failed_report"]["timestamp_utc"]).timestamp()
    estimated_liability_seconds = time.time() - failed_at + evaluation_recovery.CONTROLLER_SECONDS + 120
    if estimated_liability_seconds * float(budget["sandbox_hour_usd"]) / 3600 > .50:
        raise ValueError("old sandbox liability exceeds quarantine allowance; confirm its terminal state before launch")
    recovery_id = "qwen-eval-recovery-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{recovery_id}")
    snapshot = {p.as_posix(): p.read_text() for p in sorted(Path("verifier_rl").glob("*.py")) +
                [Path(__file__), Path("evaluation_recovery_protocol.txt")]}
    deadline = time.time() + evaluation_recovery.CONTROLLER_SECONDS
    for name, value in {"inputs": inputs, "budget": budget, "source_snapshot": snapshot,
                        "deadline": deadline, "frozen_files_checked": checked}.items():
        write_private(directory / f"{name}.json", canonical_json(value))
    call = recover_evaluation.spawn(recovery_id, inputs, budget, deadline, snapshot)
    write_private(directory / "launch.json", canonical_json({"run_id": recovery_id,
        "source_run_id": evaluation_recovery.SOURCE_RUN, "call_id": call.object_id,
        "new_model_generations": 0, "gpu_calls": 0}))
    print("CPU-only recovery:", recovery_id, "call:", call.object_id, flush=True)
    print("Five cached evaluations, nineteen pending; fixed prior/controller reserve USD", budget["fixed_reserved_usd"], flush=True)
    result = call.get()
    for name, value in result.items():
        persist_equal(directory / f"{name}.json", value)
    print("Completed:", directory / "summary.json", flush=True)
