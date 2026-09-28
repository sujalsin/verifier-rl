"""Zero/mixed synthetic-reward controls for the actual Qwen + TRL optimizer path.

This tests gradients, optimizer steps and checkpoint reload, NOT coding learning.
No generated text is executed. Coding rewards and cache evaluation are unchanged.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import modal

from verifier_rl.cli import create_run_directory, write_private
from verifier_rl.model_trial import MODEL_ID, MODEL_REVISION, validate_run_id
from verifier_rl.suites import canonical_json
from verifier_rl.training import control_rewards, grpo_kwargs, require_control_evidence, require_grader_controls

app = modal.App("verifier-rl-grpo-optimizer-control")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("modal==1.5.5", "torch==2.8.0", "transformers==4.57.1",
                      "trl==0.28.0", "datasets==3.5.1", "accelerate==1.12.0")
         .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_python_source("verifier_rl"))


@app.function(image=image, gpu="L4", cpu=2, memory=16384, timeout=600, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def run_control(run_id: str, grader_evidence: dict, snapshot: dict):
    import gc
    import hashlib
    import importlib.metadata
    import time
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    validate_run_id(run_id)
    require_grader_controls(grader_evidence)
    started = time.monotonic()
    directory = create_run_directory(f"/artifacts/{run_id}")
    plan = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "group_size": 4,
            "conditions": {"zero": 1, "mixed": 2}, "max_completion_tokens": 32,
            "seed": 20260928, "max_gpu_seconds": 600, "coding_learning_claim": False,
            "reward": "synthetic: zero, then indicator of lexicographically smallest text in each group",
            "grpo_configuration_shared_with_cache_trial": True,
            "limitations": "short non-code prompt; validates optimizer wiring, not long-code learning or a useful skill"}
    write_private(directory / "plan.json", canonical_json(plan))
    write_private(directory / "source_snapshot.json", canonical_json(snapshot))
    write_private(directory / "grader-control.json", canonical_json(grader_evidence))
    write_private(directory / "runtime.json", canonical_json({
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "gpu": torch.cuda.get_device_name(), "image_id": image.object_id}))
    artifacts.commit()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    messages = [{"role": "user", "content": "Write one short sentence describing an imaginary animal."}]

    def parameter_hash(policy):
        h = hashlib.sha256()
        for name, p in policy.named_parameters():
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    results = {}
    try:
        initial_hash = parameter_hash(model)
        for condition, steps in plan["conditions"].items():
            current = create_run_directory(str(directory / condition))
            before_hash = parameter_hash(model)
            if before_hash != initial_hash:
                raise RuntimeError("both controls must start from the same initial weights")
            calls = []

            def synthetic_reward(completions, **kwargs):
                if len(calls) >= steps:
                    raise RuntimeError("control reward-call budget exceeded")
                raw = [c[0]["content"] if isinstance(c, list) else c for c in completions]
                # Preserve raw evidence even when a degenerate group aborts the control.
                write_private(current / f"raw-{len(calls):04d}.json", canonical_json(raw))
                artifacts.commit()
                rewards = control_rewards(raw, condition)
                write_private(current / f"rewards-{len(calls):04d}.json", canonical_json(rewards))
                calls.append(rewards)
                artifacts.commit()
                print(condition, "synthetic rewards:", rewards, flush=True)
                return rewards

            set_seed(plan["seed"])
            args = GRPOConfig(**grpo_kwargs(current / "trainer", max_steps=steps,
                                           completion_tokens=32, seed=plan["seed"]))
            write_private(current / "trainer_config.json", args.to_json_string())
            trainer = GRPOTrainer(model=model, args=args, reward_funcs=synthetic_reward,
                train_dataset=Dataset.from_list([{"prompt": messages}] * 4), processing_class=tokenizer)
            trainer.train()
            after_hash = parameter_hash(model)
            metrics = {"global_step": trainer.state.global_step, "rewards": calls,
                       "log_history": trainer.state.log_history,
                       "before_parameter_hash": before_hash, "after_parameter_hash": after_hash,
                       "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                       "parameters_changed": before_hash != after_hash}
            checkpoint = current / "checkpoint"
            trainer.save_model(str(checkpoint))
            tokenizer.save_pretrained(checkpoint)
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=False,
                dtype=torch.float32, attn_implementation="sdpa").to("cuda")
            metrics["checkpoint_reload_hash_matches"] = parameter_hash(model) == after_hash
            write_private(current / "metrics.json", canonical_json(metrics))
            artifacts.commit()
            require_control_evidence(metrics, condition)
            results[condition] = metrics
            print(condition, "PASS; weights changed:", metrics["parameters_changed"], flush=True)
        result = {"run_id": run_id, "kind": "synthetic_grpo_optimizer_control", "passed": True,
                  "coding_learning_claim": False, "conditions": results,
                  "gpu_function_seconds": time.monotonic() - started, "plan": plan}
        write_private(directory / "summary.json", canonical_json(result))
        artifacts.commit()
        return result
    except Exception as exc:
        write_private(directory / "error.json", canonical_json({"type": type(exc).__name__,
                      "elapsed_seconds": time.monotonic() - started}))
        artifacts.commit()
        raise


@app.local_entrypoint()
def main(grader_report: str, allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for one bounded L4 control job")
    evidence = json.loads(Path(grader_report).read_text())
    require_grader_controls(evidence)
    run_id = "qwen-optimizer-control-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    paths = sorted(Path("verifier_rl").glob("*.py")) + [Path(__file__)]
    snapshot = {p.name if p.name == Path(__file__).name else p.as_posix(): p.read_text() for p in paths}
    write_private(directory / "source_snapshot.json", canonical_json(snapshot))
    call = run_control.spawn(run_id, evidence, snapshot)
    write_private(directory / "launch.json", canonical_json({"run_id": run_id, "call_id": call.object_id}))
    print("Optimizer control:", run_id, flush=True)
    result = call.get()
    write_private(directory / "summary.json", canonical_json(result))
    print("Controls passed:", result["passed"], "Evidence:", directory / "summary.json")
