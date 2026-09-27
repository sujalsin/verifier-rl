"""One bounded GPU smoke job and protected CPU grading, not a research run.

Requires a passing CPU trial report for the same sandbox image.
No deployed endpoint; candidate programs execute only in clean CPU sandboxes.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import modal

from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.grading import evaluate_candidate
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.model_trial import (EXTRACTION_VERSION, MAX_COMPLETION_TOKENS, MODEL_ID,
                                    canonical_prompt, extract_completion,
                                    require_scored_rewards, validate_run_id)
from verifier_rl.suites import build_suites, canonical_json, digest

app = modal.App("verifier-rl-cache-model-trial")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=True)
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("modal==1.5.5").add_local_python_source("verifier_rl"))
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("modal==1.5.5", "torch==2.8.0", "transformers==4.57.1",
                          "trl==0.28.0", "datasets==3.5.1", "accelerate==1.12.0")
             .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
             .add_local_python_source("verifier_rl"))


@app.function(image=cpu_image, cpu=1, memory=1024, timeout=1800,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def grade_batch(setup: dict, run_id: str, batch_id: str, sources: list[str], audit: bool = False):
    validate_run_id(run_id)
    if batch_id not in ("train-0", "evaluation"):
        raise ValueError("unexpected batch ID")
    if not 1 <= len(sources) <= 4 or audit != (batch_id == "evaluation"):
        raise ValueError("bounded trial batch contract violated")
    directory = create_run_directory(f"/artifacts/{run_id}/{batch_id}")
    suites = build_suites()
    selected = suites if audit else tuple(s for s in suites if s.name == "g3")
    save_suites(directory, selected)
    write_private(directory / "inputs.json", canonical_json({
        "sources": sources, "setup": setup, "concurrency": 4, "max_retries": 0,
        "audit_not_used_for_training": audit,
    }))
    artifacts.commit()

    async def evaluate():
        backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"])
        reports = []
        for index, source in enumerate(sources):
            report = await evaluate_candidate(source, selected, backend, concurrency=4, max_retries=0)
            reports.append(report)
            write_private(directory / f"candidate-{index}.json", canonical_json(report))
            await artifacts.commit.aio()
            print(f"{batch_id} candidate {index}: " + ", ".join(
                f"{s['suite']}={s['passed_count']}/{s['total']} infra={s['infrastructure_errors']}"
                for s in report["suites"]), flush=True)
            if any(s["infrastructure_errors"] for s in report["suites"]):
                raise RuntimeError("infrastructure failure: saved report, stopping batch")
        return reports

    return asyncio.run(evaluate())


@app.function(image=gpu_image, gpu="L4", cpu=2, memory=16384,
              timeout=900, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def train_smoke(setup: dict, run_id: str, prompt: str):
    import gc
    import hashlib
    import importlib.metadata
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    validate_run_id(run_id)
    started = time.monotonic()
    directory = create_run_directory(f"/artifacts/{run_id}/gpu")
    revision = HfApi().model_info(MODEL_ID).sha
    config = {"model_id": MODEL_ID, "model_revision": revision, "prompt": prompt,
              "prompt_hash": digest(prompt), "setup": setup,
              "extraction_version": EXTRACTION_VERSION,
              "max_steps": 1, "group_size": 4, "max_completion_tokens": MAX_COMPLETION_TOKENS,
              "training_suite": "g3", "loss_type": "grpo", "beta": 0.0,
              "scale_rewards": "group", "learning_rate": 1e-6, "weight_decay": 0.0,
              "seed": 20260926, "evaluation_seeds": [3001, 3002],
              "gpu": torch.cuda.get_device_name(),
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "scope": "pipeline smoke only; no SFT; no efficacy or algorithm comparison claim"}
    write_private(directory / "config.json", canonical_json(config))
    artifacts.commit()
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    messages = [{"role": "user", "content": prompt}]

    def parameter_hash(policy):
        h = hashlib.sha256()
        for name, parameter in policy.named_parameters():
            h.update(name.encode())
            h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def sample(policy):
        policy.eval()
        encoded = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(encoded, return_tensors="pt").to("cuda")
        outputs = []
        for seed in config["evaluation_seeds"]:
            set_seed(seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = policy.generate(**inputs, max_new_tokens=MAX_COMPLETION_TOKENS,
                                         do_sample=True, temperature=0.8, top_p=0.95, top_k=0,
                                         pad_token_id=tokenizer.pad_token_id,
                                         eos_token_id=tokenizer.eos_token_id)
            completion_tokens = tokens[0, inputs["input_ids"].shape[1]:]
            raw = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            extraction = extract_completion(raw)
            outputs.append({"seed": seed, "raw": raw, "source": extraction.source,
                            "extraction_status": extraction.status,
                            "extraction_version": extraction.version,
                            "tokens": len(completion_tokens),
                            "hit_token_cap": len(completion_tokens) == MAX_COMPLETION_TOKENS})
        return outputs

    try:
        before_hash = parameter_hash(model)
        before = sample(model)
        write_private(directory / "baseline.json", canonical_json(before))
        artifacts.commit()
        calls = []

        def execution_reward(completions, **kwargs):
            if calls or len(completions) != 4:
                raise RuntimeError("smoke reward-call budget exceeded")
            raw = [c[0]["content"] if isinstance(c, list) else c for c in completions]
            extractions = [extract_completion(c) for c in raw]
            sources = [e.source for e in extractions]
            write_private(directory / "rollouts.json", canonical_json({
                "raw": raw, "sources": sources, "extraction_version": EXTRACTION_VERSION,
                "extraction_statuses": [e.status for e in extractions]}))
            artifacts.commit()
            reports = grade_batch.remote(setup, run_id, "train-0", sources)
            rewards = require_scored_rewards(reports)
            calls.append(rewards)
            write_private(directory / "rewards.json", canonical_json(rewards))
            artifacts.commit()
            print("GRPO rollout rewards:", rewards, flush=True)
            return rewards

        args = GRPOConfig(output_dir=str(directory / "trainer"), max_steps=1,
                          per_device_train_batch_size=1, gradient_accumulation_steps=4,
                          num_generations=4, max_completion_length=MAX_COMPLETION_TOKENS,
                          learning_rate=1e-6, weight_decay=0.0, lr_scheduler_type="constant",
                          bf16=True, gradient_checkpointing=True, use_vllm=False,
                          beta=0.0, loss_type="grpo", scale_rewards="group",
                          temperature=0.8, top_p=0.95, top_k=0,
                          save_strategy="no", logging_steps=1, report_to="none",
                          seed=config["seed"], data_seed=config["seed"])
        write_private(directory / "trainer_config.json", args.to_json_string())
        trainer = GRPOTrainer(model=model, args=args, reward_funcs=execution_reward,
                              train_dataset=Dataset.from_list([{"prompt": messages}] * 4),
                              processing_class=tokenizer)
        trained = trainer.train()
        after_hash = parameter_hash(model)
        checkpoint = directory / "checkpoint"
        trainer.save_model(str(checkpoint))
        tokenizer.save_pretrained(checkpoint)
        metrics = {"train_metrics": trained.metrics, "log_history": trainer.state.log_history,
                   "global_step": trainer.state.global_step, "rewards": calls,
                   "before_parameter_hash": before_hash, "after_parameter_hash": after_hash,
                   "parameters_changed": before_hash != after_hash,
                   "groups_with_reward_variation": sum(len(set(r)) > 1 for r in calls)}
        write_private(directory / "training.json", canonical_json(metrics))
        artifacts.commit()
        # Reload the actual saved checkpoint before sampling it. GPU is released
        # before the independent audit; no audit feedback reaches this trainer.
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=False,
                                                       dtype=torch.float32).to("cuda")
        metrics["checkpoint_reload_hash_matches"] = parameter_hash(reloaded) == after_hash
        if not metrics["checkpoint_reload_hash_matches"]:
            raise RuntimeError("saved checkpoint parameters do not match trained model")
        after = sample(reloaded)
        metrics["gpu_function_seconds"] = time.monotonic() - started
        result = {"run_id": run_id, "config": config, "metrics": metrics,
                  "before": before, "after": after,
                  "checkpoint": f"verifier-rl-cache-artifacts/{run_id}/gpu/checkpoint"}
        write_private(directory / "summary.json", canonical_json(result))
        artifacts.commit()
        return result
    except Exception as exc:
        write_private(directory / "error.json", canonical_json({"type": type(exc).__name__,
                                                               "elapsed_seconds": time.monotonic() - started}))
        artifacts.commit()
        raise


@app.local_entrypoint()
def main(setup_file: str, cpu_report: str):
    setup = json.loads(Path(setup_file).read_text())
    conformance = json.loads(Path(cpu_report).read_text())
    if not conformance.get("passed") or conformance.get("completed_checks") != 15:
        raise ValueError("a passing 15-check CPU trial is required")
    image_ids = {a["metadata"]["image_id"] for c in conformance["checks"]
                 for s in c["report"]["suites"] for o in s["outcomes"] for a in o["attempts"]}
    if image_ids != {setup["sandbox_image_id"]}:
        raise ValueError("sandbox image differs from the conformance-tested image")
    prompt = canonical_prompt(Path("task_001_expiring_cache.txt").read_text())
    run_id = "qwen-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    local = create_run_directory(f"runs/{run_id}")
    write_private(local / "inputs.json", canonical_json({"setup": setup, "cpu_report": cpu_report,
                                                        "prompt_hash": digest(prompt)}))
    result = train_smoke.remote(setup, run_id, prompt)
    write_private(local / "model.json", canonical_json(result))
    sources = [r["source"] for r in result["before"] + result["after"]]
    reports = grade_batch.remote(setup, run_id, "evaluation", sources, audit=True)
    write_private(local / "evaluation.json", canonical_json(reports))
    print("Model trial:", run_id)
    print("Training rewards:", result["metrics"]["rewards"])
    print("Parameters changed:", result["metrics"]["parameters_changed"])
    print("Checkpoint reload verified:", result["metrics"]["checkpoint_reload_hash_matches"])
    print("Saved local records:", local)
    print("Smoke test only: two samples per checkpoint do not establish improvement.")
