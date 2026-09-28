"""One bounded SFT warm-start followed by protected, matched cache evaluation."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import modal

from verifier_rl.baseline import inspect_completion
from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.model_trial import evaluate_submission, validate_run_id
from verifier_rl.modal_backend import ModalBackend, RUNNER
from verifier_rl.partial_reward import validate_reward_design
from verifier_rl.sft_data import collate_rows, dataset_manifest, encode_example
from verifier_rl.sft_trial import evaluation_suites, pilot_plan, require_sft_evidence, summarize_pilot
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import canonical_json, digest
from verifier_rl.training import require_control_evidence

app = modal.App("verifier-rl-sft-warmstart-pilot")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
             .add_local_python_source("verifier_rl"))
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("modal==1.5.5", "torch==2.8.0", "transformers==4.57.1",
                          "trl==0.28.0", "datasets==3.5.1", "accelerate==1.12.0")
             .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
             .add_local_python_source("verifier_rl"))


@app.function(image=gpu_image, gpu="L4", cpu=2, memory=16384, timeout=900,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def train_and_generate(run_id: str, plan: dict):
    import gc
    import hashlib
    import importlib.metadata
    import time
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    validate_run_id(run_id)
    started = time.monotonic()
    directory = create_run_directory(f"/artifacts/{run_id}/gpu")
    manifest = dataset_manifest()
    if digest(canonical_json(manifest)) != plan["dataset_validation"]["manifest_hash"]:
        raise ValueError("SFT dataset changed after plan was frozen")
    write_private(directory / "plan.json", canonical_json(plan))
    write_private(directory / "dataset.json", canonical_json(manifest))
    runtime = {"gpu": torch.cuda.get_device_name(), "gpu_image_id": gpu_image.object_id,
               "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    write_private(directory / "runtime.json", canonical_json(runtime))
    artifacts.commit()

    def parameter_hash(policy):
        h = hashlib.sha256()
        for name, p in policy.named_parameters():
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    try:
        set_seed(plan["sft_seed"])
        tokenizer = AutoTokenizer.from_pretrained(plan["model_id"], revision=plan["revision"], trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        encoded = {split: [encode_example(tokenizer, e, plan["max_training_tokens"])
                           for e in manifest["examples"] if e["split"] == split] for split in ("train", "development")}
        token_stats = {split: [{"tokens": len(row["input_ids"]), "target_tokens": sum(x != -100 for x in row["labels"]),
                               "eos_supervised": tokenizer.eos_token_id in row["labels"]} for row in rows]
                       for split, rows in encoded.items()}
        write_private(directory / "tokenization.json", canonical_json({"counts": token_stats,
                      "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
                      "chat_template_hash": digest(tokenizer.chat_template)}))
        artifacts.commit()
        datasets = {split: Dataset.from_list(rows) for split, rows in encoded.items()}
        model = AutoModelForCausalLM.from_pretrained(plan["model_id"], revision=plan["revision"],
            trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
        before_hash = parameter_hash(model)
        samples = []

        def generate(policy, arm):
            policy.eval()
            tokenizer.padding_side = "left"
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": plan["prompt"]}],
                                                     tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            for seed in plan["seeds"]:
                set_seed(seed)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = policy.generate(**inputs, max_new_tokens=plan["max_completion_tokens"],
                        do_sample=True, temperature=plan["temperature"], top_p=plan["top_p"], top_k=plan["top_k"],
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
                completion = tokens[0, inputs["input_ids"].shape[1]:]
                sample = inspect_completion(tokenizer.decode(completion, skip_special_tokens=True))
                sample.update(arm=arm, seed=seed, prompt_hash=plan["prompt_hash"], tokens=len(completion),
                              hit_token_cap=len(completion) == plan["max_completion_tokens"],
                              ended_with_eos=bool(len(completion) and completion[-1].item() == tokenizer.eos_token_id))
                samples.append(sample)
                write_private(directory / f"{arm}-{seed}.json", canonical_json(sample))
                artifacts.commit()
                print("Generated", arm, seed, "syntax", sample["syntax_valid"], flush=True)

        generate(model, "before")
        if parameter_hash(model) != before_hash:
            raise RuntimeError("baseline generation changed parameters")
        tokenizer.padding_side = "right"
        model.config.use_cache = False
        set_seed(plan["sft_seed"])
        args = TrainingArguments(output_dir=str(directory / "trainer"), max_steps=16,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=2e-5, weight_decay=0.0, optim="adamw_torch", lr_scheduler_type="constant",
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy="no", eval_strategy="no", logging_steps=1, report_to="none", seed=plan["sft_seed"],
            data_seed=plan["sft_seed"], dataloader_num_workers=0, prediction_loss_only=True)
        write_private(directory / "trainer_config.json", args.to_json_string())

        def collate(rows):
            return {k: torch.tensor(v, dtype=torch.long) for k, v in collate_rows(rows, tokenizer.pad_token_id).items()}

        trainer = Trainer(model=model, args=args, train_dataset=datasets["train"], eval_dataset=datasets["development"],
                          data_collator=collate, processing_class=tokenizer)
        losses = {}
        for split in datasets:
            losses[split + "_before"] = trainer.evaluate(datasets[split])["eval_loss"]
        trainer.train()
        for split in datasets:
            losses[split + "_after"] = trainer.evaluate(datasets[split])["eval_loss"]
        after_hash = parameter_hash(model)
        metrics = {"global_step": trainer.state.global_step, "log_history": trainer.state.log_history,
                   "before_parameter_hash": before_hash, "after_parameter_hash": after_hash,
                   "supervised_losses": losses,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        checkpoint = directory / "checkpoint"
        model.config.use_cache = True
        trainer.save_model(str(checkpoint))
        tokenizer.save_pretrained(checkpoint)
        write_private(directory / "training-before-reload.json", canonical_json(metrics))
        artifacts.commit()
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=False,
                         dtype=torch.float32, attn_implementation="sdpa").to("cuda")
        metrics["checkpoint_reload_hash_matches"] = parameter_hash(reloaded) == after_hash
        write_private(directory / "training.json", canonical_json(metrics))
        artifacts.commit()
        require_sft_evidence(metrics)
        generate(reloaded, "after")
        if parameter_hash(reloaded) != after_hash:
            raise RuntimeError("post-SFT generation changed parameters")
        result = {"run_id": run_id, "plan": plan, "training": metrics, "samples": samples, "runtime": runtime,
                  "checkpoint": f"{run_id}/gpu/checkpoint", "gpu_function_seconds": time.monotonic() - started}
        write_private(directory / "summary.json", canonical_json(result))
        artifacts.commit()
        return result
    except Exception as exc:
        write_private(directory / "error.json", canonical_json({"type": type(exc).__name__,
                       "elapsed_seconds": time.monotonic() - started}))
        artifacts.commit()
        raise


@app.function(image=cpu_image, cpu=1, memory=1024, timeout=1500, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def run_pilot(run_id: str, setup: dict, conformance: dict, document: str, snapshot: dict):
    validate_run_id(run_id)
    require_current_conformance(conformance, setup["sandbox_image_id"])
    plan = pilot_plan(document)
    directory = create_run_directory(f"/artifacts/{run_id}")
    suites = evaluation_suites()
    for name, data in (("plan", plan), ("setup", setup), ("conformance", conformance),
                       ("source_snapshot", snapshot), ("reward-validation", validate_reward_design())):
        write_private(directory / f"{name}.json", canonical_json(data))
    save_suites(directory, suites)
    artifacts.commit()
    try:
        generation = train_and_generate.remote(run_id, plan)
        artifacts.reload()
        write_private(directory / "generation.json", canonical_json(generation))
        artifacts.commit()

        async def grade():
            backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=.26)
            reports = []
            for sample in generation["samples"]:
                report = await evaluate_submission(sample, suites, backend)
                report.update(arm=sample["arm"], seed=sample["seed"])
                write_private(directory / f"{sample['arm']}-{sample['seed']}-result.json", canonical_json(report))
                await artifacts.commit.aio()
                if any(s["all_passed"] is None for s in report["suites"]):
                    raise RuntimeError("unscored SFT candidate: saved evidence, stopped evaluation")
                reports.append(report)
                print("Graded", sample["arm"], sample["seed"],
                      [(s["suite"], s["passed_count"], s["total"]) for s in report["suites"]], flush=True)
            return reports

        reports = asyncio.run(grade())
        result = summarize_pilot(generation, reports)
        result["runner_hash"] = digest(RUNNER)
        write_private(directory / "reports.json", canonical_json(reports))
        write_private(directory / "summary.json", canonical_json(result))
        artifacts.commit()
        return {"generation": generation, "reports": reports, "summary": result}
    except Exception as exc:
        write_private(directory / "error.json", canonical_json({"type": type(exc).__name__}))
        artifacts.commit()
        raise


@app.local_entrypoint()
def main(setup_file: str, cpu_report: str, optimizer_report: str, allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for one bounded SFT pilot")
    setup = json.loads(Path(setup_file).read_text())
    conformance = json.loads(Path(cpu_report).read_text())
    control = json.loads(Path(optimizer_report).read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    for name in ("zero", "mixed"):
        require_control_evidence(control["conditions"][name], name)
    validate_reward_design()
    document = Path("task_001_expiring_cache.txt").read_text()
    plan = pilot_plan(document)
    run_id = "qwen-sft-pilot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    paths = sorted(Path("verifier_rl").glob("*.py")) + [Path(__file__), Path("sft_partial_protocol.txt")]
    snapshot = {p.as_posix(): p.read_text() for p in paths}
    for name, data in (("plan", plan), ("source_snapshot", snapshot), ("dataset", dataset_manifest()),
                       ("reward-validation", validate_reward_design())):
        write_private(directory / f"{name}.json", canonical_json(data))
    call = run_pilot.spawn(run_id, setup, conformance, document, snapshot)
    write_private(directory / "launch.json", canonical_json({"run_id": run_id, "call_id": call.object_id}))
    print("SFT pilot:", run_id, flush=True)
    result = call.get()
    for name, data in result.items():
        write_private(directory / f"{name}.json", canonical_json(data))
    print("SFT pilot completed; RL eligible:", result["summary"]["rl_eligible"], flush=True)
    print("Evidence:", directory / "summary.json")
