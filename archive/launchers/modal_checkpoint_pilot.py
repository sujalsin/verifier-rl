"""Generation-only pinned 1.5B comparison; no optimizer or public endpoint."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import modal

from verifier_rl.baseline import inspect_completion
from verifier_rl.checkpoint_pilot import (MODEL_ID, REVISION, plan_for, reference_packet,
                                         summarize_comparison, summarize_policy)
from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.model_trial import evaluate_submission, validate_run_id
from verifier_rl.modal_backend import ModalBackend, RUNNER
from verifier_rl.sft_trial import evaluation_suites
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import canonical_json, digest

app = modal.App("verifier-rl-checkpoint-comparison")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
             .add_local_python_source("verifier_rl"))
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("modal==1.5.5", "torch==2.8.0", "transformers==4.57.1",
                          "trl==0.28.0", "datasets==3.5.1", "accelerate==1.12.0")
             .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
             .add_local_python_source("verifier_rl"))


@app.function(image=gpu_image, gpu="L4", cpu=2, memory=16384, timeout=600,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def generate(run_id: str, plan: dict):
    import hashlib
    import importlib.metadata
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

    validate_run_id(run_id)
    if plan["model_id"] != MODEL_ID or plan["revision"] != REVISION or plan["training"] is not False:
        raise ValueError("unexpected generation checkpoint")
    started = time.monotonic()
    directory = create_run_directory(f"/artifacts/{run_id}/generation")
    write_private(directory / "plan.json", canonical_json(plan))
    artifacts.commit()

    def parameter_hash(policy):
        h = hashlib.sha256()
        for name, p in policy.named_parameters():
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if digest(tokenizer.chat_template) != plan["chat_template_hash"]:
            raise ValueError("chat template changed across checkpoints; review before comparison")
        overrides = dict(max_new_tokens=plan["max_completion_tokens"], do_sample=True,
                         temperature=plan["temperature"], top_p=plan["top_p"], top_k=plan["top_k"],
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True,
                         repetition_penalty=plan["repetition_penalty"])

        native_repetition_penalties = {}
        def effective_config(model_id, revision):
            cfg = GenerationConfig.from_pretrained(model_id, revision=revision)
            native_repetition_penalties[model_id] = cfg.repetition_penalty
            if model_id == plan["comparison_model_id"] and cfg.repetition_penalty != plan["repetition_penalty"]:
                raise ValueError("saved 0.5B baseline repetition penalty does not match comparison override")
            cfg.update(**overrides)
            result = cfg.to_dict()
            for key in ("transformers_version", "_from_model_config", "_commit_hash"):
                result.pop(key, None)
            return result

        large_defaults = effective_config(MODEL_ID, REVISION)
        small_defaults = effective_config(plan["comparison_model_id"], plan["comparison_revision"])
        if large_defaults != small_defaults:
            write_private(directory / "different-generation-defaults.json",
                          canonical_json({"small": small_defaults, "large": large_defaults}))
            artifacts.commit()
            raise ValueError("effective generation defaults differ; stop instead of silently changing decoding")
        runtime = {"gpu": torch.cuda.get_device_name(), "gpu_image_id": gpu_image.object_id,
                   "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                   "weights_dtype": "float32", "inference_autocast": "bfloat16",
                   "chat_template_hash": digest(tokenizer.chat_template),
                   "effective_generation_config": large_defaults, "decoding_defaults_match": True,
                   "native_repetition_penalties": native_repetition_penalties}
        write_private(directory / "runtime.json", canonical_json(runtime))
        artifacts.commit()
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False,
                           dtype=torch.float32, attn_implementation="sdpa").to("cuda")
        model.eval()
        write_private(directory / "model_config.json", canonical_json(model.config.to_dict()))
        before_hash = parameter_hash(model)
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": plan["prompt"]}],
                                                  tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")
        samples = []
        for seed in plan["seeds"]:
            set_seed(seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model.generate(**inputs, **overrides)
            completion = tokens[0, inputs["input_ids"].shape[1]:]
            sample = inspect_completion(tokenizer.decode(completion, skip_special_tokens=True))
            sample.update(seed=seed, prompt_hash=plan["prompt_hash"], tokens=len(completion),
                          prompt_tokens=int(inputs["input_ids"].shape[1]),
                          hit_token_cap=len(completion) == plan["max_completion_tokens"],
                          ended_with_eos=bool(len(completion) and completion[-1].item() == tokenizer.eos_token_id))
            samples.append(sample)
            write_private(directory / f"sample-{seed}.json", canonical_json(sample))
            artifacts.commit()
            print("Generated 1.5B", seed, "syntax", sample["syntax_valid"], "tokens", len(completion), flush=True)
        after_hash = parameter_hash(model)
        result = {"run_id": run_id, "plan": plan, "runtime": runtime, "samples": samples,
                  "before_parameter_hash": before_hash, "after_parameter_hash": after_hash,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "gpu_function_seconds": time.monotonic() - started}
        write_private(directory / "summary.json", canonical_json(result))
        artifacts.commit()
        if before_hash != after_hash:
            raise RuntimeError("generation unexpectedly changed parameters")
        return result
    except Exception as exc:
        write_private(directory / "error.json", canonical_json({"type": type(exc).__name__,
                      "elapsed_seconds": time.monotonic() - started}))
        artifacts.commit()
        raise


@app.function(image=cpu_image, cpu=1, memory=1024, timeout=1200, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def run_pilot(run_id: str, setup: dict, conformance: dict, document: str, reference: dict, snapshot: dict):
    validate_run_id(run_id)
    require_current_conformance(conformance, setup["sandbox_image_id"])
    plan = plan_for(document)
    summarize_policy(reference["samples"], reference["reports"], setup["sandbox_image_id"], plan["prompt_hash"])
    directory = create_run_directory(f"/artifacts/{run_id}")
    suites = evaluation_suites()
    for name, data in (("plan", plan), ("setup", setup), ("conformance", conformance),
                       ("reference", reference), ("source_snapshot", snapshot)):
        write_private(directory / f"{name}.json", canonical_json(data))
    save_suites(directory, suites)
    artifacts.commit()
    try:
        generation = generate.remote(run_id, plan)
        artifacts.reload()
        write_private(directory / "generation.json", canonical_json(generation))
        artifacts.commit()

        async def grade():
            backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=.26)
            reports = []
            for sample in generation["samples"]:
                report = await evaluate_submission(sample, suites, backend)
                report["seed"] = sample["seed"]
                write_private(directory / f"sample-{sample['seed']}-result.json", canonical_json(report))
                await artifacts.commit.aio()
                if any(s["all_passed"] is None for s in report["suites"]):
                    raise RuntimeError("unscored 1.5B candidate; saved evidence and stopped")
                reports.append(report)
                print("Graded 1.5B", sample["seed"],
                      [(s["suite"], s["passed_count"], s["total"]) for s in report["suites"]], flush=True)
            return reports

        reports = asyncio.run(grade())
        result = summarize_comparison(generation, reports, reference, setup["sandbox_image_id"])
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
def main(setup_file: str, cpu_report: str, reference_run: str, allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for one bounded 1.5B generation/evaluation job")
    setup = json.loads(Path(setup_file).read_text())
    conformance = json.loads(Path(cpu_report).read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    document = Path("task_001_expiring_cache.txt").read_text()
    plan = plan_for(document)
    prior = Path(reference_run)
    reference = reference_packet(json.loads((prior / "generation.json").read_text()),
                                 json.loads((prior / "reports.json").read_text()), plan, setup["sandbox_image_id"])
    run_id = "qwen-checkpoint-pilot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    paths = sorted(Path("verifier_rl").glob("*.py")) + [Path(__file__), Path("checkpoint_1_5b_protocol.txt")]
    snapshot = {p.as_posix(): p.read_text() for p in paths}
    for name, data in (("plan", plan), ("source_snapshot", snapshot), ("reference", reference)):
        write_private(directory / f"{name}.json", canonical_json(data))
    call = run_pilot.spawn(run_id, setup, conformance, document, reference, snapshot)
    write_private(directory / "launch.json", canonical_json({"run_id": run_id, "call_id": call.object_id}))
    print("Checkpoint pilot:", run_id, flush=True)
    result = call.get()
    for name, data in result.items():
        write_private(directory / f"{name}.json", canonical_json(data))
    print("Pilot completed:", directory / "summary.json", flush=True)
    for name in ("original_0_5b", "original_1_5b"):
        print(name, "full G3 passes", result["summary"][name]["full_g3_passes"], "/8")
