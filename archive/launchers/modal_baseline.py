"""Generation-only cache baseline: one GPU job, then bounded CPU batches."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import modal

from verifier_rl.baseline import (BASELINE_VERSION, MODEL_REVISION, SEEDS, inspect_completion,
                                 summarize, validate_recovered_report)
from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.grading import evaluate_candidate, rejected_extraction_report
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.model_trial import (EXTRACTION_VERSION, MAX_COMPLETION_TOKENS, MODEL_ID,
                                    canonical_prompt, validate_run_id)
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import build_suites, canonical_json, digest

app = modal.App("verifier-rl-cache-baseline")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=True)
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
             .add_local_python_source("verifier_rl"))
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("modal==1.5.5", "torch==2.8.0", "transformers==4.57.1",
                          "trl==0.28.0", "datasets==3.5.1", "accelerate==1.12.0")
             .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
             .add_local_python_source("verifier_rl"))


@app.function(image=gpu_image, gpu="L4", cpu=2, memory=16384, timeout=900,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def generate(run_id: str, prompt: str):
    import hashlib
    import importlib.metadata
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    validate_run_id(run_id)
    started = time.monotonic()
    directory = create_run_directory(f"/artifacts/{run_id}/generation")
    config = {"protocol": BASELINE_VERSION, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "prompt": prompt, "prompt_hash": digest(prompt), "training": False,
              "seeds": list(SEEDS), "max_completion_tokens": MAX_COMPLETION_TOKENS,
              "temperature": 0.8, "top_p": 0.95, "top_k": 0,
              "extraction_version": EXTRACTION_VERSION, "gpu": torch.cuda.get_device_name(),
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    write_private(directory / "config.json", canonical_json(config))
    artifacts.commit()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="sdpa").to("cuda")
        model.eval()

        def parameter_hash():
            h = hashlib.sha256()
            for name, parameter in model.named_parameters():
                h.update(name.encode())
                h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        before_hash = parameter_hash()
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        samples = []
        for index, seed in enumerate(SEEDS):
            set_seed(seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_TOKENS,
                                        do_sample=True, temperature=0.8, top_p=0.95, top_k=0,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
            completion = tokens[0, inputs["input_ids"].shape[1]:]
            sample = inspect_completion(tokenizer.decode(completion, skip_special_tokens=True))
            sample.update({"index": index, "seed": seed, "tokens": len(completion),
                           "hit_token_cap": len(completion) == MAX_COMPLETION_TOKENS,
                           "ended_with_eos": bool(len(completion) and completion[-1].item() == tokenizer.eos_token_id)})
            samples.append(sample)
            write_private(directory / f"sample-{index:02}.json", canonical_json(sample))
            artifacts.commit()
            print(f"Generated {index + 1}/32: {sample['extraction_status']}, "
                  f"syntax={sample['syntax_valid']}, cap={sample['hit_token_cap']}", flush=True)
        after_hash = parameter_hash()
        result = {"run_id": run_id, "config": config, "samples": samples,
                  "before_parameter_hash": before_hash, "after_parameter_hash": after_hash,
                  "parameters_unchanged": before_hash == after_hash,
                  "gpu_function_seconds": time.monotonic() - started}
        if not result["parameters_unchanged"]:
            raise RuntimeError("generation-only baseline unexpectedly changed parameters")
        write_private(directory / "summary.json", canonical_json(result))
        artifacts.commit()
        return result
    except Exception as exc:
        write_private(directory / "error.json", canonical_json({"type": type(exc).__name__}))
        artifacts.commit()
        raise


@app.function(image=cpu_image, cpu=2, memory=2048, timeout=600,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def grade_candidates(setup: dict, run_id: str, batch_index: int, samples: list[dict]):
    validate_run_id(run_id)
    if type(batch_index) is not int or not 0 <= batch_index < 16 or not 1 <= len(samples) <= 2:
        raise ValueError("baseline batch budget exceeded")
    allowed = SEEDS[batch_index * 2:batch_index * 2 + 2]
    if len({s["seed"] for s in samples}) != len(samples) or any(s["seed"] not in allowed for s in samples):
        raise ValueError("batch seeds do not match frozen protocol")
    directory = create_run_directory(f"/artifacts/{run_id}/batch-{batch_index}")
    suites = build_suites()
    save_suites(directory, suites)
    write_private(directory / "config.json", canonical_json({
        "setup": setup, "seeds": [s["seed"] for s in samples], "concurrency": 8,
        "creation_interval_seconds": 0.26,
        "max_retries": 0, "sequential_candidates": True, "training": False}))
    artifacts.commit()

    async def evaluate():
        backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=0.26)
        reports = []
        for sample in samples:
            write_private(directory / f"{sample['seed']}-started.json", canonical_json(sample))
            await artifacts.commit.aio()
            if sample["extraction_status"].startswith("rejected_"):
                report = rejected_extraction_report(sample["source"], sample["extraction_status"], suites)
            else:
                report = await evaluate_candidate(sample["source"], suites, backend,
                                                  concurrency=8, max_retries=0)
            report["sample_seed"] = sample["seed"]
            write_private(directory / f"{sample['seed']}-result.json", canonical_json(report))
            await artifacts.commit.aio()
            reports.append(report)
            print(f"Seed {sample['seed']}: " + ", ".join(
                f"{s['suite']}={s['passed_count']}/{s['total']} infra={s['infrastructure_errors']}"
                for s in report["suites"]), flush=True)
            if any(s["infrastructure_errors"] for s in report["suites"]):
                raise RuntimeError("infrastructure error: preserving candidate result and stopping baseline")
        return reports

    return asyncio.run(evaluate())


@app.local_entrypoint()
def main(setup_file: str, cpu_report: str, previous_trial: str, resume_generation: str = "",
         recover_evaluation_runs: str = ""):
    setup = json.loads(Path(setup_file).read_text())
    conformance = json.loads(Path(cpu_report).read_text())
    previous = json.loads(Path(previous_trial).read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    prompt = canonical_prompt(Path("task_001_expiring_cache.txt").read_text())
    if (previous["config"]["model_revision"] != MODEL_REVISION or
            previous["config"]["prompt_hash"] != digest(prompt)):
        raise ValueError("baseline must preserve previous model revision and task prompt")
    run_id = "qwen-baseline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    write_private(directory / "inputs.json", canonical_json({"setup": setup, "cpu_report": cpu_report,
                 "previous_trial": previous_trial, "protocol": BASELINE_VERSION,
                 "gpu_image_id": gpu_image.object_id, "cpu_image_id": cpu_image.object_id,
                 "max_candidates": 32, "max_gpu_seconds": 900,
                 "max_cpu_batch_seconds": 600, "max_batches": 16, "max_parallel_sandboxes": 8,
                 "creation_interval_seconds": 0.26, "resume_generation": resume_generation or None,
                 "recover_evaluation_runs": recover_evaluation_runs}))
    if resume_generation:
        generation = json.loads(Path(resume_generation).read_text())
        if (generation["config"]["model_revision"] != MODEL_REVISION or
                generation["config"]["prompt_hash"] != digest(prompt) or
                generation["config"]["extraction_version"] != EXTRACTION_VERSION or
                generation["config"]["max_completion_tokens"] != MAX_COMPLETION_TOKENS or
                not generation["parameters_unchanged"] or
                [s["seed"] for s in generation["samples"]] != list(SEEDS)):
            raise ValueError("saved generation does not match frozen baseline protocol")
        for sample in generation["samples"]:
            if inspect_completion(sample["raw"])["source"] != sample["source"]:
                raise ValueError("saved extracted source differs from current extraction contract")
        generation["generation_source_run_id"] = generation["run_id"]
        generation["run_id"] = run_id
        print("Resuming CPU evaluation of saved samples; no GPU function will be called.")
    else:
        generation = generate.remote(run_id, prompt)
    write_private(directory / "generation.json", canonical_json(generation))
    recovered = {}
    recovery_sources = {}
    if recover_evaluation_runs:
        roots = recover_evaluation_runs.split(",")
        if len(roots) > 4:
            raise ValueError("at most four explicitly named recovery runs")
        suites = build_suites()
        for root in roots:
            validate_run_id(root)
            for entry in artifacts.iterdir(root, recursive=True):
                if not entry.path.endswith("-result.json"):
                    continue
                report = json.loads(b"".join(artifacts.read_file(entry.path)))
                seed = report.get("sample_seed")
                if seed not in SEEDS:
                    raise ValueError("unexpected seed in recovery report")
                sample = generation["samples"][SEEDS.index(seed)]
                if validate_recovered_report(sample, report, suites, setup["sandbox_image_id"]) and seed not in recovered:
                    recovered[seed] = report
                    recovery_sources[seed] = entry.path
                    write_private(directory / f"recovered-{seed}.json", canonical_json(report))
        print(f"Recovered {len(recovered)} completed candidate reports; they will not be rerun.", flush=True)
    write_private(directory / "recovery.json", canonical_json(recovery_sources))
    by_seed = dict(recovered)
    for batch_index in range(16):
        batch = [s for s in generation["samples"][batch_index * 2:batch_index * 2 + 2]
                 if s["seed"] not in by_seed]
        if batch:
            results = grade_candidates.remote(setup, run_id, batch_index, batch)
            write_private(directory / f"grading-call-{batch_index}.json", canonical_json(results))
            by_seed.update({r["sample_seed"]: r for r in results})
    reports = [by_seed[s] for s in SEEDS]
    # Logical groups of four remain unchanged despite smaller execution batches.
    for group in range(8):
        write_private(directory / f"batch-{group}.json", canonical_json(reports[group * 4:group * 4 + 4]))
    summary = summarize(generation["samples"], reports)
    summary.update({"run_id": run_id, "gpu_function_seconds": generation["gpu_function_seconds"],
                    "parameters_unchanged": generation["parameters_unchanged"]})
    write_private(directory / "summary.json", canonical_json(summary))
    print("Baseline complete:", run_id)
    print("Syntax-valid:", summary["syntax_valid_count"], "/32")
    print("Suite full passes:", summary["suite_full_pass_counts"])
    print("Mixed reward groups:", summary["mixed_reward_groups_of_four"])
    print("Records:", directory)
