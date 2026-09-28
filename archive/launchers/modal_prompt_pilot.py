"""Eight paired seeds comparing the original prompt with interface reminders.

One bounded L4 generation job followed by CPU grading. Every completion and
candidate report is persisted before continuing; no training is performed.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import modal

from verifier_rl.baseline import inspect_completion
from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.grading import evaluate_candidate, rejected_extraction_report
from verifier_rl.modal_backend import ModalBackend, RUNNER
from verifier_rl.model_trial import canonical_prompt, validate_run_id
from verifier_rl.prompt_pilot import pilot_suites, protocol_plan, schedule, summarize
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import canonical_json, digest

app = modal.App("verifier-rl-interface-prompt-pilot")
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
def generate(run_id: str, original: str):
    import hashlib
    import importlib.metadata
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    validate_run_id(run_id)
    started = time.monotonic()
    plan = protocol_plan(original)
    directory = create_run_directory(f"/artifacts/{run_id}/generation")
    write_private(directory / "plan.json", canonical_json(plan))
    runtime = {"gpu": torch.cuda.get_device_name(), "gpu_image_id": gpu_image.object_id,
               "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
               "weights_dtype": "float32", "inference_autocast": "bfloat16"}
    write_private(directory / "runtime.json", canonical_json(runtime))
    artifacts.commit()
    tokenizer = AutoTokenizer.from_pretrained(plan["model_id"], revision=plan["model_revision"],
                                              trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        plan["model_id"], revision=plan["model_revision"], trust_remote_code=False,
        dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model.eval()

    def parameter_hash():
        h = hashlib.sha256()
        for name, parameter in model.named_parameters():
            h.update(name.encode())
            h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    before_hash = parameter_hash()
    inputs = {}
    for arm, prompt in plan["prompts"].items():
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False, add_generation_prompt=True)
        inputs[arm] = tokenizer(formatted, return_tensors="pt").to("cuda")
    samples = []
    for arm, seed in schedule():
        set_seed(seed)
        encoded = inputs[arm]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = model.generate(**encoded, max_new_tokens=plan["max_completion_tokens"],
                                    do_sample=True, temperature=plan["temperature"],
                                    top_p=plan["top_p"], top_k=plan["top_k"],
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
        completion = tokens[0, encoded["input_ids"].shape[1]:]
        sample = inspect_completion(tokenizer.decode(completion, skip_special_tokens=True))
        sample.update({"arm": arm, "seed": seed, "prompt_hash": plan["prompt_hashes"][arm],
                       "prompt_tokens": int(encoded["input_ids"].shape[1]), "tokens": len(completion),
                       "hit_token_cap": len(completion) == plan["max_completion_tokens"],
                       "ended_with_eos": bool(len(completion) and completion[-1].item() == tokenizer.eos_token_id)})
        samples.append(sample)
        write_private(directory / f"{arm}-{seed}.json", canonical_json(sample))
        artifacts.commit()
        print(f"Generated {len(samples)}/16: {arm} seed={seed}, "
              f"syntax={sample['syntax_valid']}, cap={sample['hit_token_cap']}", flush=True)
    after_hash = parameter_hash()
    result = {"run_id": run_id, "plan": plan, "runtime": runtime, "samples": samples,
              "before_parameter_hash": before_hash, "after_parameter_hash": after_hash,
              "parameters_unchanged": before_hash == after_hash,
              "gpu_function_seconds": time.monotonic() - started}
    write_private(directory / "summary.json", canonical_json(result))
    artifacts.commit()
    if not result["parameters_unchanged"]:
        raise RuntimeError("generation unexpectedly changed model parameters")
    return result


@app.function(image=cpu_image, cpu=1, memory=1024, timeout=1500,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def run_pilot(run_id: str, setup: dict, original: str, conformance: dict, source_snapshot: dict):
    validate_run_id(run_id)
    require_current_conformance(conformance, setup["sandbox_image_id"])
    started = time.monotonic()
    plan = protocol_plan(original)
    directory = create_run_directory(f"/artifacts/{run_id}")
    suites = pilot_suites()
    write_private(directory / "plan.json", canonical_json(plan))
    write_private(directory / "setup.json", canonical_json(setup))
    write_private(directory / "source_snapshot.json", canonical_json(source_snapshot))
    write_private(directory / "conformance.json", canonical_json(conformance))
    save_suites(directory, suites)
    artifacts.commit()
    try:
        generation = generate.remote(run_id, original)
        # Refresh the shared mount after another container committed generation.
        artifacts.reload()
        write_private(directory / "generation.json", canonical_json(generation))
        artifacts.commit()

        async def grade():
            backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"],
                                   creation_interval_seconds=plan["creation_interval_seconds"])
            reports = []
            for sample in generation["samples"]:
                arm, seed = sample["arm"], sample["seed"]
                write_private(directory / f"{arm}-{seed}-started.json", canonical_json(sample))
                await artifacts.commit.aio()
                if sample["extraction_status"].startswith("rejected_"):
                    report = rejected_extraction_report(sample["source"], sample["extraction_status"], suites)
                else:
                    report = await evaluate_candidate(sample["source"], suites, backend,
                                                      concurrency=plan["concurrency"], max_retries=0)
                report.update(arm=arm, seed=seed)
                write_private(directory / f"{arm}-{seed}-result.json", canonical_json(report))
                await artifacts.commit.aio()
                if any(s["infrastructure_errors"] for s in report["suites"]):
                    raise RuntimeError("unscored candidate; saved evidence and stopped pilot")
                reports.append(report)
                print(f"Graded {len(reports)}/16: {arm} seed={seed}: " + ", ".join(
                    f"{s['suite']}={s['passed_count']}/{s['total']}" for s in report["suites"]), flush=True)
            return reports

        reports = asyncio.run(grade())
        result = summarize(generation, reports)
        result.update(controller_seconds=time.monotonic() - started, runner_hash=digest(RUNNER),
                      cpu_image_id=cpu_image.object_id)
        write_private(directory / "reports.json", canonical_json(reports))
        write_private(directory / "summary.json", canonical_json(result))
        artifacts.commit()
        return {"generation": generation, "reports": reports, "summary": result}
    except Exception as exc:
        write_private(directory / "error.json", canonical_json({"type": type(exc).__name__,
                      "elapsed_seconds": time.monotonic() - started}))
        artifacts.commit()
        raise


@app.local_entrypoint()
def main(setup_file: str, cpu_report: str, allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required; pilot uses one GPU generation job and bounded CPU execution")
    setup = json.loads(Path(setup_file).read_text())
    conformance = json.loads(Path(cpu_report).read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    original = canonical_prompt(Path("task_001_expiring_cache.txt").read_text())
    plan = protocol_plan(original)
    run_id = "qwen-prompt-pilot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    snapshot_paths = sorted(Path("verifier_rl").glob("*.py")) + [Path(__file__)]
    snapshot = {p.name if p.name == Path(__file__).name else p.as_posix(): p.read_text()
                for p in snapshot_paths}
    write_private(directory / "source_snapshot.json", canonical_json(snapshot))
    write_private(directory / "plan.json", canonical_json(plan))
    write_private(directory / "setup.json", canonical_json(setup))
    write_private(directory / "conformance.json", canonical_json(conformance))
    print("Frozen plan:", directory / "plan.json", flush=True)
    call = run_pilot.spawn(run_id, setup, original, conformance, snapshot)
    write_private(directory / "launch.json", canonical_json({"run_id": run_id, "call_id": call.object_id}))
    print("Remote controller:", call.object_id, flush=True)
    result = call.get()
    for name in ("generation", "reports", "summary"):
        write_private(directory / f"{name}.json", canonical_json(result[name]))
    print("Pilot complete:", run_id)
    for arm, row in result["summary"]["arms"].items():
        print(arm, "full-suite passes:", row["full_suite_pass_counts"],
              "all outputs valid:", row["candidates_with_all_outputs_valid"], "/8")
    print("Local evidence:", directory)
