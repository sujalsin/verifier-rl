"""Audit three abnormal exits from the fixed 32-sample baseline, CPU only."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import modal

from verifier_rl.cli import create_run_directory, write_private
from verifier_rl.grading import evaluate_candidate
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.model_trial import validate_run_id
from verifier_rl.suites import Suite, build_suites, canonical_json, digest

app = modal.App("verifier-rl-baseline-exit-audit")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
             .add_local_python_source("verifier_rl"))
TARGETS = ((4026, "g3", "target/put_only"), (4028, "audit", "exhaustive/0103"),
           (4030, "audit", "exhaustive/0272"))


@app.function(image=cpu_image, cpu=1, memory=1024, timeout=180, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def recheck(parent_run: str, run_id: str):
    validate_run_id(parent_run)
    validate_run_id(run_id)
    artifacts.reload()
    parent = Path(f"/artifacts/{parent_run}")
    generation = json.loads((parent / "generation.json").read_text())
    setup = json.loads((parent / "inputs.json").read_text())["setup"]
    samples = {s["seed"]: s for s in generation["samples"]}
    suites = {s.name: s for s in build_suites()}
    directory = create_run_directory(f"/artifacts/{run_id}")
    write_private(directory / "plan.json", canonical_json({"parent_run": parent_run, "targets": TARGETS,
                 "reason": "abnormal code-137 exits; stderr is untrusted and not proof of cause"}))
    artifacts.commit()

    async def evaluate():
        backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=.26)
        records = []
        for seed, suite_name, case_name in TARGETS:
            suite = suites[suite_name]
            case = next(c for c in suite.cases if c.name == case_name)
            original = json.loads((parent / f"batch-{(seed - 4000) // 4}.json").read_text())
            original = next(r for r in original if r["sample_seed"] == seed)
            if original["candidate_hash"] != digest(samples[seed]["source"]):
                raise ValueError("source mismatch")
            prior_suite = next(s for s in original["suites"] if s["suite"] == suite_name)
            prior = next(o for o in prior_suite["outcomes"] if o["input_hash"] == case.input_hash)
            if prior["attempts"][-1]["metadata"]["returncode"] != 137:
                raise ValueError("target is not the recorded abnormal exit")
            report = await evaluate_candidate(samples[seed]["source"],
                (Suite("abnormal_exit_recheck", "audit", (case,), suite.seed),), backend,
                concurrency=1, max_retries=0)
            record = {"seed": seed, "original_suite": suite_name, "case": case_name,
                      "original_outcome": prior, "recheck": report}
            records.append(record)
            write_private(directory / f"{seed}.json", canonical_json(record))
            await artifacts.commit.aio()
        return {"run_id": run_id, "parent_run": parent_run, "records": records}

    result = asyncio.run(evaluate())
    write_private(directory / "summary.json", canonical_json(result))
    artifacts.commit()
    return result


@app.local_entrypoint()
def main(parent_run: str):
    run_id = "qwen-recheck-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    result = recheck.remote(parent_run, run_id)
    write_private(directory / "summary.json", canonical_json(result))
    print("Recheck:", run_id)
    for record in result["records"]:
        outcome = record["recheck"]["suites"][0]["outcomes"][0]
        print(record["seed"], outcome["reason"], outcome["attempts"][0]["metadata"].get("returncode"))
