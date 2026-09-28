"""Replay five saved Qwen completions through the current CPU sandbox runner.

This is a bounded diagnostic replay, not generation, training, or model evaluation.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import modal

from verifier_rl.cli import create_run_directory, write_private
from verifier_rl.grading import evaluate_candidate, rejected_extraction_report
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.model_trial import validate_run_id
from verifier_rl.suites import Suite, build_suites, canonical_json, case, digest, get, put

app = modal.App("verifier-rl-diagnostic-replay")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
             .add_local_python_source("verifier_rl"))
TARGET_SEEDS = (4001, 4005, 4028, 4030, 4031)
REPLAY_CASE = case("same_put_then_get", [put(0), get(1)])


@app.function(image=cpu_image, cpu=2, memory=2048, timeout=300,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def replay(setup: dict, parent_run: str, run_id: str):
    validate_run_id(parent_run)
    validate_run_id(run_id)
    artifacts.reload()
    generation_path = Path(f"/artifacts/{parent_run}/generation.json")
    generation = json.loads(generation_path.read_text())
    samples = {sample["seed"]: sample for sample in generation["samples"]}
    if any(seed not in samples for seed in TARGET_SEEDS):
        raise ValueError("saved generation is missing a frozen diagnostic target")
    suite = Suite("diagnostic_replay", "audit", (REPLAY_CASE,), 20260924)
    directory = create_run_directory(f"/artifacts/{run_id}")
    plan = {"run_id": run_id, "parent_run": parent_run,
            "target_seeds": TARGET_SEEDS, "suite_hash": suite.fingerprint,
            "suite_version": suite.version, "gpu": False,
            "unique_candidate_input_executions_max": len(TARGET_SEEDS),
            "input": REPLAY_CASE.input_json, "expected": REPLAY_CASE.expected,
            "purpose": "diagnostic replay; audit reward is always null"}
    write_private(directory / "plan.json", canonical_json(plan))
    artifacts.commit()

    async def evaluate():
        backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"],
                               creation_interval_seconds=0.26)
        records = []
        for seed in TARGET_SEEDS:
            sample = samples[seed]
            if sample["extraction_status"].startswith("rejected_"):
                report = rejected_extraction_report(sample["source"],
                                                     sample["extraction_status"], (suite,))
            else:
                report = await evaluate_candidate(sample["source"], (suite,), backend,
                                                  concurrency=1, max_retries=0)
            record = {"seed": seed, "parent_run": parent_run,
                      "candidate_hash": digest(sample["source"]),
                      "extraction_status": sample["extraction_status"],
                      "report": report}
            records.append(record)
            write_private(directory / f"seed-{seed}.json", canonical_json(record))
            await artifacts.commit.aio()
        return {"run_id": run_id, "parent_run": parent_run,
                "target_seeds": TARGET_SEEDS, "records": records}

    result = asyncio.run(evaluate())
    write_private(directory / "summary.json", canonical_json(result))
    artifacts.commit()
    return result


@app.local_entrypoint()
def main(setup_file: str, parent_run: str):
    setup = json.loads(Path(setup_file).read_text())
    validate_run_id(parent_run)
    run_id = "qwen-diagnostic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    local = create_run_directory(f"runs/{run_id}")
    write_private(local / "plan.json", canonical_json({
        "parent_run": parent_run, "setup": setup, "target_seeds": TARGET_SEEDS,
        "cpu_only": True, "no_generation_or_training": True}))
    result = replay.remote(setup, parent_run, run_id)
    write_private(local / "summary.json", canonical_json(result))
    print("Replay:", run_id)
    for record in result["records"]:
        outcome = record["report"]["suites"][0]["outcomes"][0]
        attempt = outcome["attempts"][0]
        metadata = attempt["metadata"]
        print(record["seed"], outcome["reason"],
              f"stage={metadata.get('runner_stage', 'extraction')}",
              f"stdout={metadata.get('stdout_preview', '')[:100]!r}")
    print("Local report:", local / "summary.json")
    print("Persistent Modal volume: verifier-rl-cache-artifacts/" + run_id)
