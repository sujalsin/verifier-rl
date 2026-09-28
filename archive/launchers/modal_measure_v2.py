"""Re-evaluate saved 1.5B programs on v2 probes, using CPU sandboxes only."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import modal

from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.measurement_v2 import (REFERENCE_RUN, measurement_plan, measurement_summary,
                                       require_measurement_report)
from verifier_rl.model_trial import evaluate_submission, validate_run_id
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.reward_v2 import behavior_suite
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import canonical_json

app = modal.App("verifier-rl-behavior-v2-measurement")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
         .add_local_python_source("verifier_rl"))


@app.function(image=image, cpu=1, memory=1024, timeout=180, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def evaluate_one(run_id: str, sample: dict, setup: dict, plan: dict):
    validate_run_id(run_id)
    from verifier_rl.suites import digest
    from verifier_rl.model_trial import validate_submission
    validate_submission(sample)
    suite = behavior_suite()
    if (plan["suite_hash"] != suite.fingerprint or sample["seed"] not in plan["seeds"]
            or digest(sample["source"]) != plan["source_hashes"][str(sample["seed"])]):
        raise ValueError("candidate/suite differs from the frozen measurement plan")
    artifacts.reload()
    directory = Path(f"/artifacts/{run_id}/cases/{sample['seed']}")
    result_path = directory / "result.json"
    if result_path.exists():
        report = json.loads(result_path.read_text())
        require_measurement_report(sample, report, setup["sandbox_image_id"])
        return report  # Idempotent replay of a complete result, no execution.
    # If an intent exists without a result, interruption left execution count
    # unknown. Stop rather than silently repeating the candidate's tests.
    directory = create_run_directory(str(directory))
    write_private(directory / "intent.json", canonical_json({"seed": sample["seed"],
                  "source_hash": digest(sample["source"]), "suite_hash": suite.fingerprint}))
    artifacts.commit()  # Persist intent BEFORE creating any candidate sandbox.
    backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=.26)
    report = asyncio.run(evaluate_submission(sample, (suite,), backend, concurrency=4))
    report["seed"] = sample["seed"]
    write_private(result_path, canonical_json(report))
    artifacts.commit()
    require_measurement_report(sample, report, setup["sandbox_image_id"])
    return report


@app.function(image=image, cpu=1, memory=1024, timeout=900, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def measure(run_id: str, generation: dict, prior_reports: list, setup: dict, conformance: dict, snapshot: dict):
    validate_run_id(run_id)
    require_current_conformance(conformance, setup["sandbox_image_id"])
    plan = measurement_plan(generation, prior_reports)
    directory = Path(f"/artifacts/{run_id}")
    artifacts.reload()
    frozen = {"plan": plan, "generation": generation, "prior_reports": prior_reports,
              "setup": setup, "conformance": conformance, "source_snapshot": snapshot}
    if directory.exists():
        for name, value in frozen.items():
            if json.loads((directory / f"{name}.json").read_text()) != value:
                raise ValueError("incomplete or conflicting controller resume state")
    else:
        create_run_directory(str(directory))
        for name, value in frozen.items():
            write_private(directory / f"{name}.json", canonical_json(value))
        save_suites(directory, (behavior_suite(),))
        artifacts.commit()
    reports = []
    for sample in generation["samples"]:
        report = evaluate_one.remote(run_id, sample, setup, plan)
        reports.append(report)
        print("Measured", sample["seed"], report["suites"][0]["passed_count"], "/34", flush=True)
    artifacts.reload()
    summary = measurement_summary(generation, prior_reports, reports, setup["sandbox_image_id"])
    summary["run_id"] = run_id
    result = {"generation": generation, "reports": reports, "summary": summary,
              "diagnostics": summary["diagnostics"]}
    for name in ("reports", "summary", "diagnostics"):
        path = directory / f"{name}.json"
        if path.exists():
            if json.loads(path.read_text()) != result[name]:
                raise ValueError("conflicting completed measurement artifact")
        else:
            write_private(path, canonical_json(result[name]))
    artifacts.commit()
    return result


@app.local_entrypoint()
def main(setup_file: str, cpu_report: str, allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for at most 272 CPU sandbox executions; no GPU")
    prior = Path("runs") / REFERENCE_RUN
    generation = json.loads((prior / "generation.json").read_text())
    prior_reports = json.loads((prior / "reports.json").read_text())
    setup = json.loads(Path(setup_file).read_text())
    conformance = json.loads(Path(cpu_report).read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    plan = measurement_plan(generation, prior_reports)
    run_id = "qwen-behavior-v2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    paths = sorted(Path("verifier_rl").glob("*.py")) + [Path(__file__), Path("measurement_v2_protocol.txt")]
    snapshot = {p.as_posix(): p.read_text() for p in paths}
    write_private(directory / "plan.json", canonical_json(plan))
    write_private(directory / "source_snapshot.json", canonical_json(snapshot))
    call = measure.spawn(run_id, generation, prior_reports, setup, conformance, snapshot)
    write_private(directory / "launch.json", canonical_json({"run_id": run_id, "call_id": call.object_id}))
    print("CPU-only measurement:", run_id, flush=True)
    result = call.get()
    for name, value in result.items():
        write_private(directory / f"{name}.json", canonical_json(value))
    print("Completed:", directory / "summary.json", flush=True)
