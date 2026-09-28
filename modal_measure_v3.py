"""Opt-in CPU-only paired v2/v3 grading of the frozen 32-program pilot."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

import modal

from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.measurement_v3 import (CANDIDATE_SECONDS, CONTROLLER_SECONDS, SOURCE_RUN,
    budget_check, control_samples, entry_id, measurement_plan, measurement_summary,
    previous_reports, repeatability_changes, require_measurement_report, saved_samples, suites)
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.model_trial import evaluate_submission, validate_run_id, validate_submission
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import canonical_json, digest

app = modal.App("verifier-rl-paired-grader-measurement")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
         .add_local_python_source("verifier_rl"))


def persist_equal(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("conflicting saved measurement artifact")
    else:
        write_private(path, canonical_json(value))


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=CANDIDATE_SECONDS,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def evaluate_one(run_id: str, sample: dict, setup: dict, plan: dict):
    validate_run_id(run_id)
    validate_submission(sample)
    entry = entry_id(sample)
    selected = suites()
    if (plan["entry_hashes"].get(entry) != digest(canonical_json(sample))
            or plan["suite_hashes"] != [s.fingerprint for s in selected]):
        raise ValueError("candidate/suites differ from frozen measurement plan")
    artifacts.reload()
    directory = Path(f"/artifacts/{run_id}/cases/{entry}")
    intent = {"entry_hash": digest(canonical_json(sample)), "plan_hash": digest(canonical_json(plan)),
              "setup_hash": digest(canonical_json(setup))}
    result_path = directory / "result.json"
    if result_path.exists():
        if json.loads((directory / "intent.json").read_text()) != intent:
            raise ValueError("conflicting completed candidate intent")
        report = json.loads(result_path.read_text())
        require_measurement_report(sample, report, setup["sandbox_image_id"])
        return report
    # An incomplete intent is ambiguous, not permission to rerun its inputs.
    directory = create_run_directory(str(directory))
    write_private(directory / "intent.json", canonical_json(intent))
    artifacts.commit()
    backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=.26)
    report = asyncio.run(evaluate_submission(sample, selected, backend, concurrency=plan["concurrency"]))
    report.update(arm=sample["arm"], seed=sample["seed"])
    write_private(result_path, canonical_json(report))
    artifacts.commit()  # Evidence precedes every validation gate.
    require_measurement_report(sample, report, setup["sandbox_image_id"])
    return report


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=CONTROLLER_SECONDS,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def measure(run_id: str, generation: dict, prior_reports: list, setup: dict,
            conformance: dict, snapshot: dict, budget: dict):
    validate_run_id(run_id)
    require_current_conformance(conformance, setup["sandbox_image_id"])
    plan = measurement_plan(generation, prior_reports)
    if budget_check(plan, budget["billing_before"], budget["rates"]) != budget:
        raise ValueError("budget record differs from frozen resource envelope")
    directory = Path(f"/artifacts/{run_id}")
    artifacts.reload()
    frozen = {"plan": plan, "generation": generation, "prior_reports": prior_reports,
              "setup": setup, "conformance": conformance, "source_snapshot": snapshot, "budget": budget}
    if directory.exists():
        for name, value in frozen.items():
            if json.loads((directory / f"{name}.json").read_text()) != value:
                raise ValueError("incomplete or conflicting controller resume state")
    else:
        create_run_directory(str(directory))
        for name, value in frozen.items():
            write_private(directory / f"{name}.json", canonical_json(value))
        save_suites(directory, suites())
        artifacts.commit()
    reports = []
    previous = previous_reports(generation, prior_reports)
    current = "not_started"
    try:
        for sample in control_samples() + saved_samples(generation):
            current = entry_id(sample)
            report = evaluate_one.remote(run_id, sample, setup, plan)
            reports.append(report)
            if current in previous and repeatability_changes(previous[current], report):
                raise ValueError("unchanged v2 inputs changed outcomes; stop for review, no retries")
            print("Measured", current, [(s["passed_count"], s["total"]) for s in report["suites"]], flush=True)
        summary = measurement_summary(generation, prior_reports, reports, setup["sandbox_image_id"])
        summary["run_id"] = run_id
        artifacts.reload()
        for name, value in (("reports", reports), ("summary", summary)):
            persist_equal(directory / f"{name}.json", value)
        artifacts.commit()
        return {"reports": reports, "summary": summary}
    except Exception as exc:
        artifacts.reload()
        persist_equal(directory / "stopped.json", {"entry": current, "error_type": type(exc).__name__,
                      "detail": str(exc)[:1000], "completed_returned_entries": len(reports),
                      "automatic_retry": False})
        artifacts.commit()
        raise


def read_billing(kind):
    result = subprocess.run([sys.executable, "-m", "modal", "billing", kind, "--json"],
                            check=True, capture_output=True, text=True, timeout=60)
    return json.loads(result.stdout)


@app.local_entrypoint()
def main(allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required; at most 1496 CPU sandboxes, no GPU/training")
    source = Path("runs") / SOURCE_RUN
    generation = json.loads((source / "generation.json").read_text())
    prior_reports = json.loads((source / "reports.json").read_text())
    setup = json.loads((source / "setup.json").read_text())
    conformance = json.loads((source / "conformance.json").read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    plan = measurement_plan(generation, prior_reports)
    budget = budget_check(plan, read_billing("summary"), read_billing("rates"))
    run_id = "qwen-grader-v3-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    paths = sorted(Path("verifier_rl").glob("*.py")) + [Path(__file__), Path("measurement_v3_protocol.txt")]
    snapshot = {p.as_posix(): p.read_text() for p in paths}
    for name, value in (("plan", plan), ("source_snapshot", snapshot), ("budget", budget),
                        ("generation", generation), ("prior_reports", prior_reports),
                        ("setup", setup), ("conformance", conformance)):
        write_private(directory / f"{name}.json", canonical_json(value))
    save_suites(directory, suites())
    call = measure.spawn(run_id, generation, prior_reports, setup, conformance, snapshot, budget)
    write_private(directory / "launch.json", canonical_json({"run_id": run_id, "call_id": call.object_id}))
    print("CPU-only paired grader measurement:", run_id, flush=True)
    print("Conservative total resource envelope including prior usage/reserve:",
          budget["total_metered_envelope_usd"], flush=True)
    result = call.get()
    for name, value in result.items():
        persist_equal(directory / f"{name}.json", value)
    print("Completed:", directory / "summary.json", flush=True)
