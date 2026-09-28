"""Preferred cloud grading entrypoint: the laptop launches; Modal controls execution."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import modal

from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.fixtures import source_for
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.model_trial import (evaluate_submission, submission_from_completion,
                                    validate_run_id, validate_submission)
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import build_suites, canonical_json

app = modal.App("verifier-rl-protected-grading")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
         .add_local_python_source("verifier_rl"))


@app.function(image=image, cpu=1, memory=1024, timeout=600, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def grade(setup: dict, conformance: dict, run_id: str, submissions: list[dict], controls: bool):
    validate_run_id(run_id)
    require_current_conformance(conformance, setup["sandbox_image_id"])
    if not 1 <= len(submissions) <= 2:
        raise ValueError("at most two submissions in this bounded G3 grader")
    for entry in submissions:
        validate_submission(entry)
    if controls and submissions != [submission_from_completion(source_for(n)) for n in ("correct", "inclusive_expiry")]:
        raise ValueError("unexpected control programs")
    directory = create_run_directory(f"/artifacts/{run_id}")
    suites = (build_suites()[2],)
    save_suites(directory, suites)
    write_private(directory / "inputs.json", canonical_json({"setup": setup, "submissions": submissions,
                  "controls": controls, "gpu": False, "concurrency": 4, "max_retries": 0}))
    artifacts.commit()

    async def evaluate():
        backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=0.26)
        reports = []
        for index, entry in enumerate(submissions):
            report = await evaluate_submission(entry, suites, backend)
            reports.append(report)
            write_private(directory / f"candidate-{index}.json", canonical_json(report))
            await artifacts.commit.aio()
            score = report["suites"][0]
            print(f"G3 candidate {index}: {score['passed_count']}/{score['total']}, reward={score['reward']}", flush=True)
            if score["all_passed"] is None:
                raise RuntimeError("infrastructure failure: evidence saved; no subsequent candidate submitted")
        return reports

    reports = asyncio.run(evaluate())
    passed = not controls or [r["suites"][0]["reward"] for r in reports] == [1, 0]
    result = {"run_id": run_id, "kind": "grader_controls" if controls else "candidate_evaluation",
              "passed": passed, "model_results": False if controls else None, "reports": reports}
    write_private(directory / "summary.json", canonical_json(result))
    artifacts.commit()
    if not passed:
        raise RuntimeError("known-correct/known-faulty G3 control failed")
    return result


@app.local_entrypoint()
def main(setup_file: str, cpu_report: str, candidate: str = "", controls: bool = False, allow_cloud: bool = False):
    if not allow_cloud or bool(candidate) == controls:
        raise ValueError("--allow-cloud and exactly one of --candidate or --controls required")
    setup = json.loads(Path(setup_file).read_text())
    conformance = json.loads(Path(cpu_report).read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    sources = [source_for(n) for n in ("correct", "inclusive_expiry")] if controls else [Path(candidate).read_text()]
    entries = [submission_from_completion(source) for source in sources]
    run_id = "qwen-grader-control-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    call = grade.spawn(setup, conformance, run_id, entries, controls)
    write_private(directory / "launch.json", canonical_json({"run_id": run_id, "call_id": call.object_id}))
    result = call.get()
    write_private(directory / "summary.json", canonical_json(result))
    print("Evidence:", directory / "summary.json")
