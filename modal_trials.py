"""Run the trusted CPU trial controller on Modal, separate from candidate sandboxes.

Launch: .venv/bin/modal run modal_trials.py --setup-file runs/SETUP/setup.json
This creates billable CPU resources. It does not start a GPU or train a model.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import modal

from verifier_rl.cli import create_run_directory, write_private
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.smoke import run_smoke
from verifier_rl.suites import canonical_json

app = modal.App("verifier-rl-cache-trials")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=True)
# This is the TRUSTED controller image. It intentionally includes grader code.
# Never use this image as the candidate execution image.
controller_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("modal==1.5.5")
    .add_local_python_source("verifier_rl")
)


@app.function(image=controller_image, cpu=1, memory=1024, timeout=900,
              retries=0, volumes={"/artifacts": artifacts}, max_containers=1)
def cpu_trial(evaluation_app: str, sandbox_image_id: str, run_id: str):
    if not run_id.startswith("cpu-") or not all(c.isalnum() or c == "-" for c in run_id):
        raise ValueError("invalid run ID")
    directory = create_run_directory(f"/artifacts/{run_id}")
    write_private(directory / "config.json", canonical_json({
        "evaluation_app": evaluation_app, "sandbox_image_id": sandbox_image_id,
        "controller_sdk": modal.__version__, "gpu": False,
        "max_executions": 24, "concurrency": 1, "max_retries": 0,
    }))
    artifacts.commit()

    async def record(result):
        write_private(directory / f"{result['check']}.json", canonical_json(result))
        await artifacts.commit.aio()

    backend = ModalBackend(evaluation_app, sandbox_image_id)
    report = asyncio.run(run_smoke(backend, record=record))
    report["run_id"] = run_id
    write_private(directory / "summary.json", canonical_json(report))
    artifacts.commit()
    return report


@app.local_entrypoint()
def main(setup_file: str):
    config = json.loads(Path(setup_file).read_text())
    run_id = "cpu-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    local = create_run_directory(f"runs/{run_id}")
    write_private(local / "setup.json", canonical_json(config))
    report = cpu_trial.remote(config["app_name"], config["sandbox_image_id"], run_id)
    write_private(local / "summary.json", canonical_json(report))
    for check in report["checks"]:
        print(check["check"], "PASS" if check["passed"] else "FAIL")
    print(f"Passed: {report['passed']}; completed checks: {report['completed_checks']}/{report['planned_checks']}")
    print(f"Local report: {local / 'summary.json'}")
    print(f"Persistent Modal volume: verifier-rl-cache-artifacts/{run_id}")
    if not report["passed"]:
        raise RuntimeError("CPU trial failed; do not proceed to model generation or training")
