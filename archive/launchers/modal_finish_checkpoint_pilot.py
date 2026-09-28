"""CPU-only recovery of at most one interrupted 1.5B candidate evaluation."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import modal

from verifier_rl.checkpoint_pilot import recovery_plan, summarize_comparison
from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.model_trial import evaluate_submission, validate_run_id
from verifier_rl.modal_backend import ModalBackend, RUNNER
from verifier_rl.sft_trial import evaluation_suites
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import canonical_json, digest

app = modal.App("verifier-rl-checkpoint-recovery")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
         .add_local_python_source("verifier_rl"))


@app.function(image=image, cpu=1, memory=1024, timeout=300, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def finish(parent_run: str, run_id: str, snapshot: dict):
    validate_run_id(parent_run)
    validate_run_id(run_id)
    artifacts.reload()
    parent = Path(f"/artifacts/{parent_run}")
    generation = json.loads((parent / "generation.json").read_text())
    if generation["run_id"] != parent_run:
        raise ValueError("generation parent identity mismatch")
    reference = json.loads((parent / "reference.json").read_text())
    setup = json.loads((parent / "setup.json").read_text())
    conformance = json.loads((parent / "conformance.json").read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    completed = [json.loads(p.read_text()) for s in generation["samples"]
                 if (p := parent / f"sample-{s['seed']}-result.json").exists()]
    policy = recovery_plan(generation, completed, setup["sandbox_image_id"])
    directory = create_run_directory(f"/artifacts/{run_id}")
    for name, value in (("recovery", policy), ("generation", generation), ("reference", reference),
                        ("setup", setup), ("conformance", conformance), ("source_snapshot", snapshot)):
        write_private(directory / f"{name}.json", canonical_json(value))
    suites = evaluation_suites()
    save_suites(directory, suites)
    artifacts.commit()

    async def grade():
        backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=.26)
        by_seed = {r["seed"]: r for r in completed}
        reports = []
        for sample in generation["samples"]:
            seed = sample["seed"]
            if seed in by_seed:
                report = by_seed[seed]
            else:
                report = await evaluate_submission(sample, suites, backend)
                report["seed"] = seed
            write_private(directory / f"sample-{seed}-result.json", canonical_json(report))
            await artifacts.commit.aio()
            if any(s["all_passed"] is None for s in report["suites"]):
                raise RuntimeError("recovery stopped unscored; no further retry")
            reports.append(report)
            print("Recovery", seed, "reused" if seed in by_seed else "evaluated", flush=True)
        return reports

    reports = asyncio.run(grade())
    summary = summarize_comparison(generation, reports, reference, setup["sandbox_image_id"])
    summary.update(run_id=run_id, generation_run_id=parent_run, recovery=policy, runner_hash=digest(RUNNER))
    write_private(directory / "reports.json", canonical_json(reports))
    write_private(directory / "summary.json", canonical_json(summary))
    artifacts.commit()
    return {"generation": generation, "reports": reports, "summary": summary, "recovery": policy}


@app.local_entrypoint()
def main(parent_run: str, allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for bounded CPU-only recovery")
    validate_run_id(parent_run)
    run_id = "qwen-checkpoint-recovery-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    paths = sorted(Path("verifier_rl").glob("*.py")) + [Path(__file__), Path("checkpoint_1_5b_protocol.txt")]
    snapshot = {p.as_posix(): p.read_text() for p in paths}
    write_private(directory / "source_snapshot.json", canonical_json(snapshot))
    call = finish.spawn(parent_run, run_id, snapshot)
    write_private(directory / "launch.json", canonical_json({"run_id": run_id, "parent_run": parent_run,
                                                            "call_id": call.object_id}))
    print("CPU recovery:", run_id, flush=True)
    result = call.get()
    for name, value in result.items():
        write_private(directory / f"{name}.json", canonical_json(value))
    print("Completed:", directory / "summary.json", flush=True)
