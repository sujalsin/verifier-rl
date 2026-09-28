"""Resume saved prompt-pilot CPU evaluations under the recorded recovery amendment."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import modal

from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.grading import ExecutionRequest, evaluate_candidate, rejected_extraction_report
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.model_trial import validate_run_id
from verifier_rl.prompt_pilot import pilot_suites, protocol_plan, summarize
from verifier_rl.prompt_recovery import PreflightOnlyRetries, RetryBudget, merge_replacements, validate_prior
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import Suite, canonical_json

app = modal.App("verifier-rl-prompt-pilot-recovery")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
         .add_local_python_source("verifier_rl"))


@app.function(image=image, cpu=1, memory=1024, timeout=900, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def finish(parent_run: str, run_id: str, source_snapshot: dict):
    validate_run_id(parent_run)
    validate_run_id(run_id)
    artifacts.reload()
    parent = Path(f"/artifacts/{parent_run}")
    generation = json.loads((parent / "generation.json").read_text())
    setup = json.loads((parent / "setup.json").read_text())
    require_current_conformance(json.loads((parent / "conformance.json").read_text()), setup["sandbox_image_id"])
    if canonical_json(generation["plan"]) != canonical_json(protocol_plan(generation["plan"]["prompts"]["original"])):
        raise ValueError("saved pilot protocol mismatch")
    suites = pilot_suites()
    directory = create_run_directory(f"/artifacts/{run_id}")
    policy = {"parent_run": parent_run, "gpu": False,
              "max_additional_preflight_attempts": 3, "max_replacements_per_input": 1,
              "eligibility": "preflight provider code -1, no candidate returncode, cleanup terminated",
              "candidate_failures_retried": False, "source_and_grading_changed": False,
              "raw_failed_attempts_retained": True, "max_controller_seconds": 900}
    write_private(directory / "recovery.json", canonical_json(policy))
    write_private(directory / "source_snapshot.json", canonical_json(source_snapshot))
    write_private(directory / "generation.json", canonical_json(generation))
    save_suites(directory, suites)
    artifacts.commit()

    async def evaluate():
        backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=0.26)
        budget = RetryBudget()
        reports = []
        for sample in generation["samples"]:
            arm, seed = sample["arm"], sample["seed"]
            filename = f"{arm}-{seed}-result.json"
            prior_path = parent / filename
            retry_backend = PreflightOnlyRetries(backend, budget)
            if prior_path.exists():
                prior = json.loads(prior_path.read_text())
                pending = validate_prior(sample, prior, suites, setup["sandbox_image_id"])
                if not pending:
                    report = prior
                else:
                    cases = {c.input_hash: c for s in suites for c in s.cases if c.input_hash in pending}
                    for case in cases.values():
                        if not retry_backend.reserve(ExecutionRequest(sample["source"], case.input_json)):
                            raise RuntimeError("preflight replacement budget exhausted")
                    replacement = await evaluate_candidate(sample["source"],
                        (Suite("preflight_recovery", "audit", tuple(cases.values()), suites[0].seed),),
                        retry_backend, concurrency=4, max_retries=1)
                    report = merge_replacements(prior, replacement)
            elif sample["extraction_status"].startswith("rejected_"):
                report = rejected_extraction_report(sample["source"], sample["extraction_status"], suites)
            else:
                report = await evaluate_candidate(sample["source"], suites, retry_backend,
                                                  concurrency=4, max_retries=1)
            report.update(arm=arm, seed=seed)
            report["recovery_parent_run"] = parent_run
            write_private(directory / filename, canonical_json(report))
            await artifacts.commit.aio()
            if any(s["infrastructure_errors"] for s in report["suites"]):
                raise RuntimeError("unresolved infrastructure error under bounded recovery policy")
            reports.append(report)
            print(f"Completed {len(reports)}/16: {arm} seed={seed}: " + ", ".join(
                f"{s['suite']}={s['passed_count']}/{s['total']}" for s in report["suites"]), flush=True)
        result = summarize(generation, reports)
        result.update(run_id=run_id, generation_run_id=parent_run, recovery=policy,
                      additional_preflight_attempts=budget.used)
        return {"generation": generation, "reports": reports, "summary": result}

    result = asyncio.run(evaluate())
    write_private(directory / "reports.json", canonical_json(result["reports"]))
    write_private(directory / "summary.json", canonical_json(result["summary"]))
    artifacts.commit()
    return result


@app.local_entrypoint()
def main(parent_run: str, allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for bounded CPU recovery")
    validate_run_id(parent_run)
    run_id = "qwen-prompt-recovery-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = create_run_directory(f"runs/{run_id}")
    paths = sorted(Path("verifier_rl").glob("*.py")) + [Path(__file__)]
    snapshot = {p.name if p.name == Path(__file__).name else p.as_posix(): p.read_text() for p in paths}
    write_private(directory / "source_snapshot.json", canonical_json(snapshot))
    call = finish.spawn(parent_run, run_id, snapshot)
    write_private(directory / "launch.json", canonical_json({"run_id": run_id, "parent_run": parent_run,
                                                            "call_id": call.object_id}))
    print("Recovery:", run_id, flush=True)
    result = call.get()
    for name in ("generation", "reports", "summary"):
        write_private(directory / f"{name}.json", canonical_json(result[name]))
    print(canonical_json({arm: row["full_suite_pass_counts"] for arm, row in result["summary"]["arms"].items()}))
    print("Local evidence:", directory)
