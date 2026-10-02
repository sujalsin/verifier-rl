"""CPU-only saved-program completion and journaled resumption; no model/GPU functions."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import modal

from verifier_rl import evaluation_completion as completion, evaluation_recovery as recovery
from verifier_rl import evaluation_journal as journal
from verifier_rl import reward_shaping as shaping
from verifier_rl.cli import create_run_directory, save_suites, write_private
from verifier_rl.modal_backend import ModalBackend
from verifier_rl.model_trial import evaluate_submission
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import canonical_json, digest

app = modal.App("verifier-rl-evaluation-completion")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
claims = modal.Dict.from_name("verifier-rl-evaluation-claims", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
         .add_local_python_source("verifier_rl"))


def artifact_directory(run_id):
    return Path("/artifacts") / run_id


def grade_saved(directory, claim_key, key, policy, sample, setup, budget, receipts, deadline,
                attempt_index=1, before_submit=None):
    """A saved raw result is reusable; an in-flight intent is never auto-retried."""
    intent = journal.intent_for(key, policy, sample, budget, receipts, attempt_index)
    item = directory / key
    existed = (item / "intent.json").exists()
    if existed:
        journal.persist(item, {"intent": intent})
        if not (item / "result.json").exists():
            raise journal.ReconciliationRequired("intent without result: preserve reservation and reconcile " + key)
        report = json.loads((item / "result.json").read_text())
        print("Reused saved raw result:", key, flush=True)
    else:
        if any((item / f"{name}.json").exists() for name in ("result", "assessment", "receipt", "started")):
            raise ValueError("orphaned execution records without an intent")
        shaping.require_deadline({"submission_deadline_epoch": deadline})
        if before_submit:
            before_submit()
        journal.persist(item, {"intent": intent})
        artifacts.commit()
        # A Volume path is not a cross-container lock. Claim at most once even
        # if two clients accidentally submit this fixed run simultaneously.
        if not claims.put(claim_key, digest(canonical_json(intent)), skip_if_exists=True):
            raise journal.ReconciliationRequired("batch already claimed; no duplicate execution: " + key)
        journal.persist(item, {"started": {"claim_key": claim_key, "intent_hash": digest(canonical_json(intent))}})
        artifacts.commit()
        print("Evaluating saved program:", key, "reserved total USD", intent["reservation"]["total_reserved_usd"], flush=True)
        backend = ModalBackend(setup["app_name"], setup["sandbox_image_id"], creation_interval_seconds=.26)
        async def evaluate():
            return await asyncio.wait_for(evaluate_submission(sample, shaping.selected_suites(True),
                                                               backend, concurrency=8), timeout=300)
        report = asyncio.run(evaluate())
        report.update(arm=sample["arm"], seed=sample["seed"])
        journal.persist(item, {"result": report})
        artifacts.commit()  # Preserve raw evidence BEFORE deriving or validating scores.
    assessment = completion.assess_report(sample, report, setup["sandbox_image_id"])
    receipt = shaping.execution_receipt(key, report)
    journal.persist(item, {"assessment": assessment, "receipt": receipt})
    artifacts.commit()
    return report, assessment, receipt, intent["reservation"]


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=completion.CONTROLLER_SECONDS,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def run_completion(run_id, inputs, budget, deadline, snapshot):
    if run_id != completion.RUN_ID:
        raise ValueError("fixed completion ID required")
    rows = completion.validate_inputs(inputs)
    if completion.completion_budget(inputs, budget["billing_before"], budget["rates"],
                                     budget["old_sandbox_terminal"]) != budget:
        raise ValueError("invalid budget")
    setup = inputs["previous"]["setup"]
    require_current_conformance(inputs["previous"]["conformance"], setup["sandbox_image_id"])
    artifacts.reload()
    directory = journal.persist(artifact_directory(run_id), {"inputs": inputs, "budget": budget,
                                "source_snapshot": snapshot, "deadline": deadline})
    # Validate the WHOLE saved journal before submitting any more work.
    state = journal.validate_checkpoint(journal.load_checkpoint(directory))
    reports, receipts, reservations = (state[n] for n in ("reports", "receipts", "reservations"))
    for key, values in state["derived"].items():
        journal.persist(directory / key, values)
    for suite in shaping.selected_suites(True):
        if not (directory / f"{suite.name}.suite.json").exists():
            save_suites(directory, (suite,))
    artifacts.commit()
    if state["interrupted"]:
        raise journal.ReconciliationRequired("unfinished batch requires reconciliation: " + ", ".join(state["interrupted"]))
    current = "initialization"
    print("Restored", len(reports), "saved program reports and their spending receipts", flush=True)
    try:
        for key, policy, sample in rows:
            if key in reports:
                continue
            current = key
            if key not in completion.PENDING_KEYS:
                raise ValueError("program is not authorized for new evaluation")
            report, assessment, receipt, reservation = grade_saved(directory, run_id + "/" + key,
                key, policy, sample, setup, budget, receipts, deadline)
            reports[key], receipts[key], reservations[key] = report, receipt, reservation
            progress = {"version": completion.VERSION, "represented_programs": len(reports),
                "pending_keys": [k for k in completion.PENDING_KEYS if k not in reports],
                "last_key": key, "last_assessment": assessment,
                "full_comparison_verified": False}
            journal.persist(directory, {f"progress-{len(reports)}": progress})
            artifacts.commit()
            print("Evaluated", len(reports), "/24:", key, assessment["status"],
                  "audit pass bounds", assessment["suites"]["audit"]["passed_cases"], flush=True)
        current = "verification"
        result = completion.verify_completion(inputs, reports, budget, receipts, reservations)
        documents = {"reports": reports, "receipts": receipts, "reservations": reservations, "summary": result}
        journal.persist(directory, documents)
        artifacts.commit()
        print("All programs evaluated; uncertainty retained:", result["policies"], flush=True)
        return documents
    except Exception as exc:
        journal.persist(directory, {f"stopped-{time.time_ns()}": {"stage": current,
            "error_type": type(exc).__name__, "detail": str(exc)[:1000], "automatic_retry": False,
            "represented_programs": len(reports), "saved_reports_unchanged": True,
            "outstanding_intent_reservation_retained": True}})
        artifacts.commit()
        raise


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=journal.RESUME_SECONDS,
              nonpreemptible=True, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_resume(checkpoint, reconciliation, budget, deadline, snapshot):
    run_id = completion.RUN_ID + "/" + journal.RESUME_ID
    # Bound controller starts as well as candidate submissions. Claims are not
    # released after a crash; lost work requires explicit reconciliation.
    if not any(claims.put(f"{run_id}/controller-start-{i}", True, skip_if_exists=True) for i in range(2)):
        raise journal.ReconciliationRequired("controller-start budget exhausted")
    state = journal.validated_resume(checkpoint, reconciliation, budget)
    recovery.check_frozen_sources(checkpoint["source_snapshot"], Path("/root"))
    setup = checkpoint["inputs"]["previous"]["setup"]
    require_current_conformance(checkpoint["inputs"]["previous"]["conformance"], setup["sandbox_image_id"])
    artifacts.reload()
    # The old journal is immutable. This amendment gets its own subdirectory,
    # manifest and deadline; automatic restarts must reuse those exact values.
    if journal.load_checkpoint(artifact_directory(completion.RUN_ID)) != checkpoint:
        raise ValueError("remote source journal changed since reconciliation")
    directory = journal.persist(artifact_directory(run_id), {"checkpoint": checkpoint,
        "reconciliation": reconciliation, "budget": budget, "deadline": deadline, "source_snapshot": snapshot})
    artifacts.commit()
    def quiescent():
        owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
        if owner.app_id != reconciliation["sandbox_app_id"]:
            raise ValueError("sandbox app identity changed")
        active = [s.object_id for s in modal.Sandbox.list(app_id=owner.app_id)]
        if active:
            raise journal.ReconciliationRequired("evaluation app has active sandboxes; inspect before submitting")
    _, policy, sample = next(r for r in state["rows"] if r[0] == journal.RESUME_KEY)
    key = journal.RESUME_KEY + "-" + journal.RESUME_ID
    report, _, receipt, reservation = grade_saved(directory, run_id + "/" + key, key, policy,
        sample, setup, budget, {}, deadline, attempt_index=2, before_submit=quiescent)
    reports = dict(state["reports"], **{journal.RESUME_KEY: report})
    result = journal.verify_resume(checkpoint, reconciliation, budget, reports, receipt, reservation)
    documents = {"reports": reports, "receipt": receipt, "reservation": reservation, "summary": result}
    journal.persist(directory, documents)
    artifacts.commit()
    print("All 24 saved programs represented; uncertainty preserved:", result["policies"], flush=True)
    return documents


@app.local_entrypoint()
def complete(allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required: six CPU-only evaluations within $20 total")
    target = Path("runs") / completion.RUN_ID
    if target.exists():
        raise ValueError("one-time run directory exists: retrieve records, do not launch again")
    source = Path("runs") / completion.SOURCE_RECOVERY
    inputs = completion.load_inputs(source)
    checked = recovery.check_frozen_sources(json.loads((source / "source_snapshot.json").read_text()), Path.cwd())
    completion.validate_inputs(inputs)
    completion.summary(inputs, inputs["reports"])
    def billing(kind):
        result = subprocess.run([sys.executable, "-m", "modal", "billing", kind, "--json"],
                                check=True, capture_output=True, text=True, timeout=45)
        return json.loads(result.stdout)
    terminal = {"sandbox_id": recovery.QUARANTINED_SANDBOX,
        "returncode": modal.Sandbox.from_id(recovery.QUARANTINED_SANDBOX).poll(),
        "observed_at_utc": datetime.now(timezone.utc).isoformat()}
    budget = completion.completion_budget(inputs, billing("summary"), billing("rates"), terminal)
    snapshot = {p.as_posix(): p.read_text() for p in sorted(Path("verifier_rl").glob("*.py"))
                + [Path(__file__), Path("evaluation_completion_protocol.txt")]}
    deadline = time.time() + completion.CONTROLLER_SECONDS
    directory = create_run_directory(str(target))
    for name, value in {"inputs": inputs, "budget": budget, "source_snapshot": snapshot,
                        "deadline": deadline, "frozen_files_checked": checked}.items():
        write_private(directory / f"{name}.json", canonical_json(value))
    call = run_completion.spawn(completion.RUN_ID, inputs, budget, deadline, snapshot)
    write_private(directory / "launch.json", canonical_json({"run_id": completion.RUN_ID,
        "call_id": call.object_id, "new_generations": 0, "gpu_calls": 0, "pending_programs": 6}))
    print("CPU-only completion:", completion.RUN_ID, "call:", call.object_id, flush=True)
    print("Full prior reservation retained; fixed USD", budget["fixed_reserved_usd"],
          "; reserve each batch separately within $20 total", flush=True)
    result = call.get()
    for name, value in result.items():
        write_private(directory / f"{name}.json", canonical_json(value))
    print("Completed with explicit uncertainty:", directory / "summary.json", flush=True)


@app.local_entrypoint()
def resume(checkpoint_dir: str, allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required: one CPU-only interrupted-batch replacement within $20 total")
    target = Path("runs") / completion.RUN_ID / journal.RESUME_ID
    if target.exists():
        raise ValueError("resume already launched: retrieve its saved artifacts; do not reset its deadline")
    checkpoint = journal.load_checkpoint(Path(checkpoint_dir))
    recovery.check_frozen_sources(checkpoint["source_snapshot"], Path.cwd())
    def read_modal(*args):
        result = subprocess.run([sys.executable, "-m", "modal", *args, "--json"],
                                check=True, capture_output=True, text=True, timeout=45)
        return json.loads(result.stdout)
    apps = read_modal("app", "list")
    old = next(a for a in apps if a["app_id"] == journal.STOPPED_APP_ID)
    setup = checkpoint["inputs"]["previous"]["setup"]
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    reconciliation = {"stopped_controller": old, "sandbox_app_name": setup["app_name"],
        "sandbox_app_id": owner.app_id, "active_sandbox_ids": [s.object_id for s in modal.Sandbox.list(app_id=owner.app_id)],
        "observed_at_utc": datetime.now(timezone.utc).isoformat(), "cause": "controller_preemption",
        "replacement_authorized": True,
        "interrupted_intent_hash": digest(canonical_json(checkpoint["items"][journal.RESUME_KEY]["intent"]))}
    budget = journal.resume_budget(checkpoint, reconciliation, read_modal("billing", "summary"), read_modal("billing", "rates"))
    snapshot = {p.as_posix(): p.read_text() for p in sorted(Path("verifier_rl").glob("*.py"))
                + [Path(__file__), Path("evaluation_completion_protocol.txt")]}
    deadline = time.time() + journal.RESUME_SECONDS
    directory = journal.persist(target, {"checkpoint": checkpoint, "reconciliation": reconciliation,
        "budget": budget, "source_snapshot": snapshot, "deadline": deadline})
    call = run_resume.spawn(checkpoint, reconciliation, budget, deadline, snapshot)
    journal.persist(directory, {"launch": {"source_run": completion.RUN_ID, "resume_id": journal.RESUME_ID,
        "call_id": call.object_id, "new_generations": 0, "gpu_calls": 0, "pending_programs": 1}})
    print("CPU-only final-program resume:", call.object_id, "; fixed reserve USD", budget["fixed_reserved_usd"], flush=True)
    result = call.get()
    journal.persist(directory, result)
    print("Completed with explicit uncertainty:", directory / "summary.json", flush=True)
