"""Two explicit CPU-only phases; the original 64 model outputs stay immutable."""

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import modal

from verifier_rl import booking_screen_recovery as recovery, supervised_execution as supervised
from verifier_rl.evaluation_journal import persist, ReconciliationRequired
from verifier_rl.grading import Status
from verifier_rl.modal_backend import Limits
from verifier_rl.panel_execution import PanelBackend, pack_result, request_for
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import BOOKING

app = modal.App("verifier-rl-booking-screen-recovery")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
claims = modal.Dict.from_name("verifier-rl-evaluation-claims", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
         .add_local_python_source("verifier_rl"))
# These files are only mounted to verify the historical source snapshot. They
# are not imported and no GPU Functions are declared by this application.
for name in ("modal_booking_study.py", "modal_booking_reward_pilot.py", "modal_booking_boundary_screen.py",
             "modal_booking_screen_recovery.py"):
    image = image.add_local_file(name, "/root/" + name)


def read(path):
    return json.loads(Path(path).read_text())


def check_snapshot(snapshot):
    for name, text in snapshot.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.read_text() != text:
            raise ValueError("recovery source changed: " + name)


def idle_owner(setup):
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("other candidate sandboxes active")


class DiagnosticBackend(supervised.SupervisedPanelBackend):
    async def execute(self, request):
        # Preserve parent stdout/stderr, including its diagnostics. Only this
        # diagnostic path bypasses normal decoding; it supplies no grades.
        return await PanelBackend.execute(self, request)


async def diagnostic_probes(source, directory, deadline):
    sample = next(s for s in source["generation"]["samples"] if s["sample_id"] == recovery.FAILED_SAMPLE)
    case = next(c for c in recovery.coverage.cases_for("training") if c.input_hash == recovery.FAILED_INPUT)
    setup, results, ids = source["setup"], {}, set(source["historical_sandbox_ids"])
    for name, program, instrumented, expected_status, expected_output in recovery.diagnostic_specs(sample):
        if time.time() >= deadline:
            raise ReconciliationRequired("diagnostic deadline reached")
        intent = {"name": name, "source_hash": digest(program), "input_hash": case.input_hash,
                  "instrumented": instrumented, "deadline": deadline, "supplies_research_scores": False}
        key = f"{recovery.RUN_ID}/diagnostic/{name}"
        if not await claims.put.aio(key + "/intent", intent, skip_if_exists=True):
            raise ReconciliationRequired("diagnostic already submitted; inspect it, do not repeat")
        persist(directory / "probes" / name, {"intent": intent})
        backend = DiagnosticBackend(setup["app_name"], setup["sandbox_image_id"], BOOKING,
                                    creation_interval_seconds=.26)
        if instrumented:
            backend.runner = recovery.diagnostic_runner()
        raw = await backend.execute(request_for(program, case))
        packed = pack_result(raw)
        await claims.put.aio(key + "/result", packed, skip_if_exists=True)
        persist(directory / "probes" / name, {"raw": packed})
        await artifacts.commit.aio()
        m, status, reason, out = raw.metadata, Status.INFRASTRUCTURE_ERROR, raw.detail, b""
        envelope = None
        if raw.status == Status.COMPLETED and m.get("cleanup") == "terminated":
            envelope = json.loads(raw.stdout)
            payload = canonical_json({"source": program, "input": case.arguments, "limits": asdict(Limits())})
            status, reason, out, _ = supervised.inspect_envelope(envelope, digest(payload))
            if raw.stdout != (canonical_json(envelope) + "\n").encode():
                raise ValueError("noncanonical diagnostic envelope")
        if m.get("sandbox_id") in ids or m.get("runner_hash") != digest(backend.runner):
            raise ValueError("diagnostic sandbox reused or runner changed")
        ids.add(m.get("sandbox_id"))
        telemetry = []
        for line in m.get("stderr_preview", "").splitlines():
            if line.startswith("SUPERVISOR_DIAGNOSTIC "):
                try:
                    telemetry.append(json.loads(line.removeprefix("SUPERVISOR_DIAGNOSTIC ")))
                except json.JSONDecodeError:
                    pass  # A capped stderr preview may end with a partial line.
        passed = (status == expected_status and m.get("cleanup") == "terminated"
                  and (expected_output is None or out.strip() == expected_output))
        results[name] = {"passed": passed, "status": status.value, "detail": reason,
                         "raw_hash": digest(canonical_json(packed)), "supervisor_report": envelope,
                         "telemetry": telemetry, "stderr_preview_truncated": m.get("stderr_bytes", 0) > 2048}
        persist(directory / "probes" / name, {"assessment": results[name]})
        await artifacts.commit.aio()
        print("DIAGNOSTIC", name, status.value, reason, "passed", passed, flush=True)
        if not passed:
            break
    return {"passed": len(results) == 6 and all(r["passed"] for r in results.values()), "probes": results,
            "historical_137_cause_proven": False, "historical_outcome_replaced": False,
            "new_generations": 0, "optimizer_updates": 0}


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=recovery.DIAGNOSTIC_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def diagnose(rates, billing, snapshot, deadline, preparation_hold):
    check_snapshot(snapshot)
    if not claims.put(recovery.RUN_ID + "/diagnostic/controller", {"deadline": deadline}, skip_if_exists=True):
        raise ReconciliationRequired("diagnostic controller already claimed")
    artifacts.reload()
    directory = Path("/artifacts") / recovery.RUN_ID
    source = recovery.load_source(Path("/artifacts") / recovery.screen.RUN_ID, claims)
    idle_owner(source["setup"])
    plan = recovery.plan_for(source)
    previous = preparation_hold["cumulative_reservation_usd"]
    if float(previous) < float(source["original_budget"]["cumulative_reservation_usd"]):
        raise ValueError("prior holds lost")
    budget = recovery.budget_quote(plan, rates, billing, previous, "diagnostic")
    persist(directory, {"source": source, "plan": plan, "source_snapshot": snapshot,
                       "diagnostic_budget": budget, "diagnostic_deadline": deadline,
                       "prior_preparation_hold": preparation_hold})
    artifacts.commit()
    print("SOURCE INVENTORY", plan["preserved"], "pending", len(plan["pending"]), "NO GPU", flush=True)
    diagnostic = asyncio.run(diagnostic_probes(source, directory / "diagnostic", deadline))
    persist(directory, {"diagnostic_result": diagnostic})
    artifacts.commit()
    return {"plan": plan, "budget": budget, "diagnostic": diagnostic}


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=recovery.RECOVERY_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def continue_screen(rates, billing, snapshot, deadline, preparation_hold):
    check_snapshot(snapshot)
    artifacts.reload()
    directory = Path("/artifacts") / recovery.RUN_ID
    source, plan = read(directory / "source.json"), read(directory / "plan.json")
    if plan != recovery.plan_for(source) or read(directory / "source_snapshot.json") != snapshot:
        raise ValueError("frozen source or plan changed")
    if read(directory / "diagnostic_result.json")["passed"] is not True:
        raise ReconciliationRequired("diagnostic controls did not pass")
    if (directory / "result.json").exists():
        return read(directory / "result.json")
    idle_owner(source["setup"])
    start = next((n for n in range(recovery.MAX_STARTS) if claims.put(f"{recovery.RUN_ID}/recovery/controller/{n}",
                 {"deadline": deadline, "snapshot_hash": digest(canonical_json(snapshot))}, skip_if_exists=True)), None)
    if start is None:
        raise ReconciliationRequired("controller start allowance exhausted")
    budget = recovery.budget_quote(plan, rates, billing,
        read(directory / "diagnostic_budget.json")["cumulative_reservation_usd"], "recovery")
    # A restarted invocation must retain the same deadline and quote.
    persist(directory, {"recovery_budget": budget, "recovery_deadline": deadline})
    artifacts.commit()
    backend = supervised.SupervisedPanelBackend(source["setup"]["app_name"], source["setup"]["sandbox_image_id"],
                                                BOOKING, creation_interval_seconds=.26)
    try:
        progress = asyncio.run(recovery.recover_inputs(source, plan, directory, deadline, backend, claims,
                                                       artifacts.commit.aio))
        state = recovery.replay_recovery(source, directory)
        result = recovery.summarize(source, state["assessments"], state["records"])
        result["progress"] = progress
        if progress["stop_reason"]:
            result["status"] = "partial_" + progress["stop_reason"]
        persist(directory, {"result": result})
        artifacts.commit()
        print("RECOVERY FINISHED", result["status"], result["summary"], result["decision"], flush=True)
        return result
    except Exception as exc:
        persist(directory, {f"stopped-{time.time_ns()}": {"type": type(exc).__name__, "detail": str(exc)[:2000]}})
        artifacts.commit()
        raise


def read_modal(*args):
    done = subprocess.run([sys.executable, "-m", "modal", *args, "--json"], check=True, capture_output=True, text=True)
    return json.loads(done.stdout)


@app.local_entrypoint()
def launch(phase: str = "diagnostic", allow_cloud: bool = False):
    if not allow_cloud or phase not in ("diagnostic", "recovery"):
        raise ValueError("explicit --allow-cloud and diagnostic/recovery phase required")
    directory = Path("runs") / recovery.RUN_ID
    if (directory / f"{phase}_launch_intent.json").exists():
        raise ReconciliationRequired("phase already launched; inspect existing call, do not duplicate")
    apps = read_modal("app", "list")
    if any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("other Modal tasks active")
    parent = Path("runs") / recovery.screen.RUN_ID
    idle_owner(read(parent / "setup.json"))
    names = [Path("modal_booking_screen_recovery.py"), Path("verifier_rl/booking_screen_recovery.py")]
    snapshot = {p.as_posix(): p.read_text() for p in names}
    # Also freeze and verify every historical implementation file at launch.
    snapshot.update(read(parent / "source_snapshot.json"))
    check_snapshot(snapshot)
    seconds = recovery.DIAGNOSTIC_SECONDS if phase == "diagnostic" else recovery.RECOVERY_SECONDS
    deadline = time.time() + seconds - 60
    rates, billing = read_modal("billing", "rates"), read_modal("billing", "summary")
    earlier = read(Path("runs/qwen-booking-boundary-recovery-20260929-v1/diagnostic_launch_intent.json"))
    preparation_hold = recovery.budget_quote({"diagnostic_starts": 6}, earlier["rates"], earlier["billing"],
                                            read(parent / "budget.json")["cumulative_reservation_usd"], "diagnostic")
    if phase == "recovery" and read(directory / "diagnostic_receipt.json")["diagnostic"]["passed"] is not True:
        raise ReconciliationRequired("diagnostic must pass before CPU recovery")
    persist(directory, {f"{phase}_launch_intent": {"deadline": deadline, "observed_apps": apps,
                       "rates": rates, "billing": billing, "snapshot_hash": digest(canonical_json(snapshot))},
                       "source_snapshot": snapshot})
    function = diagnose if phase == "diagnostic" else continue_screen
    call = function.spawn(rates, billing, snapshot, deadline, preparation_hold)
    persist(directory, {f"{phase}_launch": {"call_id": call.object_id, "created_utc": datetime.now(timezone.utc).isoformat()}})
    print("CPU ONLY", phase, "CALL", call.object_id, flush=True)
    result = call.get()
    persist(directory, {f"{phase}_receipt": result})
    if phase == "diagnostic":
        print("DIAGNOSTIC COMPLETE", result["diagnostic"]["passed"], result["plan"]["preserved"], flush=True)
