"""Evaluation-only recovery; this app contains no GPU or model-generation function."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import modal

from verifier_rl import booking_study as study, booking_two_arm as two, booking_recovery as recovery
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.evaluation_recovery import check_frozen_sources
from verifier_rl.panel_execution import PanelBackend, pack_result, request_for, unpack_result

app = modal.App("verifier-rl-booking-evaluation-recovery")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
claims = modal.Dict.from_name("verifier-rl-evaluation-claims", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
         .add_local_python_source("verifier_rl")
         .add_local_file("modal_booking_study.py", "/root/modal_booking_study.py"))


def read_modal(*args):
    return json.loads(subprocess.run([sys.executable, "-m", "modal", *args, "--json"],
                     capture_output=True, text=True, check=True, timeout=45).stdout)


def quiescent():
    apps = read_modal("app", "list")
    if any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("Modal tasks active; do not start overlapping evaluation")
    original = next(a for a in apps if a["app_id"] == recovery.SOURCE_APP)
    if original["state"] != "stopped":
        raise ReconciliationRequired("source controller not stopped")
    owner = modal.App.lookup("verifier-rl-evaluation", create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("candidate sandboxes still active")
    return {"source_app": original, "sandbox_app_id": owner.app_id, "active_sandboxes": [],
            "checked_utc": datetime.now(timezone.utc).isoformat()}


async def capture_source():
    prefix = two.RUN_ID + "/"
    paths = []
    async for entry in artifacts.iterdir.aio("/" + two.RUN_ID, recursive=True):
        path = entry.path.lstrip("/")
        if path.startswith(prefix) and recovery.selected_path(path[len(prefix):]):
            paths.append(path[len(prefix):])
    semaphore = asyncio.Semaphore(8)
    async def read(path):
        async with semaphore:
            chunks = [c async for c in artifacts.read_file.aio("/" + prefix + path)]
            return path, json.loads(b"".join(chunks))
    documents = dict(await asyncio.gather(*(read(p) for p in sorted(paths))))
    return {"run_id": two.RUN_ID, "documents": documents}


def local_source(directory):
    return {"run_id": two.RUN_ID, "documents": {p.relative_to(directory).as_posix(): json.loads(p.read_text())
        for p in sorted(directory.rglob("*.json")) if recovery.selected_path(p.relative_to(directory).as_posix())}}


def prepare():
    """Read-only cloud capture and offline replay. Does not launch a Modal Function."""
    observed = quiescent()
    source = asyncio.run(capture_source())
    parent_dir = Path("runs") / study.RUN_ID / "completed-remote" / study.RUN_ID
    parent = study.verify_run(parent_dir)
    check_frozen_sources(source["documents"]["source_snapshot.json"], Path.cwd())
    state = recovery.validate_source(source, parent)
    target = Path("runs") / recovery.RUN_ID
    persist(target, {"source": source, "parent_verification": parent, "plan": recovery.make_plan(state)})
    print("PREPARED", {"saved_programs": len(state["samples"]), "reused_complete_programs": len(state["reports"]),
        "pending_input_checks": len(state["pending"]), "source_sandbox_starts": len(state["source_sandbox_ids"]),
        "source_startup_retries": state["source_startup_retries"], "training": {k:v["evidence"] for k,v in state["arms"].items()}}, flush=True)
    return target, source, parent, state, observed


async def complete_pending(state, plan, directory, deadline, backend, store):
    """One shared queue; intent/result writes precede reuse and derived scoring."""
    cases = {c.input_hash: c for c in study.cases_for("evaluation")}
    attempts = {}
    stop = asyncio.Event()
    queue = asyncio.Queue()
    for item in state["pending"]:
        queue.put_nowait(item)
    def require_time():
        if time.time() >= deadline:
            raise ReconciliationRequired("recovery deadline reached; no new work")
    async def execute(sid, key, number):
        require_time()
        event = recovery.attempt_key(sid, key, number)
        path = directory / "attempts" / event
        intent = recovery.attempt_intent(state, plan, sid, key, number, deadline)
        if (path / "intent.json").exists() or (path / "result.json").exists():
            raise ReconciliationRequired("attempt already exists; no automatic replay: " + event)
        if not await store.put.aio(recovery.RUN_ID + "/intent/" + event, intent, skip_if_exists=True):
            raise ReconciliationRequired("attempt already claimed: " + event)
        persist(path, {"intent": intent})
        value = pack_result(await backend.execute(request_for(state["samples"][sid]["source"], cases[key])))
        if not await store.put.aio(recovery.RUN_ID + "/result/" + event, value, skip_if_exists=True):
            raise ReconciliationRequired("attempt result already exists: " + event)
        persist(path, {"result": value})
        return value
    async def worker():
        while not stop.is_set():
            try:
                sid, key = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            pair = []
            try:
                first = await execute(sid, key, 1)
                pair = attempts[sid + "/" + key] = [first]
                request = request_for(state["samples"][sid]["source"], cases[key])
                if (not stop.is_set() and (sid, key) != (recovery.FAILED_SAMPLE, recovery.FAILED_INPUT)
                        and recovery.pre_candidate_failure(unpack_result(first), request, state["setup"]["sandbox_image_id"])):
                    slot = None
                    for index in range(recovery.MAX_EXTRA_STARTUPS):
                        if await store.put.aio(recovery.RUN_ID + f"/startup-slot/{index}",
                                {"sample_id": sid, "input_hash": key, "failure": first}, skip_if_exists=True):
                            slot = index
                            persist(directory / "startup_slots", {str(index): {"sample_id": sid, "input_hash": key, "failure": first}})
                            break
                    if slot is not None:
                        pair.append(await execute(sid, key, 2))
                recovery.verify_attempt(state, sid, key, pair)
                if len(attempts) % 100 == 0:
                    print("RECOVERY INPUT PROGRESS", len(attempts), "/", len(state["pending"]), flush=True)
            except Exception as exc:
                stop.set()
                last = pair[-1] if pair else {}
                m = last.get("metadata", {})
                context = {"status": last.get("status"), "detail": last.get("detail"),
                    "sandbox_id": m.get("sandbox_id"), "preflight_returncode": m.get("preflight_returncode"),
                    "returncode": m.get("returncode"), "cleanup": m.get("cleanup")}
                raise ReconciliationRequired(f"{sid} input {key}: {type(exc).__name__}: {exc}; {context}") from exc
    # Wait for already-submitted work to finish/clean up, even after a first failure.
    outcomes = await asyncio.gather(*(worker() for _ in range(min(8, queue.qsize()))), return_exceptions=True)
    failures = [x for x in outcomes if isinstance(x, BaseException)]
    if failures:
        raise failures[0]
    return attempts


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=recovery.SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_recovery(plan, budget, snapshot, deadline):
    check_frozen_sources(snapshot, Path("/root"))
    artifacts.reload()
    directory = Path("/artifacts") / recovery.RUN_ID
    if not claims.put(recovery.RUN_ID + "/controller", {"plan": plan, "deadline": deadline}, skip_if_exists=True):
        raise ReconciliationRequired("controller already claimed; inspect saved records instead of replaying")
    stage = "source_verification"
    try:
        source = local_source(Path("/artifacts") / two.RUN_ID)
        parent = study.verify_run(Path("/artifacts") / study.RUN_ID)
        check_frozen_sources(source["documents"]["source_snapshot.json"], Path("/root"))
        state = recovery.validate_source(source, parent)
        recovery.validate_plan(plan, state)
        owner = modal.App.lookup(state["setup"]["app_name"], create_if_missing=False)
        if list(modal.Sandbox.list(app_id=owner.app_id)):
            raise ReconciliationRequired("candidate sandboxes active before recovery")
        persist(directory, {"source": source, "parent_verification": parent, "plan": plan, "budget": budget,
                "source_snapshot": snapshot, "deadline": deadline})
        artifacts.commit()
        stage = "pending_evaluations"
        print("CPU-ONLY RECOVERY", len(state["reports"]), "complete programs reused;", len(state["pending"]),
              "pending input checks; no training or generation", flush=True)
        backend = PanelBackend(state["setup"]["app_name"], state["setup"]["sandbox_image_id"], study.BOOKING,
                               creation_interval_seconds=.26)
        attempts = asyncio.run(complete_pending(state, plan, directory, deadline, backend, claims))
        artifacts.commit()
        stage = "offline_verification"
        if recovery.verify_journal(directory, state, plan, deadline) != attempts:
            raise ValueError("durable journal differs from selected attempts")
        result = recovery.verify_completion(source, parent, plan, attempts)
        if local_source(Path("/artifacts") / two.RUN_ID) != source:
            raise ValueError("original study changed during recovery")
        persist(directory, {"result": result})
        artifacts.commit()
        print("RECOVERY COMPLETE", result["policies"], flush=True)
        return result
    except Exception as exc:
        persist(directory, {f"stopped-{time.time_ns()}": {"stage": stage, "type": type(exc).__name__,
            "detail": str(exc)[:1500], "automatic_retry": False}})
        artifacts.commit()
        raise


@app.local_entrypoint()
def launch(allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for CPU-only saved-program recovery")
    target = Path("runs") / recovery.RUN_ID
    if (target / "launch_intent.json").exists():
        raise ReconciliationRequired("recovery already prepared for launch; inspect saved IDs")
    target, source, parent, state, observed = prepare()
    plan = recovery.make_plan(state)
    rates, billing = read_modal("billing", "rates"), read_modal("billing", "summary")
    budget = recovery.budget_quote(plan, rates, billing, source["documents"]["budget.json"]["cumulative_reservation_usd"])
    snapshot = {p.as_posix(): p.read_text() for p in sorted(Path("verifier_rl").glob("*.py"))
                + [Path("modal_booking_study.py"), Path("modal_booking_recovery.py")]}
    deadline = time.time() + recovery.SECONDS - 60
    persist(target, {"budget": budget, "source_snapshot": snapshot, "protocol": Path("docs/booking_recovery_protocol.txt").read_text(),
        "launch_intent": {"plan": plan, "deadline": deadline, "quiescence": observed}})
    call = run_recovery.spawn(plan, budget, snapshot, deadline)
    persist(target, {"launch": {"call_id": call.object_id, "created_utc": datetime.now(timezone.utc).isoformat()}})
    print("CPU-ONLY RECOVERY CALL", call.object_id, "CUMULATIVE RESERVATION USD", budget["cumulative_reservation_usd"], flush=True)
    persist(target, {"result": call.get()})
