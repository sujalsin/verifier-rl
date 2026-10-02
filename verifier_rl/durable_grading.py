"""Per-input execution journals shared by the new baseline and future trainer.

Only the injected remote backend may execute code. A lost or ambiguous submitted
attempt is retained as unknown. Re-entry reuses evidence and starts untouched
inputs; it never chooses a better answer by repeating a candidate.
"""

import asyncio
import json
from pathlib import Path
import time

from . import supervised_execution as supervised
from .booking_screen_recovery import validate_entry
from .evaluation_journal import ReconciliationRequired, persist
from .panel_execution import pack_result, request_for, unpack_result
from .suites import canonical_json, digest
from .task_panel import BOOKING


async def execute_batch(samples, cases, directory, key, plan, image_id, deadline,
                        backend, store, commit, *, clock=time.time):
    """Single owning grading Function; at most eight isolated executions at once.

The caller serializes grading invocations. Provider inserts are atomic even if a
controller dies before its Volume commit. Completed local journals avoid remote
reads; incomplete journals are reconciled before any decision to submit.
"""
    directory = Path(directory)
    intent = {"key": key, "sample_hash": digest(canonical_json(samples)),
              "case_hashes": [c.input_hash for c in cases], "plan_hash": digest(canonical_json(plan)),
              "image_id": image_id, "deadline": deadline}
    prefix = plan["run_id"] + "/durable/" + key
    fresh = await store.put.aio(prefix + "/batch", intent, skip_if_exists=True)
    if not fresh and await store.get.aio(prefix + "/batch") != intent:
        raise ValueError("grading batch binding/deadline changed")
    if await store.get.aio(plan["run_id"] + f"/durable/unknown-slot/{plan['unknown_circuit']-1}", None) is not None:
        raise ReconciliationRequired("unknown-outcome circuit already exhausted")
    persist(directory, {"intent": intent})
    await commit()
    entries = {s["sample_id"]: {} for s in samples}
    queue, stopped = asyncio.Queue(), asyncio.Event()
    for sample in samples:
        if not sample["extraction_status"].startswith("rejected_"):
            for case in cases:
                queue.put_nowait((sample, case))
    progress = {"inputs": 0, "new_starts": 0, "unknown_inputs": 0, "reused_inputs": 0}
    commit_lock = asyncio.Lock()

    async def read_entry(target, input_prefix):
        entry = {p.stem: json.loads(p.read_text()) for p in target.glob("*.json")}
        if not fresh and entry.get("attempt-1", {}).get("detail") == "supervisor_transport:cleanup_unconfirmed":
            receipt = await store.get.aio(input_prefix + "/cleanup-reconciliation", None)
            if receipt is not None:
                persist(target, {"cleanup-reconciliation": receipt})
                entry["cleanup-reconciliation"] = receipt
        if not fresh and ("selected" not in entry or "cleanup-reconciliation" in entry):
            for n in (1, 2):
                for suffix, name in (("intent", f"intent-{n}"), ("result", f"attempt-{n}")):
                    if name in entry or (suffix == "result" and f"intent-{n}" not in entry):
                        continue
                    value = await store.get.aio(f"{input_prefix}/{n}/{suffix}", None)
                    if value is not None:
                        entry[name] = value
            # A replacement result may survive only in the provider journal.
            # Recover its authorizing receipt before validating either attempt.
            if ("cleanup-reconciliation" not in entry and
                    entry.get("attempt-1", {}).get("detail") == "supervisor_transport:cleanup_unconfirmed"):
                receipt = await store.get.aio(input_prefix + "/cleanup-reconciliation", None)
                if receipt is not None:
                    entry["cleanup-reconciliation"] = receipt
            if entry:
                persist(target, entry)
        return entry

    async def attempt(sample, case, n, target, input_prefix, entry, slot=None):
        if stopped.is_set() or clock() >= deadline:
            raise ReconciliationRequired("execution deadline/circuit reached before submission")
        value = {"sample_id": sample["sample_id"], "source_hash": digest(sample["source"]),
                 "input_hash": case.input_hash, "attempt": n, "deadline": deadline,
                 "retry_slot": slot, "submitted_at": clock()}
        if not await store.put.aio(f"{input_prefix}/{n}/intent", value, skip_if_exists=True):
            raise ReconciliationRequired("input already submitted; reconcile, never duplicate")
        entry[f"intent-{n}"] = value
        persist(target, {f"intent-{n}": value})
        progress["new_starts"] += 1
        packed = pack_result(await backend.execute(request_for(sample["source"], case)))
        if not await store.put.aio(f"{input_prefix}/{n}/result", packed, skip_if_exists=True):
            raise ReconciliationRequired("duplicate execution result")
        entry[f"attempt-{n}"] = packed
        persist(target, {f"attempt-{n}": packed})
        return packed

    async def worker():
        while not stopped.is_set():
            if clock() >= deadline:
                stopped.set()
                raise ReconciliationRequired("grading deadline; preserve unfinished inputs")
            try:
                sample, case = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            sid, h = sample["sample_id"], case.input_hash
            target = directory / "inputs" / sid / h
            input_prefix = f"{prefix}/{sid}/{h}"
            try:
                entry = await read_entry(target, input_prefix)
                entries[sid][h] = entry
                if not entry:
                    await attempt(sample, case, 1, target, input_prefix, entry)
                else:
                    # Validate persisted evidence before considering a narrowly
                    # allowed PRE-candidate replacement.
                    validate_entry(entry, sample, case, image_id)
                    progress["reused_inputs"] += 1
                first = entry.get("attempt-1")
                reconciled = "cleanup-reconciliation" in entry
                if (first is not None and "intent-2" not in entry and ("selected" not in entry or reconciled)
                        and (reconciled or supervised.startup_retry_allowed(unpack_result(first), request_for(sample["source"], case),
                            image_id, BOOKING, policy_version=plan["startup_retry_version"]))):
                    for slot in range(plan["max_startup_retries"]):
                        reservation = {"key": key, "sample_id": sid, "input_hash": h,
                                       "first_hash": digest(canonical_json(first))}
                        if await store.put.aio(plan["run_id"] + f"/durable/startup-slot/{slot}",
                                               reservation, skip_if_exists=True):
                            persist(directory / "startup_slots", {str(slot): reservation})
                            await asyncio.sleep(1)
                            await attempt(sample, case, 2, target, input_prefix, entry, slot)
                            break
                outcome, selected = validate_entry(entry, sample, case, image_id)
                if selected is None:
                    submitted = max(v.get("submitted_at", deadline) for k,v in entry.items() if k.startswith("intent-"))
                    if clock() < submitted + 180:
                        raise ReconciliationRequired("orphan execution may still be active; wait for sandbox lifetime before re-entry")
                if selected is not None:
                    name = "selected-after-reconciliation" if reconciled and "intent-2" in entry else "selected"
                    entry[name] = selected
                    persist(target, {name: selected})
                progress["inputs"] += 1
                if outcome["passed"] is None:
                    progress["unknown_inputs"] += 1
                    print("UNKNOWN INPUT", key, sid, h, outcome["reason"], flush=True)
                    if selected and selected["metadata"].get("cleanup") != "terminated":
                        raise ReconciliationRequired("cleanup unconfirmed; stop new work")
                    # Unknowns have stable identities, so replay does not consume
                    # more slots. The controller stops on a completed batch too.
                    unknown_key = plan["run_id"] + f"/durable/unknown/{sid}/{h}"
                    if await store.put.aio(unknown_key, {"reason": outcome["reason"]}, skip_if_exists=True):
                        reserved = False
                        for slot in range(plan["unknown_circuit"]):
                            if await store.put.aio(plan["run_id"] + f"/durable/unknown-slot/{slot}",
                                                   unknown_key, skip_if_exists=True):
                                reserved = True
                                if slot == plan["unknown_circuit"] - 1:
                                    raise ReconciliationRequired("unknown-outcome circuit reached")
                                break
                        if not reserved:
                            raise ReconciliationRequired("unknown-outcome circuit already exhausted")
                if progress["inputs"] % 100 == 0:
                    print("DURABLE INPUT PROGRESS", key, progress, "remaining", queue.qsize(), flush=True)
                    async with commit_lock:
                        await commit()
            except BaseException:
                stopped.set()
                raise
    try:
        results = await asyncio.gather(*(worker() for _ in range(plan["concurrency"])), return_exceptions=True)
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
        return entries, progress
    finally:
        await commit()
