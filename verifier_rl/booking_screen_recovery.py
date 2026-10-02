"""CPU-only continuation with explicit unknowns; original screen stays immutable."""

import argparse
import asyncio
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import time

from . import booking_boundary_screen as screen, booking_boundary_contrast as contrast
from . import booking_verifier_v2 as coverage, supervised_execution as supervised
from .grading import Status
from .evaluation_journal import persist, ReconciliationRequired
from .modal_backend import Limits
from .panel_execution import request_for, unpack_result, pack_result
from .suites import canonical_json, digest
from .task_panel import BOOKING
from .verifier_quality import compare_output

VERSION = "booking-screen-recovery-0.2"
RUN_ID = "qwen-booking-boundary-recovery-20260929-v2"
FAILED_SAMPLE = "screen-17022"
FAILED_INPUT = "86d538d0d14f336b930e621ea3922ef1bb6143b4539e899bcbaf5a44c511ec79"
FAILED_SOURCE = "cd6b6f3de78e9edbdcf16ecb294529d3de9a80217b490bb6e8e393f3ce3569c4"
FAILED_SANDBOX = "sb-o8hrU9aGS5evgVkAiXCVRd"
DIAGNOSTIC_SECONDS, RECOVERY_SECONDS = 600, 3600
MAX_STARTS, MAX_STARTUP_RETRIES, UNKNOWN_CIRCUIT = 2, 16, 16


def diagnostic_runner():
    """Protected parent telemetry, remote ONLY; never eligible for research scores."""
    original = supervised.supervised_runner(BOOKING)
    telemetry = r'''
import resource
def diagnostic(event):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    row = {"event": event, "pid": os.getpid(), "cpu": round(usage.ru_utime + usage.ru_stime, 4),
           "rss_kib": usage.ru_maxrss, "cpu_limit": resource.getrlimit(resource.RLIMIT_CPU),
           "as_limit": resource.getrlimit(resource.RLIMIT_AS)}
    if event in ("start", "reaped"):
        row["cgroup"] = {}
        for name in ("memory.current", "memory.max", "memory.events"):
            try:
                with open("/sys/fs/cgroup/" + name) as stream:
                    row["cgroup"][name] = stream.read(160).strip()
            except OSError as exc:
                row["cgroup"][name] = type(exc).__name__
    print("SUPERVISOR_DIAGNOSTIC " + json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
diagnostic("start")
'''
    replacements = {
        'payload = json.loads(sys.argv[1])\nlimit = payload["limits"]':
            telemetry + '\npayload = json.loads(sys.argv[1])\nlimit = payload["limits"]',
        'started_at = time.monotonic()': 'diagnostic("spawned")\nnext_diagnostic = 1\nstarted_at = time.monotonic()',
        'elapsed = time.monotonic() - started_at':
            'elapsed = time.monotonic() - started_at\n        if elapsed >= next_diagnostic:\n'
            '            diagnostic("tick")\n            next_diagnostic += 1',
        'process.returncode = os.waitstatus_to_exitcode(status)\n                #':
            'process.returncode = os.waitstatus_to_exitcode(status)\n                diagnostic("reaped")\n                #',
    }
    for before, after in replacements.items():
        if original.count(before) != 1:
            raise ValueError("diagnostic insertion point changed")
        original = original.replace(before, after)
    return original


def diagnostic_specs(sample):
    if sample["sample_id"] != FAILED_SAMPLE or digest(sample["source"]) != FAILED_SOURCE:
        raise ValueError("diagnostic candidate changed")
    correct = "def required_capacity(bookings): return 1"
    loop = "def required_capacity(bookings):\n    while True: pass"
    # Original and instrumented observations remain SEPARATE from the screen.
    return [("original_saved", sample["source"], False, Status.COMPLETED, b"448"),
            ("original_correct", correct, False, Status.COMPLETED, b"1"),
            ("instrumented_saved_1", sample["source"], True, Status.COMPLETED, b"448"),
            ("instrumented_correct", correct, True, Status.COMPLETED, b"1"),
            ("instrumented_saved_2", sample["source"], True, Status.COMPLETED, b"448"),
            ("instrumented_cpu_limit", loop, True, Status.TIMEOUT, None)]


def assess_record(sample, case, record, image_id):
    """Only genuine execution uncertainty becomes None; invalid binding is an error."""
    result = unpack_result(record)
    m = result.metadata
    expected = {"backend": "modal", "task_id": BOOKING, "source_hash": digest(sample["source"]),
        "input_hash": case.input_hash, "image_id": image_id, "limits": asdict(Limits()),
        "runner_hash": digest(supervised.supervised_runner(BOOKING)), "execution_version": supervised.VERSION,
        "block_network": True, "reset": "fresh_sandbox_per_input", "creation_interval_seconds": .26,
        "sdk_version": "1.5.5"}
    if any(m.get(key) != value for key, value in expected.items()):
        raise ValueError("execution identity/limits changed")
    if result.status == Status.INFRASTRUCTURE_ERROR:
        if "supervisor_report" in m:
            supervised.validate_report(result, BOOKING, image_id, source=sample["source"], case=case)
        return {"passed": None, "inclusive_match": None, "reason": result.detail,
                "record_hash": digest(canonical_json(record))}
    supervised.require_evidence(result, BOOKING, image_id, source=sample["source"], case=case)
    passed = result.status == Status.COMPLETED and compare_output(case, result.stdout)[0]
    inclusive = (result.status == Status.COMPLETED and
                 contrast._observed_integer(record) == contrast.inclusive_answer(case.arguments_json))
    return {"passed": passed, "inclusive_match": inclusive, "reason": result.detail or ("passed" if passed else "wrong_output"),
            "record_hash": digest(canonical_json(record))}


def unknown(reason):
    return {"passed": None, "inclusive_match": None, "reason": reason, "record_hash": None}


def validate_entry(entry, sample, case, image_id):
    """Replay original or recovery attempts, never choose the better answer."""
    if "committed_batch" in entry:
        # Completed batches are authoritative committed execution reports, not
        # invented per-input submission intents. load_source independently
        # verifies the entire batch, including retry history and raw-file hash.
        sid_number = int(sample["sample_id"].removeprefix("screen-"))
        provenance = entry["committed_batch"]
        if (set(entry) != {"committed_batch", "selected"} or not 17000 <= sid_number < 17064
                or provenance.get("key") != f"screen-{(sid_number-17000)//4:02d}"
                or len(provenance.get("report_hash", "")) != 64
                or any(c not in "0123456789abcdef" for c in provenance["report_hash"])):
            raise ValueError("invalid committed batch provenance")
        outcome = assess_record(sample, case, entry["selected"], image_id)
        if outcome["passed"] is None:
            raise ValueError("committed complete batch contains unknown outcome")
        return outcome, entry["selected"]
    allowed = {"intent-1", "attempt-1", "intent-2", "attempt-2", "selected",
               "cleanup-reconciliation", "selected-after-reconciliation"}
    if set(entry) - allowed:
        raise ValueError("unexpected input journal document")
    for number in (1, 2):
        intent, attempt = entry.get(f"intent-{number}"), entry.get(f"attempt-{number}")
        if intent is not None:
            if (intent.get("sample_id") != sample["sample_id"] or intent.get("source_hash") != digest(sample["source"])
                    or intent.get("input_hash") != case.input_hash or intent.get("attempt") != number):
                raise ValueError("input intent binding changed")
        if attempt is not None:
            if intent is None:
                raise ValueError("execution without a recorded intent")
            assess_record(sample, case, attempt, image_id)
    first = entry.get("attempt-1")
    reconciled = "cleanup-reconciliation" in entry
    if reconciled:
        from .booking_cleanup_recovery import validate_reconciliation
        if first is None:
            raise ValueError("cleanup receipt without original attempt")
        validate_reconciliation(first, entry["cleanup-reconciliation"], sample, case)
    if "intent-2" in entry:
        if first is None or not (reconciled or supervised.startup_retry_allowed(unpack_result(first), request_for(sample["source"], case),
                image_id, BOOKING, policy_version=supervised.STARTUP_RETRY_VERSION)):
            raise ValueError("second attempt not justified by pre-candidate evidence")
        selected = entry.get("attempt-2")
        if selected and selected["metadata"].get("sandbox_id") == first["metadata"].get("sandbox_id"):
            raise ValueError("replacement reused the first sandbox")
    else:
        selected = first
    # The historical selection is immutable. A reconciled startup retry gets
    # a separate selection receipt; no failed attempt is erased or relabelled.
    if "selected" in entry and entry["selected"] != selected and not (reconciled and entry["selected"] == first):
        raise ValueError("selected execution differs from prescribed attempt")
    if "selected-after-reconciliation" in entry:
        if (not reconciled or "intent-2" not in entry or selected is None
                or entry["selected-after-reconciliation"] != selected):
            raise ValueError("reconciled selection differs from prescribed attempt")
    if selected is None:
        return unknown("unresolved_submitted_intent" if entry else "not_started"), None
    return assess_record(sample, case, selected, image_id), selected


def load_source(directory, store=None):
    """Read-only inventory, including unfinished work; run inside the CPU controller."""
    directory = Path(directory)
    def read(relative):
        return json.loads((directory / relative).read_text())
    plan, setup, generation = read("plan.json"), read("setup.json"), read("generation/result.json")
    screen.validate_plan(plan)
    if (plan["run_id"] != screen.RUN_ID or len(generation["samples"]) != 64
            or generation["parameters_unchanged"] is not True or generation["parameter_hash"] != plan["initial_parameter_hash"]
            or generation["initial_checkpoint"] != plan["initial_checkpoint"]):
        raise ValueError("saved generation identity changed")
    # Existing files named by the frozen source snapshot must still be identical.
    snapshot = read("source_snapshot.json")
    for name, content in snapshot.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or Path(name).read_text() != content:
            raise ValueError("original implementation changed: " + name)
    controls = read("grading/controls/result.json")
    checked, ids, slots = screen.verify_batch(controls, screen.controls(plan), "controls", plan, setup["sandbox_image_id"])
    if screen.validate_controls(checked) != read("preflight.json"):
        raise ValueError("saved preflight changed")
    ids += screen.supervisor_controls.validate_controls(read("supervisor_controls.json"), setup["sandbox_image_id"])
    source = {"plan": plan, "setup": setup, "generation": generation,
              "original_source_snapshot_hash": digest(canonical_json(snapshot)),
              "original_budget": read("budget.json"), "entries": {}, "completed_batches": [],
              "committed_batch_evidence": {}}
    print("SOURCE CONTROLS VALIDATED", flush=True)
    for index in range(16):
        key, samples = f"screen-{index:02d}", generation["samples"][4*index:4*index+4]
        screen.validate_batch(key, samples, plan)
        result_path = directory / f"grading/{key}/result.json"
        saved = json.loads(result_path.read_text()) if result_path.exists() else None
        submitted = (directory / f"grading/{key}/intent.json").exists()
        if store is not None:
            claimed = store.get(f"{screen.RUN_ID}/work/grade/{key}", None)
            if claimed is not None and not submitted:
                raise ValueError("batch claim without committed intent")
        if saved is not None:
            _, used, retried = screen.verify_batch(saved, samples, key, plan, setup["sandbox_image_id"])
            if read(f"grading/{key}/raw.json") != {k: v for k, v in saved.items() if k != "graded"}:
                raise ValueError("saved raw batch changed")
            source["completed_batches"].append(key)
            slots.extend(retried)
            ids.extend(used)
            source["committed_batch_evidence"][key] = {"report_hash": digest(canonical_json(saved)),
                                                       "retries": saved["retries"]}
        for sample in samples:
            sid = sample["sample_id"]
            if (read(f"generation/samples/{sid}.json") != sample or
                    read(f"generation/sample-intents/{sid}.json") != {"seed": sample["seed"], "sample_id": sid}):
                raise ValueError("saved sample or seed intent changed")
            entries = source["entries"][sid] = {}
            if saved is not None:
                provenance = {"key": key, "report_hash": source["committed_batch_evidence"][key]["report_hash"]}
                entries.update({h: {"committed_batch": provenance, "selected": record}
                                for h, record in saved["records"][sid].items()})
                continue
            if not submitted:
                continue
            # Avoid thousands of negative FUSE directory probes for work which
            # was never submitted. Reconcile only the actual interrupted batch.
            input_root = directory / "grading" / key / "inputs" / sid
            existing = {p.name for p in input_root.iterdir()} if input_root.exists() else set()
            for case in coverage.cases_for("training"):
                target = directory / "grading" / key / "inputs" / sid / case.input_hash
                entry = ({p.stem: json.loads(p.read_text()) for p in target.glob("*.json")}
                         if case.input_hash in existing else {})
                # The provider journal survives a controller crash before Volume commit.
                if store is not None and submitted and saved is None:
                    for number in (1, 2):
                        prefix = f"{screen.RUN_ID}/execution/{sid}/{case.input_hash}/{number}"
                        for suffix, name in (("intent", f"intent-{number}"), ("result", f"attempt-{number}")):
                            if name in entry:
                                continue
                            if suffix == "result" and f"intent-{number}" not in entry:
                                continue
                            if number == 2 and ("attempt-1" not in entry or not supervised.startup_retry_allowed(
                                    unpack_result(entry["attempt-1"]), request_for(sample["source"], case),
                                    setup["sandbox_image_id"], BOOKING, policy_version=supervised.STARTUP_RETRY_VERSION)):
                                continue
                            value = store.get(prefix + "/" + suffix, None)
                            if value is not None:
                                if name in entry and entry[name] != value:
                                    raise ValueError("provider and Volume journals disagree")
                                entry[name] = value
                if entry:
                    if not submitted:
                        raise ValueError("input evidence in an unsubmitted batch")
                    _, selected = validate_entry(entry, sample, case, setup["sandbox_image_id"])
                    if saved and selected != saved["records"][sid].get(case.input_hash):
                        raise ValueError("selected journal differs from completed batch")
                    entries[case.input_hash] = entry
                    ids.extend(v["metadata"]["sandbox_id"] for k, v in entry.items()
                               if k.startswith("attempt-") and v["metadata"].get("sandbox_id"))
                elif saved and not sample["extraction_status"].startswith("rejected_"):
                    raise ValueError("completed batch missing durable input evidence")
        print("SOURCE BATCH VALIDATED", key, "committed", saved is not None, "submitted", submitted, flush=True)
    if len(ids) != len(set(ids)) or len(slots) != len(set(slots)):
        raise ValueError("reused sandbox or startup retry slot")
    failed = source["entries"][FAILED_SAMPLE][FAILED_INPUT]
    m = failed["selected"]["metadata"]
    if (m.get("sandbox_id") != FAILED_SANDBOX or m.get("returncode") != 137
            or failed["selected"]["detail"] != "supervisor_transport:nonzero_exit"):
        raise ValueError("reviewed stop differs")
    source["historical_sandbox_ids"] = ids
    source["original_retry_slots"] = slots
    return source


def inventory(source):
    pending, assessments, records, counts = [], {}, {}, {"resolved": 0, "unknown": 0, "not_started": 0, "extraction_rejected": 0}
    for sample in source["generation"]["samples"]:
        sid = sample["sample_id"]
        assessments[sid], records[sid] = {}, {}
        for case in coverage.cases_for("training"):
            entry = source["entries"][sid].get(case.input_hash, {})
            if sample["extraction_status"].startswith("rejected_"):
                if entry:
                    raise ValueError("rejected extraction unexpectedly executed")
                value, record = {"passed": False, "inclusive_match": False, "reason": "extraction_rejected", "record_hash": None}, None
                counts["extraction_rejected"] += 1
            else:
                value, record = validate_entry(entry, sample, case, source["setup"]["sandbox_image_id"])
                label = "resolved" if value["passed"] is not None else ("unknown" if entry else "not_started")
                counts[label] += 1
                if not entry:
                    pending.append([sid, case.input_hash])
            assessments[sid][case.input_hash] = value
            if record is not None:
                records[sid][case.input_hash] = record
    return {"pending": pending, "assessments": assessments, "records": records, "counts": counts}


def replay_recovery(source, directory):
    """Rebuild scores only from original evidence and new per-input journals."""
    state = inventory(source)
    samples = {s["sample_id"]: s for s in source["generation"]["samples"]}
    cases = {c.input_hash: c for c in coverage.cases_for("training")}
    ids, attempts = list(source["historical_sandbox_ids"]), 0
    allowed = {tuple(p) for p in state["pending"]}
    for target in (Path(directory) / "inputs").glob("*/*"):
        sid, h = target.parent.name, target.name
        if (sid, h) not in allowed:
            raise ValueError("recovery attempted an original submitted input")
        entry = {p.stem: json.loads(p.read_text()) for p in target.glob("*.json")}
        outcome, record = validate_entry(entry, samples[sid], cases[h], source["setup"]["sandbox_image_id"])
        state["assessments"][sid][h] = outcome
        if record is not None:
            state["records"][sid][h] = record
        for name, value in entry.items():
            if name.startswith("intent-"):
                attempts += 1
            if name.startswith("attempt-") and value["metadata"].get("sandbox_id"):
                ids.append(value["metadata"]["sandbox_id"])
    if len(ids) != len(set(ids)) or attempts > len(allowed) + MAX_STARTUP_RETRIES:
        raise ValueError("sandbox reused or execution bound exceeded")
    return state


async def recover_inputs(source, plan, directory, deadline, backend, store, commit, *, clock=time.time):
    """Continue after unknowns, but never resubmit an ambiguous/incorrect execution.

    Provider intents claim starts before submission; provider results survive a
    lost Volume commit. Explicit limits bound retries, concurrency and wall time.
    """
    if plan != plan_for(source):
        raise ValueError("recovery plan changed")
    samples = {s["sample_id"]: s for s in source["generation"]["samples"]}
    cases = {c.input_hash: c for c in coverage.cases_for("training")}
    queue, stopped = asyncio.Queue(), asyncio.Event()
    for sid, h in plan["pending"]:
        queue.put_nowait((samples[sid], cases[h]))
    progress = {"completed_inputs": 0, "unknown_inputs": 0, "new_starts": 0, "stop_reason": None}
    commit_lock = asyncio.Lock()

    async def reconcile(target, prefix):
        entry = {p.stem: json.loads(p.read_text()) for p in target.glob("*.json")}
        for n in (1, 2):
            # Query only missing evidence. A retry intent can survive without its
            # first local result, so both attempt numbers are reconciled.
            for suffix, name in (("intent", f"intent-{n}"), ("result", f"attempt-{n}")):
                if name in entry:
                    continue
                if suffix == "result" and f"intent-{n}" not in entry:
                    continue
                value = await store.get.aio(f"{prefix}/{n}/{suffix}", None)
                if value is not None:
                    entry[name] = value
        if entry:
            persist(target, entry)
        return entry

    async def attempt(sample, case, number, target, prefix, entry):
        if clock() >= deadline or stopped.is_set():
            return None
        intent = {"sample_id": sample["sample_id"], "source_hash": digest(sample["source"]),
                  "input_hash": case.input_hash, "attempt": number, "deadline": deadline}
        if not await store.put.aio(f"{prefix}/{number}/intent", intent, skip_if_exists=True):
            raise ReconciliationRequired("concurrent or previously submitted input")
        persist(target, {f"intent-{number}": intent})
        entry[f"intent-{number}"] = intent
        progress["new_starts"] += 1
        result = pack_result(await backend.execute(request_for(sample["source"], case)))
        if not await store.put.aio(f"{prefix}/{number}/result", result, skip_if_exists=True):
            raise ReconciliationRequired("input result already exists")
        persist(target, {f"attempt-{number}": result})
        entry[f"attempt-{number}"] = result
        return result

    async def worker():
        while not stopped.is_set():
            if clock() >= deadline:
                progress["stop_reason"] = "deadline"
                stopped.set()
                break
            try:
                sample, case = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            sid, h = sample["sample_id"], case.input_hash
            target = Path(directory) / "inputs" / sid / h
            prefix = f"{RUN_ID}/execution/{sid}/{h}"
            try:
                entry = await reconcile(target, prefix)
                if not entry:
                    first = await attempt(sample, case, 1, target, prefix, entry)
                    if first is not None and supervised.startup_retry_allowed(unpack_result(first),
                            request_for(sample["source"], case), source["setup"]["sandbox_image_id"], BOOKING,
                            policy_version=supervised.STARTUP_RETRY_VERSION):
                        for slot in range(MAX_STARTUP_RETRIES):
                            reservation = {"sample_id": sid, "input_hash": h, "first_hash": digest(canonical_json(first))}
                            if await store.put.aio(f"{RUN_ID}/startup-slot/{slot}", reservation, skip_if_exists=True):
                                persist(Path(directory) / "startup_slots", {str(slot): reservation})
                                await asyncio.sleep(1)
                                await attempt(sample, case, 2, target, prefix, entry)
                                break
                outcome, record = validate_entry(entry, sample, case, source["setup"]["sandbox_image_id"])
                if record is not None:
                    persist(target, {"selected": record})
                progress["completed_inputs"] += 1
                if outcome["passed"] is None:
                    progress["unknown_inputs"] += 1
                    print("RECOVERY UNKNOWN", sid, h, outcome["reason"], flush=True)
                    if record and record["metadata"].get("cleanup") != "terminated":
                        progress["stop_reason"] = "cleanup_unconfirmed"
                        stopped.set()
                    if progress["unknown_inputs"] >= UNKNOWN_CIRCUIT:
                        progress["stop_reason"] = "unknown_circuit"
                        stopped.set()
                if progress["completed_inputs"] % 100 == 0:
                    print("RECOVERY PROGRESS", progress, "remaining", queue.qsize(), flush=True)
                    async with commit_lock:
                        await commit()
            except Exception:
                stopped.set()
                raise
    results = await asyncio.gather(*(worker() for _ in range(plan["concurrency"])), return_exceptions=True)
    await commit()
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        raise failures[0]
    return progress


def plan_for(source):
    state = inventory(source)
    return {"version": VERSION, "run_id": RUN_ID, "source_run_id": screen.RUN_ID,
        "source_hash": digest(canonical_json(source)), "original_plan_hash": digest(canonical_json(source["plan"])),
        "generation_hash": digest(canonical_json(source["generation"])), "pending": state["pending"],
        "preserved": state["counts"], "diagnostic_starts": 6,
        "diagnostic_runner_hash": digest(diagnostic_runner()), "diagnostic_seconds": DIAGNOSTIC_SECONDS,
        "recovery_seconds": RECOVERY_SECONDS, "max_controller_starts": MAX_STARTS,
        "max_sandbox_starts": len(state["pending"]) + MAX_STARTUP_RETRIES,
        "max_startup_retries": MAX_STARTUP_RETRIES, "unknown_circuit": UNKNOWN_CIRCUIT,
        "concurrency": 8, "creation_interval_seconds": .26, "new_generations": 0, "optimizer_updates": 0,
        "ambiguous_attempt_retries": 0, "diagnostic_results_supply_scores": False,
        "original_gate_requires_complete_evidence": True, "automatic_training": False}


def _bounds(values):
    if any(v is not None and type(v) is not bool for v in values):
        raise ValueError("boolean or explicit unknown required")
    lo, unresolved = sum(v is True for v in values), sum(v is None for v in values)
    return {"passed_bounds": [lo, lo+unresolved], "total": len(values), "unknown": unresolved,
            "reward_bounds": [lo/len(values), (lo+unresolved)/len(values)],
            "full_pass_bounds": [int(lo == len(values)), int(not any(v is False for v in values))]}


def summarize(source, assessments, records):
    cases = coverage.cases_for("training")
    kept, _ = contrast.partition(cases)
    samples, rows, exact_rows = source["generation"]["samples"], [], {}
    if set(assessments) != {s["sample_id"] for s in samples}:
        raise ValueError("all 64 samples must be represented")
    for sample in samples:
        sid, outcomes = sample["sample_id"], assessments[sample["sample_id"]]
        if set(outcomes) != {c.input_hash for c in cases}:
            raise ValueError("all input outcomes must be represented")
        ref = _bounds([outcomes[c.input_hash]["passed"] for c in cases])
        weak = _bounds([outcomes[c.input_hash]["passed"] for c in kept])
        signature = _bounds([outcomes[c.input_hash]["inclusive_match"] for c in cases])["full_pass_bounds"]
        row = {"sample_id": sid, "source_hash": digest(sample["source"]), "reference": ref,
            "endpoint_omission": weak, "inclusive_signature_bounds": signature,
            "omission_only_full_acceptance_bounds": [int(weak["full_pass_bounds"][0] and not ref["full_pass_bounds"][1]),
                                                      int(weak["full_pass_bounds"][1] and not ref["full_pass_bounds"][0])],
            "unknown_inputs": [{"input_hash": k, "reason": v["reason"]} for k, v in outcomes.items() if v["passed"] is None]}
        rows.append(row)
        if not ref["unknown"]:
            exact_rows[sid] = screen.grade(sample, records[sid], source["setup"]["sandbox_image_id"], source["plan"])["row"]
    groups = []
    for i in range(16):
        sids = [s["sample_id"] for s in samples[i*4:i*4+4]]
        if all(sid in exact_rows for sid in sids):
            groups.append(contrast.compare_group(f"screen-{i:02d}", [exact_rows[sid] for sid in sids], actual_training_group=False))
    count = lambda key: [sum(r[key][j] for r in rows) for j in (0, 1)]
    totals = {"programs": len(rows), "fully_resolved_programs": len(exact_rows),
        "unknown_input_outcomes": sum(r["reference"]["unknown"] for r in rows),
        "reference_full_pass_bounds": [sum(r["reference"]["full_pass_bounds"][j] for r in rows) for j in (0, 1)],
        "weak_full_pass_bounds": [sum(r["endpoint_omission"]["full_pass_bounds"][j] for r in rows) for j in (0, 1)],
        "false_acceptance_bounds": count("omission_only_full_acceptance_bounds"),
        "inclusive_signature_bounds": count("inclusive_signature_bounds"),
        "distinct_inclusive_source_bounds": [len({r["source_hash"] for r in rows if r["inclusive_signature_bounds"][j]}) for j in (0, 1)],
        "resolved_groups": len(groups), "changed_relative_signal_groups_lower_bound": sum(g["relative_signal_changed"] for g in groups)}
    decision = (screen.decision(contrast.summarize([exact_rows[s["sample_id"]] for s in samples], groups))
                if len(exact_rows) == 64 else {"ready_for_matched_protocol_review": False, "automatic_training": False,
                    "amplification_established": False, "reason": "original complete-evidence gate not met; retain uncertainty"})
    return {"version": VERSION, "status": "completed" if len(exact_rows) == 64 else "completed_with_uncertainty",
        "summary": totals, "programs": rows, "groups": groups, "decision": decision,
        "new_generations": 0, "optimizer_updates": 0, "audit_evaluated": False}


def budget_quote(plan, rates, billing, previous, phase):
    if phase not in ("diagnostic", "recovery"):
        raise ValueError("unknown phase")
    def number(value):
        d = Decimal(str(value))
        if not d.is_finite() or d < 0:
            raise ValueError("invalid cost")
        return d
    starts = plan["diagnostic_starts"] if phase == "diagnostic" else plan["max_sandbox_starts"]
    seconds = DIAGNOSTIC_SECONDS if phase == "diagnostic" else MAX_STARTS * RECOVERY_SECONDS
    sandbox = starts * 120 * (number(rates["cpu_hour_cost_sandbox"]) + number(rates["mem_gib_hour_cost_sandbox"])/4) / 3600
    controller = 3 * seconds * (number(rates["cpu_hour_cost"]) + 2*number(rates["mem_gib_hour_cost"])) / 3600
    total = sandbox + controller + 1
    held = max(number(previous), number(billing["metered_cost"]))
    return {"phase": phase, "rates": rates, "billing_before": billing, "previous_holds_usd": str(held),
            "sandbox_max_usd": str(sandbox), "controller_max_usd": str(controller), "contingency_usd": "1",
            "additional_reservation_usd": str(total), "cumulative_reservation_usd": str(held + total),
            "is_invoice": False, "provider_hard_cap": False, "gpu_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    source = json.loads((args.directory / "source.json").read_text())
    print(json.dumps(plan_for(source), indent=2, sort_keys=True))
