"""Evaluation-only parallel runner. No training, replacement draws or code retries.

This is a separate execution-scheduling amendment, not a relabeling of the
single-service runtime. Candidate execution and scoring reuse the frozen code.
One serialized coordinator owns all permits, unknown accounting and budgets.
Workers exclusively own disjoint immutable evidence directories.
"""

import asyncio
from copy import deepcopy
from decimal import Decimal
import json
import math
from pathlib import Path
import time

from . import booking_replication as study, program_execution as execution
from . import program_grading as old, program_storage as storage
from .evaluation_journal import ReconciliationRequired
from .program_benchmark import bounded_map
from .suites import canonical_json, digest

VERSION = "parallel-evaluation-0.1"
AMENDMENT = "parallel-evaluation-001"
WORKERS = 4
PROGRAMS_PER_WORKER = 4
PERMIT_TTL = 5.0
CANARY_INTERVAL = 7.0  # At most one new start/s alongside the old four-start/s gate.
RESEARCH_INTERVAL = 2.0  # At most four starts in any 1s window, including TTL jitter.
MAX_SECONDS = 3600
CANARY_CEILING = Decimal("5")


def fingerprint(value):
    return digest(canonical_json(value))


def runtime(parent, mode, deadline):
    if mode not in ("canary", "research") or not math.isfinite(deadline):
        raise ValueError("invalid evaluation runtime")
    return {"version": VERSION, "amendment": AMENDMENT, "mode": mode,
            "image_id": parent["image_id"], "execution_version": execution.VERSION,
            "runner_hash": digest(execution.runner()), "profile": execution.PROFILE,
            "workers": WORKERS, "programs_per_worker": PROGRAMS_PER_WORKER,
            "creation_interval": CANARY_INTERVAL if mode == "canary" else RESEARCH_INTERVAL,
            "permit_ttl": PERMIT_TTL, "deadline": deadline,
            "automatic_candidate_retries": 0}


def validate_runtime(value):
    if value != runtime(value, value["mode"], value["deadline"]):
        raise ValueError("parallel scheduling/execution contract changed")


def initialize(value, manifests):
    validate_runtime(value)
    allowed = {m["key"]: {"identity": fingerprint(m), "samples": {
        s["sample_id"]: {"rejected": s["extraction_status"].startswith("rejected_")}
        for s in m["samples"]}} for m in manifests}
    if not allowed or len(allowed) != len(manifests):
        raise ValueError("fixed, unique jobs required")
    if any(m["runtime"] != value for m in manifests):
        raise ValueError("mixed execution contracts")
    return {"runtime": value, "allowed": allowed, "owners": {}, "permits": {},
            "programs": {}, "batches": {}, "unknown": [], "last_permit": None, "stop": None}


def transition(state, action, payload, now):
    """Pure state transition. Caller must be a single non-concurrent service."""
    result = deepcopy(state)
    validate_runtime(result["runtime"])
    if action == "status":
        return result, {"batches": len(result["batches"]), "total": len(result["allowed"]),
                        "permits": len(result["permits"]), "unknown": len(result["unknown"]),
                        "stop": result["stop"]}
    key = payload["key"]
    if key not in result["allowed"] or payload["identity"] != result["allowed"][key]["identity"]:
        raise ValueError("job is not in the fixed execution manifest")
    if action == "claim":
        if result["stop"] or now >= result["runtime"]["deadline"]:
            raise ReconciliationRequired("parallel scheduler stopped or expired")
        if key in result["owners"]:
            raise ReconciliationRequired("batch already owned; never duplicate worker execution")
        result["owners"][key] = payload["owner"]
        return result, True
    if result["owners"].get(key) != payload["owner"]:
        raise ValueError("wrong batch owner")
    if action in ("permit", "program") and payload["sample_id"] not in result["allowed"][key]["samples"]:
        raise ValueError("program not in fixed manifest")
    if action == "permit":
        sid = payload["sample_id"]
        token = key + "/" + sid
        if result["allowed"][key]["samples"][sid]["rejected"]:
            raise ValueError("rejected source must never execute")
        if result["stop"] or now >= result["runtime"]["deadline"]:
            raise ReconciliationRequired("no new starts after stop/deadline")
        if token in result["permits"]:
            raise ReconciliationRequired("permit already issued; ambiguous start must not be replayed")
        previous = result["last_permit"]
        if previous is not None and now - previous < result["runtime"]["creation_interval"]:
            raise ValueError("global start gate violated")
        permit = {"issued": now, "expires": now + PERMIT_TTL, "identity": payload["identity"]}
        result["permits"][token] = permit
        result["last_permit"] = now
        return result, permit
    if action == "program":
        token = key + "/" + payload["sample_id"]
        rejected = result["allowed"][key]["samples"][payload["sample_id"]]["rejected"]
        if token not in result["permits"] and not rejected:
            raise ValueError("program evidence without execution permit")
        receipt = {k: payload[k] for k in ("document_hash", "unknown", "cleanup", "sandbox_ids")}
        for other, published in result["programs"].items():
            if other != token and set(published["sandbox_ids"]) & set(receipt["sandbox_ids"]):
                raise ValueError("sandbox reused across distinct programs")
        previous = result["programs"].get(token)
        if previous is not None and previous != receipt:
            raise ValueError("published execution evidence changed")
        result["programs"][token] = receipt
        result["unknown"] = sorted(set(result["unknown"]) | set(payload["unknown"]))[:64]
        if payload["cleanup"] != "terminated":
            result["stop"] = {"reason": "cleanup_unconfirmed", "program": token}
        elif len(result["unknown"]) >= 64:
            result["stop"] = {"reason": "unknown_circuit", "program": token}
        return result, {"stop": result["stop"]}
    if action == "batch":
        if any(key + "/" + sid not in result["programs"] for sid in result["allowed"][key]["samples"]):
            raise ValueError("cannot publish a batch before all programs")
        receipt = payload["receipt"]
        previous = result["batches"].get(key)
        if previous is not None and previous != receipt:
            raise ValueError("batch publication changed")
        result["batches"][key] = receipt
        return result, {"completed": len(result["batches"]), "total": len(result["allowed"])}
    raise ValueError("invalid coordinator operation")


def job(samples, cases, key, plan, value, *, control=False):
    validate_runtime(value)
    if not key or "/" in key or "\\" in key or key in (".", ".."):
        raise ValueError("unsafe batch identity")
    if len(samples) != 4 or len({s["sample_id"] for s in samples}) != 4:
        raise ValueError("one fixed four-program group required")
    if control != (value["mode"] == "canary"):
        raise ValueError("controls and research must have separate namespaces")
    if not control:
        study.validate_batch(key, samples, "evaluation", plan)
        if list(cases) != list(study.pilot.cases_for("evaluation")):
            raise ValueError("evaluation suite changed")
    for sample in samples:
        sid = sample["sample_id"]
        if not sid or "/" in sid or "\\" in sid or sid in (".", ".."):
            raise ValueError("unsafe sample identity")
    return {"key": key, "samples": samples, "case_hashes": [c.input_hash for c in cases],
            "plan_hash": fingerprint(plan), "runtime": value, "control": control}


def checked_program(sample, document, cases, image_id):
    if sample["extraction_status"].startswith("rejected_"):
        if document != {"rejected": True, "sample_hash": fingerprint(sample)}:
            raise ValueError("rejected program identity changed")
        return {c.input_hash: {"passed": False, "inclusive_match": False,
                              "reason": "extraction_rejected"} for c in cases}
    outcomes = execution.validate_program(document, sample["source"], cases, image_id)
    for case in cases:
        value = outcomes[case.input_hash]
        value["inclusive_match"] = (None if value["passed"] is None else
            type(value["actual"]) is int and value["actual"] == study.pilot.contrast.inclusive_answer(case.arguments_json))
    return outcomes


async def execute(manifest, cases, directory, backend_factory, rpc, commit, archive, *,
                  owner, publish_only=False, clock=time.time, progress=None):
    """Isolated worker; failed storage recovery may only publish saved evidence."""
    validate_runtime(manifest["runtime"])
    if manifest["case_hashes"] != [c.input_hash for c in cases]:
        raise ValueError("worker case set differs from fixed manifest")
    directory = Path(directory)
    identity = fingerprint(manifest)
    base = {"key": manifest["key"], "identity": identity, "owner": owner}
    if not publish_only:
        await rpc("claim", base)
    await storage.save(directory / "intent.json", manifest)
    lock = asyncio.Lock()
    async def committed():
        async with lock:
            await storage.retry(commit, label="parallel Volume commit")
    await committed()
    stats = {"inputs": 0, "new_programs": 0, "reused_programs": 0, "unknown": 0}
    seconds = Decimal(0)
    documents = {}

    async def one(sample):
        nonlocal seconds
        sid = sample["sample_id"]
        target = directory / "programs" / sid
        program_intent = {"job_hash": identity, "sample_hash": fingerprint(sample), "sample_id": sid}
        if sample["extraction_status"].startswith("rejected_"):
            document = {"rejected": True, "sample_hash": fingerprint(sample)}
        elif (target / "result.json").exists():
            if json.loads((target / "intent.json").read_text()) != program_intent:
                raise ValueError("saved program intent changed")
            document = json.loads((target / "result.json").read_text())
            stats["reused_programs"] += 1
        else:
            if publish_only or (target / "intent.json").exists():
                raise ReconciliationRequired("missing final evidence; candidate code must not be replayed")
            await storage.save(target / "intent.json", program_intent)
            await committed()
            async def gate():
                permit = await rpc("permit", dict(base, sample_id=sid))  # NEVER retry a start permit.
                if clock() > permit["expires"]:
                    raise ReconciliationRequired("expired permit; do not start a sandbox")
            count = 0
            async def record(input_hash, value):
                nonlocal count
                await storage.save(target / "inputs" / (input_hash + ".json"), value)
                count += 1
                stats["inputs"] += 1
                if count % storage.CHUNK_SIZE == 0:
                    await committed()
                if progress:
                    progress(dict(stats))
            stats["new_programs"] += 1
            document = await backend_factory(gate).execute_program(sample["source"], cases, on_record=record)
            # Save complete protected result before any final Dict publication.
            await storage.save(target / "result.json", document)
            await committed()
            seconds += min(Decimal(execution.lifetime_for(len(cases))),
                           Decimal(math.ceil(document["metadata"]["total_seconds"])))
        outcomes = checked_program(sample, document, cases, manifest["runtime"]["image_id"])
        await storage.save(target / "result.json", document)
        if "rejected" not in document:
            for h, record in document["records"].items():
                await storage.save(target / "inputs" / (h + ".json"), record)
        await committed()
        async with lock:
            await storage.retry(lambda: archive(sid, program_intent, document), label="parallel archive commit")
        if "rejected" not in document:
            if document["metadata"].get("failure", {}).get("stage") == "evidence_storage" or document["metadata"].get("storage_failure"):
                raise storage.StorageFailure("partial execution preserved; no automatic code retry")
        unknown = [sid + "/" + h for h, v in outcomes.items() if v["passed"] is None]
        answer = await storage.retry(lambda: rpc("program", dict(base, sample_id=sid,
            document_hash=fingerprint(document), unknown=unknown,
            sandbox_ids=[] if "rejected" in document else storage.sandbox_ids(document),
            cleanup="terminated" if "rejected" in document else document["metadata"]["cleanup"])),
            label="parallel result publication")
        stats["unknown"] += len(unknown)
        if answer["stop"]:
            raise ReconciliationRequired("coordinator stopped; retain unknown outcomes")
        documents[sid] = document
        return outcomes
    try:
        outcomes = await bounded_map(manifest["samples"], one, PROGRAMS_PER_WORKER)
        rows = [study.program_summary(s, o, "evaluation") for s, o in zip(manifest["samples"], outcomes)] if not manifest["control"] else []
        raw = {"manifest": manifest, "documents": documents}
        await storage.save(directory / "raw.json", raw)
        # Re-read every input file before issuing a completed receipt.
        await committed()
        for sid, document in documents.items():
            if json.loads((directory / "programs" / sid / "result.json").read_text()) != document:
                raise ValueError("durable program differs")
            for h, record in document.get("records", {}).items():
                if json.loads((directory / "programs" / sid / "inputs" / (h + ".json")).read_text()) != record:
                    raise ValueError("durable input differs")
        result = {"raw_hash": fingerprint(raw), "rows": rows, "job_hash": identity,
                  "outcome_hashes": [fingerprint(o) for o in outcomes], "stats": stats,
                  "sandbox_seconds": str(seconds)}
        # Recovery may change reuse counts/timing, but never the immutable evidence receipt.
        receipt = {k: result[k] for k in ("raw_hash", "rows", "job_hash", "outcome_hashes")}
        await storage.save(directory / "receipt.json", receipt)
        await committed()
        compact = {k: receipt[k] for k in ("raw_hash", "job_hash", "outcome_hashes")}
        await storage.retry(lambda: rpc("batch", dict(base, receipt=compact)), label="parallel batch publication")
        return result
    finally:
        await committed()


def remaining_jobs(saved_arms, plan, value, completed):
    """Fixed identities only: never choose programs from their observed scores."""
    expected = {study.label(s, a) for s in study.SEEDS for a in study.ARMS}
    if set(saved_arms) != expected:
        raise ValueError("all twelve completed policies required")
    jobs = []
    cases = study.pilot.cases_for("evaluation")
    for label, arm in sorted(saved_arms.items()):
        study.validate_arm(arm, plan)
        for step in (12, 24):
            samples = arm["evaluations"][str(step)]["samples"]
            for i in range(len(samples) // 4):
                key = f"eval-{label}-{step:02d}-{i:02d}"
                if key not in completed:
                    jobs.append(job(samples[4*i:4*i+4], cases, key, plan, value))
    return jobs
