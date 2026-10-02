"""Clock-independent start leases and evidence-only reconciliation.

The worker bounds a permit's entire RPC round trip using its own monotonic
clock. Polling never consumes a start permit. A returned but expired permit
may be retired only by its original, still pre-execution caller. Lost replies
and worker restarts remain fail-closed. Candidate execution is never retried.
"""
import asyncio
from copy import deepcopy
import math
import time
import uuid

from . import parallel_evaluation as parallel, parallel_handoff as handoff
from . import booking_replication as study, program_storage as storage
from .evaluation_journal import ReconciliationRequired
from .suites import digest

REVISION = "monotonic-permit-001"
MAX_WAIT = 240.0  # Also covers 16 conservative seven-second canary admissions.
MAX_EXPIRED_GRANTS = 3


def permit_transition(state, payload, *, now, wall, boot):
    """Serialized, nonblocking admission; monotonic epochs never cross boots."""
    result = deepcopy(state)
    parallel.validate_runtime(result["runtime"])
    key, sid = payload["key"], payload["sample_id"]
    if (key not in result["allowed"] or payload["identity"] != result["allowed"][key]["identity"]
            or result["owners"].get(key) != payload["owner"]
            or sid not in result["allowed"][key]["samples"]):
        raise ValueError("invalid permit ownership or manifest")
    if result["allowed"][key]["samples"][sid]["rejected"]:
        raise ValueError("rejected source cannot receive a permit")
    if result["stop"] or wall >= result["runtime"]["deadline"]:
        raise ReconciliationRequired("no new starts after stop/deadline")
    nonce = payload["nonce"]
    if not isinstance(nonce, str) or not nonce:
        raise ValueError("unique permit request required")
    token = key + "/" + sid
    if token in result["programs"]:
        raise ReconciliationRequired("program already published")
    previous = result["permits"].get(token)
    if previous is not None:
        if previous["nonce"] == nonce:
            return result, {"grant": previous}
        if payload.get("unused_previous") != previous["nonce"]:
            raise ReconciliationRequired("permit already owned; no ambiguous replay")
    elif payload.get("unused_previous") is not None:
        raise ValueError("unknown retired permit")
    timing = result.get("permit_timing")
    if timing is None or timing["boot"] != boot:
        # Old grants may still be in flight. Let all five-second leases expire
        # before a new coordinator process can grant any starts.
        timing = {"boot": boot, "next_at": now + parallel.PERMIT_TTL + result["runtime"]["creation_interval"]}
        result["permit_timing"] = timing
    if now < timing["next_at"]:
        return result, {"wait_seconds": min(timing["next_at"] - now, 10.0)}
    history = result.setdefault("retired_permits", {}).setdefault(token, [])
    if previous is not None:
        if len(history) >= MAX_EXPIRED_GRANTS:
            raise ReconciliationRequired("expired permit renewal bound reached")
        history.append(previous)
    grant = {"nonce": nonce, "identity": payload["identity"], "ttl": parallel.PERMIT_TTL,
             "boot": boot, "issued_monotonic": now, "issued_wall_for_audit": wall}
    result["permits"][token] = grant
    timing["next_at"] = now + result["runtime"]["creation_interval"]
    return result, {"grant": grant}


class PermitRPC:
    """Adapt the frozen worker to local monotonic expiration timestamps."""
    def __init__(self, rpc, *, clock=time.monotonic, sleep=asyncio.sleep, nonce=lambda: uuid.uuid4().hex):
        self.rpc, self.clock, self.sleep, self.nonce = rpc, clock, sleep, nonce
        self.used = set()
        self.expired_grants = 0

    async def __call__(self, action, payload):
        if action != "permit":
            return await self.rpc(action, payload)
        token = payload["key"] + "/" + payload["sample_id"]
        if token in self.used:
            raise ReconciliationRequired("start gate may only be entered once")
        self.used.add(token)
        end, request, previous, expired = self.clock() + MAX_WAIT, self.nonce(), None, 0
        while self.clock() < end:
            began = self.clock()
            # No retry after a transport exception: receipt ownership is then
            # uncertain. Only explicit no-grant/unused-grant replies can poll.
            answer = await self.rpc("try_permit", dict(payload, nonce=request, unused_previous=previous))
            if "wait_seconds" in answer:
                delay = answer["wait_seconds"]
                if not math.isfinite(delay) or not 0 <= delay <= 10:
                    raise ValueError("invalid permit polling delay")
                await self.sleep(min(delay + .05, max(0, end - self.clock())))
                continue
            grant = answer["grant"]
            if (grant["nonce"] != request or grant["identity"] != payload["identity"]
                    or grant["ttl"] != parallel.PERMIT_TTL):
                raise ValueError("permit response identity or lease changed")
            expires = began + grant["ttl"]
            if self.clock() <= expires:
                return {"expires": expires, "identity": grant["identity"], "nonce": request}
            # Still inside the gate: Sandbox.create and candidate submission
            # have not been called. Keep the old grant as retirement evidence.
            expired += 1
            self.expired_grants += 1
            if expired >= MAX_EXPIRED_GRANTS:
                raise ReconciliationRequired("bounded permit delay exceeded before sandbox creation")
            previous, request = request, self.nonce()
        raise ReconciliationRequired("bounded permit admission wait exceeded")


def require_never_started(sample, document, cases, image_id):
    """Only the reviewed expired-gate path proves Sandbox.create was not called."""
    parallel.checked_program(sample, document, cases, image_id)
    m = document["metadata"]
    if (m.get("candidate_submission_attempted") is not False or "sandbox_id" in m
            or m.get("cleanup") != "no_handle; creation_may_be_unconfirmed"
            or m.get("failure") != {"stage": "create", "type": "ReconciliationRequired",
                                    "detail": "expired permit; do not start a sandbox"}
            or any(k in m for k in ("preflight_returncode", "runtime_python", "startup_seconds"))):
        raise ReconciliationRequired("program may have started; never replay ambiguous execution")
    for case in cases:
        record = document["records"][case.input_hash]
        if (record["status"] != "infrastructure_error" or record["detail"] != "program_not_executed"
                or record["stdout_base64"] or record["retryable"]
                or record["metadata"] != {"input_hash": case.input_hash, "source_hash": digest(sample["source"])}):
            raise ReconciliationRequired("submitted or partial input cannot be replayed")


def classify_interrupted(manifest, documents, published):
    cases = study.pilot.cases_for("evaluation")
    samples = manifest["samples"]
    if set(documents) != {s["sample_id"] for s in samples} or set(published) != set(documents):
        raise ValueError("complete interrupted program evidence required")
    outcomes, unknown = [], []
    for sample in samples:
        sid = sample["sample_id"]
        doc, receipt = documents[sid], published[sid]
        values = parallel.checked_program(sample, doc, cases, manifest["runtime"]["image_id"])
        missing = [sid + "/" + h for h, v in values.items() if v["passed"] is None]
        expected = {"document_hash": parallel.fingerprint(doc), "unknown": missing,
                    "cleanup": "terminated" if "rejected" in doc else doc["metadata"]["cleanup"],
                    "sandbox_ids": [] if "rejected" in doc else storage.sandbox_ids(doc)}
        if receipt != expected:
            raise ValueError("interrupted evidence differs from published program receipt")
        outcomes.append(values)
        unknown.extend(missing)
    if unknown:
        for sample in samples:
            require_never_started(sample, documents[sample["sample_id"]], cases, manifest["runtime"]["image_id"])
        return {"status": "proven_never_started", "unknown": unknown,
                "document_hashes": {sid: parallel.fingerprint(doc) for sid, doc in documents.items()}}
    raw = {"manifest": manifest, "documents": documents}
    receipt = {"raw_hash": parallel.fingerprint(raw), "job_hash": parallel.fingerprint(manifest),
               "rows": [study.program_summary(s, v, "evaluation") for s, v in zip(samples, outcomes)],
               "outcome_hashes": [parallel.fingerprint(v) for v in outcomes]}
    return {"status": "completed", "raw": raw, "receipt": receipt, **handoff.check_new(manifest, raw, receipt)}
