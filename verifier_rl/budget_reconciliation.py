"""Transparent accounting reports and evidence-bound, offline reconciliation.

The legacy ledger's ``actual`` field is a resource-duration estimate, not an
invoice. Never replace the ledger with a dashboard total, release live holds,
or write concurrently with its serialized cloud owner.
"""
from decimal import Decimal

from . import compute_budget as budget
from .parallel_evaluation import fingerprint


def describe(ledger):
    settled = sum((budget.number(v["actual"]) for v in ledger["items"].values()
                   if v["actual"] is not None), Decimal(0))
    holds = {k: v for k, v in ledger["items"].items() if v["actual"] is None}
    reserved = sum((budget.number(v["maximum"]) for v in holds.values()), Decimal(0))
    return {"estimated_settled_compute_usd": str(settled),
            "outstanding_maximum_reservations_usd": str(reserved),
            "overhead_allowance_usd": ledger["overhead"],
            "guard_balance_usd": str(budget.committed(ledger)),
            "compute_ceiling_usd": ledger["ceiling"],
            "headroom_usd": str(budget.number(ledger["ceiling"]) - budget.committed(ledger)),
            "unsettled_reservations": len(holds),
            "is_provider_billing": False}


def completed_call_bound(item, evidence, rates, kind):
    """Bound one stopped, zero-retry CPU/GPU call using provider timestamps.

    Charge the entire interval from submission to completion (including queue
    waits), plus every reported startup and two minutes per input. This is a
    conservative estimate, not metered billing. Sandbox-creating calls are not
    eligible here; their separate holds must retain all execution liabilities.
    """
    if kind not in ("cpu", "gpu") or evidence["sandbox_starts"] != 0:
        raise ValueError("only separately reserved CPU/GPU worker calls qualify")
    if evidence["identity"] != item["identity"] or evidence["retries"] != 0:
        raise ValueError("reservation identity or retry policy changed")
    life, info = evidence["lifecycle"], evidence["call_info"]
    if (life["state"] != "APP_STATE_STOPPED" or life["app_id"] != evidence["app_id"]
            or budget.number(life["stopped_at"]) <= 0
            or info["function_call_id"] != evidence["call_id"]
            or info.get("pending_inputs", {}).get("total", 0) != 0):
        raise ValueError("terminal provider evidence required")
    categories = [info.get(k, {}) for k in
                  ("succeeded_inputs", "failed_inputs", "timeout_inputs", "cancelled_inputs")]
    if info.get("total_inputs") != 1:
        raise ValueError("single-input invocation required")
    if sum(c.get("total", 0) for c in categories) != 1:
        raise ValueError("complete terminal input inventory required")
    records = []
    for category in categories:
        if len(category.get("latest", [])) != category.get("total", 0):
            raise ValueError("partial provider input inventory")
        records.extend(category.get("latest", []))
    created = budget.number(info["created_at"])
    record = records[0]
    start, finish = budget.number(record["started_at"]), budget.number(record["finished_at"])
    if not 0 < created <= start <= finish <= budget.number(life["stopped_at"]):
        raise ValueError("invalid provider timing")
    seconds = finish - created + budget.number(record["task_startup_time"]) + 120
    bound = budget.cost(rates, kind, seconds)
    # A longer provider interval is not evidence supporting a release.
    return {"estimated_compute_usd": str(min(bound, budget.number(item["maximum"]))),
            "charged_seconds": str(seconds), "kind": kind,
            "evidence_hash": fingerprint(evidence), "identity": item["identity"]}


def reconcile_offline(saved, corrections, evidence, *, writers_stopped):
    """Pure transition; the caller must fence *all* writers before publishing.

    The full state hash is for stale-plan detection, not a distributed lock.
    """
    if not writers_stopped:
        raise ValueError("live accounting writer: refuse external ledger edit")
    if corrections["before_hash"] != fingerprint(saved):
        raise ValueError("ledger changed; rebuild the reconciliation proposal")
    if saved["ledger"]["ceiling"] != "250":
        raise ValueError("original $250 ceiling required")
    ledger = saved["ledger"]
    for key, receipt in corrections["items"].items():
        if key not in ledger["items"] or ledger["items"][key]["actual"] is not None:
            raise ValueError("only existing unresolved reservations can be reconciled")
        proof = evidence[key]
        verified = completed_call_bound(ledger["items"][key], proof, corrections["rates"], receipt["kind"])
        if verified != receipt:
            raise ValueError("receipt does not match provider evidence")
        ledger = budget.settle(ledger, key, receipt["estimated_compute_usd"], receipt["identity"])
    return {**saved, "ledger": ledger}
