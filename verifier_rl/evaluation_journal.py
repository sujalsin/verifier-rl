"""Restart-safe saved-program journals. Never execute candidate source here.

An intent without a raw result is NOT permission to try again. It requires an
explicit reconciliation amendment, while its entire spending reservation stays
held. Completed raw results can reconstruct missing derived records.
"""

import argparse
from decimal import Decimal
import json
from pathlib import Path

from . import evaluation_completion as completion, evaluation_recovery as recovery
from . import reward_shaping as shaping
from .cli import write_private
from .suites import canonical_json, digest

RESUME_ID = "resume-001"
RESUME_KEY = completion.PENDING_KEYS[-1]
RESUME_SECONDS = 600
STOPPED_APP_ID = "ap-Z6pIzFY9kCLunJqAReKxn1"
JOURNAL_FILES = ("intent", "result", "assessment", "receipt")


class ReconciliationRequired(ValueError):
    """A potentially submitted batch must not be blindly submitted again."""


def persist(directory, documents):
    """Write missing JSON only; reject changed or partially written evidence.

This is not a distributed lock. The launcher separately claims each batch using
an atomic provider-side insert before execution. Commit before spending.
"""
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Check all existing records before filling any gaps.
    for name, value in documents.items():
        path = directory / f"{name}.json"
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("saved evidence differs: " + str(path))
    for name, value in documents.items():
        path = directory / f"{name}.json"
        if not path.exists():
            write_private(path, canonical_json(value))
    return directory


def load_checkpoint(directory):
    directory = Path(directory)
    result = {name: json.loads((directory / f"{name}.json").read_text())
              for name in ("inputs", "budget", "source_snapshot", "deadline")}
    result["items"] = {key: {name: json.loads((directory / key / f"{name}.json").read_text())
        for name in JOURNAL_FILES if (directory / key / f"{name}.json").exists()}
        for key in completion.PENDING_KEYS if (directory / key).exists()}
    return result


def intent_for(key, policy, sample, budget, receipts, attempt_index=1):
    count = 0 if sample["extraction_status"].startswith("rejected_") else 432
    return {"sample": sample, "policy": policy, "attempt_index": attempt_index, "retry_allowed": False,
            "reservation": shaping.reserve_batch(budget, receipts, key, count)}


def validate_checkpoint(checkpoint):
    inputs, budget = checkpoint["inputs"], checkpoint["budget"]
    rows = completion.validate_inputs(inputs)
    if completion.completion_budget(inputs, budget["billing_before"], budget["rates"],
                                     budget["old_sandbox_terminal"]) != budget:
        raise ValueError("saved completion budget changed")
    if set(checkpoint["items"]) - set(completion.PENDING_KEYS):
        raise ValueError("unexpected program in journal")
    reports, receipts, reservations, derived, interrupted = dict(inputs["reports"]), {}, {}, {}, []
    gap = False
    for key, policy, sample in rows:
        if key not in completion.PENDING_KEYS:
            continue
        item = checkpoint["items"].get(key, {})
        if not item:
            gap = True
            continue
        if gap:
            raise ValueError("journal has work after an unfinished sequential batch")
        expected = intent_for(key, policy, sample, budget, receipts)
        if item.get("intent") != expected:
            raise ValueError("saved intent or cumulative reservation changed: " + key)
        reservations[key] = expected["reservation"]
        if "result" not in item:
            if set(item) != {"intent"}:
                raise ValueError("derived records without a raw result")
            interrupted.append(key)
            gap = True
            continue
        report = item["result"]
        assessment = completion.assess_report(sample, report, inputs["previous"]["setup"]["sandbox_image_id"])
        receipt = shaping.execution_receipt(key, report)
        derived[key] = {"assessment": assessment, "receipt": receipt}
        for name, value in derived[key].items():
            if name in item and item[name] != value:
                raise ValueError("derived journal record changed: " + key + "/" + name)
        reports[key], receipts[key] = report, receipt
    comparison = completion.summary(inputs, reports)
    return {"rows": rows, "reports": reports, "receipts": receipts, "reservations": reservations,
            "derived": derived, "interrupted": interrupted, "summary": comparison}


def require_reconciliation(checkpoint, evidence):
    state = validate_checkpoint(checkpoint)
    app = evidence["stopped_controller"]
    if (state["interrupted"] != [RESUME_KEY] or len(state["reports"]) != 23
            or app.get("app_id") != STOPPED_APP_ID or app.get("state") != "stopped"
            or str(app.get("tasks")) != "0"
            or evidence.get("active_sandbox_ids") != []
            or not str(evidence.get("sandbox_app_id", "")).startswith("ap-")
            or evidence.get("sandbox_app_name") != checkpoint["inputs"]["previous"]["setup"]["app_name"]
            or evidence.get("cause") != "controller_preemption"
            or evidence.get("replacement_authorized") is not True
            or evidence.get("interrupted_intent_hash") != digest(canonical_json(checkpoint["items"][RESUME_KEY]["intent"]))):
        raise ValueError("one reconciled controller interruption with 23 preserved reports required")
    return state


def resume_budget(checkpoint, evidence, billing, rates):
    require_reconciliation(checkpoint, evidence)
    return _resume_cost_envelope(checkpoint, billing, rates)


def _resume_cost_envelope(checkpoint, billing, rates):
    def amount(value):
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError("invalid cost")
        return result
    held = amount(checkpoint["items"][RESUME_KEY]["intent"]["reservation"]["total_reserved_usd"])
    # Two bounded controller starts are reserved, including the documented 3x
    # non-preemptible CPU/memory rate. The launcher caps starts atomically.
    controller = 2 * 3 * Decimal(RESUME_SECONDS + 60) * (
        amount(rates["cpu_hour_cost"]) + 2 * amount(rates["mem_gib_hour_cost"])) / 3600
    budget = {"billing_before": billing, "rates": rates, "total_trial_limit_usd": "20",
        "old_reservation_held_usd": str(held), "controller_resource_reserve_usd": str(controller),
        "controller_start_limit": 2, "nonpreemptible_rate_multiplier": 3,
        "fixed_reserved_usd": str(max(held, amount(billing["metered_cost"]) + 1) + controller),
        "sandbox_hour_usd": str(amount(rates["cpu_hour_cost_sandbox"]) + amount(rates["mem_gib_hour_cost_sandbox"]) / 4),
        "max_sandbox_executions": 432, "provider_hard_spending_cap": False,
        "policy": "Hold interrupted batch in full; one CPU-only replacement, no candidate-outcome retries."}
    shaping.reserve_batch(budget, {}, RESUME_KEY + "-" + RESUME_ID, 432)
    return budget


def validated_resume(checkpoint, evidence, budget):
    state = require_reconciliation(checkpoint, evidence)
    if budget != _resume_cost_envelope(checkpoint, budget["billing_before"], budget["rates"]):
        raise ValueError("resume budget changed")
    return state


def verify_resume(checkpoint, evidence, budget, reports, receipt, reservation):
    state = validated_resume(checkpoint, evidence, budget)
    if set(reports) != set(state["reports"]) | {RESUME_KEY}:
        raise ValueError("all 23 old reports and only one replacement required")
    if any(reports[key] != value for key, value in state["reports"].items()):
        raise ValueError("completed raw evidence replaced")
    key = RESUME_KEY + "-" + RESUME_ID
    expected = shaping.execution_receipt(key, reports[RESUME_KEY])
    if receipt != expected or reservation != shaping.reserve_batch(budget, {}, key, expected["executions"]):
        raise ValueError("replacement spending records changed")
    result = completion.summary(checkpoint["inputs"], reports)
    result.pop("retries")
    result["resumption"] = {"source_run": completion.RUN_ID, "resume_id": RESUME_ID,
        "reused_program_reports": 23, "controller_interruption_replacements": 1,
        "candidate_outcome_retries": 0, "interrupted_unrecorded_execution_bounds": [0, 432],
        "old_reservation_held_usd": budget["old_reservation_held_usd"],
        "replacement_reserved_total_usd": reservation["total_reserved_usd"]}
    result["limitations"].append("Controller-preempted batch replaced once; its per-input evidence is unavailable, "
                                "its full cost reservation is held, and 0..432 extra executions are unrecorded.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline resume verification; no cloud or candidate execution")
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    docs = {n: json.loads((args.run / f"{n}.json").read_text()) for n in
            ("checkpoint", "reconciliation", "budget", "reports", "receipt", "reservation", "summary")}
    result = verify_resume(*(docs[n] for n in
        ("checkpoint", "reconciliation", "budget", "reports", "receipt", "reservation")))
    if result != docs["summary"]:
        raise ValueError("summary differs from recomputation")
    print(canonical_json({"verified": True, "cloud_calls": 0, "candidate_source_executed": False,
                          "policies": result["policies"], "resumption": result["resumption"]}))


if __name__ == "__main__":
    main()
