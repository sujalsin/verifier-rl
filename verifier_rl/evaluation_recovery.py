"""CPU-only continuation of a stopped shaping audit; never execute source locally."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

from .diagnostics import diagnose_sample
from .grpo_pilot import training_evidence
from .measurement_v3 import unique_outcomes
from .reward_shaping import (FORMULAS, SEEDS, arm_plan, before_signature, budget_envelope,
    comparison_summary, execution_receipt, require_report, reserve_batch,
    rollout_rewards, selected_suites)
from .suites import canonical_json, digest

SOURCE_RUN = "qwen-shaping-20260928T051345-8cc25b49"
VERSION = "saved-shaping-evaluation-recovery-0.1"
CONTROLLER_SECONDS = 4500
QUARANTINED_SANDBOX = "sb-buXskhjQWmoQwQUOmMw3Ho"
TOTAL_LIMIT_USD = "20"
MAX_EXTRA_ATTEMPTS = 2


def entries(generations):
    rows = [("baseline", s) for s in generations["partial"]["samples"] if s["arm"] == "before"]
    rows += [(f, s) for f in FORMULAS for s in generations[f]["samples"] if s["arm"] == "after"]
    return [(f"{policy}-{sample['seed']}", policy, sample) for policy, sample in rows]


def validate_inputs(inputs):
    plan, setup, generations = (inputs[k] for k in ("plan", "setup", "generations"))
    if set(generations) != set(FORMULAS):
        raise ValueError("both saved training arms required")
    for formula in FORMULAS:
        generation = generations[formula]
        if (generation["run_id"] != SOURCE_RUN + "-" + formula.replace("_", "-")
                or generation["plan"] != arm_plan(plan, formula, SOURCE_RUN)):
            raise ValueError("training configuration changed")
        evidence = training_evidence(generation["metrics"])
        if evidence != generation["training_evidence"]:
            raise ValueError("saved training evidence changed")
        if [(s["arm"], s["seed"]) for s in generation["samples"]] != [
                (arm, seed) for arm in ("before", "after") for seed in SEEDS]:
            raise ValueError("saved evaluation population changed")
        if len(generation["rollout_records"]) != generation["metrics"]["global_step"]:
            raise ValueError("training groups missing")
        for i, group in enumerate(generation["rollout_records"]):
            rewards = rollout_rewards(group["samples"], group["reports"], setup["sandbox_image_id"], generation["plan"])
            if rewards != group["rewards"] or rewards != generation["metrics"]["rewards"][i]:
                raise ValueError("saved training rewards changed")
    if before_signature(generations["partial"]["samples"]) != before_signature(generations["completion_bonus"]["samples"]):
        raise ValueError("baseline generations differ")
    if [s["raw"] for s in generations["partial"]["rollout_records"][0]["samples"]] != [
            s["raw"] for s in generations["completion_bonus"]["rollout_records"][0]["samples"]]:
        raise ValueError("initial training groups differ")
    rows = entries(generations)
    cached = inputs["cached"]
    if set(cached) != {f"baseline-{s}" for s in range(10000, 10005)}:
        raise ValueError("exactly the five completed baseline records must be reused")
    for key, _, sample in rows:
        if key in cached:
            require_report(sample, cached[key], True, setup["sandbox_image_id"])
    failed = inputs["failed_report"]
    failed_sample = next(s for key, _, s in rows if key == "baseline-10005")
    if inputs["failed_intent"]["sample"] != failed_sample:
        raise ValueError("failed sample identity changed")
    diagnose_sample(failed_sample, failed, selected_suites(True))
    faults = [o for o in unique_outcomes(failed).values() if o["reason"] == "infrastructure_error"]
    if (len(faults) != 1 or failed["execution_config"]["attempts"] != 432
            or faults[0]["attempts"][-1]["metadata"].get("sandbox_id") != QUARANTINED_SANDBOX
            or faults[0]["attempts"][-1]["detail"] != "cleanup_unconfirmed"
            or faults[0]["attempts"][-1]["metadata"].get("preflight_stage") != "command_start"
            or "returncode" in faults[0]["attempts"][-1]["metadata"]):
        raise ValueError("failure is outside the reviewed recovery scope")
    old_budget = inputs["old_budget"]
    if budget_envelope(plan, old_budget["billing_before"], old_budget["rates"]) != old_budget:
        raise ValueError("original budget changed")
    spent = {}
    for formula in FORMULAS:
        for group in generations[formula]["rollout_records"]:
            for sample, report in zip(group["samples"], group["reports"]):
                key = f"{SOURCE_RUN}-{formula.replace('_', '-')}-training-{sample['seed']}"
                spent[key] = execution_receipt(key, report)
    for seed in range(10000, 10005):
        key = f"{SOURCE_RUN}-baseline-before-{seed}"
        spent[key] = execution_receipt(key, cached[f"baseline-{seed}"])
    expected = reserve_batch(old_budget, spent, f"{SOURCE_RUN}-baseline-before-10005", 432)
    if inputs["failed_intent"]["spending_reservation"] != expected:
        raise ValueError("original outstanding reservation does not recompute")
    return rows


def recovery_budget(inputs, billing, rates):
    """Keep the full old reservation; never silently refund the unresolved batch."""
    def amount(value):
        value = Decimal(str(value))
        if not value.is_finite() or value < 0:
            raise ValueError("invalid cost")
        return value
    old = amount(inputs["failed_intent"]["spending_reservation"]["total_reserved_usd"])
    # The old envelope includes GPU/worker ceilings, prior usage, the entire
    # failed batch and $1 contingency. Also hold $0.50 for the quarantined
    # sandbox during this bounded recovery; this is not proof of termination.
    prior = max(old, amount(billing["metered_cost"]) + 1)
    controller = Decimal(CONTROLLER_SECONDS + 60) * (
        amount(rates["cpu_hour_cost"]) + 2 * amount(rates["mem_gib_hour_cost"])) / 3600
    budget = {"billing_before": billing, "rates": rates,
        "total_trial_limit_usd": TOTAL_LIMIT_USD, "old_reservation_held_usd": str(old),
        "quarantined_sandbox_reserve_usd": "0.50", "controller_resource_reserve_usd": str(controller),
        "fixed_reserved_usd": str(prior + controller + Decimal("0.50")),
        "sandbox_hour_usd": str(amount(rates["cpu_hour_cost_sandbox"]) + amount(rates["mem_gib_hour_cost_sandbox"]) / 4),
        "max_sandbox_executions": (19 + MAX_EXTRA_ATTEMPTS) * 432, "provider_hard_spending_cap": False,
        "policy": "Hold prior reservations and each failed batch; at most one infra replacement per program, two total. No GPU calls."}
    reserve_batch(budget, {}, "preflight-budget-check", 432)
    return budget


def check_frozen_sources(snapshot, root):
    checked = []
    for key, source in snapshot.items():
        if key.startswith("verifier_rl/") and key.endswith(".py"):
            if (root / key).read_text() != source:
                raise ValueError("frozen grader/training implementation changed: " + key)
            checked.append(key)
    if "verifier_rl/reward_shaping.py" not in checked:
        raise ValueError("missing frozen grader snapshot")
    return checked


def retry_eligible(sample, report, image_id, reconciled=None):
    """Only infrastructure failures; never retry an incorrect answer or signal."""
    from dataclasses import asdict
    from .modal_backend import Limits, RUNNER
    diagnose_sample(sample, report, selected_suites(True))
    found = False
    for outcome in unique_outcomes(report).values():
        if not outcome["attempts"]:
            if outcome["reason"] != "not_executed":
                return False
            continue
        if len(outcome["attempts"]) != 1:
            return False
        attempt = outcome["attempts"][0]
        if attempt["status"] == "extraction_rejected":
            continue
        m = attempt["metadata"]
        if (m.get("image_id") != image_id or m.get("runner_hash") != digest(RUNNER)
                or m.get("limits") != asdict(Limits()) or m.get("block_network") is not True
                or m.get("reset") != "fresh_sandbox_per_input" or m.get("sdk_version") != "1.5.5"
                or m.get("creation_interval_seconds") != .26):
            return False
        code = m.get("returncode")
        if attempt["status"] == "timeout" or (type(code) is int and (code < 0 or code >= 128)):
            return False
        if attempt["status"] == "infrastructure_error":
            found = True
            sid = m.get("sandbox_id")
            if not sid or (m.get("cleanup") != "terminated"
                           and type((reconciled or {}).get(sid)) is not int):
                return False
        elif m.get("cleanup") != "terminated" or m.get("preflight_returncode") != 0:
            return False
    return found


def failed_receipt(key, report):
    """Keep the full reserved batch, even if it stopped before all submissions."""
    return {"candidate_id": key, "executions": 432, "accounted_sandbox_seconds": 432 * 120,
            "report_hash": digest(canonical_json(report)), "held_failed_batch": True,
            "executions_are_reserved_upper_bound": True, "is_provider_invoice": False}


def verify_recovery(inputs, reports, budget, receipts, reservations, attempts, reconciliations):
    rows = validate_inputs(inputs)
    expected_keys = {key for key, _, _ in rows}
    if set(reports) != expected_keys:
        raise ValueError("all 24 evaluation records required")
    grouped = {p: [] for p in ("baseline", *FORMULAS)}
    accounted, failed_keys = {}, []
    for key, policy, sample in rows:
        report = reports[key]
        require_report(sample, report, True, inputs["setup"]["sandbox_image_id"])
        grouped[policy].append(report)
        if key in inputs["cached"]:
            if report != inputs["cached"][key]:
                raise ValueError("a completed result was replaced")
            continue
        for attempt_index in (1, 2):
            attempt_key = f"{key}-attempt-{attempt_index}"
            recorded = attempts.get(attempt_key)
            if recorded is None:
                raise ValueError("missing recovery attempt")
            if recorded == report:
                receipt = execution_receipt(attempt_key, report)
            else:
                if attempt_index != 1 or not retry_eligible(sample, recorded,
                        inputs["setup"]["sandbox_image_id"], reconciliations.get(attempt_key)):
                    raise ValueError("unauthorized replacement or changed successful report")
                receipt = failed_receipt(attempt_key, recorded)
                failed_keys.append(attempt_key)
            if receipts.get(attempt_key) != receipt:
                raise ValueError("missing or changed recovery receipt")
            if reservations.get(attempt_key) != reserve_batch(budget, accounted, attempt_key, receipt["executions"]):
                raise ValueError("recovery reservation does not recompute")
            accounted[attempt_key] = receipt
            if recorded == report:
                break
    if set(receipts) != set(accounted) or set(reservations) != set(accounted):
        raise ValueError("unexpected spending record")
    if set(attempts) != set(accounted) or len(failed_keys) > MAX_EXTRA_ATTEMPTS or set(reconciliations) != set(failed_keys):
        raise ValueError("attempt/reconciliation budget differs from records")
    summary = comparison_summary(SOURCE_RUN, inputs["plan"], inputs["generations"], grouped,
                                 inputs["setup"]["sandbox_image_id"])
    accepted_reports = list(reports.values()) + [r for g in inputs["generations"].values()
        for group in g["rollout_records"] for r in group["reports"]]
    used_ids = {a["metadata"].get("sandbox_id") for report in accepted_reports
                for o in unique_outcomes(report).values() for a in o["attempts"]} - {None}
    for failed in [inputs["failed_report"], *[attempts[key] for key in failed_keys]]:
        failed_ids = {a["metadata"].get("sandbox_id") for o in unique_outcomes(failed).values()
                      for a in o["attempts"]} - {None}
        if failed_ids & used_ids:
            raise ValueError("a failed-run sandbox was reused")
        used_ids.update(failed_ids)
    summary["recovery"] = {"version": VERSION, "cached_program_records": 5,
        "new_program_evaluations": 19, "new_model_samples": 0, "new_training_updates": 0,
        "original_failed_report_hash": digest(canonical_json(inputs["failed_report"])),
        "original_failure_retained_unscored": True, "automatic_retries": len(failed_keys),
        "comparison_excludes_original_failed_batch": True,
        "all_recorded_sandbox_executions_including_failed_batch":
            summary["recorded_sandbox_executions"] + inputs["failed_report"]["execution_config"]["attempts"]
            + sum(attempts[key]["execution_config"]["attempts"] for key in failed_keys)}
    summary["limitations"] = [*summary["limitations"],
        "Evaluation resumed after infrastructure failure; one failed program batch was rerun in fresh sandboxes.",
        "One old sandbox shutdown was unconfirmed at recovery planning; the failed record and cost reserve are retained."]
    return summary


def main():
    from .cli import create_run_directory, write_private
    parser = argparse.ArgumentParser(description="Verify saved recovery without cloud or candidate execution")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    docs = {n: json.loads((args.run / f"{n}.json").read_text()) for n in (
        "inputs", "reports", "budget", "receipts", "reservations", "attempts", "reconciliations", "summary")}
    if recovery_budget(docs["inputs"], docs["budget"]["billing_before"], docs["budget"]["rates"]) != docs["budget"]:
        raise ValueError("budget changed")
    summary = verify_recovery(docs["inputs"], docs["reports"], docs["budget"], docs["receipts"], docs["reservations"],
                              docs["attempts"], docs["reconciliations"])
    if summary != docs["summary"]:
        raise ValueError("summary does not recompute")
    out = create_run_directory(args.out)
    write_private(out / "verification.json", canonical_json({"verified": True, "cloud_calls": 0,
        "candidate_source_executed": False, "policies": summary["policies"],
        "document_hashes": {k: digest(canonical_json(v)) for k, v in docs.items()}}))
    print(canonical_json(summary["policies"]))


if __name__ == "__main__":
    main()
