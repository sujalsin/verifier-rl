"""Explicit post-stop audit amendment; raw reports and training stay unchanged.

Unattributed candidate terminations have [incorrect, correct] score bounds.
They are never retried, silently accepted, or dropped from the denominator.
"""

import argparse
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

from . import evaluation_recovery as recovery, reward_shaping as shaping
from .diagnostics import diagnose_sample
from .measurement_v3 import unique_outcomes
from .modal_backend import Limits, RUNNER
from .reward_v2 import FAMILIES, WEIGHTS
from .reward_v3 import COUNTS
from .suites import canonical_json, digest

VERSION = "saved-shaping-audit-bounds-0.1"
SOURCE_RECOVERY = "qwen-eval-recovery-20260928T062109-d1a04fd6"
RUN_ID = "qwen-eval-completion-20260928-v1"
AMBIGUOUS_KEY = "completion_bonus-10001"
CONTROLLER_SECONDS = 2100
PENDING_KEYS = tuple(f"completion_bonus-{s}" for s in range(10002, 10008))


def assess_report(sample, report, image_id):
    """Validate original evidence, then derive a separate uncertainty overlay."""
    suites = shaping.selected_suites(True)
    if [s["suite_hash"] for s in report["suites"]] != [s.fingerprint for s in suites]:
        raise ValueError("incorrect evaluation suites")
    diagnostic = diagnose_sample(sample, report, suites)
    config = report["execution_config"]
    if config["max_retries"] != 0 or config.get("not_executed_inputs", 0):
        raise ValueError("retries or unfinished input coverage are outside this amendment")
    if any(s["all_passed"] is None for s in report["suites"]):
        raise ValueError("infrastructure failures remain unscored; preserve and stop")
    if set(diagnostic["stdout_evidence"]) - {"not_completed", "verified_complete_stdout"}:
        raise ValueError("incomplete output evidence")
    unknown, ids = set(), set()
    for input_hash, outcome in unique_outcomes(report).items():
        if len(outcome["attempts"]) != 1:
            raise ValueError("exactly one attempt required; no termination retries")
        attempt = outcome["attempts"][0]
        if attempt["status"] == "extraction_rejected":
            continue
        m = attempt["metadata"]
        if (m.get("image_id") != image_id or m.get("runner_hash") != digest(RUNNER)
                or m.get("limits") != asdict(Limits()) or m.get("cleanup") != "terminated"
                or m.get("block_network") is not True or m.get("reset") != "fresh_sandbox_per_input"
                or m.get("sdk_version") != "1.5.5" or m.get("creation_interval_seconds") != .26
                or m.get("preflight_returncode") != 0 or not m.get("sandbox_id")):
            raise ValueError("execution environment or confirmed cleanup differs")
        if m["sandbox_id"] in ids:
            raise ValueError("sandbox reused within program")
        ids.add(m["sandbox_id"])
        code = m.get("returncode")
        signal = type(code) is int and (code < 0 or code >= 128)
        if attempt["status"] == "timeout" or signal:
            if attempt["status"] not in ("candidate_error", "timeout") or outcome["passed"] is not False:
                raise ValueError("termination metadata contradicts recorded outcome")
            unknown.add(input_hash)
    if len(ids) != config["attempts"]:
        raise ValueError("sandbox execution count mismatch")
    if not unknown:
        shaping.require_report(sample, report, True, image_id)
    else:
        # The old validator must still reject this raw report; no retroactive fix.
        try:
            shaping.require_report(sample, report, True, image_id)
        except ValueError as exc:
            if str(exc) != "unattributed termination: stop, preserve failure, do not retry":
                raise
        else:
            raise ValueError("amendment unexpectedly differs from original termination gate")
    bounds = {}
    for suite, result in zip(suites, report["suites"]):
        outcomes = result["outcomes"]
        passed = sum(o["passed"] is True for o in outcomes)
        missing = sum(o["input_hash"] in unknown for o in outcomes)
        known_failure = any(o["passed"] is False and o["input_hash"] not in unknown for o in outcomes)
        bounds[suite.name] = {"passed_cases": [passed, passed + missing], "total": len(outcomes),
            "full_pass": [int(passed == len(outcomes)), int(not known_failure)],
            "unattributed_cases": [o["case"] for o in outcomes if o["input_hash"] in unknown]}
    v3 = report["suites"][0]
    partial = []
    for upper in (False, True):
        partial.append(float(sum(WEIGHTS[f] * sum(
            o["passed"] is True or (upper and o["input_hash"] in unknown)
            for c, o in zip(suites[0].cases, v3["outcomes"]) if c.tags == (f,)) / COUNTS[f]
            for f in FAMILIES)))
    full = bounds["balanced_v3"]["full_pass"]
    return {"status": "bounded_unattributed_termination" if unknown else "validated",
        "report_hash": digest(canonical_json(report)), "source_hash": report["candidate_hash"],
        "raw_report_unchanged": True, "original_protocol_accepted": not unknown,
        "partial_reward": partial, "bonus_reward": [(p + b) / 2 for p, b in zip(partial, full)],
        "suites": bounds, "unattributed_inputs": len(unknown)}


def load_inputs(root):
    def read(name):
        return json.loads((root / f"{name}.json").read_text())
    previous = read("inputs")
    reports = dict(previous["cached"])
    for key, _, _ in recovery.entries(previous["generations"]):
        if key in reports or key in PENDING_KEYS:
            continue
        name = "failed-completion-bonus-10001-result" if key == AMBIGUOUS_KEY else key + "-result"
        reports[key] = read(name)
    return {"source_recovery_id": SOURCE_RECOVERY, "previous": previous, "reports": reports,
        "previous_budget": read("budget"), "previous_stop": read("remote-stopped"),
        "failed_intent": read("failed-completion-bonus-10001-intent")}


def validate_inputs(inputs):
    if inputs["source_recovery_id"] != SOURCE_RECOVERY:
        raise ValueError("wrong source recovery")
    previous = inputs["previous"]
    rows = recovery.validate_inputs(previous)
    expected = {key for key, _, _ in rows} - set(PENDING_KEYS)
    if set(inputs["reports"]) != expected:
        raise ValueError("preserve exactly 18 saved reports; only six may be newly evaluated")
    old = inputs["previous_budget"]
    if recovery.recovery_budget(previous, old["billing_before"], old["rates"]) != old:
        raise ValueError("previous budget changed")
    stop = inputs["previous_stop"]
    if (stop["stage"] != AMBIGUOUS_KEY or stop["complete_evaluation_records"] != 17
            or stop["automatic_retry"] is not False
            or stop["detail"] != "unattributed termination: stop, preserve failure, do not retry"):
        raise ValueError("source stop differs from the reviewed amendment")
    receipts = {}
    for key, policy, sample in rows:
        if key in PENDING_KEYS:
            continue
        report = inputs["reports"][key]
        assessment = assess_report(sample, report, previous["setup"]["sandbox_image_id"])
        if (assessment["status"] == "validated") != (key != AMBIGUOUS_KEY):
            raise ValueError("unexpected original ambiguity")
        if key in previous["cached"]:
            if report != previous["cached"][key]:
                raise ValueError("cached report changed")
            continue
        attempt_key = key + "-attempt-1"
        if key == AMBIGUOUS_KEY:
            intent = inputs["failed_intent"]
            if (intent["sample"] != sample or intent["policy"] != policy or intent["attempt_index"] != 1
                    or intent["reservation"] != shaping.reserve_batch(old, receipts, attempt_key, 432)):
                raise ValueError("outstanding prior reservation or sample changed")
        else:
            receipts[attempt_key] = shaping.execution_receipt(attempt_key, report)
    return rows


def completion_budget(inputs, billing, rates, terminal):
    if terminal.get("sandbox_id") != recovery.QUARANTINED_SANDBOX or type(terminal.get("returncode")) is not int:
        raise ValueError("confirm the old quarantined sandbox is terminal before new work")
    def amount(value):
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError("invalid cost")
        return result
    held = amount(inputs["failed_intent"]["reservation"]["total_reserved_usd"])
    controller = Decimal(CONTROLLER_SECONDS + 60) * (
        amount(rates["cpu_hour_cost"]) + 2 * amount(rates["mem_gib_hour_cost"])) / 3600
    budget = {"billing_before": billing, "rates": rates, "old_sandbox_terminal": terminal,
        "total_trial_limit_usd": "20", "old_reservation_held_usd": str(held),
        "controller_resource_reserve_usd": str(controller),
        "fixed_reserved_usd": str(max(held, amount(billing["metered_cost"]) + 1) + controller),
        "sandbox_hour_usd": str(amount(rates["cpu_hour_cost_sandbox"]) + amount(rates["mem_gib_hour_cost_sandbox"]) / 4),
        "max_sandbox_executions": 6 * 432, "provider_hard_spending_cap": False,
        "policy": "Hold the entire prior reservation; six missing programs only, no retries, no GPU calls."}
    shaping.reserve_batch(budget, {}, "preflight", 432)
    return budget


def summary(inputs, reports):
    """Fixed eight-program denominators, even for a partially finished continuation."""
    rows = recovery.entries(inputs["previous"]["generations"])
    if set(reports) - {key for key, _, _ in rows} or not set(inputs["reports"]) <= set(reports):
        raise ValueError("missing saved reports or unexpected evaluation")
    for key, report in inputs["reports"].items():
        if reports[key] != report:
            raise ValueError("saved evidence replaced")
    used = set()
    def remember(report):
        ids = [a["metadata"]["sandbox_id"] for o in unique_outcomes(report).values()
               for a in o["attempts"] if a["metadata"].get("sandbox_id")]
        if len(set(ids)) != len(ids) or used.intersection(ids):
            raise ValueError("sandbox reused across original or new execution records")
        used.update(ids)
    for generation in inputs["previous"]["generations"].values():
        for group in generation["rollout_records"]:
            for report in group["reports"]:
                remember(report)
    remember(inputs["previous"]["failed_report"])
    policies, program_rows = {}, []
    for key, policy, sample in rows:
        p = policies.setdefault(policy, {"planned_programs": 8, "evaluated_programs": 0,
            "pending_keys": [], "ambiguous_programs": 0, "unattributed_inputs": 0,
            "audit_passed_cases": [0, 0], "full_audit_passes": [0, 0],
            "mean_partial_reward": [0., 0.], "mean_bonus_reward": [0., 0.]})
        if key in reports:
            report = reports[key]
            remember(report)
            a = assess_report(sample, report, inputs["previous"]["setup"]["sandbox_image_id"])
            p["evaluated_programs"] += 1
            p["ambiguous_programs"] += int(a["unattributed_inputs"] > 0)
            p["unattributed_inputs"] += a["unattributed_inputs"]
            audit = a["suites"]["audit"]
        else:
            p["pending_keys"].append(key)
            a = {"status": "not_evaluated", "partial_reward": [0, 1], "bonus_reward": [0, 1]}
            audit = {"passed_cases": [0, 388], "full_pass": [0, 1]}
        program_rows.append({"key": key, "policy": policy, "assessment": a})
        for side in (0, 1):
            p["audit_passed_cases"][side] += audit["passed_cases"][side]
            p["full_audit_passes"][side] += audit["full_pass"][side]
            p["mean_partial_reward"][side] += a["partial_reward"][side] / 8
            p["mean_bonus_reward"][side] += a["bonus_reward"][side] / 8
    for p in policies.values():
        p["audit_total_cases"] = 8 * 388
        p["audit_case_fraction"] = [n / (8 * 388) for n in p["audit_passed_cases"]]
    return {"version": VERSION, "all_programs_evaluated": len(reports) == 24,
        "original_protocol_fully_validated": all(p["evaluated_programs"] == 8 and p["ambiguous_programs"] == 0
                                                for p in policies.values()),
        "policies": policies, "rows": program_rows, "recorded_sandbox_executions": len(used),
        "new_generations": 0, "new_training_updates": 0, "retries": 0,
        "limitations": ["Bounds represent unresolved outcomes, NOT confidence intervals.",
            "One task, four updates, one training seed and eight sampled programs per policy.",
            "Development audit, not an untouched final benchmark.",
            "Post-failure evaluation amendment; original rejected reports are unchanged."]}


def verify_completion(inputs, reports, budget, receipts, reservations):
    rows = validate_inputs(inputs)
    if set(reports) != {key for key, _, _ in rows}:
        raise ValueError("all 24 raw program reports required")
    if completion_budget(inputs, budget["billing_before"], budget["rates"], budget["old_sandbox_terminal"]) != budget:
        raise ValueError("budget does not recompute")
    accounted = {}
    for key, _, _ in rows:
        if key not in PENDING_KEYS:
            continue
        report = reports[key]
        receipt = shaping.execution_receipt(key, report)
        if receipts.get(key) != receipt or reservations.get(key) != shaping.reserve_batch(budget, accounted, key, receipt["executions"]):
            raise ValueError("missing or changed spending record")
        accounted[key] = receipt
    if set(receipts) != set(PENDING_KEYS) or set(reservations) != set(PENDING_KEYS):
        raise ValueError("exactly six new program receipts required")
    result = summary(inputs, reports)
    if not result["all_programs_evaluated"]:
        raise ValueError("incomplete continuation")
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline verification; never execute generated programs")
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    docs = {n: json.loads((args.run / f"{n}.json").read_text())
            for n in ("inputs", "reports", "budget", "receipts", "reservations", "summary")}
    result = verify_completion(*(docs[n] for n in ("inputs", "reports", "budget", "receipts", "reservations")))
    if result != docs["summary"]:
        raise ValueError("saved summary differs from recomputation")
    print(canonical_json({"verified": True, "cloud_calls": 0, "candidate_source_executed": False,
        "all_programs_evaluated": result["all_programs_evaluated"], "policies": result["policies"]}))


if __name__ == "__main__":
    main()
