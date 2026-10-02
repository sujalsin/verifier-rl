"""Offline endpoint-omission diagnostics; never execute candidates or launch jobs.

Original reports are revalidated before counterfactual scoring. Development
cohorts, authored fault maps and incomplete groups remain explicitly separate.
"""

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

from . import booking_study as historical, booking_reward_pilot as pilot
from . import booking_verifier_v2 as coverage
from .grading import Status
from .panel_execution import authored_booking, unpack_result
from .reward_review import group_advantages
from .suites import canonical_json, digest
from .task_panel import BOOKING, FAMILIES

VERSION = "booking-endpoint-contrast-0.1"
ANALYSIS_ID = "booking-boundary-contrast-20260929-v1"
TOLERANCE = 1e-9
COHORTS = ((pilot.RUN_ID, 14), (pilot.WARM_RUN_ID, 5))


def has_shared_endpoint(case):
    if case.task_id != BOOKING:
        raise ValueError("only booking inputs are supported")
    bookings = case.arguments["bookings"]
    return bool({a for a, _ in bookings} & {b for _, b in bookings})


def partition(cases):
    if (not cases or any(c.task_id != BOOKING or c.split != "training" for c in cases)
            or len({c.input_hash for c in cases}) != len(cases)):
        raise ValueError("unique booking training cases required; audit cannot supply reward")
    kept = tuple(c for c in cases if not has_shared_endpoint(c))
    omitted = tuple(c for c in cases if has_shared_endpoint(c))
    if not kept or not omitted:
        raise ValueError("both scored and omitted inputs are required")
    return kept, omitted


def score_training(cases, outcomes):
    """Pure arithmetic AFTER evidence validation, also usable with authored maps."""
    kept, omitted = partition(cases)
    if (set(outcomes) != {c.input_hash for c in cases}
            or any(type(v) is not bool for v in outcomes.values())):
        raise ValueError("all resolved boolean outcomes required, including omitted tests")

    def score(selected):
        n = sum(outcomes[c.input_hash] for c in selected)
        return {"passed": n, "total": len(selected), "reward": n / len(selected),
                "full_pass": n == len(selected)}

    return {"reference": score(cases), "endpoint_omission": score(kept),
            "omitted_cases": {"passed": sum(outcomes[c.input_hash] for c in omitted),
                              "total": len(omitted)}}


@lru_cache(maxsize=1024)
def inclusive_answer(arguments_json):
    return authored_booking(json.loads(arguments_json)["bookings"], "inclusive_boundary")


def _observed_integer(record):
    result = unpack_result(record)
    if result.status != Status.COMPLETED:
        return None
    try:
        value = json.loads(result.stdout)
    except (ValueError, UnicodeError, RecursionError):
        return None
    return value if type(value) is int and value >= 0 else None


def _program_row(sample, records, cases, outcomes, audit=None):
    scores = score_training(cases, outcomes)
    rejected = sample["extraction_status"].startswith("rejected_")
    answers = {c.input_hash: None if rejected else _observed_integer(records[c.input_hash]) for c in cases}
    differentiating = [c for c in cases if c.expected != inclusive_answer(c.arguments_json)]
    inclusive_matches = [c for c in differentiating
                         if answers[c.input_hash] == inclusive_answer(c.arguments_json)]
    witnesses = sorted(inclusive_matches, key=lambda c: (len(c.arguments["bookings"]), c.input_hash))[:1]
    return {"sample_id": sample["sample_id"], "source_hash": digest(sample["source"]),
            "records_hash": digest(canonical_json(records)),
            "extraction_rejected": rejected, "syntax_valid": sample["syntax_valid"],
            **scores,
            "reward_difference": scores["endpoint_omission"]["reward"] - scores["reference"]["reward"],
            "omission_only_full_acceptance": scores["endpoint_omission"]["full_pass"] and not scores["reference"]["full_pass"],
            "inclusive_output_signature": bool(differentiating) and all(
                answers[c.input_hash] == inclusive_answer(c.arguments_json) for c in cases),
            "differentiating_cases": len(differentiating), "inclusive_wrong_outputs": len(inclusive_matches),
            "witnesses": [{"case_id": c.name, "family": c.family, "input_hash": c.input_hash,
                           "arguments": c.arguments, "expected": c.expected,
                           "observed": answers[c.input_hash], "inclusive_expected": inclusive_answer(c.arguments_json)}
                          for c in witnesses],
            "historical_audit": audit}


def compare_group(key, rows, *, actual_training_group):
    if len(rows) != 4 or len({row["sample_id"] for row in rows}) != 4:
        raise ValueError("four distinct ordered program records required")
    rewards = {condition: [r[condition]["reward"] for r in rows]
               for condition in ("reference", "endpoint_omission")}
    advantages = {k: group_advantages(values) for k, values in rewards.items()}
    exact_scale = {k: group_advantages(values, epsilon=0) for k, values in rewards.items()}
    difference = max(abs(a - b) for a, b in zip(*exact_scale.values()))

    def sign(value):
        return 0 if abs(value) <= TOLERANCE else (1 if value > 0 else -1)

    changed = [r["sample_id"] for r, a, b in zip(rows, *exact_scale.values()) if sign(a) != sign(b)]
    return {"group_id": key, "actual_training_group": actual_training_group,
            "sample_ids": [r["sample_id"] for r in rows], "rewards": rewards,
            "advantages_epsilon_1e_4": advantages, "advantages_epsilon_0": exact_scale,
            "mixed_rewards": {k: len(set(v)) > 1 for k, v in rewards.items()},
            "max_advantage_difference_without_epsilon": difference,
            "relative_signal_changed": difference > TOLERANCE,
            "advantage_sign_change_ids": changed}


def suite_manifest(cases):
    kept, omitted = partition(cases)
    return {"cases": len(cases), "reference_scored": len(cases), "omission_scored": len(kept),
            "omitted_count": len(omitted),
            "scoring_hash": digest(canonical_json({"version": VERSION,
                "cases": [c.manifest() for c in cases], "omitted": [c.input_hash for c in omitted]})),
            "omitted_by_family": {f: sum(c.family == f for c in omitted) for f in FAMILIES},
            "kept_input_sizes": sorted({len(c.arguments["bookings"]) for c in kept}),
            "omitted_inputs": [{"case_id": c.name, "input_hash": c.input_hash, "family": c.family}
                               for c in omitted]}


def authored_controls():
    cases = coverage.cases_for("training")
    result = {}
    for fault in ("correct", "inclusive_end", "empty_only", "constant_zero", "constant_one", "return_length"):
        outcomes = {}
        for c in cases:
            bookings, expected = c.arguments["bookings"], c.expected
            answer = {"correct": expected, "inclusive_end": inclusive_answer(c.arguments_json),
                      "empty_only": expected if bookings else 1, "constant_zero": 0,
                      "constant_one": 1, "return_length": len(bookings)}[fault]
            outcomes[c.input_hash] = answer == expected
        result[fault] = score_training(cases, outcomes)
    return result


def summarize(rows, groups):
    counts = Counter(row["source_hash"] for row in rows)
    return {"programs": len(rows), "distinct_sources": len(counts),
            "repeated_source_occurrences": sum(v - 1 for v in counts.values()),
            "extraction_rejections": sum(r["extraction_rejected"] for r in rows),
            "reference_full_passes": sum(r["reference"]["full_pass"] for r in rows),
            "omission_full_passes": sum(r["endpoint_omission"]["full_pass"] for r in rows),
            "omission_only_full_acceptances": sum(r["omission_only_full_acceptance"] for r in rows),
            "inclusive_output_signatures": sum(r["inclusive_output_signature"] for r in rows),
            "distinct_inclusive_signature_sources": len({r["source_hash"] for r in rows if r["inclusive_output_signature"]}),
            "changed_scalar_rewards": sum(abs(r["reward_difference"]) > TOLERANCE for r in rows),
            "groups": len(groups), "changed_relative_signal_groups": sum(g["relative_signal_changed"] for g in groups),
            "groups_with_sign_changes": sum(bool(g["advantage_sign_change_ids"]) for g in groups),
            "mixed_reference_groups": sum(g["mixed_rewards"]["reference"] for g in groups),
            "mixed_omission_groups": sum(g["mixed_rewards"]["endpoint_omission"] for g in groups)}


def _comparable_report(report):
    # This top-level list was assembled in asynchronous completion order. JSON
    # persistence sorts the records dictionary, so replay changes that order.
    # Compare the exact ID multiset (including multiplicity), not a set. All
    # source/input bindings, outcomes, sample order and nested reports stay exact.
    ids = report.get("sandbox_ids")
    if not isinstance(ids, list) or any(not isinstance(sid, str) for sid in ids):
        raise ValueError("invalid saved sandbox identity list")
    return dict(report, sandbox_ids=sorted(ids))


def replay_cohort(read, *, completed_groups=None):
    """Read callback enables synthetic tests without cloud, models, or subprocesses."""
    plan, setup = read("plan.json"), read("setup.json")
    legacy = completed_groups is None
    (historical.validate_plan if legacy else pilot.validate_plan)(plan)
    cases = historical.suites()["training"] if legacy else coverage.cases_for("training")
    n = 8 if legacy else completed_groups
    rows, groups, sandbox_ids, evidence = [], [], [], []
    for index in range(n):
        key = f"calibration-{index:02d}" if legacy else f"train-linear-{index:02d}"
        role = "calibration" if legacy else "training"
        raw = read(f"grading/{key}/result.json")
        samples = raw["samples"]
        if legacy:
            if [s["sample_id"] for s in samples] != historical.batch_ids(key, role):
                raise ValueError("original calibration sample order changed")
            if [s["seed"] for s in samples] != list(range(12000 + index * 4, 12004 + index * 4)):
                raise ValueError("original calibration seeds changed")
        else:
            pilot.validate_batch(key, role, samples, plan)
        if (raw["key"] != key or raw["role"] != role or raw["plan_hash"] != digest(canonical_json(plan))
                or set(raw["records"]) != {s["sample_id"] for s in samples}):
            raise ValueError("batch plan, record membership or provenance mismatch")
        grade = historical.grade if legacy else pilot.grade
        reports = [grade(s, raw["records"][s["sample_id"]], role, setup["sandbox_image_id"], plan) for s in samples]
        if [_comparable_report(r) for r in reports] != [_comparable_report(r) for r in raw["reports"]]:
            raise ValueError("original report differs from independently replayed records: " + key)
        if not legacy:
            prior_ids, _ = pilot.verify_retries(raw, setup["sandbox_image_id"], plan)
            sandbox_ids.extend(prior_ids)
        current = []
        for sample, report in zip(samples, reports):
            sandbox_ids.extend(report["sandbox_ids"])
            outcomes = ({c.input_hash: report["outcomes"][c.input_hash]["passed"] for c in cases} if legacy
                        else report["reports"]["training"]["outcomes"])
            audit = ({"role": "previously_used_calibration_audit", "passed": report["audit_case_passes"],
                      "total": len(historical.suites()["calibration"]), "full_pass": report["audit_passed"]}
                     if legacy else None)
            current.append(_program_row(sample, raw["records"][sample["sample_id"]], cases, outcomes, audit))
        groups.append(compare_group(key, current, actual_training_group=not legacy))
        rows.extend(current)
        evidence.append({"batch_id": key, "batch_hash": digest(canonical_json(raw))})
    if len(sandbox_ids) != len(set(sandbox_ids)):
        raise ValueError("reused execution identity between batches")
    unresolved = []
    if not legacy:
        key = f"train-linear-{n:02d}"
        failed_samples = read(f"arms/linear/rollouts/{key}.json")
        pilot.validate_batch(key, "training", failed_samples, plan)
        unresolved = [{"sample_id": s["sample_id"], "source_hash": digest(s["source"]),
                       "reward": None, "reason": "interrupted_group_not_scored_in_this_complete_group_analysis"}
                      for s in failed_samples]
    return {"run_id": plan["run_id"], "plan_hash": digest(canonical_json(plan)),
            "initial_policy_hash": plan["initial_parameter_hash"], "original_version": plan["version"],
            "population": "old_untouched_policy_calibration" if legacy else "changing_policies_completed_training_groups",
            "fresh_confirmation": False, "suite": suite_manifest(cases), "summary": summarize(rows, groups),
            "unscored_programs": unresolved, "groups": groups, "programs": rows,
            "report_comparison": "exact except top-level sandbox ID multiset ordering",
            "validated_execution_ids": len(sandbox_ids), "source_batches": evidence}


def make_report(root, evidence_root):
    root, evidence_root = Path(root).resolve(), Path(evidence_root).resolve()
    files = []

    def reader(run_id, legacy=False):
        def read(relative):
            if legacy:
                path = root / "runs" / run_id / "completed-remote" / run_id / relative
            elif relative in ("plan.json", "setup.json"):
                path = root / "runs" / run_id / relative
            else:
                path = evidence_root / run_id / relative
            value = path.read_text()
            files.append({"run_id": run_id, "path": relative, "sha256": digest(value)})
            return json.loads(value)
        return read

    cohorts = [replay_cohort(reader(historical.RUN_ID, legacy=True))]
    cohorts += [replay_cohort(reader(run), completed_groups=n) for run, n in COHORTS]
    controls = authored_controls()
    signal = any(c["summary"]["changed_relative_signal_groups"] for c in cohorts[1:])
    witness = any(c["summary"]["inclusive_output_signatures"] for c in cohorts[1:])
    return {"version": VERSION, "analysis_id": ANALYSIS_ID, "status": "offline_development_report",
            "new_candidate_executions": 0, "new_model_generations": 0, "new_optimizer_steps": 0,
            "historical_artifacts_modified": False, "authored_mathematical_controls": controls,
            "proposed_training_suite": suite_manifest(coverage.cases_for("training")),
            "unchanged_development_audit": {"cases": 192, "suite_hash": coverage.suite_hash("audit")},
            "cohorts": cohorts, "files": files, "files_hash": digest(canonical_json(files)),
            "source_code_hashes": {path.name: digest(path.read_text())
                                   for path in sorted((root / "verifier_rl").glob("*.py"))},
            "decision": {"observed_development_signal_contrast": signal,
                         "observed_v2_inclusive_output_signature": witness,
                         "next_step": "review_and_freeze_fresh_screen" if signal and witness else "review_inconclusive_development_contrast",
                         "fresh_screen_completed": False, "training_ready": False},
            "limitations": [
                "Post-hoc defect selection and historical rescoring, not prospective calibration or learning evidence.",
                "Cohorts differ in suite and policy; do not pool them as independent samples of one policy.",
                "Interrupted groups remain unscored; completed groups can be a selected subset.",
                "Inclusive output signatures on finite inputs are not automatic source-level confirmation.",
                "Counterfactual scalar advantages are not measured gradients or counterfactual RL trajectories.",
                "There are no new audit evaluations, training updates, or reward-hacking findings."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evidence", type=Path, help="downloaded fixed-cohort result JSON root")
    parser.add_argument("--controls-only", action="store_true", help="no artifact reads, only trusted mathematical checks")
    args = parser.parse_args()
    if args.controls_only:
        report = {"version": VERSION, "model_samples": 0, "authored_mathematical_controls": authored_controls(),
                  "proposed_training_suite": suite_manifest(coverage.cases_for("training"))}
    else:
        report = make_report(args.root, args.evidence or args.root / "runs" / ANALYSIS_ID / "evidence")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
