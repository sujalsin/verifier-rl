"""Offline reward audit: saved JSON evidence and fixed authored controls only.

Never import or execute generated source, call Modal, or change training defaults.
Counterfactual scalar rewards/advantages are NOT alternative training runs.
"""

import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean, stdev

from .diagnostics import diagnose_run, diagnose_sample
from .fixtures import fixture_impl
from .grading import ExecutionResult, Status, score_suite
from .reward_v2 import CONTROLS, SEED, WEIGHTS, behavior_scores, behavior_suite, control_output
from .reward_v3 import coverage_scores, coverage_suite
from .suites import build_suites, canonical_json, coverage, digest

VERSION = "cache-reward-review-0.1"
NEW_CONTROLS = ("short_inputs_only", "single_character_keys_only", "decrement_expiry_on_read")
FORMULAS = ("v2_partial", "binary", "completion_bonus")


def authored_output(operations, name):
    """Explicit project-authored functions, NOT a generated-code execution path."""
    if name in CONTROLS:
        return control_output(operations, name)
    if name not in NEW_CONTROLS:
        raise ValueError("unknown authored control")
    if name in NEW_CONTROLS[:2]:
        unsupported = (len(operations) > 6 if name == "short_inputs_only"
                       else any(len(op["key"]) > 1 for op in operations))
        return ([None for op in operations if op["op"] == "get"] if unsupported
                else fixture_impl(operations))
    stored, answers = {}, []
    for op in operations:
        key, t = op["key"], op["time"]
        if op["op"] == "put":
            stored[key] = op["value"], t + op["ttl"]
        else:
            entry, answer = stored.get(key), None
            if entry is not None and t < entry[1]:
                answer = entry[0]
                stored[key] = entry[0], entry[1] - 1
            answers.append(answer)
    return answers


def authored_report(suite, name):
    executions = {c.input_hash: (ExecutionResult(
        Status.COMPLETED, canonical_json(authored_output(c.operations, name)).encode(),
        metadata={"backend": "trusted_authored_control_only"}),) for c in suite.cases}
    return score_suite(suite, executions)


def formula_scores(scores):
    p, b = scores["partial"], scores["binary"]
    if p is None and b is None:
        return {f: None for f in FORMULAS}
    if (type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
            or type(b) is not int or b not in (0, 1)):
        raise ValueError("invalid scalar rewards")
    if (p == 1) != (b == 1):
        raise ValueError("full partial credit and full-suite acceptance must agree")
    return {"v2_partial": p, "binary": b, "completion_bonus": (p + b) / 2}


def group_advantages(rewards, epsilon=1e-4):
    """Diagnostic TRL group/sample-std formula; no Torch/optimizer dependency."""
    if (len(rewards) < 2 or not math.isfinite(epsilon) or epsilon < 0
            or any(type(r) not in (int, float) or not math.isfinite(r) for r in rewards)):
        raise ValueError("finite reward group with at least two elements required")
    center, spread = mean(rewards), stdev(rewards)
    return [(r - center) / (spread + epsilon) for r in rewards] if spread else [0.0] * len(rewards)


def rank_diagnostics(rows):
    """Descriptive program-pair counts, not independent statistical trials."""
    counts = Counter()
    for a, b in combinations(rows, 2):
        if a["audit_case_accuracy"] is None or b["audit_case_accuracy"] is None:
            counts["unscored_pairs"] += 1
            continue
        reward_delta = a["scores"]["partial"] - b["scores"]["partial"]
        accuracy_delta = a["audit_case_accuracy"] - b["audit_case_accuracy"]
        if not reward_delta and not accuracy_delta:
            category = "tied_both"
        elif not reward_delta:
            category = "reward_tie_only"
        elif not accuracy_delta:
            category = "audit_tie_only"
        else:
            category = "concordant" if reward_delta * accuracy_delta > 0 else "discordant"
        counts[category] += 1
    return {"program_records": len(rows), "unique_sources": len({r["source_hash"] for r in rows}),
            "pair_counts": dict(counts),
            "interpretation": "Observed development records only; pairs and repeated sources are dependent."}


def suite_domain(suite):
    result = coverage(suite)
    result["key_lengths"] = sorted({len(op["key"]) for c in suite.cases for op in c.operations})
    result["suite_hash"] = suite.fingerprint
    return result


def review_controls(seeds=(SEED, SEED + 1, SEED + 2)):
    rows = []
    for seed in seeds:
        old, new = behavior_suite(seed), coverage_suite(seed)
        for name in CONTROLS + NEW_CONTROLS:
            before, after = authored_report(old, name), authored_report(new, name)
            scores_before, scores_after = behavior_scores(before, old), coverage_scores(after, new)
            if name == "correct" and scores_after["binary"] != 1:
                raise AssertionError("correct authored implementation must pass v3")
            if name != "correct" and scores_after["binary"] != 0:
                raise AssertionError(f"v3 false full acceptance: {name}")
            if name in NEW_CONTROLS[:2] and (scores_before["binary"] != 1 or scores_after["binary"] != 0):
                raise AssertionError("coverage-gap demonstration did not reproduce")
            if name in ("always_none", "always_empty", "constant_seven", "random_answers"):
                if scores_after["partial"] > .05:
                    raise AssertionError("trivial output exceeds edge-only credit allowance")
            rows.append({"seed": seed, "control": name, "v2": scores_before, "v3": scores_after,
                         "v3_failed_cases": [o["case"] for o in after["outcomes"] if not o["passed"]]})
    return {"model_results": False, "authored_functions_only": True, "controls_passed": True,
            "seeds": list(seeds), "rows": rows,
            "domains": {"v2": suite_domain(behavior_suite()), "v3": suite_domain(coverage_suite())}}


def review_pilot(generation, reports):
    reward, audit = behavior_suite(), build_suites()[-1]
    # Static source/extraction checks and strict replay of complete saved stdout.
    # This does not execute candidate source, salvage outputs, or repair scores.
    diagnostics = diagnose_run(generation, reports, (reward, audit))
    evaluations = []
    for report in reports:
        if [r["suite_hash"] for r in report["suites"]] != [reward.fingerprint, audit.fingerprint]:
            raise ValueError("reward review requires the exact v2 and development audit suites")
        r, a = report["suites"]
        scores = behavior_scores(r, reward)
        if scores["partial"] is None:
            raise ValueError("unresolved reward record cannot supply a comparison")
        evaluations.append({"arm": report["arm"], "seed": report["seed"],
            "source_hash": report["candidate_hash"], "scores": scores, "formulas": formula_scores(scores),
            "audit_full_pass": a["all_passed"], "audit_passed_cases": a["passed_count"],
            "audit_total_cases": a["total"],
            "audit_case_accuracy": a["passed_count"] / a["total"] if a["all_passed"] is not None else None,
            "unresolved_137_cases": [o["case"] for o in a["outcomes"]
                if any(attempt["metadata"].get("returncode") == 137 for attempt in o["attempts"])]})
    groups, training_rows, training_diagnostics, identities = [], [], [], set()
    records = generation["rollout_records"]
    if (not 1 <= len(records) <= 4 or len(records) != generation["metrics"]["global_step"]
            or len(records) != len(generation["metrics"]["rewards"])):
        raise ValueError("training group/step coverage mismatch")
    for step, record in enumerate(records, 1):
        if len(record["samples"]) != 4 or len(record["reports"]) != 4 or len(record["rewards"]) != 4:
            raise ValueError("expected four complete rollouts per training group")
        selected = []
        for sample, report in zip(record["samples"], record["reports"]):
            if (sample["arm"] != "training" or sample["seed"] in identities
                    or sample["prompt_hash"] != generation["plan"]["prompt_hash"]
                    or len(report["suites"]) != 1):
                raise ValueError("training identity or suite mismatch")
            identities.add(sample["seed"])
            training_diagnostics.append(diagnose_sample(sample, report, (reward,)))
            saved = report["suites"][0]
            scores = behavior_scores(saved, reward)
            if scores["partial"] is None:
                raise ValueError("unscored training reward")
            edge = float(WEIGHTS["edges"]) * scores["family_pass_rates"]["edges"]
            selected.append({"step": step, "rollout_id": sample["seed"],
                "source_hash": report["candidate_hash"], "scores": scores,
                "formulas": formula_scores(scores), "edge_credit": edge,
                "substantive_credit": scores["partial"] - edge,
                "case_outcomes": dict(Counter(o["reason"] for o in saved["outcomes"]))})
        recomputed = [r["scores"]["partial"] for r in selected]
        if recomputed != record["rewards"] or recomputed != generation["metrics"]["rewards"][step - 1]:
            raise ValueError("saved training rewards disagree with recorded case outcomes")
        formulas = {}
        for name in FORMULAS:
            values = [r["formulas"][name] for r in selected]
            advantages = group_advantages(values)
            formulas[name] = {"rewards": values, "advantages": advantages,
                "uniform": len(set(values)) == 1,
                "positive_advantage_rollout_ids": [r["rollout_id"] for r, a in zip(selected, advantages) if a > 0],
                "positive_advantage_nonfull_ids": [r["rollout_id"] for r, a in zip(selected, advantages)
                                                   if a > 0 and not r["scores"]["binary"]],
                "positive_advantage_edge_only_ids": [r["rollout_id"] for r, a in zip(selected, advantages)
                                                      if a > 0 and r["substantive_credit"] <= 1e-12]}
        groups.append({"step": step, "formulas": formulas})
        training_rows.extend(selected)
    return {"run_id": generation["run_id"], "new_model_samples": 0,
            "generation_document_hash": digest(canonical_json(generation)),
            "reports_document_hash": digest(canonical_json(reports)),
            "recorded_rewards_recomputed_exactly": True,
            "training_rows": training_rows, "groups": groups, "evaluation_rows": evaluations,
            "observed_v2_rankings": rank_diagnostics(evaluations),
            "evaluation_diagnostics": diagnostics, "training_diagnostics": training_diagnostics}


def main(argv=None):
    from .cli import create_run_directory, save_suites, write_private
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed real-reward pilot directory")
    parser.add_argument("--out", required=True, help="NEW local sidecar directory; exclusive creation")
    args = parser.parse_args(argv)
    try:
        generation = json.loads((args.run / "generation.json").read_text())
        reports = json.loads((args.run / "reports.json").read_text())
        pilot = review_pilot(generation, reports)
        controls = review_controls()
        proposal, audit = coverage_suite(), build_suites()[-1]
        if {c.input_hash for c in proposal.cases} & {c.input_hash for c in audit.cases}:
            raise ValueError("proposed reward cases overlap development audit inputs")
        report = {"version": VERSION, "source_directory": str(args.run), "pilot": pilot,
                  "controls": controls, "candidate_source_executed": False,
                  "official_scores_changed": False, "training_defaults_changed": False,
                  "cloud_calls": 0, "new_model_samples": 0,
                  "proposal": {"v3_audit_exact_overlap": 0, "status": "offline_validated_not_enabled"},
                  "limitations": [
                      "Known coverage weaknesses do not establish the cause of the learning regression.",
                      "Authored controls are not observed model exploitation or proof against arbitrary programs.",
                      "Only sixteen paired evaluation records; repeated sources and program pairs are dependent.",
                      "The exit-137 audit outcome is retained and flagged, not silently converted to a pass.",
                      "No model candidate has been executed on the new v3 inputs.",
                      "Formula comparisons reuse v2 outputs; they do not predict a counterfactual RL trajectory.",
                      "Both v3 probes and previously seen audit results are development data."]}
        directory = create_run_directory(args.out)
        save_suites(directory, (behavior_suite(), proposal))
        write_private(directory / "reward_review.json", canonical_json(report))
        write_private(directory / "implementation_hashes.json", canonical_json({
            name: digest((Path(__file__).resolve().parent / name).read_text())
            for name in ("reward_review.py", "reward_v3.py", "reward_v2.py", "grading.py")}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print("Reward audit complete; no cloud execution or training. Historical scores unchanged.")
    print(canonical_json(pilot["observed_v2_rankings"]))
    for row in controls["rows"]:
        if row["seed"] == SEED and row["control"] in NEW_CONTROLS:
            print(row["control"], "v2", row["v2"]["partial"], "v3", row["v3"]["partial"])
    for group in pilot["groups"]:
        print("step", group["step"], "positive nonfull advantages", canonical_json({
            name: record["positive_advantage_nonfull_ids"] for name, record in group["formulas"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
