"""Frozen generation-only baseline protocol and non-executing result analysis."""

import ast
import argparse
from collections import Counter
import json
from pathlib import Path

from .model_trial import EXTRACTION_VERSION, extract_completion
from .suites import canonical_json, digest

MODEL_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
SEEDS = tuple(range(4000, 4032))
BASELINE_VERSION = "cache-generation-baseline-0.1"


def validate_recovered_report(sample, report, suites, image_id):
    if report.get("sample_seed") != sample["seed"] or report["candidate_hash"] != digest(sample["source"]):
        raise ValueError("recovery source/seed mismatch")
    expected = {s.name: s.fingerprint for s in suites}
    if {s["suite"]: s["suite_hash"] for s in report["suites"]} != expected:
        raise ValueError("recovery suite mismatch")
    if any(s["infrastructure_errors"] for s in report["suites"]):
        return False  # Retain but do not reuse incomplete/unscored attempts.
    for suite in report["suites"]:
        for outcome in suite["outcomes"]:
            for attempt in outcome["attempts"]:
                metadata = attempt["metadata"]
                if metadata.get("image_id") != image_id or metadata.get("cleanup") != "terminated":
                    raise ValueError("recovery image/cleanup mismatch")
    return True


def inspect_completion(raw):
    extraction = extract_completion(raw)
    rejected = extraction.status.startswith("rejected_")
    result = {"raw": raw, "source": extraction.source,
              "extraction_status": extraction.status, "extraction_version": EXTRACTION_VERSION,
              "syntax_valid": False, "has_top_level_entrypoint": False,
              "syntax_error": None}
    if not rejected:
        try:
            # Compile/parse only: never exec or import candidate source here.
            compile(extraction.source, "<candidate>", "exec")
            tree = ast.parse(extraction.source)
            result["syntax_valid"] = True
            result["has_top_level_entrypoint"] = any(
                isinstance(n, ast.FunctionDef) and n.name == "simulate_cache" for n in tree.body)
        except (SyntaxError, ValueError) as exc:
            result["syntax_error"] = {"type": type(exc).__name__, "line": getattr(exc, "lineno", None)}
    return result


def summarize(samples, reports):
    if len(samples) != len(SEEDS) or [s["seed"] for s in samples] != list(SEEDS):
        raise ValueError("baseline requires all 32 samples in frozen seed order")
    if len(reports) != len(samples):
        raise ValueError("incomplete baseline evaluations")
    suite_names = ("g1", "g2", "g3", "audit")
    accepted = {name: [] for name in suite_names}
    passed_cases = Counter()
    total_cases = Counter()
    disagreements = Counter()
    reasons = Counter()
    statuses = Counter()
    execution_seconds = 0.0
    any_completed = all_completed = any_valid_output = 0
    candidate_rows = []
    for sample, report in zip(samples, reports):
        if digest(sample["source"]) != report["candidate_hash"]:
            raise ValueError("candidate/report hash mismatch")
        if tuple(s["suite"] for s in report["suites"]) != suite_names:
            raise ValueError("all four baseline suites required in canonical order")
        if any(s["infrastructure_errors"] for s in report["suites"]):
            raise ValueError("infrastructure errors must not be summarized as candidate failures")
        unique = {}
        scores = {}
        for suite in report["suites"]:
            name = suite["suite"]
            passed_cases[name] += suite["passed_count"]
            total_cases[name] += suite["total"]
            success = suite["passed_count"] == suite["total"]
            accepted[name].append(success)
            scores[name] = {"passed": suite["passed_count"], "total": suite["total"], "all_passed": success}
            for outcome in suite["outcomes"]:
                unique.setdefault(outcome["input_hash"], outcome)
        completed = 0
        valid_output = False
        for outcome in unique.values():
            reasons[outcome["reason"]] += 1
            attempt = outcome["attempts"][-1]
            statuses[attempt["status"]] += 1
            execution_seconds += sum(a["metadata"].get("total_seconds", 0.0) for a in outcome["attempts"])
            completed += attempt["status"] == "completed"
            valid_output |= outcome["reason"] in ("pass", "wrong_answer")
        any_completed += completed > 0
        all_completed += completed == len(unique)
        any_valid_output += valid_output
        for name in ("g1", "g2", "g3"):
            disagreements[name] += scores[name]["all_passed"] and not scores["audit"]["all_passed"]
        candidate_rows.append({"seed": sample["seed"], "extraction_status": sample["extraction_status"],
                               "syntax_valid": sample["syntax_valid"], "hit_token_cap": sample["hit_token_cap"],
                               "scores": scores})
    return {
        "protocol": BASELINE_VERSION, "sample_count": len(samples), "training": False,
        "extraction_statuses": dict(Counter(s["extraction_status"] for s in samples)),
        "extractable_count": sum(not s["extraction_status"].startswith("rejected_") for s in samples),
        "syntax_valid_count": sum(s["syntax_valid"] for s in samples),
        "top_level_entrypoint_count": sum(s["has_top_level_entrypoint"] for s in samples),
        "hit_token_cap_count": sum(s["hit_token_cap"] for s in samples),
        "token_cap_without_eos_count": sum(s["hit_token_cap"] and not s["ended_with_eos"] for s in samples),
        "candidates_with_any_completed_execution": any_completed,
        "candidates_with_all_executions_completed": all_completed,
        "candidates_with_any_valid_output": any_valid_output,
        "suite_full_pass_counts": {name: sum(values) for name, values in accepted.items()},
        "suite_case_pass_counts": dict(passed_cases), "suite_case_totals": dict(total_cases),
        "false_acceptance_counts_vs_development_audit": dict(disagreements),
        "mixed_reward_groups_of_four": {
            name: sum(len(set(values[i:i + 4])) > 1 for i in range(0, len(values), 4))
            for name, values in accepted.items() if name != "audit"},
        "group_count": 8, "unique_execution_reasons": dict(reasons),
        "unique_execution_statuses": dict(statuses),
        "unique_executions": sum(statuses.values()),
        "summed_invocation_seconds": execution_seconds,
        "candidates": candidate_rows,
        "limitations": ["One task and one fixed prompt, not task generalization.",
                        "Development audit is not an untouched final evaluation.",
                        "Input-weighted case pass rates are diagnostics, not RL rewards.",
                        "Completed execution does not imply valid output or correctness.",
                        "A token-cap hit is not proof that the Python code itself is incomplete."]}


def main():
    parser = argparse.ArgumentParser(description="Recompute baseline statistics from saved evidence; no execution")
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, help="optional NEW summary file; never overwrite evidence")
    args = parser.parse_args()
    generation = json.loads((args.run / "generation.json").read_text())
    reports = []
    for index in range(8):
        reports.extend(json.loads((args.run / f"batch-{index}.json").read_text()))
    result = summarize(generation["samples"], reports)
    result.update({"run_id": generation["run_id"],
                   "gpu_function_seconds": generation["gpu_function_seconds"],
                   "parameters_unchanged": generation["parameters_unchanged"]})
    if args.out:
        from .cli import write_private
        write_private(args.out, canonical_json(result))
    print(canonical_json(result))


if __name__ == "__main__":
    main()
