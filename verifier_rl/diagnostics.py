"""Non-executing, strict-output diagnostics stored separately from official scores.

Never interpret a last stdout line as a result or run/rewrite a candidate.
Exceptions/stage strings are untrusted hints, not authenticated failure causes.
"""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re

from .grading import MAX_OUTPUT_BYTES, compare_output
from .model_trial import extract_completion
from .partial_reward import balanced_scores, balanced_suite
from .reward_v2 import behavior_scores, behavior_suite
from .suites import build_suites, canonical_json, digest

VERSION = "cache-execution-diagnostics-0.1"
BUCKETS = {
    "pass": "correct", "wrong_answer": "wrong_answer",
    "invalid_json": "output_protocol_failure", "invalid_schema": "output_protocol_failure",
    "candidate_error": "candidate_execution_error", "timeout": "execution_timeout_unattributed",
    "output_limit": "candidate_output_limit", "infrastructure_error": "infrastructure_failure",
    "not_executed": "not_executed", "extraction_rejected": "extraction_failure",
}


def identity(item):
    seed = item.get("seed", item.get("sample_seed"))
    if type(seed) is not int:
        raise ValueError("saved sample/report needs an integer seed")
    return item.get("arm", "policy"), seed


def output_evidence(outcome):
    """Recheck complete previews only; incomplete previews are never salvaged."""
    if not outcome["attempts"] or outcome["attempts"][-1]["status"] != "completed":
        return "not_completed"
    metadata = outcome["attempts"][-1]["metadata"]
    preview, count = metadata.get("stdout_preview"), metadata.get("stdout_bytes")
    if not isinstance(preview, str) or type(count) is not int or count < 0:
        return "unavailable"
    raw = preview.encode("utf-8")
    if len(raw) != count:
        return "preview_incomplete"
    if metadata.get("stdout_sha256") != digest(preview):
        raise ValueError("stdout preview hash mismatch")
    if count > MAX_OUTPUT_BYTES:
        raise ValueError("saved completed output exceeded the output bound")
    passed, reason, actual = compare_output(raw, outcome["expected"])
    if (passed, reason, actual) != (outcome["passed"], outcome["reason"], outcome["actual"]):
        raise ValueError("saved score disagrees with complete stdout")
    return "verified_complete_stdout"


def diagnose_sample(sample, report, suites):
    if identity(sample) != identity(report):
        raise ValueError("sample/report identity mismatch")
    extracted = extract_completion(sample["raw"], version=sample["extraction_version"])
    if (extracted.source != sample["source"] or extracted.status != sample["extraction_status"]
            or report["candidate_hash"] != digest(sample["source"])):
        raise ValueError("sample extraction/source hash mismatch")
    accepted = not extracted.status.startswith("rejected_")
    syntax_valid, entrypoint = False, False
    if accepted:
        try:
            compile(sample["source"], "<static-only>", "exec")  # Compile only, never execute.
            tree = ast.parse(sample["source"])
            syntax_valid = True
            entrypoint = any(isinstance(n, ast.FunctionDef) and n.name == "simulate_cache" for n in tree.body)
        except (SyntaxError, ValueError):
            pass
    known = {s.fingerprint: s for s in suites}
    unique, scores = {}, {}
    for saved in report["suites"]:
        suite = known.get(saved["suite_hash"])
        if suite is None or saved["suite"] != suite.name or suite.name in scores:
            raise ValueError("unknown or duplicate suite; provide its exact manifest-backed definition")
        if saved["total"] != len(suite.cases) or len(saved["outcomes"]) != len(suite.cases):
            raise ValueError("suite coverage mismatch")
        flags = []
        for case, outcome in zip(suite.cases, saved["outcomes"]):
            if (outcome["case"] != case.name or outcome["input_hash"] != case.input_hash
                    or outcome["expected"] != case.expected):
                raise ValueError("case/expected-output identity mismatch")
            passed, reason = outcome["passed"], outcome["reason"]
            if (passed is not None and type(passed) is not bool) or reason not in BUCKETS:
                raise ValueError("invalid correctness flag or outcome reason")
            if passed is not (None if reason in ("infrastructure_error", "not_executed") else reason == "pass"):
                raise ValueError("reason/correctness disagreement")
            attempts = outcome["attempts"]
            if reason == "not_executed":
                if attempts: raise ValueError("not-executed outcome has execution attempts")
            elif not attempts:
                raise ValueError("missing execution evidence")
            else:
                status = attempts[-1]["status"]
                expected_status = "completed" if reason in ("pass", "wrong_answer", "invalid_json", "invalid_schema") else reason
                if status != expected_status:
                    raise ValueError("status/reason disagreement")
                if (reason == "extraction_rejected") == accepted:
                    raise ValueError("extraction evidence disagreement")
            flags.append(passed)
            key = case.input_hash
            if key in unique:
                if any(unique[key][f] != outcome[f] for f in ("passed", "reason", "actual", "expected", "attempts")):
                    raise ValueError("shared input has contradictory recorded outcomes")
            else:
                unique[key] = outcome
        all_passed = None if None in flags else all(flags)
        expected_reward = None if suite.purpose != "training" or all_passed is None else int(all_passed)
        if (saved["passed_count"] != sum(v is True for v in flags) or saved["all_passed"] is not all_passed
                or saved["reward"] != expected_reward
                or saved["infrastructure_errors"] != sum(o["reason"] == "infrastructure_error" for o in saved["outcomes"])
                or saved.get("not_executed_count", 0) != sum(o["reason"] == "not_executed" for o in saved["outcomes"])):
            raise ValueError("inconsistent saved suite aggregates")
        scores[suite.name] = {k: saved[k] for k in ("suite_hash", "passed_count", "total", "reward", "all_passed")}
        if suite.name == "balanced":
            scores[suite.name]["derived_partial"] = balanced_scores(saved, suite)["partial"]
        elif suite.name == "balanced_v2":
            scores[suite.name]["derived_partial"] = behavior_scores(saved, suite)["partial"]
    if not scores:
        raise ValueError("no suites to diagnose")
    buckets, evidence, hints = Counter(), Counter(), Counter()
    attempts_count = 0
    for outcome in unique.values():
        buckets[BUCKETS[outcome["reason"]]] += 1
        evidence[output_evidence(outcome)] += 1
        attempts_count += sum(a["status"] != "extraction_rejected" for a in outcome["attempts"])
        if outcome["reason"] == "candidate_error":
            metadata = outcome["attempts"][-1].get("metadata", {})
            matches = re.findall(r"(?m)^([A-Za-z]+Error):", metadata.get("stderr_preview", ""))
            hints[(metadata.get("runner_stage", "unknown"), matches[-1] if matches else "unknown")] += 1
    if report["execution_config"]["attempts"] != attempts_count:
        raise ValueError("execution-attempt count mismatch")
    observed = buckets["correct"] + buckets["wrong_answer"]
    return {"arm": identity(sample)[0], "seed": identity(sample)[1], "source_hash": report["candidate_hash"],
            "extraction_accepted": accepted, "syntax_valid": syntax_valid, "static_entrypoint_present": entrypoint,
            "unique_inputs": len(unique), "recorded_execution_attempts": attempts_count,
            "outcome_counts": dict(buckets), "stdout_evidence": dict(evidence),
            "semantic_observations": {"valid_output_cases": observed, "correct_cases": buckets["correct"],
                                      "wrong_answer_cases": buckets["wrong_answer"],
                                      "unobservable_cases": len(unique) - observed,
                                      "conditional_accuracy": buckets["correct"] / observed if observed else None},
            "execution_error_hints_untrusted": [{"stage": stage, "exception": exc, "count": n}
                                                 for (stage, exc), n in sorted(hints.items())],
            "official_suite_scores_unchanged": scores}


def diagnose_run(generation, reports, suites=None):
    suites = suites or (*build_suites(), balanced_suite(), behavior_suite())
    samples = generation["samples"]
    by_key = {identity(r): r for r in reports}
    if (len(by_key) != len(reports) or len({identity(s) for s in samples}) != len(samples)
            or set(by_key) != {identity(s) for s in samples}):
        raise ValueError("missing, duplicate, or unexpected sample/report")
    if not samples:
        raise ValueError("no samples to diagnose")
    rows = []
    for sample in samples:
        if sample.get("prompt_hash") != generation["plan"]["prompt_hash"]:
            raise ValueError("sample/prompt mismatch")
        rows.append(diagnose_sample(sample, by_key[identity(sample)], suites))
    arms = {}
    for arm in sorted({r["arm"] for r in rows}):
        selected = [r for r in rows if r["arm"] == arm]
        counts = Counter()
        for row in selected: counts.update(row["outcome_counts"])
        arms[arm] = {"program_samples": len(selected), "syntax_valid": sum(r["syntax_valid"] for r in selected),
                     "all_inputs_have_valid_output": sum(r["semantic_observations"]["unobservable_cases"] == 0 for r in selected),
                     "unique_candidate_inputs": sum(r["unique_inputs"] for r in selected), "outcome_counts": dict(counts)}
    return {"version": VERSION, "generation_run_id": generation["run_id"], "new_model_samples": 0,
            "candidate_source_executed": False, "official_scores_changed": False,
            "generation_document_hash": digest(canonical_json(generation)),
            "reports_document_hash": digest(canonical_json(reports)), "arms": arms, "samples": rows,
            "limitations": ["Valid-output-only accuracy has selection bias; always report its denominator and missing cases.",
                            "Invalid output is not proof that the underlying function is semantically wrong or right.",
                            "Stage/exception text is untrusted diagnostic evidence, not authenticated attribution.",
                            "Timeouts without further evidence are not attributed to model slowness.",
                            "No last-line parsing, source repair, rescoring on new inputs, or training occurs here.",
                            "Unrecorded work lost to controller interruption is outside these persisted-record counts."]}


def main():
    from .cli import create_run_directory, write_private
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="existing generation.json/reports.json directory")
    parser.add_argument("--out", required=True, help="new sidecar directory; never overwrite a run")
    args = parser.parse_args()
    try:
        generation = json.loads((args.run / "generation.json").read_text())
        reports = json.loads((args.run / "reports.json").read_text())
        result = diagnose_run(generation, reports)
        result["source_directory"] = str(args.run)
        directory = create_run_directory(args.out)
        write_private(directory / "diagnostics.json", canonical_json(result))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(canonical_json(result["arms"]))
    print("Official scores unchanged; no model/candidate execution.")


if __name__ == "__main__":
    main()
