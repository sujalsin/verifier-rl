"""Summarize saved smoke evidence without executing or repairing candidate code."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .suites import canonical_json, digest


def summarize(model, evaluations):
    samples = model["before"] + model["after"]
    if len(evaluations) != len(samples):
        raise ValueError("missing evaluation records")
    candidates = []
    total_seconds = 0.0
    execution_count = 0
    for index, (sample, report) in enumerate(zip(samples, evaluations)):
        if digest(sample["source"]) != report["candidate_hash"]:
            raise ValueError("evaluation/source hash mismatch")
        unique_inputs = {}
        scores = []
        for suite in report["suites"]:
            scores.append({"suite": suite["suite"], "passed": suite["passed_count"],
                           "total": suite["total"], "reward": suite["reward"],
                           "infrastructure_errors": suite["infrastructure_errors"],
                           "reasons": dict(Counter(o["reason"] for o in suite["outcomes"]))})
            for outcome in suite["outcomes"]:
                unique_inputs.setdefault(outcome["input_hash"], outcome["attempts"])
        for attempts in unique_inputs.values():
            execution_count += len(attempts)
            total_seconds += sum(a["metadata"].get("total_seconds", 0) for a in attempts)
        candidates.append({"phase": "before" if index < len(model["before"]) else "after",
                           "seed": sample["seed"], "hit_token_cap": sample["hit_token_cap"],
                           "scores": scores})
    return {
        "run_id": model["run_id"], "kind": "integration_smoke_not_efficacy_study",
        "model_id": model["config"]["model_id"],
        "model_revision": model["config"]["model_revision"],
        "metrics": model["metrics"], "candidates": candidates,
        "evaluation_executions": execution_count,
        "evaluation_summed_invocation_seconds": total_seconds,
        "evaluation_mean_invocation_seconds": total_seconds / execution_count if execution_count else None,
        "limitations": [
            "Two samples per checkpoint cannot establish improvement or task difficulty.",
            "Development audit is not an untouched final evaluation.",
            "Invocation timing is not billed runtime; billing may lag.",
            "Zero reward does not by itself distinguish format, execution, or logical failure.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(canonical_json(summarize(json.loads((args.run / "model.json").read_text()),
                                   json.loads((args.run / "evaluation.json").read_text()))))


if __name__ == "__main__":
    main()
