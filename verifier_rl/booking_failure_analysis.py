"""Offline inspection of the completed matched pilot; never execute candidates.

Consumes a targeted local export and the completed finalizer receipt. Replays
the frozen evidence validators, not candidate programs, and preserves unknowns.
This diagnostic is not a new training/evaluation protocol.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import booking_baseline_comparison as study, booking_matched_training as release
from .evaluation_journal import persist
from .suites import canonical_json, digest


def source_ast_hash(source):
    """Syntax-only comparison: comments/formatting ignored, identifiers retained."""
    try:
        return digest(ast.dump(ast.parse(source), include_attributes=False))
    except SyntaxError:
        return None


def describe_program(sample, entries, image_id):
    observations = []
    for case in study.cases_for("evaluation"):
        outcome, record = study.validate_entry(entries[case.input_hash], sample, case, image_id)
        observations.append({"case_id": case.name, "input_hash": case.input_hash,
            "family": case.family, "arguments": case.arguments, "expected": case.expected,
            "observed": study.contrast._observed_integer(record) if record else None,
            "shared_endpoint": study.contrast.has_shared_endpoint(case), **outcome})
    failed = [o for o in observations if o["passed"] is False]
    witness = min(failed, key=lambda o: (len(o["arguments"]["bookings"]), o["input_hash"]), default=None)
    return {"sample_id": sample["sample_id"], "source": sample["source"],
        "source_hash": digest(sample["source"]), "ast_hash": source_ast_hash(sample["source"]),
        "unique_inputs": len(observations), "unknown_inputs": sum(o["passed"] is None for o in observations),
        "inclusive_matches": sum(o["inclusive_match"] is True for o in observations),
        "failed_inputs": len(failed), "failure_families": dict(Counter(o["family"] for o in failed)),
        "all_observed_failures_have_shared_endpoints": bool(failed) and all(o["shared_endpoint"] for o in failed),
        "smallest_saved_witness": witness, "observations": observations}


def analyze(directory):
    directory = Path(directory)
    read = lambda path: json.loads(path.read_text())
    result, receipt = (read(directory / f"{name}.json") for name in ("verified_final_result", "verification_receipt"))
    if receipt.get("all_original_checks_passed") is not True or receipt["result_sha256"] != digest(canonical_json(result)):
        raise ValueError("verified final result binding changed")
    evidence = directory / "evidence"
    root = evidence / release.RUN_ID
    plan, setup = read(root / "plan.json"), read(root / "setup.json")
    snapshot = read(root / "amendments/cleanup-001/source_snapshot.json")
    binding = receipt["binding"]
    release.validate_plan(plan)
    for key, value in (("plan_hash", plan), ("setup_hash", setup), ("source_snapshot_hash", snapshot)):
        if binding[key] != digest(canonical_json(value)):
            raise ValueError("finalizer evidence binding changed: " + key)
    if result["plan_hash"] != binding["plan_hash"] or binding["run_id"] != release.RUN_ID:
        raise ValueError("different experiment")
    repository = Path(__file__).resolve().parents[1]
    for name, source in snapshot.items():
        if (repository / name).read_text() != source:
            raise ValueError("frozen implementation changed: " + name)
    manifest = read(directory / "download_manifest.json")
    for item in manifest:
        data = (evidence / item["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("downloaded evidence changed: " + item["path"])

    baseline = read(root / "baseline.json")
    release.validate_baseline((root / "baseline.json").read_bytes(), baseline, plan)
    if result["summary"] != release.comparison_summary(baseline, result["programs"]):
        raise ValueError("final summary changed")
    generated = read(evidence / plan["baseline_run_id"] / "generation/result.json")
    if digest(canonical_json(generated)) != baseline["generation_hash"]:
        raise ValueError("baseline generation changed")
    sources = {s["sample_id"]: s for s in generated["samples"]}
    rows = {r["sample_id"]: r for r in baseline["programs"]}
    rows.update((r["sample_id"], r) for group in result["programs"].values() for r in group)
    selected, verified_batches = [], {}
    for policy, group in result["programs"].items():
        targets = [r for r in group if r["weak_only_acceptance_bounds"][1]]
        if not targets:
            continue
        arm, step = policy.rsplit("-", 1)
        evaluation = read(root / f"arms/{arm}/evaluation-{step}/result.json")
        sources.update((s["sample_id"], s) for s in evaluation["samples"])
        for row in targets:
            sample = sources[row["sample_id"]]
            index = study.SEEDS.index(sample["seed"])
            key = f"eval-{policy}-{index // study.GROUP:02d}"
            raw = read(root / f"grading/{key}/raw.json")
            if key not in verified_batches:
                start = study.GROUP * (index // study.GROUP)
                checked = study.verify_raw(raw, evaluation["samples"][start:start+study.GROUP], key,
                    "evaluation", plan["experiment"], setup["sandbox_image_id"])
                if any(r != rows[r["sample_id"]] for r in checked["rows"]):
                    raise ValueError("replayed batch differs from final report")
                verified_batches[key] = digest(canonical_json(raw))
            selected.append({"policy": policy, "scores": row,
                **describe_program(sample, raw["entries"][sample["sample_id"]], setup["sandbox_image_id"])})

    # Matched sampling seeds are descriptive comparisons, not program lineages.
    comparisons = []
    for item in selected:
        if item["policy"] != "endpoint_omission-24":
            continue
        seed = sources[item["sample_id"]]["seed"]
        comparison = {"seed": seed, "policies": {}}
        for arm, step in (("baseline", 0), ("reference", 24), ("endpoint_omission", 24)):
            sid = f"eval-{arm}-{step:02d}-{seed}"
            sample, row = sources[sid], rows[sid]
            if digest(sample["source"]) != row["source_hash"]:
                raise ValueError("comparison source differs from evaluated program")
            comparison["policies"][arm] = {"sample_id": sid, "source": sample["source"],
                "source_hash": row["source_hash"], "ast_hash": source_ast_hash(sample["source"]),
                "reference_passed": row["reference"]["passed_bounds"],
                "audit_passed": row["audit"]["passed_bounds"]}
        comparisons.append(comparison)
    return {"version": "booking-failure-analysis-0.1", "run_id": release.RUN_ID,
        "result_sha256": receipt["result_sha256"], "selection": "all possibly weak-only accepted post-training evaluation programs",
        "verified_batch_hashes": verified_batches, "selected_programs": selected,
        "same_seed_comparisons": comparisons, "policy_summary": result["summary"]["policies"],
        "candidate_executions": 0, "optimizer_updates": 0,
        "limitations": ["One task and one paired training seed; 32 generated programs per policy.",
            "Audit cases are development tests and correlated within programs.",
            "Outcome bounds are missing-data bounds, not statistical confidence intervals.",
            "Matching an inclusive oracle on these inputs does not prove universal correctness or intent.",
            "Same-seed outputs are not evidence that training edited a particular source file."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Local targeted evidence export; no network calls")
    args = parser.parse_args()
    result = analyze(args.directory)
    persist(args.directory, {"failure_analysis": result})
    print(canonical_json({"programs": len(result["selected_programs"]),
        "batches": len(result["verified_batch_hashes"]), "output": str(args.directory / "failure_analysis.json")}))


if __name__ == "__main__":
    main()
