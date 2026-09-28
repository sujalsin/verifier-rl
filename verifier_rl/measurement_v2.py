"""Frozen CPU-only remeasurement of the existing eight untouched-1.5B samples."""

from .checkpoint_pilot import MODEL_ID, REVISION
from .diagnostics import diagnose_run, diagnose_sample
from .model_trial import validate_submission
from .modal_backend import RUNNER
from .partial_reward import balanced_scores
from .reward_v2 import behavior_scores, behavior_suite
from .suites import canonical_json, digest

GENERATION_RUN = "qwen-checkpoint-pilot-20260927T234301-fc4ee586"
REFERENCE_RUN = "qwen-checkpoint-recovery-20260927T235622-8129215d"
PARAMETER_HASH = "0e7b709defb712125da1ce7b7e307b56bd00bd889074107f0aa6c56184a7e8ac"
PROMPT_HASH = "df1f13179b5192e02418e6e7ab3c815904581e6ff760f95277b585f5303d1c81"
SEEDS = tuple(range(4000, 4008))
VERSION = "saved-1.5b-behavior-measurement-0.1"


def measurement_plan(generation, prior_reports):
    plan = generation["plan"]
    if (generation["run_id"] != GENERATION_RUN or plan["model_id"] != MODEL_ID
            or plan["revision"] != REVISION or plan["training"] is not False
            or plan["prompt_hash"] != PROMPT_HASH
            or generation["before_parameter_hash"] != PARAMETER_HASH
            or generation["after_parameter_hash"] != PARAMETER_HASH
            or [s["seed"] for s in generation["samples"]] != list(SEEDS)):
        raise ValueError("measurement must use the eight frozen untouched-1.5B samples")
    diagnose_run(generation, prior_reports)  # Validate saved identities, outputs and old scores.
    for sample in generation["samples"]:
        validate_submission(sample)
    suite = behavior_suite()
    return {"version": VERSION, "generation_run_id": GENERATION_RUN, "reference_run_id": REFERENCE_RUN,
            "generation_hash": digest(canonical_json(generation)), "prior_reports_hash": digest(canonical_json(prior_reports)),
            "suite_hash": suite.fingerprint, "suite_version": suite.version, "suite_cases": len(suite.cases),
            "seeds": list(SEEDS), "source_hashes": {str(s["seed"]): digest(s["source"]) for s in generation["samples"]},
            "model_id": MODEL_ID, "revision": REVISION, "training": False, "new_model_samples": 0,
            "max_sandbox_executions": 8 * len({c.input_hash for c in suite.cases}), "concurrency": 4,
            "max_retries": 0, "creation_interval_seconds": .26, "max_controller_seconds": 900,
            "max_candidate_controller_seconds": 180, "runner_hash": digest(RUNNER),
            "no_automatic_training": True}


def require_measurement_report(sample, report, image_id):
    if len(report["suites"]) != 1:
        raise ValueError("v2 remeasurement uses exactly one suite")
    row = diagnose_sample(sample, report, (behavior_suite(),))
    if report["suites"][0]["all_passed"] is None:
        raise ValueError("unresolved measurement infrastructure failure")
    for outcome in report["suites"][0]["outcomes"]:
        for attempt in outcome["attempts"]:
            metadata = attempt["metadata"]
            if (metadata.get("image_id") != image_id or metadata.get("runner_hash") != digest(RUNNER)
                    or metadata.get("cleanup") != "terminated" or metadata.get("block_network") is not True):
                raise ValueError("measurement environment/cleanup mismatch")
    return row


def measurement_summary(generation, prior_reports, reports, image_id):
    plan = measurement_plan(generation, prior_reports)
    if len(reports) != 8 or [r["seed"] for r in reports] != list(SEEDS):
        raise ValueError("all eight v2 reports required in frozen seed order")
    previous = {r["seed"]: r for r in prior_reports}
    rows = []
    for sample, report in zip(generation["samples"], reports):
        require_measurement_report(sample, report, image_id)
        old = next(s for s in previous[sample["seed"]]["suites"] if s["suite"] == "balanced")
        new = report["suites"][0]
        rows.append({"seed": sample["seed"], "source_hash": report["candidate_hash"],
                     "v1": balanced_scores(old), "v2": behavior_scores(new),
                     "v2_passed_cases": new["passed_count"], "v2_total_cases": new["total"]})
    groups = [[r["v2"]["partial"] for r in rows[i:i + 4]] for i in (0, 4)]
    return {"version": VERSION, "plan": plan, "training": False, "new_model_samples": 0,
            "rows": rows, "partial_groups": groups,
            "full_v2_passes": sum(r["v2"]["binary"] for r in rows),
            "mean_v1_partial": sum(r["v1"]["partial"] for r in rows) / 8,
            "mean_v2_partial": sum(r["v2"]["partial"] for r in rows) / 8,
            "mixed_groups_above_edge_allowance": sum(len(set(g)) > 1 and max(g) > .05 for g in groups),
            "recorded_sandbox_executions": sum(r["execution_config"]["attempts"] for r in reports),
            "diagnostics": diagnose_run(generation, reports, (behavior_suite(),)),
            "limitations": ["Same eight programs on different test distributions; score changes are not learning.",
                            "No new generation, optimizer update or independent strict audit.",
                            "A mixed reward group is not proof of a good learning signal or reward-hacking resistance.",
                            "No automatic follow-up training or extra samples."]}
