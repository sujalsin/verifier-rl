"""Original-policy verifier comparison: frozen protocol and evidence-only scoring.

The executable first phase measures the original policy. Training is a separate
release, requiring a live full-state resume test; no old warm-start gate is waived.
Candidate source is data and is never executed in this module.
"""

import argparse
from decimal import Decimal
from functools import lru_cache
import json
from pathlib import Path
import re

from . import booking_study as original, booking_verifier_v2 as coverage
from . import booking_boundary_contrast as contrast, supervised_execution as supervised
from .booking_screen_recovery import validate_entry, _bounds
from .evaluation_journal import ReconciliationRequired
from .suites import canonical_json, digest
from .task_panel import BOOKING, prompt_for

VERSION = "booking-baseline-comparison-0.1"
RUN_ID = "qwen-booking-baseline-comparison-20260929-v1"
ARMS = ("reference", "endpoint_omission")
SEEDS = tuple(range(18000, 18032))
TRAIN_SEED, STEPS, GROUP = 20261003, 24, 4
GPU_SECONDS, GRADE_SECONDS, CONTROLLER_SECONDS = 1500, 1800, 10800
MAX_STARTUP_RETRIES, UNKNOWN_CIRCUIT = 16, 16
SCORING_HASH = "73db78349be855fe843d5d27334899972b12180c1693aac9e51fe092f83f5b24"


@lru_cache(maxsize=2)
def cases_for(role):
    if role not in ("training", "evaluation"):
        raise ValueError("unknown case role")
    cases = coverage.cases_for("training")
    if role == "evaluation":
        cases = (*cases, *coverage.cases_for("audit"))
    # Only the mandatory empty input overlaps. Execute that input once.
    return tuple({c.input_hash: c for c in cases}.values())


def trainer_kwargs(directory):
    return dict(original.trainer_kwargs(directory), seed=TRAIN_SEED, data_seed=TRAIN_SEED,
                save_strategy="steps", save_steps=1, save_only_model=False, save_total_limit=2)


def _plan(prompt):
    if digest(prompt) != original.PROMPT_HASH:
        raise ValueError("booking prompt changed")
    if contrast.suite_manifest(cases_for("training"))["scoring_hash"] != SCORING_HASH:
        raise ValueError("verifier scoring rule changed")
    return {"version": VERSION, "run_id": RUN_ID, "task_id": BOOKING,
        "model_id": original.MODEL_ID, "revision": original.REVISION,
        "initial_parameter_hash": original.PARAMETER_HASH, "parameter_count": original.PARAMETERS,
        "initial_checkpoint": None, "project_sft": False, "prior_project_rl_updates": 0,
        "baseline": "original pinned Qwen Instruct policy; not raw pretrained or step-12 weights",
        "chat_template_hash": original.CHAT_TEMPLATE_HASH, "prompt": prompt,
        "prompt_hash": original.PROMPT_HASH, "arms": list(ARMS), "group_size": GROUP,
        "steps_per_arm": STEPS, "training_seed": TRAIN_SEED, "optimizer_start": "fresh in both arms",
        "trainer": trainer_kwargs("FROZEN_OUTPUT_DIRECTORY"),
        "evaluation_seeds": list(SEEDS), "evaluation_samples_per_policy": len(SEEDS),
        "checkpoint_steps": [0, 12, 24], "checkpoint_selection": "fixed final step 24; never best audit",
        "checkpoint_retention": "last two complete trainer states plus separate step-12/24 model snapshots",
        "max_completion_tokens": 512, "temperature": .8, "top_p": .95, "top_k": 0,
        "repetition_penalty": original.REPETITION_PENALTY,
        "verifier_version": coverage.VERSION, "scoring_hash": SCORING_HASH,
        "suite_hashes": {r: coverage.suite_hash(r) for r in coverage.COUNTS},
        "training_cases": 96, "weak_scored_cases": 57, "audit_cases": 192,
        "unique_evaluation_inputs": len(cases_for("evaluation")), "reward_shape": "linear per-case fraction",
        "audit_access": "baseline after protocol freeze; no audit-derived rewards, tuning or selection",
        "execution_version": supervised.VERSION,
        "runner_hash": digest(supervised.supervised_runner(BOOKING)),
        "startup_retry_version": supervised.STARTUP_RETRY_VERSION,
        "concurrency": 8, "creation_interval_seconds": .26,
        "max_startup_retries": MAX_STARTUP_RETRIES, "unknown_circuit": UNKNOWN_CIRCUIT,
        "max_controller_starts": 2, "max_gpu_starts": 1, "max_grading_starts_per_batch": 2,
        "max_gpu_seconds": GPU_SECONDS, "max_grading_seconds": GRADE_SECONDS,
        "max_controller_seconds": CONTROLLER_SECONDS,
        "baseline_max_sandbox_starts": 12 + 3*96 + len(SEEDS)*len(cases_for("evaluation")) + MAX_STARTUP_RETRIES,
        "baseline_grading_batches": 1 + len(SEEDS)//GROUP,
        "baseline_optimizer_updates": 0, "automatic_training": False, "automatic_expansion": False,
        "training_release_requires": ["live full-state interruption/resume equivalence test",
            "pending rollout/token/reward identity preservation", "same original weights and fresh optimizers",
            "unknown rewards block the update; no zero imputation or replacement sampling"],
        "hypothesis": "Endpoint omission may select inclusive-end errors more strongly than reference verification.",
        "primary_outcomes": ["development-audit full-program pass rate", "inclusive-end signature frequency"],
        "secondary_outcomes": ["audit case pass rate", "weak-only full acceptance", "reference score",
            "syntax validity", "token-cap rate", "unknown outcome bounds", "tokens and resource use"],
        "historical_screen_gate_passed": False,
        "protocol_status": "new exploratory comparison, not promotion of the failed step-12 screen",
        "limitations": ["One task, one paired training seed, 24 updates and 32 draws per policy: exploratory only.",
            "The 192-case audit is development evaluation, not an untouched final benchmark.",
            "Tests within a program are correlated; programs, not executions, are the sampling units.",
            "No random-noise arm, equal-initial-accuracy claim or multi-algorithm ranking.",
            "Full signatures are behavioral evidence, not intent or universal correctness proofs.",
            "A null result must remain a null result; no automatic resampling or longer training."]}


def make_plan(root):
    return _plan(prompt_for(BOOKING, root))


def validate_plan(plan):
    if plan != _plan(plan["prompt"]):
        raise ValueError("frozen original-policy comparison changed")


def identities(policy="baseline", step=0):
    if (policy == "baseline" and step != 0) or (policy != "baseline" and (policy not in ARMS or step not in (12, 24))):
        raise ValueError("policy outside frozen evaluation schedule")
    return [(f"eval-{policy}-{step:02d}-{seed}", seed) for seed in SEEDS]


def controls(plan):
    return [original.sample_from_text(original.controls()[name], sid="control-" + name, plan=plan)
            for name in ("correct", "inclusive", "constant")]


def validate_batch(key, samples, role, plan):
    validate_plan(plan)
    if key == "controls":
        if role != "training" or samples != controls(plan):
            raise ValueError("authored grading controls changed")
        return
    if role == "training":
        match = re.fullmatch(r"train-(reference|endpoint_omission)-(\d{2})", key)
        if not match or not 0 <= int(match[2]) < STEPS:
            raise ValueError("unexpected training group")
        if [s["sample_id"] for s in samples] != [f"{key}-{i}" for i in range(GROUP)]:
            raise ValueError("training group identity changed")
    elif role == "evaluation":
        match = re.fullmatch(r"eval-(baseline|reference|endpoint_omission)-(00|12|24)-(\d{2})", key)
        if not match or not 0 <= int(match[3]) < len(SEEDS)//GROUP:
            raise ValueError("unexpected evaluation group")
        start = GROUP * int(match[3])
        if [(s["sample_id"], s["seed"]) for s in samples] != identities(match[1], int(match[2]))[start:start+GROUP]:
            raise ValueError("fixed evaluation seeds/order changed")
    else:
        raise ValueError("unknown batch role")
    for sample in samples:
        original.validate_sample(sample, plan)
        if sample["tokens"] is None or type(sample["ended_with_eos"]) is not bool:
            raise ValueError("generation metadata missing")


def program_summary(sample, outcomes, role):
    required = cases_for(role)
    if set(outcomes) != {c.input_hash for c in required}:
        raise ValueError("complete fixed input denominator required")
    train = cases_for("training")
    kept, omitted = contrast.partition(train)
    score = lambda cases: _bounds([outcomes[c.input_hash]["passed"] for c in cases])
    ref, weak = score(train), score(kept)
    signature = _bounds([outcomes[c.input_hash]["inclusive_match"] for c in train])["full_pass_bounds"]
    row = {"sample_id": sample["sample_id"], "source_hash": digest(sample["source"]),
        "reference": ref, "endpoint_omission": weak, "omitted": score(omitted),
        "inclusive_signature_bounds": signature,
        "weak_only_acceptance_bounds": [int(weak["full_pass_bounds"][0] and not ref["full_pass_bounds"][1]),
            int(weak["full_pass_bounds"][1] and not ref["full_pass_bounds"][0])],
        "unknown_inputs": sum(outcomes[c.input_hash]["passed"] is None for c in required),
        "unknown_reasons": sorted({o["reason"] for o in outcomes.values() if o["passed"] is None}),
        "syntax_valid": sample["syntax_valid"], "hit_token_cap": sample["hit_token_cap"],
        "extraction_status": sample["extraction_status"]}
    if role == "evaluation":
        row["audit"] = score(coverage.cases_for("audit"))
    return row


def verify_raw(raw, samples, key, role, plan, image_id):
    """Recompute bounds from immutable attempts, never trust an asserted score."""
    validate_batch(key, samples, role, plan)
    if (raw["samples"] != samples or raw["key"] != key or raw["role"] != role
            or raw["plan_hash"] != digest(canonical_json(plan))
            or set(raw["entries"]) != {s["sample_id"] for s in samples}):
        raise ValueError("batch provenance mismatch")
    cases = cases_for(role)
    rows, ids, attempts = [], [], 0
    for sample in samples:
        entries, outcomes = raw["entries"][sample["sample_id"]], {}
        rejected = sample["extraction_status"].startswith("rejected_")
        if set(entries) != (set() if rejected else {c.input_hash for c in cases}):
            raise ValueError("missing/extra per-input journal")
        for case in cases:
            if rejected:
                outcome = {"passed": False, "inclusive_match": False, "reason": "extraction_rejected", "record_hash": None}
            else:
                entry = entries[case.input_hash]
                if not entry or "committed_batch" in entry:
                    raise ValueError("input must be submitted, not silently missing")
                outcome, _ = validate_entry(entry, sample, case, image_id)
                for n in (1, 2):
                    if f"intent-{n}" in entry:
                        attempts += 1
                    record = entry.get(f"attempt-{n}")
                    if record is not None and record["metadata"].get("sandbox_id"):
                        ids.append(record["metadata"]["sandbox_id"])
            outcomes[case.input_hash] = outcome
        rows.append(program_summary(sample, outcomes, role))
    if len(ids) != len(set(ids)):
        raise ValueError("sandbox reused across inputs/programs")
    return {"rows": rows, "sandbox_ids": ids, "submitted_attempts": attempts,
            "unknown_inputs": sum(row["unknown_inputs"] for row in rows)}


def validate_controls(checked):
    rows = checked["rows"]
    if ([r["reference"]["passed_bounds"] for r in rows] != [[96,96], [64,64], [1,1]]
            or [r["endpoint_omission"]["passed_bounds"] for r in rows] != [[57,57], [57,57], [1,1]]
            or rows[1]["inclusive_signature_bounds"] != [1,1] or checked["unknown_inputs"]):
        raise ValueError("live verifier controls failed or unresolved")
    return {"passed": True, "reference_counts": [96,64,1], "weak_counts": [57,57,1]}


def rewards_from_raw(raw, samples, key, arm, plan, image_id):
    if arm not in ARMS or not key.startswith(f"train-{arm}-"):
        raise ValueError("wrong reward arm")
    checked = verify_raw(raw, samples, key, "training", plan, image_id)
    # Do not let omitted unknowns silently allow one arm to train when the other
    # would be blocked. No audit inputs are even accepted by this function.
    if checked["unknown_inputs"]:
        raise ReconciliationRequired("unknown training outcome; preserve group and full checkpoint, do not update")
    return [r[arm]["reward_bounds"][0] for r in checked["rows"]]


def policy_summary(rows, policy="baseline", step=0):
    if [r["sample_id"] for r in rows] != [sid for sid, _ in identities(policy, step)]:
        raise ValueError("all 32 predeclared policy samples required")
    def bounds(key, item):
        return [sum(r[key][item][i] for r in rows) for i in (0, 1)]
    result = {"programs": len(rows), "unique_sources": len({r["source_hash"] for r in rows}),
        "unknown_input_outcomes": sum(r["unknown_inputs"] for r in rows),
        "fully_resolved_programs": sum(not r["unknown_inputs"] for r in rows),
        "syntax_valid": sum(r["syntax_valid"] for r in rows),
        "hit_token_cap": sum(r["hit_token_cap"] for r in rows)}
    for key, total in (("reference",96), ("endpoint_omission",57), ("audit",192)):
        passes = bounds(key, "full_pass_bounds")
        cases = bounds(key, "passed_bounds")
        result[key] = {"full_pass_bounds": passes, "full_pass_rate_bounds": [n/len(rows) for n in passes],
            "case_pass_bounds": cases, "case_pass_rate_bounds": [n/(len(rows)*total) for n in cases]}
    for key in ("inclusive_signature_bounds", "weak_only_acceptance_bounds"):
        result[key] = [sum(r[key][i] for r in rows) for i in (0,1)]
    result["amplification_established"] = False
    result["optimizer_updates"] = 0 if policy == "baseline" else step
    return result


def budget_quote(plan, rates, billing):
    """Bound this phase only; historical unused reservations are not invoices."""
    validate_plan(plan)
    def number(value):
        d = Decimal(str(value))
        if not d.is_finite() or d < 0:
            raise ValueError("invalid resource rate")
        return d
    cpu, mem = number(rates["cpu_hour_cost"]), number(rates["mem_gib_hour_cost"])
    sandbox = plan["baseline_max_sandbox_starts"]*120*(number(rates["cpu_hour_cost_sandbox"])
                + number(rates["mem_gib_hour_cost_sandbox"])/4)/3600
    gpu = GPU_SECONDS*(number(rates["gpu_hour_cost_l40s"])+2*cpu+32*mem)/3600
    seconds = 2*CONTROLLER_SECONDS + 2*plan["baseline_grading_batches"]*GRADE_SECONDS
    controllers = 3*seconds*(cpu+2*mem)/3600
    return {"rates": rates, "billing_before": billing, "sandbox_max_usd": str(sandbox),
        "gpu_max_usd": str(gpu), "controllers_max_usd": str(controllers), "contingency_usd": "2",
        "additional_resource_envelope_usd": str(sandbox+gpu+controllers+2),
        "is_invoice": False, "provider_hard_cap": False,
        "scope": "one fixed 32-sample original-policy baseline and controls; no RL or automatic expansion"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(make_plan(Path.cwd()), indent=2, sort_keys=True))
