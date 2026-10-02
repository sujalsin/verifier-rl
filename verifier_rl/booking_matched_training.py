"""Release bounds and evidence checks for the frozen two-arm training study."""

from decimal import Decimal
import hashlib
from pathlib import Path

from . import booking_baseline_comparison as experiment
from .grpo_recovery import VERSION as RECOVERY_VERSION
from .suites import canonical_json, digest

VERSION = "booking-matched-training-0.3"
RUN_ID = "qwen-booking-matched-training-20260929-v1"
CONTROL_ID = "qwen-booking-full-state-control-20260929-v3"
BASELINE_SHA256 = "990f7bd577b07959859974ff311a56e721361acf93ba864eea05bfee30d3e8b8"
CONTROL_SECONDS, TRAIN_SECONDS, GRADE_SECONDS, CONTROLLER_SECONDS = 1800, 9000, 1800, 43200
CONTROL_STEPS = 3
MAX_SANDBOX_STARTS = 12 + 3*96 + 2*24*4*96 + 2*2*32*287 + 16


def make_plan(root):
    return _plan(experiment.make_plan(root))


def _plan(experiment_plan):
    return {"version": VERSION, "run_id": RUN_ID, "control_id": CONTROL_ID,
            "experiment": experiment_plan, "recovery_version": RECOVERY_VERSION,
            "baseline_run_id": experiment.RUN_ID, "baseline_sha256": BASELINE_SHA256,
            "max_sandbox_starts": MAX_SANDBOX_STARTS, "control_steps": CONTROL_STEPS,
            "control_parts": ["uninterrupted", "interrupt", "resume"],
            "control_seconds_per_part": CONTROL_SECONDS, "train_seconds_per_call": TRAIN_SECONDS,
            "max_train_starts_per_arm": 2, "max_grading_starts_per_batch": 2,
            "grading_seconds": GRADE_SECONDS, "controller_seconds": CONTROLLER_SECONDS,
            "max_controller_starts": 2, "grading_batches": 1 + 2*24 + 2*2*8,
            "control_reward": [0.0, 1/3, 2/3, 1.0],
            "control_interruption": "after group 1 reward is persisted, before its optimizer update",
            "control_is_research_data": False, "new_baseline_samples": 0,
            "runtime_amendment": "deterministic CUDA training for exact recovery; fixed logging directory",
            "evaluation_runtime": "original nondeterministic-algorithm flag; restore training flag afterwards",
            "automatic_sampling_expansion": False}


def validate_plan(plan):
    experiment.validate_plan(plan["experiment"])
    expected = _plan(plan["experiment"])
    if plan != expected:
        raise ValueError("training release/configuration changed")


def execution_plan(plan):
    # Execution namespacing is separate from the immutable research contract.
    return dict(plan["experiment"], run_id=plan["run_id"])


def validate_baseline(data, report, plan):
    if hashlib.sha256(data).hexdigest() != plan["baseline_sha256"]:
        raise ValueError("baseline anchor changed")
    if (report["plan_hash"] != digest(canonical_json(plan["experiment"]))
            or report["optimizer_updates"] != 0
            or report["summary"] != experiment.policy_summary(report["programs"])):
        raise ValueError("baseline report differs from frozen experiment")
    return report


def validate_control(parts, plan):
    if set(parts) != set(plan["control_parts"]):
        raise ValueError("all three live control parts required")
    reference, interrupted, resumed = (parts[n] for n in plan["control_parts"])
    if (reference["steps"] != 3 or interrupted["steps"] != 1 or resumed["steps"] != 3
            or interrupted["interruption_observed"] is not True
            or resumed["replayed_groups"] != [1]
            or reference["replayed_groups"] != [] or interrupted["replayed_groups"] != []):
        raise ValueError("interruption/replay control incomplete")
    fields = ("parameter_hash", "optimizer_hash", "scheduler_hash", "rng_hash", "trl_step")
    for step in (1, 2, 3):
        if reference["boundaries"][str(step)]["trl_step"] != 4*step:
            raise ValueError("control did not use frozen gradient accumulation")
        for field in fields:
            if reference["boundaries"][str(step)][field] != resumed["boundaries"][str(step)][field]:
                raise ValueError(f"resume control diverged: step {step}, {field}")
    for group in range(3):
        for field in ("tokens_hash", "reward_hash", "after_rng_hash"):
            if reference["groups"][str(group)][field] != resumed["groups"][str(group)][field]:
                raise ValueError(f"resume rollout/reward diverged: group {group}, {field}")
    if (reference["boundaries"]["1"]["parameter_hash"] == plan["experiment"]["initial_parameter_hash"]
            or not reference["nonzero_gradient_steps"] or not resumed["nonzero_gradient_steps"]):
        raise ValueError("control did not demonstrate a real optimizer update")
    return {"passed": True, "version": RECOVERY_VERSION, "steps_compared": [1, 2, 3],
            "pending_group_replayed": 1, "fresh_next_group_equal": True,
            "weights_optimizer_scheduler_rng_equal": True, "coding_learning_claim": False}


def quote(plan, rates, billing):
    def rate(name):
        value = Decimal(str(rates[name]))
        if not value.is_finite() or value < 0:
            raise ValueError("invalid provider rate")
        return value
    gpu_seconds = 3*CONTROL_SECONDS + 2*plan["max_train_starts_per_arm"]*TRAIN_SECONDS
    cpu_seconds = 3*CONTROL_SECONDS + 300 + plan["max_controller_starts"]*CONTROLLER_SECONDS + plan["grading_batches"]*2*GRADE_SECONDS
    sandbox = Decimal(MAX_SANDBOX_STARTS*120)/3600 * (rate("cpu_hour_cost_sandbox") + rate("mem_gib_hour_cost_sandbox")/4)
    gpu = Decimal(gpu_seconds)/3600 * (rate("gpu_hour_cost_l40s") + 2*rate("cpu_hour_cost") + 32*rate("mem_gib_hour_cost"))
    cpu = 3*Decimal(cpu_seconds)/3600 * (rate("cpu_hour_cost") + 2*rate("mem_gib_hour_cost"))
    return {"rates": rates, "billing_before": billing, "max_compute_envelope_usd": str(sandbox+gpu+cpu),
            "sandbox_envelope_usd": str(sandbox), "gpu_envelope_usd": str(gpu), "cpu_envelope_usd": str(cpu),
            "not_invoice_or_expected_spend": True, "provider_hard_cap": False,
            "storage_and_other_account_activity_excluded": True,
            "scope": "three bounded recovery-control calls; two 24-update arms; four fixed 32-program evaluations"}


def validate_arm(result, plan):
    arm, metrics = result["arm"], result["metrics"]
    if (arm not in experiment.ARMS or metrics["global_step"] != 24
            or metrics["before_parameter_hash"] != plan["experiment"]["initial_parameter_hash"]
            or metrics["finite_parameters"] is not True or metrics["final_reload_verified"] is not True
            or set(result["evaluations"]) != {"12", "24"} or set(result["boundaries"]) != {str(i) for i in range(1,25)}):
        raise ValueError("incomplete matched training arm")
    if metrics["after_parameter_hash"] != result["boundaries"]["24"]["parameter_hash"]:
        raise ValueError("final weights differ from checkpoint")
    for step in (12, 24):
        evaluation = result["evaluations"][str(step)]
        if evaluation["parameter_hash"] != result["boundaries"][str(step)]["parameter_hash"]:
            raise ValueError("evaluation used different weights")
        if len(evaluation["samples"]) != 32:
            raise ValueError("fixed 32-sample evaluation required")
        for i in range(8):
            experiment.validate_batch(f"eval-{arm}-{step:02d}-{i:02d}", evaluation["samples"][4*i:4*i+4],
                                      "evaluation", plan["experiment"])
    return result


def comparison_summary(baseline, policies):
    expected = {f"{arm}-{step}" for arm in experiment.ARMS for step in (12,24)}
    if set(policies) != expected:
        raise ValueError("four complete post-training policy measurements required")
    summary = {"baseline": baseline["summary"]}
    for arm in experiment.ARMS:
        for step in (12, 24):
            summary[f"{arm}-{step}"] = experiment.policy_summary(policies[f"{arm}-{step}"], arm, step)
    def difference(first, second):
        return [first[0]-second[1], first[1]-second[0]]
    final = {}
    for name, key in (("audit_full_pass", "audit"), ("inclusive_signature", None), ("weak_only_acceptance", None)):
        def bounds(policy):
            return (summary[policy][key]["full_pass_rate_bounds"] if key else
                    [v/32 for v in summary[policy][name+"_bounds"]])
        final[name] = {arm+"_minus_baseline": difference(bounds(arm+"-24"), bounds("baseline")) for arm in experiment.ARMS}
        final[name]["omission_minus_reference"] = difference(bounds("endpoint_omission-24"), bounds("reference-24"))
    return {"policies": summary, "final_rate_differences": final,
            "amplification_established": False, "automatic_expansion": False,
            "interpretation_required": "One task and one training-seed pair; inspect failures and report nulls/uncertainty."}
