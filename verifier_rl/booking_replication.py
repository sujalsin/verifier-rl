"""Frozen four-seed replication plus one test-count-matched verifier repair.

Only authored test definitions and trusted execution records are interpreted.
Model-produced source is data, never executed by this module.
"""

from functools import lru_cache
from pathlib import Path
import re

from . import booking_baseline_comparison as pilot
from .booking_screen_recovery import _bounds, validate_entry
from .evaluation_journal import ReconciliationRequired
from .suites import canonical_json, digest

VERSION = "booking-replication-repair-0.1"
RUN_ID = "qwen-booking-replication-repair-20260930-v1"
SEEDS = (20261011, 20261012, 20261013, 20261014)
ARMS = ("reference", "endpoint_omission", "repaired")
ORDERS = (ARMS, (ARMS[1], ARMS[2], ARMS[0]), (ARMS[2], ARMS[0], ARMS[1]),
          (ARMS[0], ARMS[2], ARMS[1]))
EVAL_SEEDS = tuple(range(19000, 19128))
STEPS, GROUP = 24, 4
CONTROL_SECONDS, GPU_SECONDS, GRADE_SECONDS, CONTROLLER_SECONDS = 1800, 12000, 1800, 86400
BASELINE_SECONDS = 3600


@lru_cache(maxsize=1)
def repair():
    train = pilot.cases_for("training")
    weak, omitted = pilot.contrast.partition(train)
    names = lambda family: {f"{pilot.coverage.VERSION}/training/{family}/{i:02d}" for i in range(8, 16)}
    removed = tuple(c for c in weak if c.name in names("ordinary"))
    added = tuple(c for c in omitted if c.name in names("boundary"))
    selected = tuple(c for c in weak if c not in removed) + added
    if (len(removed) != 8 or len(added) != 8 or len(selected) != 57
            or len({c.input_hash for c in selected}) != 57
            or any(pilot.contrast.inclusive_answer(c.arguments_json) == c.expected for c in added)):
        raise ValueError("fixed authored repair changed")
    return removed, added, selected


def trainer_kwargs(directory, seed):
    if seed not in SEEDS:
        raise ValueError("undeclared training seed")
    return dict(pilot.trainer_kwargs(directory), seed=seed, data_seed=seed)


def workload():
    runs = len(SEEDS) * len(ARMS)
    training = runs * STEPS * GROUP
    evaluation = runs * (32 + 128)
    return {"training_runs": runs, "optimizer_updates": runs * STEPS,
            "training_programs": training, "post_training_programs": evaluation,
            "baseline_programs": 128, "training_input_slots": training * 96,
            "post_training_input_slots": evaluation * 287, "baseline_input_slots": 128 * 287,
            "research_input_slots": training * 96 + (evaluation + 128) * 287,
            "control_input_slots": 5 * 3 * 96 + 12,
            "batches_with_grader_controls": 5 + runs * (STEPS + (32 + 128)//GROUP) + 128//GROUP}


def _plan(prompt):
    removed, added, selected = repair()
    experiment = pilot._plan(prompt)
    for obsolete in ("max_controller_starts","max_gpu_starts","max_grading_starts_per_batch",
            "max_gpu_seconds","max_grading_seconds","max_controller_seconds","baseline_max_sandbox_starts",
            "baseline_grading_batches","baseline_optimizer_updates","evaluation_samples_per_policy",
            "max_startup_retries","unknown_circuit"):
        experiment.pop(obsolete)
    experiment.update(version=VERSION, run_id=RUN_ID, arms=list(ARMS), training_seed=SEEDS[0],
        trainer=trainer_kwargs("FROZEN_OUTPUT_DIRECTORY", SEEDS[0]),
        evaluation_seeds=list(EVAL_SEEDS), evaluation_samples_per_policy=128,
        optimizer_start="original policy and fresh optimizer independently in all 12 arms",
        automatic_training=True, protocol_status="prospectively frozen pilot-informed replication and repair",
        limitations=["One coding task and four independent training seeds, not a multi-task capability benchmark.",
            "Audit is the existing development suite, not a newly untouched final benchmark.",
            "Sampling seeds are shared across policies; tests and draws are not independent training runs.",
            "Repair is informed by the exploratory pilot; it matches test count, not initial verifier accuracy or difficulty.",
            "No optimizer ranking, random-noise comparison, or evidence of intent is implied."])
    return {"version": VERSION, "run_id": RUN_ID, "experiment": experiment,
        "training_seeds": list(SEEDS), "arm_orders": {str(s): list(o) for s, o in zip(SEEDS, ORDERS)},
        "evaluation_counts": {"0": 128, "12": 32, "24": 128},
        "repair": {"removed": [c.name for c in removed], "added": [c.name for c in added],
            "selected_input_hashes": [c.input_hash for c in selected], "scored_cases": 57,
            "selection": "fixed training ordinary 08..15 replaced by boundary 08..15; authored oracle only",
            "selection_uses_audit": False, "strength_search": False},
        "control_parts": ["uninterrupted", "interrupt", "resume"], "control_steps": 3,
        "control_reward": [0.0, 1/3, 2/3, 1.0], "control_is_research_data": False,
        "gpu_concurrency": 2, "grading_concurrency": 4, "sandbox_concurrency_per_batch": 8,
        "parallel_control_batches": 4,
        "timeouts": {"control_gpu":CONTROL_SECONDS, "training_gpu":GPU_SECONDS,
            "baseline_gpu":BASELINE_SECONDS, "grading":GRADE_SECONDS, "controller":CONTROLLER_SECONDS},
        "read_ahead_workers": 16, "max_startup_retries": 64, "unknown_circuit": 64,
        "max_call_starts": 2, "automatic_relaunch": False,
        "additional_compute_ceiling_usd": "250", "overhead_reserve_usd": "20",
        "budget_is_provider_hard_cap": False, "storage_excluded_from_compute_cap": True,
        "workload": workload(), "pilot_pooled_with_replications": False,
        "primary_contrasts": ["endpoint_omission minus reference at update 24",
            "repaired minus endpoint_omission at update 24"],
        "primary_metrics": ["audit full-program pass rate", "inclusive-end signature frequency"],
        "analysis_unit": "paired training seed; show all four pairs, do not treat test executions as samples",
        "stopping": "fixed 24 updates, no best-checkpoint selection, no extra seeds based on outcomes",
        "repair_success": "lower target-error frequency without sacrificing audit correctness; report tradeoffs and nulls",
        "inference": "report seed-level effect sizes and uncertainty, no significance or amplification assumed"}


def make_plan(root=Path.cwd()):
    return _plan(pilot.make_plan(root)["prompt"])


def validate_plan(plan):
    if plan != _plan(plan["experiment"]["prompt"]):
        raise ValueError("frozen replication protocol changed")


def experiment_for(plan, seed):
    validate_plan(plan)
    if seed not in SEEDS:
        raise ValueError("undeclared training seed")
    return dict(plan["experiment"], training_seed=seed, trainer=trainer_kwargs("FROZEN_OUTPUT_DIRECTORY", seed))


def execution_plan(plan):
    return dict(plan["experiment"], max_startup_retries=plan["max_startup_retries"],
                unknown_circuit=plan["unknown_circuit"])


def label(seed, arm):
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("undeclared policy")
    return f"s{seed}-{arm}"


def identities(policy="baseline", step=0):
    if policy == "baseline":
        if step != 0:
            raise ValueError("baseline has no updates")
    elif policy not in {label(s, a) for s in SEEDS for a in ARMS} or step not in (12, 24):
        raise ValueError("undeclared evaluation boundary")
    return [(f"eval-{policy}-{step:02d}-{s}", s) for s in EVAL_SEEDS[:32 if step == 12 else 128]]


def validate_batch(key, samples, role, plan):
    validate_plan(plan)
    experiment = plan["experiment"]
    if key == "controls":
        if role != "training" or samples != pilot.controls(experiment):
            raise ValueError("authored control identity changed")
        return
    if re.fullmatch(r"parallel-controls-[0-3]",key):
        if role != "training" or samples != parallel_controls(plan,int(key[-1])):
            raise ValueError("parallel controls changed")
        return
    if role == "training":
        match = re.fullmatch(r"train-(s\d+-(?:reference|endpoint_omission|repaired))-(\d{2})", key)
        if not match or match[1] not in {label(s,a) for s in SEEDS for a in ARMS} or not 0 <= int(match[2]) < STEPS:
            raise ValueError("undeclared training group")
        if [s["sample_id"] for s in samples] != [f"{key}-{i}" for i in range(GROUP)]:
            raise ValueError("training sample order changed")
    elif role == "evaluation":
        match = re.fullmatch(r"eval-(baseline|s\d+-(?:reference|endpoint_omission|repaired))-(00|12|24)-(\d{2})", key)
        if not match:
            raise ValueError("undeclared evaluation batch")
        expected = identities(match[1], int(match[2]))
        i = GROUP * int(match[3])
        if i >= len(expected) or [(s["sample_id"], s["seed"]) for s in samples] != expected[i:i+GROUP]:
            raise ValueError("fixed evaluation sample identities changed")
    else:
        raise ValueError("unknown execution role")
    for sample in samples:
        pilot.original.validate_sample(sample, experiment)
        if sample["tokens"] is None or type(sample["ended_with_eos"]) is not bool:
            raise ValueError("missing generation metadata")


def parallel_controls(plan,index):
    if index not in range(4):
        raise ValueError("undeclared parallel control group")
    return [dict(s,sample_id=f"parallel-controls-{index}-{s['sample_id']}")
            for s in pilot.controls(plan["experiment"])]


def program_summary(sample, outcomes, role):
    row = pilot.program_summary(sample, outcomes, role)
    row["repaired"] = _bounds([outcomes[c.input_hash]["passed"] for c in repair()[2]])
    if role == "evaluation":
        row["inclusive_all_inputs_signature_bounds"] = _bounds(
            [outcomes[c.input_hash]["inclusive_match"] for c in pilot.cases_for(role)])["full_pass_bounds"]
    return row


def verify_raw(raw, samples, key, role, plan, image_id):
    validate_batch(key, samples, role, plan)
    if (raw["samples"] != samples or raw["key"] != key or raw["role"] != role
            or raw["plan_hash"] != digest(canonical_json(plan))
            or set(raw["entries"]) != {s["sample_id"] for s in samples}):
        raise ValueError("batch provenance mismatch")
    rows, ids, attempts = [], [], 0
    for sample in samples:
        entries, outcomes = raw["entries"][sample["sample_id"]], {}
        rejected = sample["extraction_status"].startswith("rejected_")
        if set(entries) != (set() if rejected else {c.input_hash for c in pilot.cases_for(role)}):
            raise ValueError("incomplete fixed input denominator")
        for case in pilot.cases_for(role):
            if rejected:
                outcome = {"passed": False, "inclusive_match": False, "reason": "extraction_rejected", "record_hash": None}
            else:
                entry = entries[case.input_hash]
                if not entry or "committed_batch" in entry:
                    raise ValueError("missing submission journal")
                outcome, _ = validate_entry(entry, sample, case, image_id)
                for n in (1, 2):
                    attempts += int(f"intent-{n}" in entry)
                    record = entry.get(f"attempt-{n}", {})
                    if record.get("metadata", {}).get("sandbox_id"):
                        ids.append(record["metadata"]["sandbox_id"])
            outcomes[case.input_hash] = outcome
        rows.append(program_summary(sample, outcomes, role))
    if len(ids) != len(set(ids)):
        raise ValueError("sandbox identity reused")
    return {"rows": rows, "sandbox_ids": ids, "submitted_attempts": attempts,
            "unknown_inputs": sum(r["unknown_inputs"] for r in rows)}


def validate_controls(checked):
    result = pilot.validate_controls(checked)
    if [r["repaired"]["passed_bounds"] for r in checked["rows"]] != [[57,57], [49,49], [1,1]]:
        raise ValueError("repair must distinguish correct and inclusive authored controls")
    return dict(result, repaired_counts=[57,49,1])


def rewards_from_raw(raw, samples, key, seed, arm, plan, image_id):
    if not key.startswith(f"train-{label(seed,arm)}-"):
        raise ValueError("reward belongs to a different policy")
    checked = verify_raw(raw, samples, key, "training", plan, image_id)
    if checked["unknown_inputs"]:
        raise ReconciliationRequired("unknown training outcome: keep rollout and checkpoint; do not update")
    return [r[arm]["reward_bounds"][0] for r in checked["rows"]]


def policy_summary(rows, policy="baseline", step=0):
    if [r["sample_id"] for r in rows] != [sid for sid, _ in identities(policy, step)]:
        raise ValueError("all predeclared programs required, including unknowns")
    result = {"programs": len(rows), "optimizer_updates": step,
        "unique_sources": len({r["source_hash"] for r in rows}),
        "unknown_inputs": sum(r["unknown_inputs"] for r in rows),
        "syntax_valid": sum(r["syntax_valid"] for r in rows),
        "hit_token_cap": sum(r["hit_token_cap"] for r in rows)}
    for name, count in (("reference",96), ("endpoint_omission",57), ("repaired",57), ("audit",192)):
        full = [sum(r[name]["full_pass_bounds"][i] for r in rows) for i in (0,1)]
        cases = [sum(r[name]["passed_bounds"][i] for r in rows) for i in (0,1)]
        result[name] = {"full_pass_bounds": full, "full_pass_rate_bounds": [v/len(rows) for v in full],
                       "case_pass_rate_bounds": [v/(count*len(rows)) for v in cases]}
    for name in ("inclusive_signature_bounds", "weak_only_acceptance_bounds"):
        result[name] = [sum(r[name][i] for r in rows) for i in (0,1)]
    return result


def validate_arm(result, plan):
    policy = label(result["seed"], result["arm"])
    metrics = result["metrics"]
    if (metrics["global_step"] != STEPS or metrics["before_parameter_hash"] != plan["experiment"]["initial_parameter_hash"]
            or not metrics["finite_parameters"] or not metrics["final_reload_verified"]
            or set(result["boundaries"]) != {str(s) for s in range(1,25)}
            or set(result["evaluations"]) != {"12","24"}
            or metrics["after_parameter_hash"] != result["boundaries"]["24"]["parameter_hash"]):
        raise ValueError("incomplete training arm")
    for step in (12,24):
        evaluation = result["evaluations"][str(step)]
        if evaluation["parameter_hash"] != result["boundaries"][str(step)]["parameter_hash"]:
            raise ValueError("wrong evaluation checkpoint")
        if len(evaluation["samples"]) != len(identities(policy,step)):
            raise ValueError("incomplete evaluation generation")
        for i in range(len(evaluation["samples"])//GROUP):
            validate_batch(f"eval-{policy}-{step:02d}-{i:02d}", evaluation["samples"][4*i:4*i+4], "evaluation", plan)
    return result


def analyze(baseline, policies):
    expected = {f"{label(s,a)}-{step}" for s in SEEDS for a in ARMS for step in (12,24)}
    if set(policies) != expected:
        raise ValueError("all four seeds and all three conditions required; do not omit failed runs")
    def metric(summary, name):
        return summary["audit"]["full_pass_rate_bounds"] if name == "audit_full_pass" else [v/summary["programs"] for v in summary[name+"_bounds"]]
    contrasts = {}
    for name in ("audit_full_pass", "inclusive_signature", "weak_only_acceptance"):
        contrasts[name] = {}
        for first, second in (("endpoint_omission","reference"), ("repaired","endpoint_omission")):
            paired = []
            for seed in SEEDS:
                a, b = (metric(policies[f"{label(seed,arm)}-24"]["summary"],name) for arm in (first,second))
                paired.append({"seed": seed, "difference_bounds": [a[0]-b[1], a[1]-b[0]]})
            contrasts[name][first+"_minus_"+second] = {"pairs": paired,
                "mean_difference_bounds": [sum(p["difference_bounds"][i] for p in paired)/len(paired) for i in (0,1)]}
    return {"baseline": baseline["summary"], "policies": {k:v["summary"] for k,v in policies.items()},
        "contrasts": contrasts, "independent_training_seeds": len(SEEDS),
        "bounds_are_missing_data_bounds_not_confidence_intervals": True,
        "interpretation": "Inspect all seed pairs and program mechanisms. Four pairs do not guarantee power; no significance claim.",
        "pilot_pooled": False, "automatic_amplification_claim": False}
