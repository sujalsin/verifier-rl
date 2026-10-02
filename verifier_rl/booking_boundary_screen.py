"""Frozen generation-only shared-endpoint feasibility screen; no optimizer."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

from . import booking_boundary_contrast as contrast, booking_reward_pilot as parent
from . import booking_study as original, booking_verifier_v2 as coverage
from . import supervised_execution as supervised, supervisor_controls
from .panel_execution import request_for, unpack_result
from .suites import canonical_json, digest
from .task_panel import BOOKING, prompt_for

VERSION = "booking-boundary-screen-0.1"
RUN_ID = "qwen-booking-boundary-screen-20260929-v1"
SEEDS = tuple(range(17000, 17064))
GROUP = 4
SECONDS, GPU_SECONDS, GRADE_SECONDS = 7200, 1800, 600
SCORING_HASH = "73db78349be855fe843d5d27334899972b12180c1693aac9e51fe092f83f5b24"
GATES = {"minimum_reference_full_passes": 2, "minimum_reference_failures": 2,
         "minimum_distinct_inclusive_signature_sources": 2,
         "minimum_mixed_groups_each_verifier": 4, "minimum_changed_relative_signal_groups": 2}


def _plan(prompt):
    if digest(prompt) != original.PROMPT_HASH:
        raise ValueError("booking prompt changed")
    manifest = contrast.suite_manifest(coverage.cases_for("training"))
    if manifest["scoring_hash"] != SCORING_HASH:
        raise ValueError("endpoint scoring rule changed")
    return {"version": VERSION, "run_id": RUN_ID, "task_id": BOOKING,
        "model_id": original.MODEL_ID, "revision": original.REVISION,
        "initial_checkpoint": parent.WARM_CHECKPOINT, "initial_parameter_hash": parent.WARM_HASH,
        "parameter_count": original.PARAMETERS, "chat_template_hash": original.CHAT_TEMPLATE_HASH,
        "prompt": prompt, "prompt_hash": original.PROMPT_HASH,
        "parent_run_id": parent.RUN_ID, "parent_arm": "linear", "parent_step": 12,
        "population": "saved step-12 model; 12 previous linear-reward updates, not untouched Qwen",
        "seeds": list(SEEDS), "sample_count": len(SEEDS), "group_size": GROUP,
        "max_completion_tokens": 512, "temperature": .8, "top_p": .95, "top_k": 0,
        "repetition_penalty": original.REPETITION_PENALTY,
        "suite_hash": coverage.suite_hash("training"), "scoring_hash": SCORING_HASH,
        "reference_cases": 96, "weak_cases": 57, "audit_cases": 0,
        "execution_version": supervised.VERSION, "startup_retry_version": supervised.STARTUP_RETRY_VERSION,
        "gates": dict(GATES), "optimizer_updates": 0, "automatic_training": False,
        "automatic_expansion": False, "supervisor_controls": 12, "grading_controls": 3,
        "concurrency": 8, "creation_interval_seconds": .26,
        "max_startup_retries": 16, "max_replacements_per_input": 1, "retry_backoff_seconds": 1,
        "max_sandbox_executions": 12 + (3 + len(SEEDS)) * 96 + 16,
        "max_gpu_calls": 1, "max_gpu_seconds": GPU_SECONDS,
        "max_controller_seconds": SECONDS, "max_grading_seconds": GRADE_SECONDS,
        "max_grading_calls": 1 + len(SEEDS) // GROUP,
        "limitations": ["Fresh samples, but the defect and tests were selected using development evidence.",
            "No audit evaluation, task generalization, training improvement or amplification claim.",
            "Consecutive groups are hypothetical GRPO groups, not optimizer updates.",
            "Feasibility thresholds are engineering choices, not statistical significance thresholds.",
            "Inclusive output signatures require source review before causal mechanism claims."]}


def make_plan(root):
    return _plan(prompt_for(BOOKING, root))


def validate_plan(plan):
    if plan != _plan(plan["prompt"]):
        raise ValueError("frozen boundary screen changed")


def identities():
    return [(f"screen-{seed}", seed) for seed in SEEDS]


def controls(plan):
    return [original.sample_from_text(original.controls()[name], sid="control-" + name, plan=plan)
            for name in ("correct", "inclusive", "constant")]


def validate_batch(key, samples, plan):
    validate_plan(plan)
    if key == "controls":
        if samples != controls(plan):
            raise ValueError("authored controls changed")
        return
    schedule = {f"screen-{i:02d}": identities()[i * GROUP:(i + 1) * GROUP]
                for i in range(len(SEEDS) // GROUP)}
    if key not in schedule or [(s["sample_id"], s["seed"]) for s in samples] != schedule[key]:
        raise ValueError("sample count, order or seeds changed")
    for sample in samples:
        original.validate_sample(sample, plan)
        if sample["tokens"] is None or type(sample["ended_with_eos"]) is not bool:
            raise ValueError("generation metadata missing")


def grade(sample, records, image_id, plan):
    original.validate_sample(sample, plan)
    report = coverage.grade_submission(sample, records, "training", image_id,
                                        execution_version=plan["execution_version"])
    row = contrast._program_row(sample, records, coverage.cases_for("training"), report["outcomes"])
    return {"report": report, "row": row}


def verify_retries(raw, image_id):
    selected = {r["metadata"]["sandbox_id"] for records in raw["records"].values() for r in records.values()}
    cases = {c.input_hash: c for c in coverage.cases_for("training")}
    samples = {s["sample_id"]: s for s in raw["samples"]}
    ids, slots, pairs = [], [], set()
    for receipt in raw["retries"]:
        sid, key, slot = receipt["sample_id"], receipt["input_hash"], receipt["slot"]
        first = unpack_result(receipt["first"])
        if ((sid, key) in pairs or type(slot) is not int or not 0 <= slot < 16
                or not supervised.startup_retry_allowed(first, request_for(samples[sid]["source"], cases[key]),
                    image_id, BOOKING, policy_version=supervised.STARTUP_RETRY_VERSION)
                or receipt["replacement"] != raw["records"][sid][key]):
            raise ValueError("invalid startup replacement evidence")
        ids.append(first.metadata["sandbox_id"])
        slots.append(slot)
        pairs.add((sid, key))
    if len(ids) != len(set(ids)) or set(ids) & selected or len(slots) != len(set(slots)):
        raise ValueError("reused retry sandbox or slot")
    return ids, slots


def verify_batch(raw, samples, key, plan, image_id):
    validate_batch(key, samples, plan)
    if (raw["samples"] != samples or raw["key"] != key or raw["role"] != "training"
            or raw["plan_hash"] != digest(canonical_json(plan))
            or set(raw["records"]) != {s["sample_id"] for s in samples}):
        raise ValueError("batch provenance mismatch")
    graded = [grade(s, raw["records"][s["sample_id"]], image_id, plan) for s in samples]
    if graded != raw["graded"]:
        raise ValueError("scores differ from replayed execution records")
    ids, slots = verify_retries(raw, image_id)
    ids += [sid for row in graded for sid in row["report"]["sandbox_ids"]]
    if len(ids) != len(set(ids)):
        raise ValueError("reused sandbox across programs")
    return graded, ids, slots


def validate_controls(graded):
    rows = [g["row"] for g in graded]
    if (len(rows) != 3 or [r["reference"]["passed"] for r in rows] != [96, 64, 1]
            or [r["endpoint_omission"]["passed"] for r in rows] != [57, 57, 1]
            or rows[1]["inclusive_output_signature"] is not True):
        raise ValueError("live grading controls failed")
    return {"passed": True, "reference_counts": [96, 64, 1], "weak_counts": [57, 57, 1]}


def decision(summary):
    if summary["programs"] != 64 or summary["groups"] != 16:
        raise ValueError("complete fixed-size population required, not a successful subset")
    checks = {
        "reference_success_and_failure": (GATES["minimum_reference_full_passes"] <= summary["reference_full_passes"]
                                           <= 64 - GATES["minimum_reference_failures"]),
        "distinct_inclusive_witnesses": summary["distinct_inclusive_signature_sources"] >= GATES["minimum_distinct_inclusive_signature_sources"],
        "mixed_groups_each_verifier": min(summary["mixed_reference_groups"], summary["mixed_omission_groups"]) >= GATES["minimum_mixed_groups_each_verifier"],
        "different_normalized_signals": summary["changed_relative_signal_groups"] >= GATES["minimum_changed_relative_signal_groups"],
    }
    return {"checks": checks, "ready_for_matched_protocol_review": all(checks.values()),
            "automatic_training": False, "amplification_established": False,
            "action": "review sources and freeze a separate matched RL protocol" if all(checks.values()) else
                      "report the insufficient contrast; no automatic resampling or RL"}


def verify_run(directory):
    directory = Path(directory)
    def read(path):
        return json.loads((directory / path).read_text())
    plan, setup = read("plan.json"), read("setup.json")
    validate_plan(plan)
    ids = supervisor_controls.validate_controls(read("supervisor_controls.json"), setup["sandbox_image_id"])
    slots, rows, groups = [], [], []
    def batch(key, samples):
        raw = read(f"grading/{key}/result.json")
        checked = verify_batch(raw, samples, key, plan, setup["sandbox_image_id"])
        if read(f"grading/{key}/raw.json") != {k: v for k, v in raw.items() if k != "graded"}:
            raise ValueError("saved raw batch changed")
        for sample in samples:
            for input_hash, record in raw["records"][sample["sample_id"]].items():
                path = f"grading/{key}/inputs/{sample['sample_id']}/{input_hash}/selected.json"
                if read(path) != record:
                    raise ValueError("selected input journal changed")
        return checked
    checked, used, retries = batch("controls", controls(plan))
    if read("preflight.json") != validate_controls(checked):
        raise ValueError("saved preflight differs")
    ids.extend(used)
    slots.extend(retries)
    generation = read("generation/result.json")
    samples = generation["samples"]
    if (generation["parameters_unchanged"] is not True or generation["parameter_hash"] != plan["initial_parameter_hash"]
            or generation["initial_checkpoint"] != plan["initial_checkpoint"] or len(samples) != 64):
        raise ValueError("generation provenance differs")
    for index in range(16):
        selected_samples, key = samples[4 * index:4 * index + 4], f"screen-{index:02d}"
        for sample in selected_samples:
            if read(f"generation/samples/{sample['sample_id']}.json") != sample:
                raise ValueError("saved sample changed")
            if read(f"generation/sample-intents/{sample['sample_id']}.json") != {"seed": sample["seed"], "sample_id": sample["sample_id"]}:
                raise ValueError("generation intent changed")
        checked, used, retries = batch(key, selected_samples)
        current = [g["row"] for g in checked]
        groups.append(contrast.compare_group(key, current, actual_training_group=False))
        rows.extend(current)
        ids.extend(used)
        slots.extend(retries)
    if len(ids) != len(set(ids)) or len(slots) != len(set(slots)) or len(ids) > plan["max_sandbox_executions"]:
        raise ValueError("execution identities or resource bounds violated")
    summary = contrast.summarize(rows, groups)
    return {"version": VERSION, "status": "completed", "plan_hash": digest(canonical_json(plan)),
            "source_snapshot_hash": digest(canonical_json(read("source_snapshot.json"))),
            "generation_hash": digest(canonical_json(generation)), "summary": summary,
            "decision": decision(summary), "programs": rows, "groups": groups,
            "sandbox_executions": len(ids), "startup_replacements": len(slots),
            "optimizer_updates": 0, "audit_evaluated": False}


def budget_quote(plan, rates, billing, prior):
    validate_plan(plan)
    def number(value):
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError("invalid resource cost")
        return result
    cpu, mem = number(rates["cpu_hour_cost"]), number(rates["mem_gib_hour_cost"])
    sandbox = plan["max_sandbox_executions"] * 120 * (number(rates["cpu_hour_cost_sandbox"]) + number(rates["mem_gib_hour_cost_sandbox"]) / 4) / 3600
    gpu = GPU_SECONDS * (number(rates["gpu_hour_cost_l40s"]) + 2 * cpu + 32 * mem) / 3600
    controllers = 3 * (SECONDS + plan["max_grading_calls"] * GRADE_SECONDS) * (cpu + 2 * mem) / 3600
    held = max(number(prior), number(billing["metered_cost"]))
    additional = sandbox + gpu + controllers + 2
    return {"rates": rates, "billing_before": billing, "prior_reservation_held_usd": str(held),
            "sandbox_max_usd": str(sandbox), "gpu_max_usd": str(gpu), "controllers_max_usd": str(controllers),
            "storage_build_contingency_usd": "2", "additional_reservation_usd": str(additional),
            "cumulative_reservation_usd": str(held + additional), "is_invoice": False, "provider_hard_cap": False,
            "scope": "64 new programs once, 96 inputs each, controls and at most 16 pre-candidate replacements; no RL"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = verify_run(args.verify) if args.verify else make_plan(Path.cwd())
    print(json.dumps(result, indent=2, sort_keys=True))
