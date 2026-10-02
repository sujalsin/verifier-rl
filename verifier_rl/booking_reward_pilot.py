"""Matched linear/log reward pilot contracts and offline evidence replay."""

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import re

from . import booking_study as old, booking_verifier_v2 as verifier, supervised_execution as supervised
from .booking_recovery import pre_candidate_failure
from .model_trial import submission_from_completion
from .panel_execution import pack_result, request_for, require_evidence, unpack_result
from .suites import canonical_json, digest
from .task_panel import BOOKING, prompt_for

VERSION = "booking-reward-shape-0.1"
RUN_ID = "qwen-booking-reward-shape-20260928-v1"
ARMS = ("linear", "logarithmic")
STEPS, GROUP = 24, 4
SEEDS = tuple(range(15000, 15016))
SECONDS, GPU_SECONDS, GRADE_SECONDS = 21600, 7200, 1800
MAX_STARTUP_RETRIES = 16
SAVED_ID = "eval-structured-12-14002"
SAVED_SOURCE_HASH = "d424bdf53f68104a58c0ecb0b95aa4f4b5a6062a8e34f82c3cd61961cd6dc2b1"
WARM_VERSION = "booking-reward-warmstart-0.3"
WARM_RUN_ID = "qwen-booking-reward-warmstart-20260929-v3"
FAILED_WARM_RUN_ID = "qwen-booking-reward-warmstart-20260929-v2"
WARM_HASH = "9fb03b169af3d5e4391716390c1504f66f74a678f1136e7efed739fc116c52ba"
WARM_CHECKPOINT = f"/artifacts/{RUN_ID}/arms/linear/checkpoint-12"
WARM_SEEDS = tuple(range(16000, 16016))


def _plan(prompt):
    if digest(prompt) != old.PROMPT_HASH:
        raise ValueError("booking prompt changed")
    return {"version": VERSION, "run_id": RUN_ID, "prompt": prompt, "prompt_hash": old.PROMPT_HASH,
            "model_id": old.MODEL_ID, "revision": old.REVISION, "task_id": BOOKING,
            "initial_parameter_hash": old.PARAMETER_HASH, "parameter_count": old.PARAMETERS,
            "chat_template_hash": old.CHAT_TEMPLATE_HASH, "arms": list(ARMS),
            "verifier_condition": "reference", "verifier_version": verifier.VERSION,
            "suite_hashes": {r: verifier.suite_hash(r) for r in verifier.COUNTS},
            "log_strength": verifier.LOG_STRENGTH, "steps": STEPS, "group_size": GROUP,
            "evaluation_seeds": list(SEEDS), "training_seed": old.TRAIN_SEED,
            "max_completion_tokens": 512, "temperature": .8, "top_p": .95, "top_k": 0,
            "repetition_penalty": old.REPETITION_PENALTY, "sft": False,
            "trainer": old.trainer_kwargs("FROZEN_OUTPUT_DIRECTORY"),
            "saved_control_id": SAVED_ID, "saved_control_source_hash": SAVED_SOURCE_HASH,
            "concurrency": 8, "creation_interval_seconds": .26,
            "max_startup_retries": MAX_STARTUP_RETRIES, "max_gpu_calls": 2,
            "max_gpu_seconds_per_arm": GPU_SECONDS, "max_controller_seconds": SECONDS,
            "max_grading_seconds": GRADE_SECONDS,
            "max_sandbox_executions": 4 * 96 + 2 * STEPS * GROUP * 96 + 3 * 16 * 287 + MAX_STARTUP_RETRIES,
            "audit_access": "after both training arms finish; never used for rewards or selection",
            "automatic_expansion": False,
            "limitations": ["One task, one seed, 24 steps and 16 probes/policy: descriptive pilot.",
                            "No binary training arm: cannot isolate partial vs binary from historical coverage changes.",
                            "Fresh development audit inputs, not an untouched task-generalization benchmark.",
                            "Reward means on different scales are not evidence of improved correctness."]}


def _warm_plan(prompt):
    plan = _plan(prompt)
    plan.update(version=WARM_VERSION, run_id=WARM_RUN_ID, initial_parameter_hash=WARM_HASH,
        initial_checkpoint=WARM_CHECKPOINT, execution_version=supervised.VERSION,
        evaluation_seeds=list(WARM_SEEDS), parent_run_id=RUN_ID, parent_arm="linear", parent_step=12,
        optimizer_start="fresh; not an exact resume", baseline="saved step-12 policy before further RL",
        supervisor_controls=12, max_sandbox_executions=plan["max_sandbox_executions"] + 12)
    plan["trainer"] = trainer_kwargs(plan, "FROZEN_OUTPUT_DIRECTORY")
    plan["limitations"] += [
        "Both arms inherit 12 prior linear-reward updates; this compares subsequent shaping, not training from base.",
        "New execution version; old ambiguous exits and stopped results are not reclassified.",
        "Fresh optimizer for each arm; 24 NEW updates from the shared saved step-12 weights."]
    return plan


def is_warm(plan):
    return plan.get("version") == WARM_VERSION


def trainer_kwargs(plan, directory):
    args = old.trainer_kwargs(directory)
    if is_warm(plan):
        args.update(save_strategy="steps", save_steps=12, save_only_model=False, save_total_limit=None)
    return args


def make_plan(root, *, warm_start=False):
    return (_warm_plan if warm_start else _plan)(prompt_for(BOOKING, root))


def validate_plan(plan):
    if plan != (_warm_plan if is_warm(plan) else _plan)(plan["prompt"]):
        raise ValueError("matched reward protocol changed")


def cases_for(role):
    if role in ("preflight", "training"):
        return verifier.cases_for("training")
    if role == "evaluation":
        return tuple({c.input_hash: c for r in ("training", "audit") for c in verifier.cases_for(r)}.values())
    raise ValueError("invalid execution role")


def batch_ids(key, role, plan=None):
    if role == "preflight" and key == "preflight":
        return ["control-" + x for x in ("correct", "inclusive", "constant", "saved_false_accept")]
    match = re.fullmatch(r"train-(linear|logarithmic)-(\d{2})", key)
    if role == "training" and match and int(match[2]) < STEPS:
        return [f"{key}-{j}" for j in range(GROUP)]
    match = re.fullmatch(r"eval-(baseline|linear|logarithmic)-0([0-3])", key)
    if role == "evaluation" and match:
        seeds = plan["evaluation_seeds"] if plan is not None else SEEDS
        return [f"eval-{match[1]}-{seed}" for seed in seeds[4 * int(match[2]):4 * int(match[2]) + 4]]
    raise ValueError("batch outside declared schedule")


def validate_batch(key, role, samples, plan):
    validate_plan(plan)
    if [s["sample_id"] for s in samples] != batch_ids(key, role, plan):
        raise ValueError("sample identities differ from schedule")
    for sample in samples:
        old.validate_sample(sample, plan)
        if role == "evaluation" and sample["seed"] != int(sample["sample_id"].rsplit("-", 1)[1]):
            raise ValueError("evaluation seed mismatch")


def preflight_samples(saved, plan):
    if saved["sample_id"] != SAVED_ID or digest(saved["source"]) != SAVED_SOURCE_HASH:
        raise ValueError("saved coverage witness differs")
    old.validate_sample(saved, plan)
    sources = {k: v for k, v in old.controls().items() if k in ("correct", "inclusive", "constant")}
    sources["saved_false_accept"] = saved["raw"]
    return [old.sample_from_text(sources[name.removeprefix("control-")], sid=name, plan=plan)
            for name in batch_ids("preflight", "preflight")]


def grade(sample, records, role, image_id, plan):
    old.validate_sample(sample, plan)
    rejected = sample["extraction_status"].startswith("rejected_")
    required = set() if rejected else {c.input_hash for c in cases_for(role)}
    if set(records) != required:
        raise ValueError("incomplete or extra input evidence")
    reports = {r: verifier.grade_submission(sample, {c.input_hash: records[c.input_hash]
                for c in verifier.cases_for(r)} if not rejected else {}, r, image_id,
                execution_version=plan.get("execution_version"))
               for r in (("training", "audit") if role == "evaluation" else ("training",))}
    ids = [r["metadata"]["sandbox_id"] for r in records.values()]
    if len(ids) != len(set(ids)):
        raise ValueError("sandbox reused between inputs")
    return {"sample_id": sample["sample_id"], "role": role, "reports": reports, "sandbox_ids": ids}


def reward(report, arm):
    if report["role"] != "training" or arm not in ARMS or set(report["reports"]) != {"training"}:
        raise ValueError("only training reports supply matched rewards")
    return verifier.training_reward(report["reports"]["training"], "reference", arm)


def validate_preflight(samples, reports):
    if [s["sample_id"] for s in samples] != batch_ids("preflight", "preflight") or len(reports) != 4:
        raise ValueError("four fixed preflight controls required")
    for sample, report in zip(samples, reports):
        if report["sample_id"] != sample["sample_id"] or report["role"] != "preflight":
            raise ValueError("preflight identity mismatch")
        name = sample["sample_id"].removeprefix("control-")
        expected = (SAVED_SOURCE_HASH if name == "saved_false_accept"
                    else digest(submission_from_completion(old.controls()[name])["source"]))
        if digest(sample["source"]) != expected:
            raise ValueError("preflight source changed")
    counts = [r["reports"]["training"]["passed"] for r in reports]
    if counts[:3] != [96, 64, 1] or not 0 <= counts[3] < 96:
        raise ValueError("live controls or known coverage regression failed: " + str(counts))
    return {"passed": True, "case_passes": dict(zip(batch_ids("preflight", "preflight"), counts)),
            "saved_bug_rejected_by_new_training_suite": True}


def retry_allowed(result, request, image_id, plan=None):
    if plan is not None and is_warm(plan):
        return supervised.startup_retry_allowed(result, request, image_id, BOOKING)
    return pre_candidate_failure(result, request, image_id)


def verify_retries(raw, image_id, plan=None):
    selected = {value["metadata"]["sandbox_id"] for records in raw["records"].values() for value in records.values()}
    prior, slots, pairs = [], [], set()
    cases = {c.input_hash: c for c in cases_for(raw["role"])}
    samples = {s["sample_id"]: s for s in raw["samples"]}
    for receipt in raw["retries"]:
        sid, key, slot = receipt["sample_id"], receipt["input_hash"], receipt["slot"]
        first, second = unpack_result(receipt["first"]), receipt["replacement"]
        request = request_for(samples[sid]["source"], cases[key])
        if ((sid, key) in pairs or type(slot) is not int or not 0 <= slot < MAX_STARTUP_RETRIES
                or not retry_allowed(first, request, image_id, plan)
                or second != raw["records"][sid][key]):
            raise ValueError("invalid or repeated startup replacement")
        prior.append(first.metadata["sandbox_id"])
        slots.append(slot)
        pairs.add((sid, key))
    if len(prior) != len(set(prior)) or set(prior) & selected or len(slots) != len(set(slots)):
        raise ValueError("reused startup sandbox or retry slot")
    return prior, slots


def training_evidence(metrics, plan=None):
    initial_hash = plan["initial_parameter_hash"] if plan is not None else old.PARAMETER_HASH
    groups = metrics["rewards"]
    gradients = [row["grad_norm"] for row in metrics["log_history"] if "grad_norm" in row]
    if (metrics["global_step"] != STEPS or len(groups) != STEPS or len(gradients) != STEPS
            or any(len(g) != GROUP or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
                                         for x in g) for g in groups)
            or any(not math.isfinite(g) or g < 0 for g in gradients)
            or metrics["before_parameter_hash"] != initial_hash
            or metrics["finite_parameters"] is not True or metrics["checkpoint_reload_verified"] is not True):
        raise ValueError("incomplete optimizer evidence")
    mixed = sum(len(set(g)) > 1 for g in groups)
    changed = metrics["after_parameter_hash"] != initial_hash
    if not mixed and (any(gradients) or changed):
        raise ValueError("uniform-only rewards unexpectedly moved the initial policy")
    return {"steps": STEPS, "mixed_groups": mixed, "nonzero_gradient_steps": sum(g > 0 for g in gradients),
            "parameters_changed": changed, "checkpoint_reload_verified": True}


def policy_summary(samples, reports, plan=None):
    seeds = plan["evaluation_seeds"] if plan is not None else list(SEEDS)
    if len(samples) != 16 or len(reports) != 16 or [s["seed"] for s in samples] != seeds:
        raise ValueError("sixteen predeclared evaluation samples required")
    for s, r in zip(samples, reports):
        if r["sample_id"] != s["sample_id"] or r["role"] != "evaluation":
            raise ValueError("evaluation sample/report mismatch")
    return {"programs": 16, "audit_full_passes": sum(r["reports"]["audit"]["full_pass"] for r in reports),
            "audit_case_passes": sum(r["reports"]["audit"]["passed"] for r in reports),
            "audit_cases": 16 * 192, "training_case_passes": sum(r["reports"]["training"]["passed"] for r in reports),
            "training_cases": 16 * 96,
            "training_full_passes": sum(r["reports"]["training"]["full_pass"] for r in reports),
            "unique_sources": len({digest(s["source"]) for s in samples}),
            "syntax_valid": sum(s["syntax_valid"] for s in samples),
            "extraction_rejected": sum(s["extraction_status"].startswith("rejected_") for s in samples)}


def budget_quote(plan, rates, billing, prior):
    validate_plan(plan)
    def number(value):
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError("invalid resource price")
        return result
    cpu, mem = number(rates["cpu_hour_cost"]), number(rates["mem_gib_hour_cost"])
    sandbox = plan["max_sandbox_executions"] * 120 * (number(rates["cpu_hour_cost_sandbox"])
                + number(rates["mem_gib_hour_cost_sandbox"]) / 4) / 3600
    gpu = 2 * GPU_SECONDS * (number(rates["gpu_hour_cost_l40s"]) + 2 * cpu + 32 * mem) / 3600
    controllers = 3 * (SECONDS + 61 * GRADE_SECONDS) * (cpu + 2 * mem) / 3600
    held = max(number(prior), number(billing["metered_cost"]))
    contingency = Decimal(10 if is_warm(plan) else 2)
    return {"prior_reservation_held_usd": str(held), "rates": rates, "billing_before": billing,
            "sandbox_max_usd": str(sandbox), "gpu_max_usd": str(gpu), "controllers_max_usd": str(controllers),
            "storage_build_contingency_usd": str(contingency), "additional_reservation_usd": str(sandbox + gpu + controllers + contingency),
            "cumulative_reservation_usd": str(held + sandbox + gpu + controllers + contingency),
            "is_invoice": False, "provider_hard_cap": False,
            "scope": "one matched 24-step linear/log pilot; no automatic extra arms, steps or retries"}


def verify_run(directory):
    directory = Path(directory)
    def read(path):
        return json.loads((directory / path).read_text())
    return verify_with_reader(read)


def verify_with_reader(read):
    """The same replay against files or a read-only artifact mapping in tests."""
    plan, setup = read("plan.json"), read("setup.json")
    validate_plan(plan)
    ids, slots = [], []
    if is_warm(plan):
        from .supervisor_controls import validate_controls
        controls = read("supervisor_controls.json")
        ids.extend(validate_controls(controls, setup["sandbox_image_id"]))
    def batch(key, role, samples):
        validate_batch(key, role, samples, plan)
        raw = read(f"grading/{key}/result.json")
        if (raw["samples"] != samples or raw["key"] != key or raw["role"] != role
                or raw["plan_hash"] != digest(canonical_json(plan))):
            raise ValueError("batch provenance mismatch")
        reports = [grade(s, raw["records"][s["sample_id"]], role, setup["sandbox_image_id"], plan) for s in samples]
        if reports != raw["reports"]:
            raise ValueError("saved report differs from raw records")
        for sample in samples:
            for key_hash, value in raw["records"][sample["sample_id"]].items():
                journal = read(f"grading/{key}/inputs/{sample['sample_id']}/{key_hash}/selected.json")
                if journal != value:
                    raise ValueError("input journal differs from selected result")
        prior, used = verify_retries(raw, setup["sandbox_image_id"], plan)
        ids.extend(prior)
        slots.extend(used)
        ids.extend(sid for r in reports for sid in r["sandbox_ids"])
        return reports
    controls = read("preflight_samples.json")
    checked = validate_preflight(controls, batch("preflight", "preflight", controls))
    if checked != read("preflight.json"):
        raise ValueError("preflight changed")
    arms = {arm: read(f"arms/{arm}/result.json") for arm in ARMS}
    for arm, result in arms.items():
        if result["arm"] != arm or result["evidence"] != training_evidence(result["metrics"], plan):
            raise ValueError("optimizer evidence changed")
        tokens = 0
        for step in range(STEPS):
            key = f"train-{arm}-{step:02d}"
            group = read(f"arms/{arm}/rollouts/{key}.json")
            reports = batch(key, "training", group)
            values = [reward(r, arm) for r in reports]
            if (read(f"arms/{arm}/rewards/{key}.json") != {"rewards": values, "reports": reports}
                    or result["metrics"]["rewards"][step] != values):
                raise ValueError("consumed reward differs from execution evidence")
            if step == 0 and result["first_rollouts"] != [s["raw"] for s in group]:
                raise ValueError("first rollout evidence differs")
            tokens += sum(s["tokens"] for s in group)
        if result["metrics"]["generated_rollout_tokens"] != tokens:
            raise ValueError("rollout token accounting differs")
        if result["checkpoint_hashes"]["24"] != result["metrics"]["after_parameter_hash"]:
            raise ValueError("checkpoint identity differs")
    if (arms["linear"]["first_rollouts"] != arms["logarithmic"]["first_rollouts"]
            or arms["linear"]["checkpoint_hashes"]["0"] != plan["initial_parameter_hash"]):
        raise ValueError("initial policies/rollouts differ")
    populations = {"baseline": arms["linear"]["evaluations"]["0"],
                   **{arm: result["evaluations"]["24"] for arm, result in arms.items()}}
    summaries = {}
    for policy, samples in populations.items():
        reports = []
        owner, step = ("linear", "00") if policy == "baseline" else (policy, "24")
        for sample in samples:
            if read(f"arms/{owner}/evaluation-{step}/samples/{sample['sample_id']}.json") != sample:
                raise ValueError("saved generation differs from evaluated sample")
        for index in range(4):
            reports.extend(batch(f"eval-{policy}-{index:02d}", "evaluation", samples[index * 4:index * 4 + 4]))
        summaries[policy] = policy_summary(samples, reports, plan)
    if len(ids) != len(set(ids)) or len(slots) != len(set(slots)) or len(ids) > plan["max_sandbox_executions"]:
        raise ValueError("execution bounds or isolation identity violated")
    return {"version": plan["version"], "run_id": plan["run_id"], "offline_verified": True, "policies": summaries,
            "training": {a: r["evidence"] for a, r in arms.items()}, "sandbox_starts": len(ids),
            "startup_retries": len(slots), "limitations": plan["limitations"]}


def main():
    parser = argparse.ArgumentParser(description="Offline matched booking-reward verification")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    result = verify_run(args.run) if args.run else make_plan(Path.cwd())
    if args.run and json.loads((args.run / "result.json").read_text()) != result:
        raise ValueError("final result differs from independent replay")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
