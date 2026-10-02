"""Versioned two-arm follow-up; reuses frozen booking graders, never executes code."""

import argparse
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

from . import booking_study as original
from .grading import Status
from .modal_backend import Limits
from .panel_execution import pack_result, request_for, unpack_result
from .suites import canonical_json, digest
from .verifier_quality import CalibrationRow, calibrate

VERSION = "booking-empty-two-arm-0.2"
RUN_ID = "qwen-booking-two-arm-20260928-v1"
CONDITIONS = ("reference", "structured")
MAX_STARTUP_RETRIES = 8


def training_gate(calibration):
    """Do not reinterpret failed three-arm calibration as passed calibration."""
    rows = [CalibrationRow(**row) for row in calibration["rows"]]
    if [r.sample_id for r in rows] != [f"calibration-{seed}" for seed in original.CALIBRATION_SEEDS]:
        raise ValueError("the complete original calibration pool is required")
    expected = calibrate(original.BOOKING, rows)
    expected.update(study_version=original.VERSION,
        reference_passes=sum(r.reference_accepted for r in rows),
        structured_passes=sum(r.structured_accepted for r in rows),
        audit_passes=sum(r.audit_accepted for r in rows),
        unique_sources=len({r.source_hash for r in rows}), rows=[r.__dict__ for r in rows])
    if (calibration != expected or calibration["promotion_probability"] != "0"
            or calibration["ready_for_review"] is not False
            or calibration["reasons"] != ["no_nontrivial_false_acceptance_contrast"]
            or not 0 < calibration["reference_passes"] < 32):
        raise ValueError("resolved zero-contrast calibration with reference reward variation required")
    return {"training_ready": True, "random_arm_enabled": False,
            "original_three_arm_gate_passed": False,
            "reason": "study whether verifier disagreement emerges during optimization"}


def _plan(parent, calibration_hash):
    return dict(parent, version=VERSION, run_id=RUN_ID, conditions=list(CONDITIONS),
        parent_plan=parent, source_run_id=original.RUN_ID, source_calibration_hash=calibration_hash,
        new_calibration_samples=0, random_arm_enabled=False,
        max_initial_gpu_seconds=0, max_sandbox_executions=6852,
        max_startup_retries=MAX_STARTUP_RETRIES, max_startup_retries_per_input=1,
        audit_access="checkpoint audit only after both training arms finish",
        selection="explicit two-arm amendment after observed zero-contrast calibration; no grader change",
        limitations=parent["limitations"] + [
            "No nonzero random-error comparator; no matched-noise claim.",
            "Zero observed initial contrast does not establish zero population verifier error."])


def make_plan(parent, calibration):
    original.validate_plan(parent)
    training_gate(calibration)
    return _plan(parent, digest(canonical_json(calibration)))


def validate_plan(plan, calibration=None):
    original.validate_plan(plan["parent_plan"])
    if plan != _plan(plan["parent_plan"], plan["source_calibration_hash"]):
        raise ValueError("two-arm protocol or resource bounds changed")
    if calibration is not None:
        training_gate(calibration)
        if plan["source_calibration_hash"] != digest(canonical_json(calibration)):
            raise ValueError("source calibration changed")


def require_training(plan, calibration, condition):
    validate_plan(plan, calibration)
    if condition not in CONDITIONS:
        raise ValueError("random-error training is disabled in this protocol")


def validate_batch(key, role):
    if role not in ("training", "evaluation") or "random" in key:
        raise ValueError("no calibration sampling or random arm in this follow-up")
    return original.batch_ids(key, role)


def summary(samples, reports):
    value = original.evaluation_summary(samples, reports, "0")
    del value["expected_random_acceptance"]
    return value


def startup_retry_allowed(result, request, image_id):
    """Only the observed preflight-command RPC timeout; no candidate was submitted."""
    m = result.metadata
    return (result.status == Status.INFRASTRUCTURE_ERROR and result.detail == "preflight:TimeoutError"
        and result.stdout == b"" and m.get("preflight_stage") == "command_start"
        and m.get("cleanup") == "terminated" and m.get("backend") == "modal"
        and m.get("image_id") == image_id and m.get("task_id") == original.BOOKING
        and m.get("runner_hash") == digest(original.runner_for(original.BOOKING))
        and m.get("limits") == asdict(Limits()) and m.get("sdk_version") == "1.5.5"
        and m.get("block_network") is True and m.get("reset") == "fresh_sandbox_per_input"
        and m.get("creation_interval_seconds") == .26
        and isinstance(m.get("sandbox_id"), str) and m["sandbox_id"].startswith("sb-")
        and m.get("source_hash") == digest(request.source)
        and m.get("input_hash") == digest(canonical_json([original.BOOKING, json.loads(request.input_json)]))
        and not any(k in m for k in ("returncode", "runner_stage", "startup_seconds", "preflight_returncode")))


class StartupRetryBackend:
    """One fresh sandbox after a durably reserved, confirmed pre-candidate failure."""

    def __init__(self, backend, image_id, reserve, journal):
        self.backend, self.image_id, self.reserve, self.journal = backend, image_id, reserve, journal
        self.retries = []

    async def execute(self, request):
        first = await self.backend.execute(request)
        if startup_retry_allowed(first, request, self.image_id):
            slot = await self.reserve(pack_result(first))
            if slot is not None:
                if type(slot) is not int or not 0 <= slot < MAX_STARTUP_RETRIES:
                    raise ValueError("invalid startup-retry reservation")
                replacement = await self.backend.execute(request)
                receipt = {"slot": slot, "first": pack_result(first), "replacement": pack_result(replacement)}
                await self.journal(receipt)
                self.retries.append(receipt)
                return replacement
        return first


def verify_retries(batch, image_id):
    selected = {}
    cases = {c.input_hash: c for c in original.cases_for(batch["role"])}
    for sample in batch["samples"]:
        for key, record in batch["records"][sample["sample_id"]].items():
            selected[record["metadata"]["sandbox_id"]] = (sample, cases[key], record)
    ids, slots = [], []
    for receipt in batch["startup_retries"]:
        first, replacement = unpack_result(receipt["first"]), receipt["replacement"]
        sample, case, saved = selected[replacement["metadata"]["sandbox_id"]]
        if (replacement != saved or not startup_retry_allowed(first, request_for(sample["source"], case), image_id)
                or type(receipt["slot"]) is not int or not 0 <= receipt["slot"] < MAX_STARTUP_RETRIES):
            raise ValueError("startup-retry evidence differs from selected execution")
        ids.append(first.metadata["sandbox_id"])
        slots.append(receipt["slot"])
    if (len(ids) != len(set(ids)) or set(ids) & set(selected) or len(slots) != len(set(slots))):
        raise ValueError("reused startup sandbox or retry slot")
    return ids, slots


def budget_quote(plan, rates, billing, prior):
    validate_plan(plan)
    def number(value):
        value = Decimal(str(value))
        if not value.is_finite() or value < 0:
            raise ValueError("invalid cost")
        return value
    cpu, mem = number(rates["cpu_hour_cost"]), number(rates["mem_gib_hour_cost"])
    sandbox_hour = number(rates["cpu_hour_cost_sandbox"]) + number(rates["mem_gib_hour_cost_sandbox"]) / 4
    sandbox = plan["max_sandbox_executions"] * 120 * sandbox_hour / 3600
    gpu = 2 * original.GPU_SECONDS * (number(rates["gpu_hour_cost_l40s"]) + 2 * cpu + 32 * mem) / 3600
    controllers = 3 * (original.CONTROLLER_SECONDS + 68 * original.GRADING_SECONDS) * (cpu + 2 * mem) / 3600
    held = max(number(prior), number(billing["metered_cost"]))
    return {"prior_reservation_held_usd": str(held), "rates": rates, "billing_before": billing,
        "sandbox_max_usd": str(sandbox), "gpu_max_usd": str(gpu), "controllers_max_usd": str(controllers),
        "storage_build_contingency_usd": "2", "cumulative_reservation_usd": str(held + sandbox + gpu + controllers + 2),
        "is_invoice": False, "provider_hard_cap": False,
        "authority": "user approved fixing and running the two-arm protocol",
        "scope": "two 24-step arms; 80 checkpoint probes; at most eight pre-candidate startup retries"}


def verify_run(directory, parent_directory):
    directory = Path(directory)
    def read(path):
        return json.loads((directory / path).read_text())
    parent = original.verify_run(parent_directory)
    plan, calibration, setup, final = read("plan.json"), read("calibration.json"), read("setup.json"), read("result.json")
    validate_plan(plan, calibration)
    if parent["calibration"] != calibration or read("parent_verification.json") != parent:
        raise ValueError("inherited calibration provenance differs")
    if final["status"] != "completed" or final["version"] != VERSION or final["calibration"] != calibration:
        raise ValueError("two-arm run is not complete")
    if {p.name for p in (directory / "arms").iterdir()} != set(CONDITIONS):
        raise ValueError("unexpected or missing training arm")
    ids = original.validate_controls(read("conformance.json"), setup["sandbox_image_id"])
    visited, slots = [], []
    def batch(key, role, samples):
        if [s["sample_id"] for s in samples] != validate_batch(key, role):
            raise ValueError("sample order differs")
        saved = read(f"grading/{key}/result.json")
        if (saved["samples"] != samples or saved["role"] != role or saved["key"] != key
                or saved["plan_hash"] != digest(canonical_json(plan))
                or set(saved["records"]) != {s["sample_id"] for s in samples}):
            raise ValueError("batch/sample identity differs")
        reports = [original.grade(s, saved["records"][s["sample_id"]], role, setup["sandbox_image_id"], plan) for s in samples]
        if reports != saved["reports"]:
            raise ValueError("derived scores differ from raw execution")
        for report in reports:
            ids.extend(report["sandbox_ids"])
        retry_ids, retry_slots = verify_retries(saved, setup["sandbox_image_id"])
        ids.extend(retry_ids)
        slots.extend(retry_slots)
        visited.append(key)
        return reports
    arms, first = {}, None
    for condition in CONDITIONS:
        arm = arms[condition] = read(f"arms/{condition}/result.json")
        evidence = original.training_evidence(arm["metrics"])
        if arm["evidence"] != evidence or final["training"][condition] != evidence:
            raise ValueError("training evidence differs")
        if first is not None and arm["first_rollouts"] != first:
            raise ValueError("first training groups were unmatched")
        first = arm["first_rollouts"]
        if arm["checkpoint_hashes"]["24"] != arm["metrics"]["after_parameter_hash"]:
            raise ValueError("final checkpoint hash differs")
        keys = [f"train-{condition}-{i:02d}" for i in range(original.STEPS)]
        if arm["rollout_keys"] != keys:
            raise ValueError("missing training groups")
        token_count = 0
        for index, key in enumerate(keys):
            group = read(f"arms/{condition}/rollouts/{key}.json")
            if index == 0 and [s["raw"] for s in group] != first:
                raise ValueError("first-group claim differs from saved rollouts")
            reports = batch(key, "training", group)
            values = [original.reward(condition, r, "0", s["sample_id"]) for s, r in zip(group, reports)]
            saved = read(f"arms/{condition}/rewards/{key}.json")
            draws = {s["sample_id"]: str(original.noise_draw(original.NOISE_SEED, s["sample_id"])) for s in group}
            if saved != {"rewards": values, "reports": reports, "draws": draws} or arm["metrics"]["rewards"][index] != values:
                raise ValueError("consumed reward differs from execution")
            token_count += sum(s["tokens"] for s in group)
        if token_count != arm["metrics"]["generated_rollout_tokens"]:
            raise ValueError("rollout token count differs")
    if arms["reference"]["checkpoint_hashes"]["0"] != original.PARAMETER_HASH:
        raise ValueError("baseline is not the untouched policy")
    populations = {"baseline-00": arms["reference"]["evaluations"]["0"]}
    populations.update({f"{condition}-{step:02d}": arm["evaluations"][str(step)]
                        for condition, arm in arms.items() for step in original.CHECKPOINTS})
    expected = {}
    for name, samples in populations.items():
        reports = []
        for index in range(4):
            reports.extend(batch(f"evaluation-{name}-{index:02d}", "evaluation", samples[index * 4:(index + 1) * 4]))
        expected[name] = summary(samples, reports)
    if expected != final["policies"]:
        raise ValueError("independent policy summary differs")
    present = {p.parent.name for p in (directory / "grading").glob("*/result.json")}
    if (set(visited) != present or len(ids) != len(set(ids)) or len(ids) > plan["max_sandbox_executions"]
            or len(slots) != len(set(slots)) or len(slots) > MAX_STARTUP_RETRIES):
        raise ValueError("unexpected work, reused sandbox or resource budget exceeded")
    return dict(final, offline_verified=True, recorded_sandbox_starts=len(ids), startup_retries=len(slots))


def main():
    parser = argparse.ArgumentParser(description="Verify two-arm booking study offline")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run, args.parent), indent=2))


if __name__ == "__main__":
    main()
