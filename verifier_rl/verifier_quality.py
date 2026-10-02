"""Offline verifier-quality controls and calibration; no cloud or model execution.

The three rewards are binary. Per-family partial correctness is diagnostic only.
An execution adapter must validate source, runner, cleanup and report identities
before supplying ExecutionResults here; this module is not that trust boundary.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import json
import re

from .grading import ExecutionResult, MAX_OUTPUT_BYTES, Status
from .checkpoint_pilot import CHAT_TEMPLATE_HASH, MODEL_ID, REPETITION_PENALTY, REVISION
from .suites import canonical_json, digest
from .task_panel import (FAMILIES, TASK_IDS, VERSION as PANEL_VERSION, build_panel,
                         control_answer, prompt_for, suite_hash, task_for, valid_output)

VERSION = "verifier-quality-0.1"
CONDITIONS = ("reference", "random_false_acceptance", "structured_false_acceptance")
SCREEN_SAMPLES = 8
SCREEN_SEEDS = tuple(range(11000, 11008))
CALIBRATION_MIN_WRONG = 8


def compare_output(case, stdout):
    """Exact task-aware comparison. In particular, bool is not interchangeable with int."""
    if type(stdout) is not bytes:
        raise ValueError("raw output bytes required")
    if len(stdout) > MAX_OUTPUT_BYTES:
        return False, "output_limit"
    try:
        answer = json.loads(stdout)
    except (ValueError, UnicodeError, RecursionError):
        return False, "invalid_json"
    if not valid_output(case.task_id, answer):
        return False, "invalid_schema"
    return (True, "pass") if answer == case.expected else (False, "wrong_answer")


def grade(cases, executions):
    """Score supplied execution records; never runs candidate code or fills missing results."""
    if (len(cases) != 16 or len({c.task_id for c in cases}) != 1
            or len({c.split for c in cases}) != 1
            or Counter(c.family for c in cases) != dict.fromkeys(FAMILIES, 4)
            or len({c.input_hash for c in cases}) != 16):
        raise ValueError("one complete sixteen-case panel suite required")
    if set(executions) - {case.input_hash for case in cases}:
        raise ValueError("unexpected execution identities")
    outcomes = []
    for case in cases:
        result = executions.get(case.input_hash)
        if result is None:
            passed, reason = None, "not_executed"
        elif not isinstance(result, ExecutionResult) or not isinstance(result.status, Status):
            raise ValueError("typed execution record required")
        elif result.status == Status.INFRASTRUCTURE_ERROR:
            passed, reason = None, "infrastructure_error"
        elif (result.status == Status.TIMEOUT
              or (type(result.metadata.get("returncode")) is int
                  and (result.metadata["returncode"] < 0 or result.metadata["returncode"] >= 128))):
            passed, reason = None, "unattributed_termination"
        elif result.status == Status.COMPLETED:
            if result.metadata.get("returncode", 0) != 0:
                raise ValueError("completed execution has a nonzero exit code")
            passed, reason = compare_output(case, result.stdout)
        else:
            passed, reason = False, result.status.value
        outcomes.append({"case": case.name, "input_hash": case.input_hash,
                         "family": case.family, "passed": passed, "reason": reason})
    resolved = all(row["passed"] is not None for row in outcomes)
    reference = all(row["passed"] for row in outcomes) if resolved else None
    structured = (all(row["passed"] for row in outcomes if row["family"] != "boundary")
                  if resolved and cases[0].split == "training" else None)
    return {"version": VERSION, "task_id": cases[0].task_id, "split": cases[0].split,
            "suite_hash": suite_hash(cases), "resolved": resolved,
            "reference_accepted": reference, "structured_accepted": structured,
            "case_passes": sum(row["passed"] is True for row in outcomes), "total": len(outcomes),
            "family_passes": {family: sum(row["passed"] is True for row in outcomes
                                          if row["family"] == family) for family in FAMILIES},
            "outcomes": outcomes}


def noise_draw(noise_seed, event_id):
    """Replayable, content-independent coin. Allocate the event ID BEFORE generation.

    Both values stay in the trusted experiment journal. Never use a source hash,
    reward, output text, or retry number in the event ID; retries reuse the draw.
    This is a seeded experimental randomization, not a cryptographic secrecy claim.
    """
    if any(type(value) is not str or not value or len(value) > 256 for value in (noise_seed, event_id)):
        raise ValueError("bounded nonempty noise seed and event ID required")
    value = int(digest(canonical_json([VERSION, noise_seed, event_id])), 16)
    return Fraction(value, 2 ** 256)


def training_reward(condition, training, *, probability=None, draw=None):
    if condition not in CONDITIONS:
        raise ValueError("unknown verifier condition")
    if training.get("version") != VERSION or training.get("split") != "training":
        raise ValueError("development/final reports cannot enter the reward function")
    if training.get("resolved") is not True:
        raise ValueError("unresolved execution is not a zero reward or a random acceptance")
    reference, structured = training["reference_accepted"], training["structured_accepted"]
    if any(type(value) is not bool for value in (reference, structured)) or (reference and not structured):
        raise ValueError("invalid nested-verifier decisions")
    if condition == "reference":
        return int(reference)
    if condition == "structured_false_acceptance":
        return int(structured)
    if (not isinstance(probability, Fraction) or not 0 <= probability <= 1
            or not isinstance(draw, Fraction) or not 0 <= draw < 1):
        raise ValueError("an exact calibrated probability and a persisted draw are required")
    return int(reference or draw < probability)


@dataclass(frozen=True)
class CalibrationRow:
    sample_id: str
    source_hash: str
    task_id: str
    pool: str
    reference_accepted: bool
    structured_accepted: bool
    audit_accepted: bool

    def __post_init__(self):
        task_for(self.task_id)
        if (not self.sample_id or re.fullmatch(r"[0-9a-f]{64}", self.source_hash) is None
                or self.pool not in ("screen", "calibration")
                or any(type(value) is not bool for value in
                       (self.reference_accepted, self.structured_accepted, self.audit_accepted))
                or (self.reference_accepted and not self.structured_accepted)):
            raise ValueError("complete identified nested-verifier results required")


def calibrate(task_id, rows):
    """Match empirical false-acceptance rates IN EXPECTATION on a separate pool.

    Some finite-reference passes can still fail the independent audit. If among
    N audit failures, H pass reference and S pass structured, q=(S-H)/(N-H).
    Using q=S/N would over-promote when H>0. The audit is never a training oracle.
    """
    task_for(task_id)
    if (not rows or any(row.pool != "calibration" or row.task_id != task_id for row in rows)
            or len({row.sample_id for row in rows}) != len(rows)):
        raise ValueError("one task's distinct calibration sample IDs required; no screen reuse")
    wrong = [row for row in rows if not row.audit_accepted]
    n = len(wrong)
    h = sum(row.reference_accepted for row in wrong)
    s = sum(row.structured_accepted for row in wrong)
    eligible = n - h
    probability = Fraction(s - h, eligible) if eligible else None
    reasons = []
    if n < CALIBRATION_MIN_WRONG:
        reasons.append("fewer_than_eight_audit_rejected_samples")
    if len({row.source_hash for row in wrong}) < 4:
        reasons.append("fewer_than_four_distinct_audit_rejected_sources")
    if probability is None or not 0 < probability < 1:
        reasons.append("no_nontrivial_false_acceptance_contrast")
    expected_random = Fraction(h, n) + Fraction(eligible, n) * probability if n and probability is not None else None
    return {"version": VERSION, "task_id": task_id, "samples": len(rows),
            "audit_rejected": n, "reference_false_accepts": h, "structured_false_accepts": s,
            "promotion_probability": str(probability) if probability is not None else None,
            "structured_false_acceptance_rate": str(Fraction(s, n)) if n else None,
            "expected_random_false_acceptance_rate": str(expected_random) if expected_random is not None else None,
            "ready_for_review": not reasons, "reasons": reasons,
            "matching": "in expectation on this calibration pool only; not a measured population equality",
            "pool_hash": digest(canonical_json([row.__dict__ for row in rows]))}


def screen_gate(task_id, rows):
    task_for(task_id)
    if (len(rows) != SCREEN_SAMPLES or any(row.pool != "screen" or row.task_id != task_id for row in rows)
            or len({row.sample_id for row in rows}) != len(rows)):
        raise ValueError("exactly eight distinct, resolved screen samples required")
    rewards = [row.reference_accepted for row in rows]
    mixed = sum(len(set(rewards[index:index + 4])) > 1 for index in (0, 4))
    reasons = []
    if not 0 < sum(rewards) < SCREEN_SAMPLES:
        reasons.append("reference_rewards_have_no_variation")
    if not mixed:
        reasons.append("no_mixed_reference_group_of_four")
    if not any(row.audit_accepted for row in rows):
        reasons.append("no_full_development_audit_pass")
    if len({row.source_hash for row in rows}) < 2:
        reasons.append("fewer_than_two_distinct_sources")
    return {"task_id": task_id, "ready_for_calibration_review": not reasons, "reasons": reasons,
            "reference_passes": sum(rewards), "audit_passes": sum(row.audit_accepted for row in rows),
            "structured_passes": sum(row.structured_accepted for row in rows),
            "mixed_groups_of_four": mixed, "samples": SCREEN_SAMPLES}


def screen_budget_preview(rates, metered_cost, prior_reserved):
    """A conservative quote, NOT a launch guard or permission to incur charges.

    Preserve outstanding reservations even if current billing is lower. A future
    launcher must enforce these lifetimes/start counts and journal each intent.
    """
    def amount(value):
        number = Decimal(str(value))
        if not number.is_finite() or number < 0:
            raise ValueError("nonnegative finite accounting values required")
        return number
    cpu, memory = amount(rates["cpu_hour_cost"]), amount(rates["mem_gib_hour_cost"])
    sandbox_hour = amount(rates["cpu_hour_cost_sandbox"]) + amount(rates["mem_gib_hour_cost_sandbox"]) / 4
    held = max(amount(prior_reserved), amount(metered_cost))
    screen_executions = len(TASK_IDS) * SCREEN_SAMPLES * 32
    # Three authored programs x two cases x three task entrypoints.
    conformance_executions = 18
    sandbox = (screen_executions + conformance_executions) * 120 * sandbox_hour / 3600
    gpu = 2 * 600 * (amount(rates["gpu_hour_cost_l40s"]) + 2 * cpu + 32 * memory) / 3600
    controller = 2 * 1200 * 3 * (cpu + 2 * memory) / 3600
    contingency = Decimal("0.50")
    total = held + sandbox + gpu + controller + contingency
    return {"kind": "unlaunched_screen_quote", "prior_costs_held_usd": str(held),
            "model_programs": len(TASK_IDS) * SCREEN_SAMPLES,
            "model_input_executions": screen_executions,
            "authored_conformance_executions": conformance_executions,
            "sandbox_reserve_usd": str(sandbox), "gpu_reserve_usd": str(gpu),
            "controller_reserve_usd": str(controller), "contingency_usd": str(contingency),
            "cumulative_reserve_usd": str(total), "total_limit_usd": "20",
            "fits_existing_limit": total <= Decimal("20"), "provider_hard_spending_cap": False,
            "enforcement_implemented": False, "calibration_or_training_included": False}


def preflight(root):
    panel = build_panel()
    tasks = []
    for task_id, suites in panel.items():
        controls = []
        for fault in ("correct", "inclusive_boundary", "constant"):
            scores = {}
            for split, cases in suites.items():
                results = {case.input_hash: ExecutionResult(Status.COMPLETED,
                           canonical_json(control_answer(task_id, case.arguments, fault)).encode())
                           for case in cases}
                scores[split] = grade(cases, results)
            controls.append({"authored_control": fault,
                             "reference_reward": training_reward("reference", scores["training"]),
                             "structured_reward": training_reward("structured_false_acceptance", scores["training"]),
                             "development_full_pass": scores["development"]["reference_accepted"],
                             "development_case_passes": scores["development"]["case_passes"]})
        expected = [(1, 1, True), (0, 1, False), (0, 0, False)]
        if [(c["reference_reward"], c["structured_reward"], c["development_full_pass"]) for c in controls] != expected:
            raise ValueError("authored verifier controls failed")
        tasks.append({"task_id": task_id, "entrypoint": task_for(task_id).entrypoint,
                      "prompt_hash": digest(prompt_for(task_id, root)),
                      "suite_hashes": {split: suite_hash(cases) for split, cases in suites.items()},
                      "cases_per_split": {split: len(cases) for split, cases in suites.items()},
                      "controls": controls})
    return {"version": VERSION, "panel_version": PANEL_VERSION, "offline_controls_passed": True,
            "model_samples": 0, "calibrated": False, "cloud_ready": False,
            "screen_generation": {"model_id": MODEL_ID, "revision": REVISION,
                                  "chat_template_hash": CHAT_TEMPLATE_HASH,
                                  "seeds_per_task": list(SCREEN_SEEDS), "max_completion_tokens": 512,
                                  "temperature": .8, "top_p": .95, "top_k": 0,
                                  "repetition_penalty": REPETITION_PENALTY,
                                  "initial_policy": "untouched base instruct checkpoint; no SFT or RL"},
            "screen_samples_planned": len(TASK_IDS) * SCREEN_SAMPLES,
            "screen_execution_upper_bound": len(TASK_IDS) * SCREEN_SAMPLES * 32,
            "tasks": tasks,
            "next_gate": "task-aware sandbox adapter/conformance and fresh budget check before baseline screen"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root for canonical task specifications")
    args = parser.parse_args()
    print(json.dumps(preflight(args.root), indent=2))


if __name__ == "__main__":
    main()
