"""Matched partial/completion-bonus pilot contracts; no local candidate execution."""

import argparse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import json
import math
from pathlib import Path
import time

from .diagnostics import diagnose_sample
from .grpo_pilot import pilot_plan, training_evidence
from .measurement_v3 import unique_outcomes
from .modal_backend import Limits, RUNNER
from .reward_review import formula_scores, group_advantages
from .reward_v3 import coverage_scores, coverage_suite
from .suites import build_suites, canonical_json, digest

VERSION = "cache-v3-reward-shaping-pilot-0.1"
FORMULAS = ("partial", "completion_bonus")
SEEDS = tuple(range(10000, 10008))
CONTROLLER_SECONDS = 5400
SOURCE_RUN = "qwen-grpo-partial-20260928T010255-4a0e6d52"
MEASUREMENT_RUN = "qwen-grader-v3-20260928T041709-7e90839d"


def selected_suites(audit=False):
    reward = coverage_suite()
    if not audit:
        return (reward,)
    independent = build_suites()[-1]
    if {c.input_hash for c in reward.cases} & {c.input_hash for c in independent.cases}:
        raise ValueError("reward/audit inputs overlap")
    return reward, independent


def study_plan(document):
    plan = pilot_plan(document)
    reward, audit = selected_suites(True)
    plan.update(version=VERSION, formulas=list(FORMULAS), seeds=list(SEEDS),
        reward_suite_hash=reward.fingerprint, reward_cases=len(reward.cases),
        max_training_samples=32, max_training_executions=32 * len(reward.cases),
        max_evaluation_executions=24 * (len(reward.cases) + len(audit.cases)),
        unique_evaluation_records=24, generated_evaluation_records=32,
        max_new_model_samples=64, controller_seconds=CONTROLLER_SECONDS,
        baseline_policy="Both initial checkpoints and all eight before completions must match; grade once.",
        checkpoint_selection="last completed step, no audit-based selection",
        audit_access="after BOTH arms finish training, reload and generation",
        limitations=["Two four-update arms, one training seed and eight evaluation seeds per policy: pilot only.",
                     "Eight baseline generations are duplicated to check initial-generation repeatability, not independent samples.",
                     "Same initial RNG seed does not keep later on-policy samples identical after the policies diverge.",
                     "The development audit is not an untouched final benchmark or a general coding test.",
                     "Positive advantage for an incomplete solution is not itself a GRPO bug.",
                     "Bonus changes score gaps only when some completions fully pass; scaling otherwise mostly cancels.",
                     "No KL regularization, SFT, algorithm expansion, adaptive tuning, or automatic follow-up."])
    plan["max_sandbox_executions"] = plan["max_training_executions"] + plan["max_evaluation_executions"]
    return plan


def arm_plan(plan, formula, study_id):
    if plan["version"] != VERSION or formula not in FORMULAS:
        raise ValueError("unknown study or reward formula")
    return dict(plan, reward_formula=formula, study_run_id=study_id)


def reward_value(scores, formula):
    if formula not in FORMULAS:
        raise ValueError("unknown reward formula")
    values = formula_scores(scores)  # Validates finite partial/full agreement, including unscored values.
    return values["v2_partial" if formula == "partial" else "completion_bonus"]


def require_report(sample, report, audit, image_id):
    suites = selected_suites(audit)
    if [s["suite_hash"] for s in report["suites"]] != [s.fingerprint for s in suites]:
        raise ValueError("incorrect suites or audit leakage into reward")
    row = diagnose_sample(sample, report, suites)
    if any(s["all_passed"] is None for s in report["suites"]):
        raise ValueError("unresolved infrastructure failure, not a zero reward")
    if report["execution_config"]["max_retries"] != 0:
        raise ValueError("no retries permitted")
    if set(row["stdout_evidence"]) - {"not_completed", "verified_complete_stdout"}:
        raise ValueError("incomplete output evidence")
    ids = set()
    for outcome in unique_outcomes(report).values():
        if len(outcome["attempts"]) != 1:
            raise ValueError("exactly one attempt per input required")
        attempt = outcome["attempts"][0]
        if attempt["status"] == "extraction_rejected":
            continue
        m = attempt["metadata"]
        if (m.get("image_id") != image_id or m.get("runner_hash") != digest(RUNNER)
                or m.get("limits") != asdict(Limits()) or m.get("cleanup") != "terminated"
                or m.get("block_network") is not True or m.get("reset") != "fresh_sandbox_per_input"
                or m.get("sdk_version") != "1.5.5" or m.get("creation_interval_seconds") != .26
                or m.get("preflight_returncode") != 0 or not m.get("sandbox_id")):
            raise ValueError("execution environment or cleanup differs from protocol")
        if m["sandbox_id"] in ids:
            raise ValueError("sandbox reused within candidate")
        ids.add(m["sandbox_id"])
        code = m.get("returncode")
        if attempt["status"] == "timeout" or (type(code) is int and (code < 0 or code >= 128)):
            raise ValueError("unattributed termination: stop, preserve failure, do not retry")
    if len(ids) != report["execution_config"]["attempts"]:
        raise ValueError("sandbox count mismatch")
    coverage_scores(report["suites"][0])
    return row


def rollout_rewards(samples, reports, image_id, plan):
    if len(samples) != 4 or len(reports) != 4:
        raise ValueError("four complete rollouts required")
    rewards = []
    for sample, report in zip(samples, reports):
        if sample["arm"] != "training" or sample["prompt_hash"] != plan["prompt_hash"]:
            raise ValueError("wrong rollout identity")
        require_report(sample, report, False, image_id)
        rewards.append(reward_value(coverage_scores(report["suites"][0]), plan["reward_formula"]))
    return rewards


def before_signature(samples):
    before = [s for s in samples if s["arm"] == "before"]
    if [s["seed"] for s in before] != list(SEEDS):
        raise ValueError("eight initial-policy samples required")
    fields = ("seed", "raw", "source", "extraction_status", "extraction_version", "prompt_hash", "tokens")
    return [{k: s[k] for k in fields} for s in before]


def budget_envelope(plan, billing, rates):
    def amount(value):
        number = Decimal(str(value))
        if not number.is_finite() or number < 0:
            raise ValueError("invalid nonnegative billing value")
        return number
    cpu, memory = amount(rates["cpu_hour_cost"]), amount(rates["mem_gib_hour_cost"])
    sandbox_hour = amount(rates["cpu_hour_cost_sandbox"]) + amount(rates["mem_gib_hour_cost_sandbox"]) / 4
    gpu_ceiling = 2 * plan["max_gpu_seconds"] * (amount(rates["gpu_hour_cost_l40s"]) + 2 * cpu + 32 * memory) / 3600
    # Reserve every CPU function's full timeout, independently of controller lifetime.
    worker_ceiling = (32 + 24) * plan["max_grading_call_seconds"] * (cpu + memory) / 3600
    controller_ceiling = plan["controller_seconds"] * (cpu + 2 * memory) / 3600
    fixed = amount(billing["metered_cost"]) + gpu_ceiling + worker_ceiling + controller_ceiling + Decimal("1")
    reserve = (plan["reward_cases"] + plan["audit_cases"]) * Limits().sandbox_lifetime_seconds * sandbox_hour / 3600
    if fixed + reserve > Decimal("10"):
        raise ValueError("budget cannot safely reserve even one maximum-size candidate batch")
    return {"billing_before": billing, "rates": rates, "total_trial_limit_usd": "10",
        "gpu_resource_reserve_usd": str(gpu_ceiling), "worker_resource_reserve_usd": str(worker_ceiling),
        "controller_resource_reserve_usd": str(controller_ceiling), "lag_build_storage_reserve_usd": "1",
        "fixed_reserved_usd": str(fixed), "sandbox_hour_usd": str(sandbox_hour),
        "largest_batch_reserve_usd": str(reserve), "max_sandbox_executions": plan["max_sandbox_executions"],
        "provider_hard_spending_cap": False,
        "policy": "Before EACH batch, reserve all its sandbox lifetimes plus previously accounted sandbox time; stop if over $10."}


def reserve_batch(budget, receipts, candidate_id, executions):
    if type(executions) is not int or not 0 <= executions <= 432:
        raise ValueError("invalid batch execution bound")
    if candidate_id in receipts:
        raise ValueError("candidate already accounted; do not resubmit")
    used_seconds, used_executions = 0, 0
    for key, receipt in receipts.items():
        if key != receipt["candidate_id"]:
            raise ValueError("budget receipt identity mismatch")
        seconds, count = receipt["accounted_sandbox_seconds"], receipt["executions"]
        if (type(seconds) is not int or type(count) is not int or not 0 <= count <= 432
                or not count <= seconds <= count * Limits().sandbox_lifetime_seconds):
            raise ValueError("invalid budget receipt")
        used_seconds += seconds
        used_executions += count
    if used_executions + executions > budget["max_sandbox_executions"]:
        raise ValueError("study execution budget exceeded")
    reserved_seconds = executions * Limits().sandbox_lifetime_seconds
    total = Decimal(budget["fixed_reserved_usd"]) + Decimal(budget["sandbox_hour_usd"]) * (used_seconds + reserved_seconds) / 3600
    if total > Decimal(budget["total_trial_limit_usd"]):
        raise ValueError("total trial spending guard: stop before new sandbox creation")
    return {"candidate_id": candidate_id, "previous_accounted_seconds": used_seconds,
            "previous_executions": used_executions, "max_new_executions": executions,
            "reserved_sandbox_seconds": reserved_seconds, "total_reserved_usd": str(total)}


def execution_receipt(candidate_id, report):
    seconds, count = 0, 0
    for outcome in unique_outcomes(report).values():
        for attempt in outcome["attempts"]:
            if attempt["status"] == "extraction_rejected":
                continue
            elapsed = attempt["metadata"].get("total_seconds")
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("missing trusted timing evidence; keep full reservation and stop")
            count += 1
            # Round up observed lifecycle time (includes startup/RPC/cleanup),
            # bounded by the provider's configured sandbox lifetime. Not an invoice.
            seconds += min(Limits().sandbox_lifetime_seconds, max(1, math.ceil(elapsed)))
    if count != report["execution_config"]["attempts"]:
        raise ValueError("receipt attempt count mismatch")
    return {"candidate_id": candidate_id, "executions": count, "accounted_sandbox_seconds": seconds,
            "report_hash": digest(canonical_json(report)), "is_provider_invoice": False}


def require_deadline(plan):
    deadline = plan.get("submission_deadline_epoch")
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.time() >= deadline:
        raise ValueError("study submission deadline reached or missing")


def study_execution_records(study_id, generations, reports):
    """Canonical execution order; baseline counted once, never as both before arms."""
    records = []
    for formula in FORMULAS:
        owner = study_id + "-" + formula.replace("_", "-")
        for group in generations[formula]["rollout_records"]:
            for sample, report in zip(group["samples"], group["reports"]):
                records.append((owner, sample["arm"], sample["seed"], report))
    for policy in ("baseline", *FORMULAS):
        owner = study_id + "-" + policy.replace("_", "-")
        for report in reports[policy]:
            records.append((owner, report["arm"], report["seed"], report))
    return records


def verify_budget_records(study_id, generations, reports, budget, receipts, reservations):
    seen = {}
    for owner, arm, seed, report in study_execution_records(study_id, generations, reports):
        key = f"{owner}-{arm}-{seed}"
        expected = execution_receipt(key, report)
        if receipts.get(key) != expected:
            raise ValueError("budget receipt differs from execution evidence")
        if reservations.get(key) != reserve_batch(budget, seen, key, expected["executions"]):
            raise ValueError("spending reservation differs from sequential accounting")
        seen[key] = expected
    if set(seen) != set(receipts) or set(seen) != set(reservations):
        raise ValueError("missing or extra budget records")
    return reserve_batch(budget, seen, "end-of-study-check", 0)


def comparison_summary(study_id, plan, generations, reports, image_id):
    if set(generations) != set(FORMULAS) or set(reports) != {"baseline", *FORMULAS}:
        raise ValueError("both formulas and common baseline required")
    if before_signature(generations[FORMULAS[0]]["samples"]) != before_signature(generations[FORMULAS[1]]["samples"]):
        raise ValueError("initial-policy generations differ; cannot silently share baseline")
    first_groups = [[s["raw"] for s in generations[f]["rollout_records"][0]["samples"]] for f in FORMULAS]
    if first_groups[0] != first_groups[1]:
        raise ValueError("initial on-policy rollout groups differ")
    rows, training_rows, evidence, all_ids = [], [], {}, set()
    def check_ids(report):
        ids = {a["metadata"]["sandbox_id"] for o in unique_outcomes(report).values()
               for a in o["attempts"] if a["status"] != "extraction_rejected"}
        if ids & all_ids:
            raise ValueError("reused sandbox across independent program evaluations")
        all_ids.update(ids)
    for formula in FORMULAS:
        generation = generations[formula]
        if (generation["plan"] != arm_plan(plan, formula, study_id)
                or generation["run_id"] != study_id + "-" + formula.replace("_", "-")):
            raise ValueError("arm differs from the frozen common plan")
        evidence[formula] = training_evidence(generation["metrics"])
        records = generation["rollout_records"]
        if len(records) != generation["metrics"]["global_step"]:
            raise ValueError("training group/step mismatch")
        for step, record in enumerate(records):
            if [s["seed"] for s in record["samples"]] != list(range(9000 + 4 * step, 9004 + 4 * step)):
                raise ValueError("rollout order mismatch")
            rewards = rollout_rewards(record["samples"], record["reports"], image_id, generation["plan"])
            if rewards != record["rewards"] or rewards != generation["metrics"]["rewards"][step]:
                raise ValueError("training rewards differ from exact execution evidence")
            advantages = group_advantages(rewards)
            scores = [coverage_scores(r["suites"][0]) for r in record["reports"]]
            training_rows.append({"formula": formula, "step": step + 1, "rewards": rewards,
                "partial_scores": [s["partial"] for s in scores], "full_passes": sum(s["binary"] for s in scores),
                "advantages": advantages, "positive_nonfull_ids": [s["seed"] for s, score, a
                    in zip(record["samples"], scores, advantages) if a > 0 and not score["binary"]]})
            for report in record["reports"]: check_ids(report)
        if [(s["arm"], s["seed"]) for s in generation["samples"]] != [
                (arm, seed) for arm in ("before", "after") for seed in SEEDS]:
            raise ValueError("matched before/after samples missing")
    first = generations[FORMULAS[0]]
    populations = {"baseline": [s for s in first["samples"] if s["arm"] == "before"],
        **{f: [s for s in generations[f]["samples"] if s["arm"] == "after"] for f in FORMULAS}}
    for policy, samples in populations.items():
        if len(reports[policy]) != len(SEEDS):
            raise ValueError("eight evaluation reports required for each policy")
        for sample, report in zip(samples, reports[policy]):
            diagnostic = require_report(sample, report, True, image_id)
            check_ids(report)
            score, audit = coverage_scores(report["suites"][0]), report["suites"][1]
            rows.append({"policy": policy, "seed": sample["seed"], "source_hash": report["candidate_hash"],
                "partial_reward": score["partial"], "bonus_reward": reward_value(score, "completion_bonus"),
                "reward_full_pass": score["binary"], "audit_full_pass": audit["all_passed"],
                "audit_passed_cases": audit["passed_count"], "audit_total": audit["total"], "diagnostic": diagnostic})
    policies = {}
    for policy in populations:
        selected = [r for r in rows if r["policy"] == policy]
        counts = Counter()
        for row in selected: counts.update(row["diagnostic"]["outcome_counts"])
        policies[policy] = {"samples": len(selected), "unique_sources": len({r["source_hash"] for r in selected}),
            "full_audit_passes": sum(r["audit_full_pass"] for r in selected),
            "full_reward_passes": sum(r["reward_full_pass"] for r in selected),
            "mean_partial_reward": sum(r["partial_reward"] for r in selected) / len(selected),
            "mean_bonus_reward": sum(r["bonus_reward"] for r in selected) / len(selected),
            "audit_passed_cases": sum(r["audit_passed_cases"] for r in selected),
            "audit_total_cases": sum(r["audit_total"] for r in selected), "outcome_counts": dict(counts)}
    return {"run_id": study_id, "plan": plan, "initial_generation_match": True,
            "initial_rollout_group_match": True, "policies": policies,
            "rows": rows, "training_groups": training_rows, "training_evidence": evidence,
            "recorded_sandbox_executions": len(all_ids), "limitations": plan["limitations"]}


def main(argv=None):
    from .cli import create_run_directory, write_private
    parser = argparse.ArgumentParser(description="Offline verification of a completed reward-shaping study")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    docs = {n: json.loads((args.run / f"{n}.json").read_text())
            for n in ("plan", "generations", "reports", "setup", "summary", "budget",
                      "budget-receipts", "spending-reservations")}
    result = comparison_summary(docs["summary"]["run_id"], docs["plan"], docs["generations"],
                                docs["reports"], docs["setup"]["sandbox_image_id"])
    if result != docs["summary"]:
        raise ValueError("saved summary does not match independent recomputation")
    budget = docs["budget"]
    if budget_envelope(docs["plan"], budget["billing_before"], budget["rates"]) != budget:
        raise ValueError("saved spending plan does not match recomputation")
    accounting = verify_budget_records(result["run_id"], docs["generations"], docs["reports"], budget,
                                      docs["budget-receipts"], docs["spending-reservations"])
    out = create_run_directory(args.out)
    write_private(out / "verification.json", canonical_json({"verified": True, "run_id": result["run_id"],
        "cloud_calls": 0, "candidate_source_executed": False, "policies": result["policies"],
        "accounting": accounting,
        "document_hashes": {n: digest(canonical_json(v)) for n, v in docs.items()}}))
    print(canonical_json(result["policies"]))


if __name__ == "__main__":
    main()
