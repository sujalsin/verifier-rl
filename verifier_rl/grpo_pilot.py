"""Frozen real-reward 1.5B GRPO pilot contracts; no model/candidate execution."""

import math

from .checkpoint_pilot import CHAT_TEMPLATE_HASH, MODEL_ID, REPETITION_PENALTY, REVISION, plan_for
from .diagnostics import diagnose_run, diagnose_sample
from .measurement_v2 import PARAMETER_HASH, PROMPT_HASH
from .modal_backend import RUNNER
from .model_trial import validate_submission
from .reward_v2 import behavior_scores, behavior_suite
from .suites import build_suites, digest
from .training import grpo_kwargs

VERSION = "cache-1.5b-grpo-partial-pilot-0.1"
SEEDS = tuple(range(8000, 8008))
TRAIN_SEED = 20260930
PARAMETERS = 1543714304


def evaluation_suites():
    reward, audit = behavior_suite(), build_suites()[-1]
    if {c.input_hash for c in reward.cases} & {c.input_hash for c in audit.cases}:
        raise ValueError("training and independent audit inputs must not overlap")
    return reward, audit


def pilot_plan(document):
    prior = plan_for(document)
    if prior["prompt_hash"] != PROMPT_HASH:
        raise ValueError("pilot must preserve the reviewed prompt")
    reward, audit = evaluation_suites()
    return {
        "version": VERSION, "model_id": MODEL_ID, "revision": REVISION,
        "initial_parameter_hash": PARAMETER_HASH, "parameter_count": PARAMETERS,
        "prompt": prior["prompt"], "prompt_hash": PROMPT_HASH,
        "chat_template_hash": CHAT_TEMPLATE_HASH, "training": True, "sft": False,
        "seeds": list(SEEDS), "training_seed": TRAIN_SEED,
        "max_steps": 4, "group_size": 4, "num_iterations": 1,
        "learning_rate": 1e-6, "weight_decay": 0.0, "beta": 0.0,
        "loss_type": "grpo", "scale_rewards": "group",
        "max_completion_tokens": 512, "temperature": .8, "top_p": .95,
        "top_k": 0, "repetition_penalty": REPETITION_PENALTY,
        "weights_dtype": "float32", "autocast_dtype": "bfloat16",
        "optimizer": "AdamW, foreach=False", "gradient_checkpointing": True,
        "gpu": "L40S", "max_gpu_seconds": 1200, "max_audit_seconds": 3600,
        "max_grading_call_seconds": 300, "concurrency": 8,
        "creation_interval_seconds": .26, "max_retries": 0,
        "reward_suite_hash": reward.fingerprint, "audit_suite_hash": audit.fingerprint,
        "reward_cases": len(reward.cases), "audit_cases": len(audit.cases),
        "max_training_samples": 16, "evaluation_samples_per_arm": 8,
        "max_training_executions": 16 * len(reward.cases),
        "max_evaluation_executions": 16 * sum(len(s.cases) for s in (reward, audit)),
        "runner_hash": digest(RUNNER), "stop_after_uniform_reward_group": True,
        "audit_access": "only after training and checkpoint selection have finished",
        "no_automatic_followup": True,
        "limitations": [
            "Four updates and eight samples per arm: an integration pilot, not an efficacy study.",
            "One cache prompt; no task-generalization or algorithm-ranking claim.",
            "The audit is development data, not an untouched final evaluation.",
            "No KL penalty/reference model in this bounded beta=0 pilot.",
            "Partial reward variation is not proof of useful learning or exploit resistance.",
        ],
    }


def trainer_kwargs(output_dir):
    return dict(grpo_kwargs(output_dir, max_steps=4, completion_tokens=512, seed=TRAIN_SEED),
                repetition_penalty=REPETITION_PENALTY, optim="adamw_torch",
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_grad_norm=1.0, epsilon=.2, importance_sampling_level="token",
                mask_truncated_completions=False, dataloader_num_workers=0)


def require_report(sample, report, suites, image_id):
    validate_submission(sample)
    if [s["suite_hash"] for s in report["suites"]] != [s.fingerprint for s in suites]:
        raise ValueError("wrong suites or audit leakage into training")
    row = diagnose_sample(sample, report, suites)
    if any(s["all_passed"] is None for s in report["suites"]):
        raise ValueError("unresolved infrastructure failure: not a zero reward")
    for saved in report["suites"]:
        for outcome in saved["outcomes"]:
            for attempt in outcome["attempts"]:
                if attempt["status"] == "extraction_rejected":
                    continue
                m = attempt["metadata"]
                if (m.get("image_id") != image_id or m.get("runner_hash") != digest(RUNNER)
                        or m.get("cleanup") != "terminated" or m.get("block_network") is not True
                        or m.get("reset") != "fresh_sandbox_per_input"):
                    raise ValueError("pilot environment/cleanup mismatch")
    return row


def rollout_rewards(samples, reports, image_id):
    if len(samples) != 4 or len(reports) != 4:
        raise ValueError("a training group requires four complete candidates")
    rewards = []
    for sample, report in zip(samples, reports):
        if sample.get("arm") != "training" or sample.get("prompt_hash") != PROMPT_HASH:
            raise ValueError("wrong rollout identity")
        require_report(sample, report, (behavior_suite(),), image_id)
        rewards.append(behavior_scores(report["suites"][0])["partial"])
    return rewards


def training_evidence(metrics):
    groups = metrics["rewards"]
    steps = metrics["global_step"]
    grads = [r["grad_norm"] for r in metrics["log_history"] if "grad_norm" in r]
    if (type(steps) is not int or not 1 <= steps <= 4 or len(groups) != steps or len(grads) != steps
            or any(len(g) != 4 or any(type(r) not in (int, float) or not math.isfinite(r) or not 0 <= r <= 1
                                     for r in g) for g in groups)
            or any(not math.isfinite(g) or g < 0 for g in grads)
            or metrics["before_parameter_hash"] != PARAMETER_HASH
            or metrics.get("checkpoint_reload_hash_matches") is not True
            or metrics.get("finite_parameters") is not True
            or metrics.get("trainable_parameters") != PARAMETERS):
        raise ValueError("invalid/incomplete training or checkpoint evidence")
    uniform = [i for i, g in enumerate(groups) if len(set(g)) == 1]
    if uniform and uniform != [steps - 1]:
        raise ValueError("training continued after a uniform reward group")
    if steps < 4 and not uniform:
        raise ValueError("unexplained early stop")
    changed = metrics["after_parameter_hash"] != metrics["before_parameter_hash"]
    mixed = sum(len(set(g)) > 1 for g in groups)
    # A uniform first group with beta=0/weight_decay=0 must be a true no-op.
    if not mixed and (changed or any(grads)):
        raise ValueError("uniform-only trial unexpectedly changed weights")
    return {"mixed_reward_groups": mixed, "nonzero_gradient_steps": sum(g > 0 for g in grads),
            "parameters_changed": changed, "checkpoint_reload_verified": True,
            "optimizer_update_verified": bool(mixed and any(g > 0 for g in grads) and changed),
            "stop_reason": "uniform_rewards" if uniform else "step_budget"}


def summarize_pilot(generation, reports, image_id):
    plan = generation["plan"]
    if (plan["version"] != VERSION or plan["model_id"] != MODEL_ID or plan["revision"] != REVISION
            or plan["prompt_hash"] != PROMPT_HASH or plan["seeds"] != list(SEEDS)):
        raise ValueError("pilot plan identity mismatch")
    samples = generation["samples"]
    identities = [(arm, seed) for arm in ("before", "after") for seed in SEEDS]
    if ([(s.get("arm"), s["seed"]) for s in samples] != identities
            or [(r.get("arm"), r["seed"]) for r in reports] != identities):
        raise ValueError("all sixteen matched before/after evaluations required")
    evidence = training_evidence(generation["metrics"])
    suites = evaluation_suites()
    arms = {}
    rows = []
    for sample, report in zip(samples, reports):
        require_report(sample, report, suites, image_id)
        reward, audit = report["suites"]
        rows.append({"arm": sample["arm"], "seed": sample["seed"],
                     "source_hash": report["candidate_hash"],
                     "partial_reward": behavior_scores(reward)["partial"],
                     "reward_full_pass": reward["all_passed"],
                     "audit_full_pass": audit["all_passed"],
                     "audit_passed_cases": audit["passed_count"], "audit_total": audit["total"]})
    for arm in ("before", "after"):
        selected = [r for r in rows if r["arm"] == arm]
        arms[arm] = {"samples": 8, "full_audit_passes": sum(r["audit_full_pass"] for r in selected),
                     "full_reward_passes": sum(r["reward_full_pass"] for r in selected),
                     "mean_partial_reward": sum(r["partial_reward"] for r in selected) / 8,
                     "audit_passed_cases": sum(r["audit_passed_cases"] for r in selected),
                     "audit_total_cases": sum(r["audit_total"] for r in selected)}
    return {"run_id": generation["run_id"], "version": VERSION, "plan": plan,
            "training_evidence": evidence, "arms": arms, "rows": rows,
            "paired_audit_case_deltas": [rows[i+8]["audit_passed_cases"] - rows[i]["audit_passed_cases"] for i in range(8)],
            "unchanged_source_pairs": sum(rows[i]["source_hash"] == rows[i+8]["source_hash"] for i in range(8)),
            "diagnostics": diagnose_run(generation, reports, suites),
            "recorded_evaluation_executions": sum(r["execution_config"]["attempts"] for r in reports),
            "limitations": plan["limitations"]}
