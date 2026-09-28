"""Pure contracts for the bounded SFT pilot and its pre-specified RL gate."""

import math

from .model_trial import MODEL_ID, MODEL_REVISION, canonical_prompt
from .partial_reward import balanced_scores, balanced_suite
from .prompt_pilot import REMINDERS
from .sft_data import validate_dataset
from .suites import build_suites, digest

SEEDS = tuple(range(4000, 4008))
SFT_SEED = 20260930
ARMS = ("before", "after")


def evaluation_suites():
    return (build_suites()[2], balanced_suite())


def pilot_plan(task_document):
    prompt = canonical_prompt(task_document) + "\n\n" + REMINDERS.strip()
    suites = evaluation_suites()
    return {"version": "sft-cache-pilot-0.1", "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "dataset_validation": validate_dataset(), "prompt": prompt, "prompt_hash": digest(prompt),
            "suite_hashes": {s.name: s.fingerprint for s in suites}, "seeds": list(SEEDS),
            "sft_seed": SFT_SEED, "sft_steps": 16, "max_training_tokens": 1024,
            "learning_rate": 2e-5, "weight_decay": 0.0, "batch_size": 1, "gradient_accumulation": 4,
            "max_completion_tokens": 512, "temperature": .8, "top_p": .95, "top_k": 0,
            "max_gpu_seconds": 900, "max_controller_seconds": 1500,
            "max_sandbox_executions": 16 * len({c.input_hash for s in suites for c in s.cases}),
            "concurrency": 4, "creation_interval_seconds": .26, "max_retries": 0,
            "limitations": ["Eight samples per policy, one task; development evidence only.",
                            "SFT data has ten functions with four wording variants, not forty independent tasks.",
                            "Readiness uses balanced reward variation above the .05 edge-only ceiling, not a learning claim."]}


def require_sft_evidence(metrics):
    grads = [r["grad_norm"] for r in metrics["log_history"] if "grad_norm" in r]
    if (metrics["global_step"] != 16 or len(grads) != 16 or not all(math.isfinite(g) and g > 0 for g in grads)
            or metrics["before_parameter_hash"] == metrics["after_parameter_hash"]
            or metrics.get("checkpoint_reload_hash_matches") is not True):
        raise ValueError("SFT optimizer/checkpoint evidence failed")
    losses = metrics["supervised_losses"]
    if (set(losses) != {f"{s}_{when}" for s in ("train", "development") for when in ("before", "after")}
            or any(not math.isfinite(v) for v in losses.values())):
        raise ValueError("nonfinite supervised loss")


def summarize_pilot(generation, reports):
    require_sft_evidence(generation["training"])
    expected_keys = {(a, s) for a in ARMS for s in SEEDS}
    samples = {(s["arm"], s["seed"]): s for s in generation["samples"]}
    by_key = {(r["arm"], r["seed"]): r for r in reports}
    if (len(generation["samples"]) != 16 or len(reports) != 16
            or set(samples) != expected_keys or set(by_key) != expected_keys):
        raise ValueError("missing/duplicate SFT pilot evidence")
    suite_hashes = {s.name: s.fingerprint for s in evaluation_suites()}
    rows = {a: [] for a in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            sample, report = samples[arm, seed], by_key[arm, seed]
            if report["candidate_hash"] != digest(sample["source"]):
                raise ValueError("SFT sample/report mismatch")
            if {s["suite"]: s["suite_hash"] for s in report["suites"]} != suite_hashes:
                raise ValueError("SFT evaluation suite mismatch")
            if any(s["all_passed"] is None for s in report["suites"]):
                raise ValueError("SFT pilot has unscored infrastructure outcomes")
            g3, balanced = report["suites"]
            scores = balanced_scores(balanced)
            unique = {o["input_hash"]: o for s in report["suites"] for o in s["outcomes"]}
            rows[arm].append({"seed": seed, "syntax_valid": sample["syntax_valid"],
                              "hit_token_cap": sample["hit_token_cap"],
                              "all_outputs_valid": all(o["reason"] in ("pass", "wrong_answer") for o in unique.values()),
                              "g3_passed_cases": g3["passed_count"], "g3_all_passed": g3["all_passed"],
                              "balanced": scores})
    groups = [[r["balanced"]["partial"] for r in rows["after"][i:i+4]] for i in (0, 4)]
    eligible = any(len(set(g)) > 1 and max(g) > .05 for g in groups)
    arms = {a: {"sample_count": 8, "syntax_valid": sum(r["syntax_valid"] for r in rows[a]),
                "valid_on_all_inputs": sum(r["all_outputs_valid"] for r in rows[a]),
                "full_g3_passes": sum(r["g3_all_passed"] for r in rows[a]),
                "full_balanced_passes": sum(r["balanced"]["binary"] for r in rows[a]),
                "mean_balanced_partial": sum(r["balanced"]["partial"] for r in rows[a]) / 8,
                "rows": rows[a]} for a in ARMS}
    return {"run_id": generation["run_id"], "version": generation["plan"]["version"],
            "arms": arms, "rl_eligible": eligible, "after_partial_groups": groups,
            "eligibility_rule": "at least one mixed group with a candidate above .05 edge-only credit",
            "checkpoint": generation["checkpoint"], "checkpoint_hash": generation["training"]["after_parameter_hash"],
            "actual_sandbox_executions": sum(r["execution_config"]["attempts"] for r in reports),
            "limitations": generation["plan"]["limitations"]}
