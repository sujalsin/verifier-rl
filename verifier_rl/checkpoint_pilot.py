"""Frozen untouched-1.5B baseline compared with saved untouched-0.5B outputs."""

from .model_trial import MODEL_ID as SMALL_MODEL, MODEL_REVISION as SMALL_REVISION, canonical_prompt, validate_submission
from .modal_backend import RUNNER
from .partial_reward import balanced_scores
from .prompt_pilot import REMINDERS
from .sft_trial import SEEDS, evaluation_suites
from .suites import digest

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
VERSION = "cache-checkpoint-comparison-0.2"
DECODE = {"max_completion_tokens": 512, "temperature": .8, "top_p": .95, "top_k": 0}
REPETITION_PENALTY = 1.05  # Original 0.5B's pinned generation_config.json default.
CHAT_TEMPLATE_HASH = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"


def plan_for(document):
    prompt = canonical_prompt(document) + "\n\n" + REMINDERS.strip()
    suites = evaluation_suites()
    return {"version": VERSION, "model_id": MODEL_ID, "revision": REVISION,
            "comparison_model_id": SMALL_MODEL, "comparison_revision": SMALL_REVISION,
            "prompt": prompt, "prompt_hash": digest(prompt), "seeds": list(SEEDS), **DECODE,
            "chat_template_hash": CHAT_TEMPLATE_HASH, "training": False,
            "repetition_penalty": REPETITION_PENALTY,
            "suite_hashes": {s.name: s.fingerprint for s in suites},
            "max_gpu_seconds": 600, "max_controller_seconds": 1200,
            "max_sandbox_executions": 8 * len({c.input_hash for s in suites for c in s.cases}),
            "concurrency": 4, "creation_interval_seconds": .26, "max_retries": 0,
            "scope": "eight generation-only samples; no SFT, RL, audit selection, or automatic follow-up",
            "limitations": ["One task and eight samples; not a definitive model ranking.",
                            "Reuse saved original 0.5B outputs, not new independent baseline samples.",
                            "A checkpoint comparison, not an isolated causal effect of parameter count.",
                            "G3 and balanced probes are development tests, not an untouched final evaluation."]}


def summarize_policy(samples, reports, image_id, prompt_hash):
    suites = evaluation_suites()
    if len(samples) != 8 or len(reports) != 8 or [s["seed"] for s in samples] != list(SEEDS):
        raise ValueError("eight fixed-seed samples and reports required")
    if [r["seed"] for r in reports] != list(SEEDS):
        raise ValueError("report seed/order mismatch")
    rows = []
    for sample, report in zip(samples, reports):
        validate_submission(sample)
        if sample["prompt_hash"] != prompt_hash or report["candidate_hash"] != digest(sample["source"]):
            raise ValueError("checkpoint pilot source/prompt mismatch")
        if len(report["suites"]) != len(suites):
            raise ValueError("checkpoint pilot suite count mismatch")
        rejected = sample["extraction_status"].startswith("rejected_")
        for suite, result in zip(suites, report["suites"]):
            if result["suite_hash"] != suite.fingerprint or len(result["outcomes"]) != len(suite.cases):
                raise ValueError("checkpoint pilot suite identity/coverage mismatch")
            if result["all_passed"] is None or result["infrastructure_errors"]:
                raise ValueError("unscored candidate; no model failure assigned")
            if result["passed_count"] != sum(o["passed"] is True for o in result["outcomes"]):
                raise ValueError("checkpoint pilot score count mismatch")
            for case, outcome in zip(suite.cases, result["outcomes"]):
                if outcome["input_hash"] != case.input_hash or outcome["expected"] != case.expected:
                    raise ValueError("checkpoint pilot case identity mismatch")
                if not outcome["attempts"]:
                    raise ValueError("missing attempt evidence")
                for attempt in outcome["attempts"]:
                    if rejected:
                        if attempt["status"] != "extraction_rejected" or report["execution_config"]["attempts"] != 0:
                            raise ValueError("rejected extraction unexpectedly executed")
                    else:
                        m = attempt["metadata"]
                        if m.get("runner_hash") != digest(RUNNER) or m.get("image_id") != image_id or m.get("cleanup") != "terminated":
                            raise ValueError("checkpoint pilot environment/cleanup mismatch")
        unique = {o["input_hash"]: o for s in report["suites"] for o in s["outcomes"]}
        g3, balanced = report["suites"]
        rows.append({"seed": sample["seed"], "source_hash": report["candidate_hash"],
                     "syntax_valid": sample["syntax_valid"], "hit_token_cap": sample["hit_token_cap"],
                     "tokens": sample["tokens"],
                     "valid_on_all_inputs": all(o["reason"] in ("pass", "wrong_answer") for o in unique.values()),
                     "g3_passed_cases": g3["passed_count"], "g3_all_passed": g3["all_passed"],
                     "balanced": balanced_scores(balanced)})
    partial_groups = [[r["balanced"]["partial"] for r in rows[i:i+4]] for i in (0, 4)]
    binary_groups = [[r["balanced"]["binary"] for r in rows[i:i+4]] for i in (0, 4)]
    return {"sample_count": 8, "syntax_valid": sum(r["syntax_valid"] for r in rows),
            "valid_on_all_inputs": sum(r["valid_on_all_inputs"] for r in rows),
            "full_g3_passes": sum(r["g3_all_passed"] for r in rows),
            "full_balanced_passes": sum(r["balanced"]["binary"] for r in rows),
            "mean_balanced_partial": sum(r["balanced"]["partial"] for r in rows) / 8,
            "unique_sources": len({r["source_hash"] for r in rows}),
            "cap_hits": sum(r["hit_token_cap"] for r in rows),
            "binary_groups": binary_groups, "partial_groups": partial_groups,
            "mixed_binary_groups": sum(len(set(g)) > 1 for g in binary_groups),
            "meaningful_mixed_partial_groups": sum(len(set(g)) > 1 and max(g) > .05 for g in partial_groups),
            "actual_sandbox_executions": sum(r["execution_config"]["attempts"] for r in reports), "rows": rows}


def reference_packet(generation, reports, plan, image_id):
    old_plan = generation["plan"]
    if old_plan["model_id"] != SMALL_MODEL or old_plan["revision"] != SMALL_REVISION:
        raise ValueError("reference must be the original pinned 0.5B checkpoint")
    if old_plan["prompt_hash"] != plan["prompt_hash"] or any(old_plan[k] != v for k, v in DECODE.items()):
        raise ValueError("reference prompt/decoding mismatch")
    samples = [s for s in generation["samples"] if s["arm"] == "before"]
    selected = [r for r in reports if r["arm"] == "before"]
    summary = summarize_policy(samples, selected, image_id, plan["prompt_hash"])
    return {"source_run_id": generation["run_id"], "model_id": SMALL_MODEL, "revision": SMALL_REVISION,
            "parameter_hash": generation["training"]["before_parameter_hash"],
            "summary": summary, "samples": samples, "reports": selected, "runtime": generation["runtime"]}


def summarize_comparison(generation, reports, reference, image_id):
    if generation["before_parameter_hash"] != generation["after_parameter_hash"]:
        raise ValueError("generation changed model weights")
    plan = generation["plan"]
    if plan["model_id"] != MODEL_ID or plan["revision"] != REVISION or plan["training"] is not False:
        raise ValueError("unexpected 1.5B model/protocol")
    small = summarize_policy(reference["samples"], reference["reports"], image_id, plan["prompt_hash"])
    large = summarize_policy(generation["samples"], reports, image_id, plan["prompt_hash"])
    return {"run_id": generation["run_id"], "version": VERSION, "training": False,
            "reference_run_id": reference["source_run_id"], "original_0_5b": small, "original_1_5b": large,
            "parameters_unchanged": True, "limitations": plan["limitations"]}


def recovery_plan(generation, completed, image_id):
    """Allow at most one missing report after a controller interruption, no new samples."""
    from .prompt_recovery import validate_prior

    plan = generation["plan"]
    if (plan["version"] != VERSION or plan["model_id"] != MODEL_ID or plan["revision"] != REVISION
            or plan["training"] is not False or generation["before_parameter_hash"] != generation["after_parameter_hash"]):
        raise ValueError("recovery requires the frozen untouched 1.5B generation")
    samples = generation["samples"]
    if [s["seed"] for s in samples] != list(SEEDS):
        raise ValueError("recovery requires all eight original samples in order")
    suites = evaluation_suites()
    by_seed = {r["seed"]: r for r in completed}
    if len(by_seed) != len(completed) or set(by_seed) - set(SEEDS):
        raise ValueError("duplicate or unexpected completed seed")
    pending = [s["seed"] for s in samples if s["seed"] not in by_seed]
    if len(pending) > 1:
        raise ValueError("recovery capped at one missing candidate report")
    for sample in samples:
        validate_submission(sample)
        if sample["prompt_hash"] != plan["prompt_hash"]:
            raise ValueError("recovery prompt mismatch")
        if sample["seed"] not in by_seed:
            continue
        report = by_seed[sample["seed"]]
        # Reuse the existing evidence gate. 'arm' is only a composite identity
        # field here; neither the saved sample nor its saved report is changed.
        unresolved = validate_prior({**sample, "arm": "checkpoint"}, {**report, "arm": "checkpoint"},
                                    suites, image_id)
        if unresolved or any(s["all_passed"] is None or s["infrastructure_errors"] for s in report["suites"]):
            raise ValueError("recovery never replaces a persisted unscored/candidate report")
    return {"parent_run": generation["run_id"], "gpu": False, "pending_seeds": pending,
            "reused_reports": len(completed), "max_additional_sandbox_executions": 54 * len(pending),
            "max_controller_seconds": 300, "max_retries": 0,
            "interrupted_candidate_attempts": "unknown; up to 54 may have run without a persisted report",
            "unrecorded_cleanup": "not confirmed; original sandboxes have a 120-second maximum lifetime",
            "new_samples": 0, "source_and_grading_changed": False}
