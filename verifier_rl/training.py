"""Bounded GRPO contracts, shared by the cache trial and optimizer controls."""

import math
import re

MAX_STEPS = 4
GROUP_SIZE = 4


def validate_steps(steps):
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"max_steps must be 1..{MAX_STEPS}")


def training_batch_id(index):
    if type(index) is not int or not 0 <= index < MAX_STEPS:
        raise ValueError("training batch index outside bounded trial")
    return f"train-{index:04d}"


def validate_batch_id(batch_id, audit):
    if batch_id == "evaluation" and audit is True:
        return
    if audit is not False or re.fullmatch(r"train-000[0-3]", batch_id) is None:
        raise ValueError("invalid training/evaluation batch ID")


def grpo_kwargs(output_dir, *, max_steps=1, completion_tokens=512, seed=20260926):
    validate_steps(max_steps)
    if type(completion_tokens) is not int or not 1 <= completion_tokens <= 512:
        raise ValueError("completion token budget must be 1..512")
    return dict(output_dir=str(output_dir), max_steps=max_steps,
                per_device_train_batch_size=1, gradient_accumulation_steps=GROUP_SIZE,
                num_generations=GROUP_SIZE, steps_per_generation=GROUP_SIZE, num_iterations=1,
                max_completion_length=completion_tokens, learning_rate=1e-6, weight_decay=0.0,
                lr_scheduler_type="constant", bf16=True, gradient_checkpointing=True,
                use_vllm=False, beta=0.0, loss_type="grpo", scale_rewards="group",
                temperature=0.8, top_p=0.95, top_k=0, save_strategy="no", logging_steps=1,
                report_to="none", seed=seed, data_seed=seed)


def control_rewards(raw, condition):
    """Synthetic optimizer test only: NOT coding rewards or capability evidence."""
    if len(raw) != GROUP_SIZE or not all(isinstance(x, str) for x in raw):
        raise ValueError("control requires exactly four text completions")
    if condition == "zero":
        return [0.0] * GROUP_SIZE
    if condition != "mixed":
        raise ValueError("unknown control condition")
    if len(set(raw)) < 2:
        raise ValueError("identical control completions: cannot test a differential update")
    # Pick a content class rather than assigning conflicting labels to identical text.
    chosen = min(raw)
    return [float(text == chosen) for text in raw]


def require_control_evidence(metrics, condition):
    expected_steps = 1 if condition == "zero" else 2
    if condition not in ("zero", "mixed") or metrics["global_step"] != expected_steps:
        raise ValueError("control step count mismatch")
    groups = metrics["rewards"]
    grads = [row["grad_norm"] for row in metrics["log_history"] if "grad_norm" in row]
    if (len(groups) != expected_steps or len(grads) != expected_steps
            or any(len(g) != GROUP_SIZE for g in groups)
            or not all(math.isfinite(x) for x in grads)
            or metrics.get("checkpoint_reload_hash_matches") is not True):
        raise ValueError("incomplete/nonfinite control evidence")
    changed = metrics["before_parameter_hash"] != metrics["after_parameter_hash"]
    if condition == "zero":
        if changed or any(x != 0 for x in grads) or any(any(g) for g in groups):
            raise ValueError("zero-reward control unexpectedly updated")
    elif not changed or not all(x > 0 for x in grads) or not all(set(g) == {0.0, 1.0} for g in groups):
        raise ValueError("mixed-reward control did not demonstrate updates")


def require_grader_controls(report):
    from .fixtures import source_for
    from .modal_backend import RUNNER
    from .model_trial import submission_from_completion
    from .suites import build_suites, digest
    if report.get("kind") != "grader_controls" or report.get("passed") is not True or len(report.get("reports", [])) != 2:
        raise ValueError("passing known-correct/known-faulty G3 controls required")
    suite = build_suites()[2]
    images = set()
    for name, candidate, reward in zip(("correct", "inclusive_expiry"), report["reports"], (1, 0)):
        # Evidence hashes the extracted bytes that actually ran, not the raw
        # fixture (whose surrounding whitespace extraction removes).
        expected = submission_from_completion(source_for(name))
        if candidate["candidate_hash"] != digest(expected["source"]) or len(candidate["suites"]) != 1:
            raise ValueError("grader control source mismatch")
        if candidate.get("extraction") != {"status": expected["extraction_status"],
                                            "version": expected["extraction_version"], "accepted": True}:
            raise ValueError("grader control extraction mismatch")
        result = candidate["suites"][0]
        if (result["suite_hash"] != suite.fingerprint or result["reward"] != reward
                or result["infrastructure_errors"] or len(result["outcomes"]) != len(suite.cases)):
            raise ValueError("grader control score/coverage mismatch")
        for case, outcome in zip(suite.cases, result["outcomes"]):
            if outcome["input_hash"] != case.input_hash or outcome["expected"] != case.expected or not outcome["attempts"]:
                raise ValueError("grader control case mismatch")
            for attempt in outcome["attempts"]:
                m = attempt["metadata"]
                images.add(m.get("image_id"))
                if m.get("runner_hash") != digest(RUNNER) or m.get("cleanup") != "terminated":
                    raise ValueError("grader control runner/cleanup mismatch")
    if len(images) != 1 or not all(isinstance(x, str) and x.startswith("im-") for x in images):
        raise ValueError("grader control image mismatch")
