"""Frozen eight-seed prompt comparison and analysis without candidate execution."""

from collections import Counter

from .baseline import MODEL_REVISION
from .model_trial import EXTRACTION_VERSION, MAX_COMPLETION_TOKENS, MODEL_ID
from .suites import build_suites, digest

PROTOCOL = "cache-interface-prompt-pilot-0.1"
SEEDS = tuple(range(4000, 4008))
ARMS = ("original", "interface_reminders")
REMINDERS = """Submission checklist:
- Define the function with the exact name simulate_cache.
- A get dictionary has only op, time and key. It has no value or ttl fields.
  Access value and ttl only when processing a put operation.
- Accumulate one result for every get, and return the complete list after all
  operations. Do not return a scalar or stop at the first get.
- Submit Python function definitions and any necessary imports only. Do not add
  demonstration calls, example operations, print statements, tests or prose.
"""


def prompts_for(original):
    if not original.strip():
        raise ValueError("nonempty original task prompt required")
    return {"original": original, "interface_reminders": original + "\n\n" + REMINDERS.strip()}


def pilot_suites():
    return tuple(s for s in build_suites() if s.name in ("g1", "g3"))


def schedule():
    # Alternate which arm is generated first; reset the RNG for each draw.
    return [(arm, seed) for i, seed in enumerate(SEEDS)
            for arm in (ARMS if i % 2 == 0 else ARMS[::-1])]


def protocol_plan(original):
    prompts = prompts_for(original)
    suites = pilot_suites()
    inputs = {c.input_hash for suite in suites for c in suite.cases}
    return {"protocol": PROTOCOL, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "prompts": prompts, "prompt_hashes": {arm: digest(p) for arm, p in prompts.items()},
            "seeds": list(SEEDS), "schedule": schedule(), "samples_per_arm": len(SEEDS),
            "max_completion_tokens": MAX_COMPLETION_TOKENS, "temperature": 0.8,
            "top_p": 0.95, "top_k": 0, "extraction_version": EXTRACTION_VERSION,
            "suite_hashes": {s.name: s.fingerprint for s in suites},
            "max_sandbox_executions": len(SEEDS) * len(ARMS) * len(inputs),
            "concurrency": 4, "creation_interval_seconds": 0.26, "max_retries": 0,
            "max_gpu_seconds": 600, "max_controller_seconds": 1500,
            "training": False, "primary_outcome": "G3 complete-suite passes per 8 programs",
            "decision": "No automatic RL or extra samples; inspect complete-suite and interface outcomes.",
            "limitations": [
                "Eight paired seeds per prompt: a small development pilot.",
                "One task; G3 is a development suite, not an independent final audit.",
                "The treatment bundles interface reminders; individual reminder effects are not isolated.",
                "Matched RNG seeds do not make different prompts generate identical trajectories.",
                "Prompt length changes; decoding settings and completion-token limit stay fixed.",
                "No candidate source repair or removal of demonstration calls.",
            ]}


def summarize(generation, reports):
    plan = generation["plan"]
    if plan != protocol_plan(plan["prompts"]["original"]):
        # JSON roundtrips turn schedule tuples into lists.
        from .suites import canonical_json
        if canonical_json(plan) != canonical_json(protocol_plan(plan["prompts"]["original"])):
            raise ValueError("generation plan differs from frozen pilot protocol")
    if generation.get("parameters_unchanged") is not True:
        raise ValueError("pilot must not change model weights")
    expected = set(schedule())
    samples = {(s["arm"], s["seed"]): s for s in generation["samples"]}
    by_key = {(r["arm"], r["seed"]): r for r in reports}
    if (set(samples) != expected or len(generation["samples"]) != len(expected)
            or set(by_key) != expected or len(reports) != len(expected)):
        raise ValueError("missing, duplicate, or unexpected pilot samples/reports")
    arms = {}
    for arm in ARMS:
        rows, reasons, stages = [], Counter(), Counter()
        actual_executions = 0
        for seed in SEEDS:
            sample, report = samples[arm, seed], by_key[arm, seed]
            if (report["candidate_hash"] != digest(sample["source"])
                    or sample["prompt_hash"] != plan["prompt_hashes"][arm]):
                raise ValueError("sample/report source or prompt mismatch")
            if {s["suite"]: s["suite_hash"] for s in report["suites"]} != plan["suite_hashes"]:
                raise ValueError("pilot evaluation suite mismatch")
            if any(s["infrastructure_errors"] for s in report["suites"]):
                raise ValueError("unresolved infrastructure errors cannot count as candidate failures")
            unique = {o["input_hash"]: o for s in report["suites"] for o in s["outcomes"]}
            if not unique:
                raise ValueError("missing case outcomes")
            reasons.update(o["reason"] for o in unique.values())
            stages.update({a["metadata"].get("runner_stage", "unknown")
                           for o in unique.values() if not o["passed"]
                           for a in o["attempts"]})
            actual_executions += report["execution_config"]["attempts"]
            rows.append({"seed": seed, "source_hash": report["candidate_hash"],
                         "extraction_status": sample["extraction_status"],
                         "syntax_valid": sample["syntax_valid"],
                         "hit_token_cap": sample["hit_token_cap"],
                         "any_valid_output": any(o["reason"] in ("pass", "wrong_answer") for o in unique.values()),
                         "all_outputs_valid": all(o["reason"] in ("pass", "wrong_answer") for o in unique.values()),
                         "scores": {s["suite"]: {"passed": s["passed_count"], "total": s["total"],
                                                 "all_passed": s["all_passed"]}
                                    for s in report["suites"]}})
        arms[arm] = {"sample_count": len(rows), "unique_source_count": len({r["source_hash"] for r in rows}),
                     "extractable_count": sum(not r["extraction_status"].startswith("rejected_") for r in rows),
                     "syntax_valid_count": sum(r["syntax_valid"] for r in rows),
                     "hit_token_cap_count": sum(r["hit_token_cap"] for r in rows),
                     "candidates_with_any_valid_output": sum(r["any_valid_output"] for r in rows),
                     "candidates_with_all_outputs_valid": sum(r["all_outputs_valid"] for r in rows),
                     "full_suite_pass_counts": {name: sum(r["scores"][name]["all_passed"] for r in rows)
                                                for name in ("g1", "g3")},
                     "case_pass_counts": {name: sum(r["scores"][name]["passed"] for r in rows)
                                          for name in ("g1", "g3")},
                     "assigned_case_totals": {name: sum(r["scores"][name]["total"] for r in rows)
                                              for name in ("g1", "g3")},
                     "mixed_g3_groups_of_four": sum(len({r["scores"]["g3"]["all_passed"]
                                                          for r in rows[i:i + 4]}) > 1
                                                    for i in range(0, len(rows), 4)),
                     "group_count": 2, "actual_sandbox_executions": actual_executions,
                     "assigned_input_reasons": dict(reasons),
                     "candidates_by_failure_stage_untrusted": dict(stages), "candidates": rows}
    return {"protocol": PROTOCOL, "run_id": generation["run_id"], "training": False,
            "parameters_unchanged": True, "gpu_function_seconds": generation["gpu_function_seconds"],
            "arms": arms, "limitations": plan["limitations"]}
