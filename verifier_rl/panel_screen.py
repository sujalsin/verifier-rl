"""Bounded generation-only screen contracts and offline evidence verification."""

import argparse
import ast
import asyncio
import json
import math
from pathlib import Path
import time

from .grading import ExecutionResult, Status
from .model_trial import submission_from_completion, validate_submission
from .panel_execution import (fixture_source, pack_result, request_for, require_evidence,
                              runner_for, unpack_result)
from .suites import canonical_json, digest
from .task_panel import TASK_IDS, build_panel, prompt_for, suite_hash, task_for
from .verifier_quality import (CalibrationRow, SCREEN_SEEDS, grade, preflight, screen_gate)

VERSION = "panel-screen-0.1"
RUN_ID = "qwen-panel-screen-20260928-v1"
CONTROLLER_SECONDS = 1200
GPU_SECONDS = 600
MAX_STARTS = 2


def make_plan(root):
    check = preflight(root)
    return {"version": VERSION, "run_id": RUN_ID, "preflight": check,
            "generation": check["screen_generation"],
            "prompts": {task_id: prompt_for(task_id, root) for task_id in TASK_IDS},
            "runner_hashes": {task_id: digest(runner_for(task_id)) for task_id in TASK_IDS},
            "concurrency": 8, "creation_interval_seconds": .26, "max_retries": 0,
            "training": False, "calibration": False, "max_programs": 24,
            "max_sandbox_executions": 786, "max_gpu_starts": MAX_STARTS,
            "max_controller_starts": MAX_STARTS, "max_gpu_seconds": GPU_SECONDS,
            "max_controller_seconds": CONTROLLER_SECONDS,
            "sample_ids": [sample_id(task_id, seed) for task_id in TASK_IDS for seed in SCREEN_SEEDS]}


def validate_plan(plan):
    from .checkpoint_pilot import CHAT_TEMPLATE_HASH, MODEL_ID, REPETITION_PENALTY, REVISION
    expected_decode = {"model_id": MODEL_ID, "revision": REVISION, "chat_template_hash": CHAT_TEMPLATE_HASH,
                       "seeds_per_task": list(SCREEN_SEEDS), "max_completion_tokens": 512,
                       "temperature": .8, "top_p": .95, "top_k": 0, "repetition_penalty": REPETITION_PENALTY,
                       "initial_policy": "untouched base instruct checkpoint; no SFT or RL"}
    if (plan["version"] != VERSION or plan["run_id"] != RUN_ID or plan["generation"] != expected_decode
            or plan["training"] is not False or plan["calibration"] is not False
            or plan["max_programs"] != 24 or plan["max_sandbox_executions"] != 786
            or plan["max_gpu_starts"] != MAX_STARTS or plan["max_controller_starts"] != MAX_STARTS
            or plan["max_gpu_seconds"] != GPU_SECONDS or plan["max_controller_seconds"] != CONTROLLER_SECONDS
            or plan["concurrency"] != 8 or plan["max_retries"] != 0 or plan["creation_interval_seconds"] != .26
            or set(plan["prompts"]) != set(TASK_IDS)
            or plan["sample_ids"] != [sample_id(t, s) for t in TASK_IDS for s in SCREEN_SEEDS]):
        raise ValueError("screen plan differs from fixed contract")
    panel = build_panel()
    if [row["task_id"] for row in plan["preflight"]["tasks"]] != list(TASK_IDS):
        raise ValueError("task identity/order mismatch")
    for row in plan["preflight"]["tasks"]:
        task_id = row["task_id"]
        if (digest(plan["prompts"][task_id]) != row["prompt_hash"]
                or row["suite_hashes"] != {split: suite_hash(cases) for split, cases in panel[task_id].items()}
                or plan["runner_hashes"][task_id] != digest(runner_for(task_id))):
            raise ValueError("screen prompt, suite or runner identity mismatch")
    return panel


def sample_id(task_id, seed):
    task_for(task_id)
    if type(seed) is not int or seed not in SCREEN_SEEDS:
        raise ValueError("unexpected screen generation seed")
    return f"{task_id}-{seed}"


def inspect_completion(task_id, raw):
    sample = submission_from_completion(raw)
    sample.update(syntax_valid=False, has_top_level_entrypoint=False, syntax_error=None)
    if not sample["extraction_status"].startswith("rejected_"):
        try:
            # Compile/parse only. No exec, import or calls into candidate code.
            compile(sample["source"], "<candidate>", "exec")
            tree = ast.parse(sample["source"])
            sample["syntax_valid"] = True
            sample["has_top_level_entrypoint"] = any(isinstance(node, ast.FunctionDef)
                and node.name == task_for(task_id).entrypoint for node in tree.body)
        except (SyntaxError, ValueError) as exc:
            sample["syntax_error"] = {"type": type(exc).__name__, "line": getattr(exc, "lineno", None)}
    return sample


def validate_sample(sample, plan):
    validate_submission(sample)
    if (sample["sample_id"] != sample_id(sample["task_id"], sample["seed"])
            or sample["prompt_hash"] != digest(plan["prompts"][sample["task_id"]])
            or type(sample["tokens"]) is not int or not 0 < sample["tokens"] <= 512
            or type(sample["ended_with_eos"]) is not bool
            or sample["hit_token_cap"] is not (sample["tokens"] == 512)):
        raise ValueError("sample identity/decoding record mismatch")
    inspected = inspect_completion(sample["task_id"], sample["raw"])
    if any(sample[key] != value for key, value in inspected.items()):
        raise ValueError("source inspection differs from saved sample")


def control_items(panel):
    for task_id in TASK_IDS:
        cases = panel[task_id]["training"]
        for fault in ("correct", "inclusive_boundary", "constant"):
            yield {"id": f"control-{task_id}-{fault}", "task_id": task_id, "fault": fault,
                   "source": fixture_source(task_id, fault), "cases": (cases[0], cases[12])}


async def execute_cases(source, cases, backend, image_id, deadline, record, concurrency=8):
    """Bounded scheduler with per-input raw evidence callback and no retries.

    A callback failure or lost input never grants permission to repeat execution.
    In-flight sandboxes finish/clean up after another worker reports an error.
    """
    queue = asyncio.Queue()
    for case in cases:
        case.expected
        queue.put_nowait(case)
    stopped = asyncio.Event()
    results = {}

    async def worker():
        while not stopped.is_set():
            if time.time() >= deadline:
                stopped.set()
                return
            try:
                case = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = await backend.execute(request_for(source, case))
            results[case.input_hash] = result
            await record(case, result)  # Preserve raw bytes before validation.
            try:
                require_evidence(result, case.task_id, image_id, source=source, case=case)
            except ValueError:
                stopped.set()
            queue.task_done()
    workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(cases)))]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker_task in workers:
            if not worker_task.done():
                worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    return results


def program_report(sample, cases_by_split, records, image_id):
    cases = tuple(case for split in ("training", "development") for case in cases_by_split[split])
    rejected = sample["extraction_status"].startswith("rejected_")
    if rejected:
        if records:
            raise ValueError("rejected extraction was unexpectedly executed")
        results = {case.input_hash: ExecutionResult(Status.EXTRACTION_REJECTED) for case in cases}
        ids = []
    else:
        if set(records) != {case.input_hash for case in cases}:
            raise ValueError("incomplete program: preserve missing/uncertain observations")
        results = {key: unpack_result(value) for key, value in records.items()}
        ids = [require_evidence(results[case.input_hash], sample["task_id"], image_id,
                                source=sample["source"], case=case) for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("sandbox reused across inputs")
    scores = {split: grade(selected, {case.input_hash: results[case.input_hash] for case in selected})
              for split, selected in cases_by_split.items()}
    if not all(score["resolved"] for score in scores.values()):
        raise ValueError("unresolved screen program")
    return {"version": VERSION, "sample_id": sample["sample_id"], "task_id": sample["task_id"],
            "candidate_hash": digest(sample["source"]), "scores": scores, "executions": len(ids),
            "sandbox_ids": ids, "extraction_rejected": rejected,
            "record_hash": digest(canonical_json(records)),
            "accounted_sandbox_seconds": sum(min(120, max(1, math.ceil(r.metadata["total_seconds"])))
                                               for r in results.values()) if not rejected else 0}


def validate_controls(plan, controls, image_id):
    from .verifier_quality import compare_output
    expected = tuple(control_items(validate_plan(plan)))
    if set(controls) != {item["id"] for item in expected}:
        raise ValueError("all nine controls required before generation")
    ids = []
    for item in expected:
        saved = controls[item["id"]]
        if (saved["task_id"] != item["task_id"] or saved["source"] != item["source"]
                or set(saved["records"]) != {case.input_hash for case in item["cases"]}):
            raise ValueError("conformance identity mismatch")
        observed = []
        for case in item["cases"]:
            result = unpack_result(saved["records"][case.input_hash])
            ids.append(require_evidence(result, item["task_id"], image_id, source=item["source"], case=case))
            observed.append(compare_output(case, result.stdout)[0] if result.status == Status.COMPLETED else False)
        wanted = {"correct": [True, True], "inclusive_boundary": [True, False], "constant": [False, False]}[item["fault"]]
        if observed != wanted:
            raise ValueError("live authored control disagrees: " + item["id"])
    if len(ids) != 18 or len(set(ids)) != 18:
        raise ValueError("live control sandbox count/reset mismatch")
    return ids


def summarize(plan, generation, controls, program_records, image_id):
    from .measurement_v2 import PARAMETER_HASH
    panel = validate_plan(plan)
    ids = validate_controls(plan, controls, image_id)
    if (generation["plan_hash"] != digest(canonical_json(plan))
            or generation["before_parameter_hash"] != PARAMETER_HASH
            or generation["after_parameter_hash"] != PARAMETER_HASH
            or generation["parameters_unchanged"] is not True
            or [s["sample_id"] for s in generation["samples"]] != plan["sample_ids"]
            or set(program_records) != set(plan["sample_ids"])):
        raise ValueError("incomplete/mismatched screen or initial policy")
    rows, gates, reports = [], {}, {}
    for sample in generation["samples"]:
        validate_sample(sample, plan)
        result = program_report(sample, panel[sample["task_id"]], program_records[sample["sample_id"]], image_id)
        reports[sample["sample_id"]] = result
        ids.extend(result["sandbox_ids"])
        training, audit = result["scores"]["training"], result["scores"]["development"]
        row = CalibrationRow(sample["sample_id"], result["candidate_hash"], sample["task_id"], "screen",
                             training["reference_accepted"], training["structured_accepted"], audit["reference_accepted"])
        rows.append(row)
    if len(ids) != len(set(ids)) or len(ids) > plan["max_sandbox_executions"]:
        raise ValueError("reused sandbox or execution count exceeded")
    for task_id in TASK_IDS:
        selected = [row for row in rows if row.task_id == task_id]
        gate = screen_gate(task_id, selected)
        samples = [sample for sample in generation["samples"] if sample["task_id"] == task_id]
        gate.update(unique_sources=len({row.source_hash for row in selected}),
                    syntax_valid=sum(sample["syntax_valid"] for sample in samples),
                    token_cap_hits=sum(sample["hit_token_cap"] for sample in samples),
                    extraction_rejections=sum(sample["extraction_status"].startswith("rejected_") for sample in samples),
                    structured_only_acceptances=sum(row.structured_accepted and not row.reference_accepted for row in selected),
                    reference_false_accepts=sum(row.reference_accepted and not row.audit_accepted for row in selected),
                    structured_false_accepts=sum(row.structured_accepted and not row.audit_accepted for row in selected),
                    audit_rejected=sum(not row.audit_accepted for row in selected),
                    development_family_passes={family: sum(reports[row.sample_id]["scores"]["development"]["family_passes"][family]
                                                             for row in selected)
                                               for family in ("ordinary", "state", "interaction", "boundary")})
        gates[task_id] = gate
    return {"version": VERSION, "run_id": RUN_ID, "verified": True, "training": False,
            "calibrated": False, "model_samples": len(rows), "sandbox_executions": len(ids),
            "parameters_unchanged": True, "gates": gates, "programs": [row.__dict__ for row in rows],
            "all_tasks_screen_eligible": all(gate["ready_for_calibration_review"] for gate in gates.values()),
            "limitations": ["Eight outputs/task are a feasibility screen, not efficacy evidence.",
                            "Development suite is not a final held-out benchmark.",
                            "Known-boundary false acceptance alone is not RL-induced exploitation.",
                            "No calibration, reward randomization or optimizer update occurred."]}


def verify_directory(directory, *, continuation=None):
    def read(relative):
        return json.loads((directory / relative).read_text())
    plan, setup = read("plan.json"), read("setup.json")
    panel = validate_plan(plan)
    controls = {item["id"]: read(f"controls/{item['id']}/result.json")
                for item in control_items(panel)}
    generation = read("gpu/generation.json")
    if continuation is None:
        paths = {sid: f"programs/{sid}/result.json" for sid in plan["sample_ids"]}
        summary_path = "summary.json"
    elif continuation == "continuation-001":
        paths = read(f"{continuation}/selected_paths.json")
        summary_path = f"{continuation}/summary.json"
        if set(paths) != set(plan["sample_ids"]):
            raise ValueError("continuation must select exactly the original 24 samples")
        # Do not trust arbitrary paths, or allow a completed original result to
        # be silently replaced with a preferable new execution.
        for sid, selected_path in paths.items():
            if selected_path not in (f"programs/{sid}/result.json",
                                    f"{continuation}/programs/{sid}/result.json",
                                    f"{continuation}/reconstructed/{sid}/result.json"):
                raise ValueError("selected path outside declared screen records")
    else:
        raise ValueError("unknown screen continuation")
    raws = {sid: read(path) for sid, path in paths.items()}
    records = {sid: raw["records"] for sid, raw in raws.items()}
    result = summarize(plan, generation, controls, records, setup["sandbox_image_id"])
    if read(summary_path) != result:
        raise ValueError("saved summary differs from independently recomputed evidence")
    if continuation is not None:
        from .verifier_quality import compare_output
        reconciliation = read(f"{continuation}/reconciliation.json")
        if (reconciliation.get("original_app_id") != "ap-CKoxLnZYt91BKvGnTm3Fh1"
                or reconciliation.get("original_state") != "stopped"
                or reconciliation.get("original_tasks") != "0"
                or reconciliation.get("active_sandbox_ids") != []
                or reconciliation.get("original_call_error_type") != "FunctionTimeoutError"
                or reconciliation.get("reason") != "controller_submission_or_function_deadline"):
            raise ValueError("missing controller deadline reconciliation")
        ids = set(validate_controls(plan, controls, setup["sandbox_image_id"]))
        for sample in generation["samples"]:
            raw = raws[sample["sample_id"]]
            if raw["source"] != sample["source"] or raw["task_id"] != sample["task_id"]:
                raise ValueError("selected result source/task mismatch")
            report = program_report(sample, panel[sample["task_id"]], raw["records"], setup["sandbox_image_id"])
            ids.update(report["sandbox_ids"])
        pending, partials = [], {}
        for sample in generation["samples"]:
            sid = sample["sample_id"]
            original_path = f"programs/{sid}/result.json"
            original_dir = directory / "programs" / sid
            original = read(original_path) if (directory / original_path).exists() else None
            leaves = {path.stem: json.loads(path.read_text()) for path in (original_dir / "inputs").glob("*.json")}
            if original is not None and (len(original["records"]) == 32
                                          or sample["extraction_status"].startswith("rejected_")):
                if paths[sid] != original_path:
                    raise ValueError("completed original result was replaced")
                continue
            if len(leaves) == 32:
                if (paths[sid] != f"{continuation}/reconstructed/{sid}/result.json"
                        or raws[sid]["records"] != {key: value["execution"] for key, value in leaves.items()}):
                    raise ValueError("complete input records were not reused unchanged")
                continue
            pending.append(sid)
            if paths[sid] != f"{continuation}/programs/{sid}/result.json":
                raise ValueError("pending program selection mismatch")
            if not (original_dir / "intent.json").exists():
                if leaves or original is not None:
                    raise ValueError("original observations lack an intent")
                continue
            intent = json.loads((original_dir / "intent.json").read_text())
            cases = {case.input_hash: case for split in panel[sample["task_id"]].values() for case in split}
            if (intent["source_hash"] != digest(sample["source"]) or intent["task_id"] != sample["task_id"]
                    or intent["input_hashes"] != list(cases)):
                raise ValueError("interrupted intent differs from frozen inputs")
            if original is not None and any(key not in leaves or leaves[key]["execution"] != value
                                            for key, value in original["records"].items()):
                raise ValueError("partial batch and input records disagree")
            partials[sid] = leaves
            for key, value in leaves.items():
                if key not in cases:
                    raise ValueError("unknown partial input")
                case = cases[key]
                before = unpack_result(value["execution"])
                identity = require_evidence(before, sample["task_id"], setup["sandbox_image_id"],
                                            source=sample["source"], case=case)
                if identity in ids:
                    raise ValueError("partial sandbox ID reused")
                ids.add(identity)
                after = unpack_result(records[sid][key])
                def outcome(value):
                    return compare_output(case, value.stdout)[0] if value.status == Status.COMPLETED else False
                if outcome(before) != outcome(after):
                    raise ValueError("replacement disagrees with a recorded original outcome")
        if (len(partials) > 1 or read(f"{continuation}/pending.json") != pending
                or read(f"{continuation}/original_partial_records.json") != partials):
            raise ValueError("original partial observations or pending set changed")
        known_partial = sum(len(raw) for raw in partials.values())
        accounting = {"reused_complete_programs": 24 - len(pending), "new_program_batches": len(pending),
                      "controller_deadline_replacements": len(partials), "new_model_samples": 0,
                      "recorded_original_partial_inputs": known_partial,
                      "total_execution_bounds": [result["sandbox_executions"] + known_partial,
                                                  result["sandbox_executions"] + 32 * len(partials)],
                      "repeated_recorded_outcomes_agree": True}
        if read(f"{continuation}/accounting.json") != accounting:
            raise ValueError("saved continuation accounting differs from raw evidence")
        result = dict(result, continuation_accounting=accounting)
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify a completed local screen without cloud or candidate execution")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--continuation", choices=["continuation-001"])
    args = parser.parse_args()
    print(json.dumps(verify_directory(args.run, continuation=args.continuation), indent=2))


if __name__ == "__main__":
    main()
