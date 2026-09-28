"""Complete at most eight pending baseline samples with a remote orchestrator.

Use modal run --detach modal_finish_baseline.py::finish ... so loss of the
laptop connection does not stop the remote orchestration loop.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from modal_baseline import app, artifacts, cpu_image, grade_candidates
from verifier_rl.baseline import MODEL_REVISION, SEEDS, inspect_completion, summarize, validate_recovered_report
from verifier_rl.cli import create_run_directory, write_private
from verifier_rl.model_trial import EXTRACTION_VERSION, MAX_COMPLETION_TOKENS, canonical_prompt, validate_run_id
from verifier_rl.suites import build_suites, canonical_json, digest

finisher_image = cpu_image.add_local_python_source("modal_baseline")


@app.function(image=finisher_image, cpu=1, memory=1024, timeout=3000,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def finish_remaining(setup: dict, run_id: str, generation: dict, recovery_runs: list[str]):
    validate_run_id(run_id)
    if not 1 <= len(recovery_runs) <= 4:
        raise ValueError("one to four explicit recovery runs required")
    for root in recovery_runs:
        validate_run_id(root)
    artifacts.reload()
    directory = create_run_directory(f"/artifacts/{run_id}")
    started = time.monotonic()
    write_private(directory / "generation.json", canonical_json(generation))
    write_private(directory / "inputs.json", canonical_json({
        "setup": setup, "recovery_runs": recovery_runs, "max_pending_samples": 8,
        "gpu_called": False, "orchestration": "remote", "timeout_seconds": 3000}))
    artifacts.commit()
    try:
        suites = build_suites()
        recovered, sources = {}, {}
        for root in recovery_runs:
            for path in sorted(Path(f"/artifacts/{root}").rglob("*-result.json")):
                report = json.loads(path.read_text())
                seed = report.get("sample_seed")
                if seed not in SEEDS:
                    raise ValueError("unexpected seed in recovery records")
                sample = generation["samples"][SEEDS.index(seed)]
                if validate_recovered_report(sample, report, suites, setup["sandbox_image_id"]) and seed not in recovered:
                    recovered[seed] = report
                    sources[seed] = str(path)
                    write_private(directory / f"recovered-{seed}.json", canonical_json(report))
        if len(recovered) < 24:
            raise ValueError("finisher requires at least 24 completed candidates; refuses larger spending scope")
        write_private(directory / "recovery.json", canonical_json(sources))
        artifacts.commit()
        print(f"Recovered {len(recovered)}/32; completing only {32 - len(recovered)} remaining samples.", flush=True)
        by_seed = dict(recovered)
        for batch_index in range(16):
            batch = [s for s in generation["samples"][batch_index * 2:batch_index * 2 + 2]
                     if s["seed"] not in by_seed]
            if batch:
                results = grade_candidates.remote(setup, run_id, batch_index, batch)
                write_private(directory / f"grading-call-{batch_index}.json", canonical_json(results))
                by_seed.update({r["sample_seed"]: r for r in results})
                artifacts.commit()
                print(f"Completed {len(by_seed)}/32", flush=True)
        reports = [by_seed[s] for s in SEEDS]
        for group in range(8):
            write_private(directory / f"batch-{group}.json", canonical_json(reports[group * 4:group * 4 + 4]))
        summary = summarize(generation["samples"], reports)
        summary.update({"run_id": run_id, "generation_source_run_id": generation["generation_source_run_id"],
                        "gpu_function_seconds": generation["gpu_function_seconds"],
                        "parameters_unchanged": generation["parameters_unchanged"],
                        "recovered_count": len(recovered), "orchestration_seconds": time.monotonic() - started})
        write_private(directory / "summary.json", canonical_json(summary))
        artifacts.commit()
        print("Baseline finished:", summary["suite_full_pass_counts"], flush=True)
        return summary
    except Exception as exc:
        write_private(directory / "error.json", canonical_json({"type": type(exc).__name__,
                                                               "seconds": time.monotonic() - started}))
        artifacts.commit()
        raise


@app.local_entrypoint()
def finish(setup_file: str, cpu_report: str, saved_generation: str, recovery_runs: str):
    setup = json.loads(Path(setup_file).read_text())
    conformance = json.loads(Path(cpu_report).read_text())
    generation = json.loads(Path(saved_generation).read_text())
    if not conformance.get("passed") or conformance.get("completed_checks") != 15:
        raise ValueError("passing conformance report required")
    tested_images = {a["metadata"]["image_id"] for c in conformance["checks"]
                     for s in c["report"]["suites"] for o in s["outcomes"] for a in o["attempts"]}
    if tested_images != {setup["sandbox_image_id"]}:
        raise ValueError("sandbox image differs from conformance-tested image")
    prompt = canonical_prompt(Path("task_001_expiring_cache.txt").read_text())
    if (generation["config"]["model_revision"] != MODEL_REVISION or
            generation["config"]["prompt_hash"] != digest(prompt) or
            generation["config"]["extraction_version"] != EXTRACTION_VERSION or
            generation["config"]["max_completion_tokens"] != MAX_COMPLETION_TOKENS or
            not generation["parameters_unchanged"] or
            [s["seed"] for s in generation["samples"]] != list(SEEDS)):
        raise ValueError("saved generation differs from baseline protocol")
    for sample in generation["samples"]:
        if inspect_completion(sample["raw"])["source"] != sample["source"]:
            raise ValueError("saved source differs from extraction contract")
    run_id = "qwen-baseline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    generation["generation_source_run_id"] = generation.get("generation_source_run_id", generation["run_id"])
    generation["run_id"] = run_id
    directory = create_run_directory(f"runs/{run_id}")
    write_private(directory / "generation.json", canonical_json(generation))
    roots = recovery_runs.split(",")
    write_private(directory / "launch.json", canonical_json({"setup": setup, "recovery_runs": roots,
                  "saved_generation": saved_generation, "remote_summary": f"{run_id}/summary.json"}))
    print("Remote completion run:", run_id, flush=True)
    summary = finish_remaining.remote(setup, run_id, generation, roots)
    write_private(directory / "summary.json", canonical_json(summary))
    print("Finished:", canonical_json({"run_id": run_id, "passes": summary["suite_full_pass_counts"]}))
