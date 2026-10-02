"""CPU-only, read-only evidence verification. Never trains or regrades programs."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import modal

import modal_booking_matched_training as frozen
from verifier_rl import booking_cleanup_recovery as cleanup
from verifier_rl import booking_matched_training as release
from verifier_rl import booking_baseline_comparison as study
from verifier_rl.parallel_finalization import (
    VERSION, WORKERS, PrefetchJSONReader, journal_records, parallel_reader,
)
from verifier_rl.evaluation_journal import persist
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import canonical_json, digest

app = modal.App("verifier-rl-booking-parallel-finalization")
SERIAL_APP = "ap-I3H8dy8IdlgnvItfooQtFO"
FINALIZATION_ID = "parallel-002"
BENCH_KEY = "eval-endpoint_omission-24-06"
BENCH_SECONDS, FINAL_SECONDS = 1800, 7200
OUTPUT_VOLUME = "verifier-rl-finalization-reports"
ADDED_SOURCES = ("modal_booking_parallel_finalize.py", "verifier_rl/parallel_finalization.py")
image = frozen.cpu_image.add_local_file("modal_booking_parallel_finalize.py", "/root/modal_booking_parallel_finalize.py")
reports = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)
volumes = {"/artifacts": frozen.artifacts.with_mount_options(read_only=True), "/reports": reports}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def output_directory():
    return Path("/reports") / release.RUN_ID / FINALIZATION_ID


def source_binding(root):
    return {name: digest((root / name).read_text()) for name in ADDED_SOURCES}


def prepare(sources):
    """Validate unchanged research source, completed arms, and fixed baseline."""
    if source_binding(Path("/root")) != sources:
        raise ValueError("parallel reader implementation differs from launch")
    root = Path("/artifacts") / release.RUN_ID
    read = frozen.recovery.read_json
    plan, setup = read(root / "plan.json"), read(root / "setup.json")
    release.validate_plan(plan)
    original = read(root / "source_snapshot.json")
    snapshot = read(root / "amendments/cleanup-001/source_snapshot.json")
    if cleanup.validate_amendment(original, snapshot) != read(root / "amendments/cleanup-001/amendment.json"):
        raise ValueError("execution amendment differs")
    frozen.check_snapshot(snapshot, Path("/root"))
    frozen.require_live_control(plan, snapshot)
    baseline_path = Path("/artifacts") / plan["baseline_run_id"] / "result.json"
    baseline = release.validate_baseline(baseline_path.read_bytes(), read(baseline_path), plan)
    if read(root / "baseline.json") != baseline:
        raise ValueError("saved baseline differs")
    arms = {arm: release.validate_arm(read(root / "arms" / arm / "result.json"), plan) for arm in study.ARMS}
    # These exist only after all four fixed-policy grading stages have finished.
    measurements = {f"{arm}-{step}": read(root / f"measurements/{arm}-{step}.json")
                    for arm in study.ARMS for step in (12, 24)}
    binding = {"run_id": release.RUN_ID, "source_snapshot_hash": digest(canonical_json(snapshot)),
               "plan_hash": digest(canonical_json(plan)), "setup_hash": digest(canonical_json(setup)),
               "arm_hashes": {a: digest(canonical_json(v)) for a, v in arms.items()},
               "baseline_sha256": plan["baseline_sha256"], "sources": sources,
               "measurements_hash": digest(canonical_json(measurements))}
    return root, plan, setup, arms, baseline, measurements, binding


def batch_keys():
    yield "controls"
    for arm in study.ARMS:
        for index in range(24):
            yield f"train-{arm}-{index:02d}"
        for step in (12, 24):
            for index in range(8):
                yield f"eval-{arm}-{step:02d}-{index:02d}"


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), nonpreemptible=True,
              timeout=BENCH_SECONDS, retries=0, max_containers=1, scaledown_window=2, volumes=volumes)
def benchmark(sources):
    frozen.claim(f"parallel-finalization/{FINALIZATION_ID}/benchmark", maximum=1)
    root, plan, setup, arms, baseline, measurements, binding = prepare(sources)
    output = output_directory()
    persist(output, {"binding": binding})
    reports.commit()
    read = frozen.recovery.read_json
    with ProgressLog(release.RUN_ID, label="PARALLEL_BENCHMARK") as progress:
        progress.stage("inventory_saved_batches", total=81, unit="batches")
        counts = {}
        for key in batch_keys():
            raw = read(root / f"grading/{key}/raw.json")
            counts[key] = sum(1 for _ in journal_records(key, raw))
            progress.advance()
        key = BENCH_KEY
        raw = read(root / f"grading/{key}/raw.json")
        start_index = 4 * int(key.rsplit("-", 1)[1])
        samples = arms["endpoint_omission"]["evaluations"]["24"]["samples"][start_index:start_index+4]
        checked = study.verify_raw(raw, samples, key, "evaluation", plan["experiment"], setup["sandbox_image_id"])
        if read(root / f"grading/{key}/result.json") != {"raw": raw, "checked": checked}:
            raise ValueError("benchmark saved batch differs from frozen score replay")
        records = list(journal_records(key, raw))
        if len(records) < 256:
            raise ValueError("benchmark requires a representative full batch")
        progress.stage("serial_sample", total=64, unit="files")
        start = time.monotonic()
        serial_values = {}
        for relative, expected in records[:64]:
            serial_values[relative] = read(root / relative)
            if serial_values[relative] != expected:
                raise ValueError("serial benchmark record differs")
            progress.advance()
        serial_seconds = time.monotonic() - start
        progress.stage("parallel_full_batch", total=len(records), unit="files")
        start = time.monotonic()
        with PrefetchJSONReader(root) as reader:
            if reader(root / f"grading/{key}/raw.json") != raw:
                raise ValueError("benchmark raw changed during read")
            for relative, expected in records:
                observed = reader(root / relative)
                if observed != expected or (relative in serial_values and observed != serial_values[relative]):
                    raise ValueError("parallel benchmark differs from serial/evidence")
                progress.advance()
        parallel_seconds = time.monotonic() - start
        rate = len(records) / parallel_seconds
        # This estimate is deliberately padded; it is not a completion guarantee.
        estimated_full_seconds = sum(counts.values()) / rate + 600
        passed = rate >= 2 * 64 / serial_seconds and estimated_full_seconds < .8 * FINAL_SECONDS
        result = {"version": VERSION, "binding": binding, "passed": passed, "utc": utc_now(),
                  "batch": key, "serial_files": 64, "serial_seconds": serial_seconds,
                  "parallel_files": len(records), "parallel_seconds": parallel_seconds,
                  "parallel_files_per_second": rate, "all_batch_file_counts": counts,
                  "estimated_full_seconds": estimated_full_seconds, "reader": reader.receipt(),
                  "serial_parallel_values_equal": True, "frozen_batch_replay_passed": True,
                  "timing_caveat": "serial subset and full parallel batch differ in size/cache state",
                  "candidate_executions": 0, "optimizer_updates": 0}
        persist(output, {"benchmark": result})
        reports.commit()
    print("PARALLEL BENCHMARK", json.dumps(result, sort_keys=True), flush=True)
    return result


async def require_serial_stopped():
    # Read-only use of the pinned SDK's exact App lifecycle, not absence in a list.
    from modal.client import _Client
    from modal_proto import api_pb2
    client = await _Client.from_env()
    response = await asyncio.wait_for(client.stub.AppGetLifecycle(
        api_pb2.AppGetLifecycleRequest(app_id=SERIAL_APP)), timeout=20)
    if response.lifecycle.app_state != api_pb2.APP_STATE_STOPPED:
        raise ValueError("serial finalizer is not confirmed stopped; no handoff")
    return {"app_id": SERIAL_APP, "state": "stopped", "checked_utc": utc_now(),
            "stopped_at": response.lifecycle.stopped_at}


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), nonpreemptible=True,
              timeout=FINAL_SECONDS, retries=0, max_containers=1, scaledown_window=2, volumes=volumes)
def finish(sources):
    reports.reload()
    root, plan, setup, arms, baseline, measurements, binding = prepare(sources)
    output = output_directory()
    benchmark_result = frozen.recovery.read_json(output / "benchmark.json")
    if benchmark_result["passed"] is not True or benchmark_result["binding"] != binding:
        raise ValueError("passed benchmark for these exact sources/artifacts required")
    stopped = asyncio.run(require_serial_stopped())
    frozen.claim(f"parallel-finalization/{FINALIZATION_ID}/finish", maximum=1)
    persist(output, {"handoff": stopped})
    reports.commit()
    started = time.monotonic()
    try:
        with ProgressLog(release.RUN_ID, label="PARALLEL_FINALIZATION") as progress:
            with parallel_reader(frozen, root) as reader:
                result = frozen.verify_comparison(root, plan, setup, arms, baseline, progress)
            if (root / "result.json").exists() and frozen.recovery.read_json(root / "result.json") != result:
                raise ValueError("serial final report differs from parallel replay")
            for name, measured in measurements.items():
                if result["summary"]["policies"][name] != measured:
                    raise ValueError("replayed policy summary differs from completed measurement")
            receipt = {"version": VERSION, "binding": binding, "serial_handoff": stopped,
                       "elapsed_seconds": time.monotonic() - started, "completed_utc": utc_now(),
                       "reader": reader.receipt(), "all_original_checks_passed": True,
                       "result_sha256": digest(canonical_json(result)),
                       "candidate_executions": 0, "optimizer_updates": 0,
                       "evidence_mount_read_only": True, "skipped_batches": 0}
            progress.stage("write_separate_verified_report")
            persist(output, {"result": result, "verification_receipt": receipt})
            reports.commit()
        print("PARALLEL COMPARISON VERIFIED", json.dumps(result["summary"], sort_keys=True), flush=True)
        return {"result": result, "verification_receipt": receipt}
    except Exception as exc:
        persist(output, {"failure": {"type": type(exc).__name__, "detail": str(exc)[:1000], "utc": utc_now()}})
        reports.commit()
        raise


@app.local_entrypoint()
def launch(phase: str = "benchmark", allow_cloud: bool = False):
    if not allow_cloud or phase not in {"benchmark", "finish"}:
        raise ValueError("explicit --allow-cloud and benchmark/finish phase required")
    target = Path("runs") / release.RUN_ID / "finalization" / FINALIZATION_ID / phase
    if (target / "launch_intent.json").exists():
        raise ValueError("phase already launched; inspect saved call rather than duplicating")
    sources = source_binding(Path.cwd())
    if phase == "finish":
        asyncio.run(require_serial_stopped())
    persist(target, {"launch_intent": {"phase": phase, "sources": sources, "utc": utc_now()}})
    function = benchmark if phase == "benchmark" else finish
    call = function.spawn(sources)
    persist(target, {"launch": {"call_id": call.object_id, "phase": phase, "utc": utc_now()}})
    print("PARALLEL PHASE LAUNCHED", phase, call.object_id, flush=True)
