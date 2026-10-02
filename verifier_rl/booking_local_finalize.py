"""Export saved baseline evidence and replay its frozen verifier locally.

No candidate execution, model loading, training, or remote Function invocation.
The download command only reads an existing Modal Volume. The verify command
uses the original verifier unchanged, including every per-input journal check.
"""

import argparse
import asyncio
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import time

from .cli import write_private
from .suites import canonical_json

RUN_ID = "qwen-booking-baseline-comparison-20260929-v1"
VOLUME = "verifier-rl-cache-artifacts"
VERSION = "booking-local-finalization-0.1"
ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def relative_path(value):
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or not path.parts
            or path.suffix != ".json" or str(path) != value):
        raise ValueError("invalid evidence path: " + value)
    return path


def local_path(root, relative):
    path = root.joinpath(*relative_path(relative).parts)
    if root.is_symlink() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink in evidence path")
    return path


async def export_run(destination, volume, *, concurrency=16):
    """Copy files only, avoiding the CLI's directory/child creation race."""
    if not 1 <= concurrency <= 32:
        raise ValueError("download concurrency must be between 1 and 32")
    destination = Path(destination)
    evidence = destination / RUN_ID
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (destination / "download_manifest.json").exists() or any(p.is_file() or p.is_symlink() for p in evidence.rglob("*")):
        raise FileExistsError("export must not replace existing evidence; choose a new directory")
    started, entries = time.monotonic(), []
    async for item in volume.iterdir.aio(RUN_ID, recursive=True):
        # Modal FileEntryType.FILE is 1; directories need no separate copy.
        if int(item.type) == 2:
            continue
        if int(item.type) != 1:
            raise ValueError("unexpected non-file entry")
        relative = str(PurePosixPath(item.path).relative_to(RUN_ID))
        local_path(evidence, relative)
        if not 0 < item.size <= 32 * 1024 * 1024:
            raise ValueError("unexpected evidence file size")
        entries.append((item.path, relative, item.size))
    if not entries or len(entries) > 40000 or sum(e[2] for e in entries) > 1024**3:
        raise ValueError("export exceeds the bounded baseline evidence size")
    if len({e[1] for e in entries}) != len(entries):
        raise ValueError("duplicate remote file path")
    print(f"EXPORT INVENTORY {len(entries)} files, {sum(e[2] for e in entries)} bytes", flush=True)
    queue = asyncio.Queue()
    for entry in entries:
        queue.put_nowait(entry)
    manifest = {}

    async def worker():
        while not queue.empty():
            remote, relative, expected_size = queue.get_nowait()

            async def receive():
                data = bytearray()
                async for chunk in volume.read_file.aio(remote):
                    data.extend(chunk)
                    if len(data) > expected_size:
                        raise ValueError("remote file grew while downloading: " + relative)
                return data

            data = await asyncio.wait_for(receive(), timeout=60)
            if len(data) != expected_size:
                raise ValueError("incomplete download: " + relative)
            path = local_path(evidence, relative)
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("xb") as stream:
                path.chmod(0o600)
                stream.write(data)
            manifest[relative] = {"bytes": len(data), "sha256": sha256(data).hexdigest()}
            if len(manifest) % 1000 == 0 or len(manifest) == len(entries):
                print(f"EXPORT {len(manifest)}/{len(entries)} files, {time.monotonic()-started:.1f}s", flush=True)

    # TaskGroup cancels remaining downloads if a copy fails; no completion
    # manifest is written for an incomplete export. Already copied data stays.
    async with asyncio.TaskGroup() as tasks:
        for _ in range(concurrency):
            tasks.create_task(worker())
    receipt = {"version": VERSION, "volume": VOLUME, "run_id": RUN_ID,
               "completed_utc": utc_now(), "files": manifest,
               "elapsed_seconds": time.monotonic()-started, "cloud_execution_started": False}
    write_private(destination / "download_manifest.json", canonical_json(receipt))
    return receipt


def check_download(destination):
    """Check all copied bytes and the presence of every completed batch."""
    destination = Path(destination)
    evidence = destination / RUN_ID
    manifest = json.loads((destination / "download_manifest.json").read_text())
    if (manifest["version"] != VERSION or manifest["volume"] != VOLUME
            or manifest["run_id"] != RUN_ID or manifest["cloud_execution_started"] is not False):
        raise ValueError("export identity changed")
    for relative, record in manifest["files"].items():
        data = local_path(evidence, relative).read_bytes()
        if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("downloaded evidence changed: " + relative)
    files = {p.relative_to(evidence).as_posix() for p in evidence.rglob("*") if p.is_file()}
    if files != set(manifest["files"]):
        raise ValueError("export inventory changed")
    required = {"plan.json", "setup.json", "source_snapshot.json", "preflight.json",
                "supervisor_controls.json", "generation/result.json"}
    keys = ["controls"] + [f"eval-baseline-00-{i:02d}" for i in range(8)]
    required.update(f"grading/{key}/{name}.json" for key in keys for name in ("raw", "result"))
    if not required.issubset(files):
        raise ValueError("missing required baseline evidence")
    for key in keys:
        raw = json.loads((evidence / f"grading/{key}/raw.json").read_text())
        result = json.loads((evidence / f"grading/{key}/result.json").read_text())
        if result["raw"] != raw:
            raise ValueError("raw/result copies differ: " + key)
        for sid, inputs in raw["entries"].items():
            for h, entry in inputs.items():
                if not {f"grading/{key}/inputs/{sid}/{h}/{name}.json" for name in entry}.issubset(files):
                    raise ValueError("missing per-input journal")
    # Importing the authored launcher registers local definitions only. No
    # .remote/.spawn/.run calls occur; verify_baseline is a plain Python function.
    from modal_booking_baseline_comparison import check_snapshot, validate_generation
    from .booking_baseline_comparison import validate_plan
    check_snapshot(json.loads((evidence / "source_snapshot.json").read_text()), ROOT)
    plan = json.loads((evidence / "plan.json").read_text())
    if plan["run_id"] != RUN_ID:
        raise ValueError("wrong baseline run")
    validate_plan(plan)
    validate_generation(json.loads((evidence / "generation/result.json").read_text()), plan)
    return {"files": len(files), "bytes": sum(r["bytes"] for r in manifest["files"].values()),
            "grading_batches": len(keys), "programs": 32}


def verify_local(destination, output):
    """Run the exact frozen verifier, instrumenting only its JSON reader."""
    destination, output = Path(destination), Path(output)
    evidence = (destination / RUN_ID).resolve()
    if output.resolve().is_relative_to(evidence):
        raise ValueError("derived output must be outside the immutable evidence directory")
    if output.exists():
        raise FileExistsError("verification output already exists")
    started = time.monotonic()
    inventory = check_download(destination)
    import modal_booking_baseline_comparison as frozen
    original_reader = frozen.read_json
    seen, reads = set(), 0

    def observed_reader(path):
        nonlocal reads
        relative = Path(path).resolve().relative_to(evidence).as_posix()
        seen.add(relative)
        reads += 1
        if reads % 1000 == 0 or relative.endswith("/raw.json"):
            print(f"VERIFY {reads} reads, {time.monotonic()-started:.1f}s, {relative}", flush=True)
        return original_reader(path)

    frozen.read_json = observed_reader
    try:
        result = frozen.verify_baseline(evidence)
    finally:
        frozen.read_json = original_reader
    if (evidence / "result.json").exists() and original_reader(evidence / "result.json") != result:
        raise ValueError("cloud final result differs from local replay")
    receipt = {"version": VERSION, "run_id": RUN_ID, "completed_utc": utc_now(),
               "evidence_directory": str(evidence), "inventory": inventory,
               "json_reads": reads, "unique_json_reads": len(seen),
               "elapsed_seconds": time.monotonic()-started, "all_original_checks_passed": True,
               "candidate_executions": 0, "optimizer_updates": 0, "cloud_execution_started": False,
               "download_manifest_sha256": sha256((destination / "download_manifest.json").read_bytes()).hexdigest(),
               "frozen_verifier_sha256": sha256((ROOT / "modal_booking_baseline_comparison.py").read_bytes()).hexdigest(),
               "result_sha256": sha256(canonical_json(result).encode()).hexdigest()}
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_private(output / "verification_receipt.json", canonical_json(receipt))
    write_private(output / "result.json", canonical_json(result))
    print("LOCAL BASELINE VERIFIED", json.dumps(result["summary"], sort_keys=True), flush=True)
    return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--destination", type=Path, required=True)
    download.add_argument("--allow-cloud-read", action="store_true")
    download.add_argument("--concurrency", type=int, default=16)
    check = commands.add_parser("check")
    check.add_argument("--destination", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--destination", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        if not args.allow_cloud_read:
            parser.error("download requires --allow-cloud-read; no remote code will run")
        import modal
        volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
        asyncio.run(asyncio.wait_for(export_run(args.destination, volume, concurrency=args.concurrency), timeout=900))
    elif args.command == "check":
        print(json.dumps(check_download(args.destination), sort_keys=True))
    else:
        verify_local(args.destination, args.output)


if __name__ == "__main__":
    main()
