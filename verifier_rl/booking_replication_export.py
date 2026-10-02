"""Download only saved failure evidence for offline analysis; no executions.

Read-only Modal access, four downloads at a time, 2 MiB per file and 128 MiB
total. Model/checkpoint files are never eligible. Existing local copies are
compared with remote bytes, not silently overwritten.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

from . import booking_replication as study
from .parallel_evaluation import fingerprint


def targets(result, inventory):
    base = PurePosixPath(study.RUN_ID)/"amendments/program-sandbox-001"
    selected = {"baseline-generation": str(base/"baseline/result.json")}
    groups = {"baseline-0": inventory["baseline"], **result["policies"]}
    for policy, group in groups.items():
        for row in group["programs"]:
            if not row["weak_only_acceptance_bounds"][1]:
                continue
            sid = row["sample_id"]
            draw = int(sid.rsplit("-", 1)[1])
            label, step = policy.rsplit("-", 1)
            key = f"eval-{label}-{int(step):02d}-{(draw-19000)//4:02d}"
            if label == "baseline":
                directory = base/"grading"/key
            elif key in inventory["completed"]:
                source = PurePosixPath(inventory["completed"][key]["source_path"])
                if source.parts[1] != "artifacts" or "recovered" in source.parts:
                    raise ValueError("unreviewed evidence location")
                directory = PurePosixPath(*source.parts[2:]).parent
            else:
                directory = base/"parallel-evaluation-001/research-004/grading"/key
            selected[sid] = str(directory/"programs"/sid/"result.json")
    return selected


async def download(directory):
    import modal

    bundle = directory/"modal-results"
    result, inventory = [json.loads((bundle/f"{name}.json").read_text())
                         for name in ("result", "inventory")]
    selected = targets(result, inventory)
    if len(selected) > 64:
        raise ValueError("small export cap exceeded")
    if shutil.disk_usage(directory).free < 2*1024**3:
        raise OSError("less than 2 GiB free; no download")
    output = directory/"failure-evidence"
    output.mkdir(exist_ok=True)
    volume = modal.Volume.from_name("verifier-rl-booking-replication-work", create_if_missing=False)
    semaphore = asyncio.Semaphore(4)
    records = {}
    total = 0

    async def one(name, remote):
        nonlocal total
        async with semaphore:
            data = bytearray()
            async for chunk in volume.read_file.aio("/"+remote):
                data.extend(chunk)
                total += len(chunk)
                if len(data) > 2*1024**2 or total > 128*1024**2:
                    raise ValueError("small evidence download limit exceeded")
            json.loads(data)
            path = output/f"{name}.json"
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("existing local evidence differs: "+name)
            else:
                with path.open("xb") as target:
                    target.write(data)
            records[name] = {"path": path.name, "remote_path": "/"+remote,
                             "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            print("DOWNLOADED", name, len(data), flush=True)

    await asyncio.gather(*(one(name, remote) for name, remote in selected.items()))
    manifest = {"source_volume": "verifier-rl-booking-replication-work",
                "result_hash": fingerprint(result), "inventory_hash": fingerprint(inventory),
                "selection": "all baseline and post-training weak-only acceptances, plus baseline generation",
                "candidate_executions": 0, "total_bytes": total, "files": records}
    payload = json.dumps(manifest, sort_keys=True, indent=2)+"\n"
    path = output/"manifest.json"
    if path.exists() and path.read_text() != payload:
        raise ValueError("export manifest changed")
    path.write_text(payload)
    print("COMPLETE", len(records), "files", total, "bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    asyncio.run(download(args.directory))


if __name__ == "__main__":
    main()
