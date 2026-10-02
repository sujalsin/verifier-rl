"""Isolated test of a short accounting-queue maintenance window.

No research inputs, candidate sandboxes, or real ledger edits. The only real
budget operation is an ordinary reservation through the existing sole writer.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import modal

from verifier_rl.evaluation_journal import persist
from verifier_rl.parallel_evaluation import fingerprint

LIVE_APP = "ap-mDjiaxflNVbgsxtuzGoo4X"
RUN = "qwen-booking-replication-repair-20260930-v1"
REL = RUN + "/amendments/program-sandbox-001/parallel-evaluation-001/research-001"
PREFIX = RUN + "/budget-reconciliation-001/autoscaler-probe-001"
app = modal.App("verifier-rl-budget-maintenance-probe")
store = modal.Dict.from_name("verifier-rl-booking-replication-journal", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5").add_local_python_source("verifier_rl")


async def live_budget(config, action, ticket, amount, identity):
    from modal.client import _Client
    from modal._functions import _Function
    from modal_proto import api_pb2
    if modal.__version__ != "1.5.5":
        raise ValueError("pinned SDK required")
    client = await _Client.from_env()
    layout = (await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=LIVE_APP), timeout=30, retry=None)).app_layout
    fid = layout.function_ids["handed_budget"]
    owner = await store.get.aio(RUN + "/program-sandbox-001/parallel-evaluation-001/research-001/budget-owner")
    if owner["function_id"] != fid or owner["config_hash"] != fingerprint(config):
        raise ValueError("live budget owner changed")
    meta = [o.function_handle_metadata for o in layout.objects if o.object_id == fid and o.HasField("function_handle_metadata")]
    if len(meta) != 1:
        raise ValueError("budget handle unavailable")
    function = _Function._new_hydrated(fid, client, meta[0])
    return await function.remote(config, action, ticket, amount, identity)


@app.function(image=image, cpu=(1, 1), memory=(128, 128), timeout=60,
              retries=0, max_containers=1, scaledown_window=2)
def fake_writer(label, delay):
    value = store.get(PREFIX + "/fake-ledger")
    store.put(PREFIX + "/entered/" + label, {"started": time.time()}, skip_if_exists=True)
    time.sleep(delay)
    store.put(PREFIX + "/fake-ledger", value + 1)
    return value + 1


@app.function(image=image, cpu=(1, 1), memory=(128, 128), timeout=240,
              retries=0, max_containers=1, scaledown_window=2)
def test_queue():
    if not store.put(PREFIX + "/claim", {"started": time.time()}, skip_if_exists=True):
        raise ValueError("probe already submitted; inspect saved result")
    store.put(PREFIX + "/fake-ledger", 0, skip_if_exists=True)
    started = time.time()
    result = {"passed": False, "real_ledger_edits": 0, "candidate_executions": 0}
    try:
        first = fake_writer.spawn("first", 10)
        while store.get(PREFIX + "/entered/first", None) is None:
            if time.time() - started > 45:
                raise TimeoutError("probe first input did not enter")
            time.sleep(0.5)
        if fake_writer.get_current_stats().num_running_inputs != 1:
            raise ValueError("active probe input not positively observed")
        result["zero_settings"] = asdict(fake_writer.update_autoscaler(min_containers=0, max_containers=0, buffer_containers=0))
        if first.get(timeout=30) != 1:
            raise ValueError("active accounting input was interrupted")
        deadline = time.time() + 30
        while True:
            stats = fake_writer.get_current_stats()
            if stats.num_total_runners == 0 and stats.num_running_inputs == 0:
                break
            if time.time() > deadline:
                raise TimeoutError("zero setting did not drain the service")
            time.sleep(1)
        queued = fake_writer.spawn("queued", 0)
        time.sleep(10)
        stats = fake_writer.get_current_stats()
        result["paused_stats"] = asdict(stats)
        if stats.num_running_inputs or stats.num_total_runners or store.get(PREFIX + "/entered/queued", None) is not None:
            raise ValueError("zero max does not prevent new inputs")
        if stats.backlog != 1:
            raise ValueError("queued probe not positively observed")
        store.put(PREFIX + "/fake-ledger", 100)
        fake_writer.update_autoscaler(min_containers=0, max_containers=1, buffer_containers=0)
        if queued.get(timeout=45) != 101:
            raise ValueError("queued input lost the maintenance correction")
        result["passed"] = True
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "detail": str(exc)}
    finally:
        fake_writer.update_autoscaler(min_containers=0, max_containers=1, buffer_containers=0)
    result["seconds"] = time.time() - started
    store.put(PREFIX + "/result", result, skip_if_exists=True)
    print("ACCOUNTING MAINTENANCE PROBE", json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main():
    from modal._utils.async_utils import synchronizer
    config = json.loads((Path("runs") / REL / "lifecycle-bridge-001/config.json").read_text())
    identity = fingerprint({"source": Path(__file__).read_text(), "probe": PREFIX})
    ticket = "budget-reconciliation-001/autoscaler-probe-001"
    if store.get(PREFIX + "/claim", None) is not None:
        raise ValueError("probe already exists; no duplicate launch")
    status = synchronizer.create_blocking(live_budget)(config, "reserve", ticket, "1", identity)
    print("PROBE RESERVED WITHIN EXISTING CEILING", status, flush=True)
    call = test_queue.spawn()
    directory = Path("runs") / RUN / "budget-reconciliation-001/autoscaler-probe-001"
    persist(directory, {"launch": {"call_id": call.object_id, "reservation_identity": identity}})
    print("ACCOUNTING PROBE CALL", call.object_id, flush=True)
    result = call.get()
    # Retain the full $1 hold pending complete accounting; no guessed refund.
    persist(directory, {"result": result})
    print(json.dumps(result, indent=2))
