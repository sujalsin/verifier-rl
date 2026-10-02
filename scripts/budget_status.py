"""Read-only live status and provider-evidence audit. No cloud execution starts.

Run ``.venv/bin/python -m scripts.budget_status`` from the repository root.
The report keeps provider workspace
usage, estimated compute, reservations and overhead separate. No ledger writes.
"""
import asyncio
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

import modal
from google.protobuf.json_format import MessageToDict
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal_proto import api_pb2

from verifier_rl import budget_reconciliation as reconcile
from verifier_rl.evaluation_journal import persist
from verifier_rl.parallel_evaluation import fingerprint

RUN = "qwen-booking-replication-repair-20260930-v1"
PREFIX = RUN + "/program-sandbox-001/parallel-evaluation-001/research-001"
CONFIG = Path("runs") / RUN / "amendments/program-sandbox-001/parallel-evaluation-001/research-001/lifecycle-bridge-001/config.json"


def serial(value):
    if is_dataclass(value):
        return serial(asdict(value))
    if isinstance(value, (Decimal, datetime)):
        return str(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def call_location(key):
    if key.startswith("program-sandbox-001/"):
        ticket = key.removeprefix("program-sandbox-001/")
        namespace = RUN + "/program-sandbox-001"
    else:
        namespace, ticket = RUN, key
    operation = ticket.split("/", 1)[0]
    # These source-reviewed calls create no candidate sandboxes directly.
    if operation not in ("training", "controller", "seed"):
        return None
    return namespace + "/calls/" + ticket, "gpu" if operation == "training" else "cpu"


async def inspect():
    if modal.__version__ != "1.5.5":
        raise ValueError("provider evidence adapter requires pinned SDK 1.5.5")
    client = await _Client.from_env()
    store = modal.Dict.from_name("verifier-rl-booking-replication-journal", create_if_missing=False)
    saved, scheduler = await asyncio.gather(store.get.aio(RUN + "/budget"), store.get.aio(PREFIX + "/state"))
    captured_utc = datetime.now(timezone.utc).isoformat()
    evaluation_prefix, retained = PREFIX, 47
    successor_prefix = PREFIX.rsplit("/", 1)[0] + "/research-002"
    successor_ready, successor_state = await asyncio.gather(store.get.aio(successor_prefix + "/ready", None),
                                                           store.get.aio(successor_prefix + "/state", None))
    if successor_ready is not None:
        retained = successor_ready["completed"]
        scheduler = successor_state or {"batches":{}, "allowed":range(successor_ready["remaining"]),
                                        "unknown":[], "stop":None}
        evaluation_prefix = successor_prefix
    # Identity-only resume keeps the research-002 reconciliation receipt,
    # but publishes evaluation progress in a new immutable namespace.
    identity_prefix = PREFIX.rsplit("/", 1)[0] + "/research-003"
    identity_ready, identity_state = await asyncio.gather(store.get.aio(identity_prefix + "/ready", None),
                                                        store.get.aio(identity_prefix + "/state", None))
    if identity_ready is not None:
        retained = identity_ready["completed"]
        scheduler = identity_state or {"batches":{}, "allowed":range(identity_ready["remaining"]),
                                       "unknown":[], "stop":None}
        evaluation_prefix = identity_prefix
    permit_prefix = PREFIX.rsplit("/", 1)[0] + "/research-004"
    permit_ready, permit_state = await asyncio.gather(store.get.aio(permit_prefix + "/ready", None),
                                                    store.get.aio(permit_prefix + "/state", None))
    if permit_ready is not None:
        retained = permit_ready["completed"]
        scheduler = permit_state or {"batches":{}, "allowed":range(permit_ready["remaining"]),
                                     "unknown":[], "stop":None}
        evaluation_prefix = permit_prefix
    config = json.loads(CONFIG.read_text())
    evidence, candidates, excluded = {}, {}, {}
    semaphore = asyncio.Semaphore(3)

    async def one(key, item):
        if item["actual"] is not None or call_location(key) is None:
            return
        async with semaphore:
            try:
                location, kind = call_location(key)
                call = await store.get.aio(location, None)
                if call is None or call["identity"] != item["identity"]:
                    raise ValueError("missing or mismatched saved invocation")
                handle = await client.stub.FunctionCallFromId(api_pb2.FunctionCallFromIdRequest(
                    function_call_id=call["call_id"]), timeout=20, retry=None)
                info, life = await asyncio.gather(
                    client.stub.FunctionCallGetInfo(api_pb2.FunctionCallGetInfoRequest(
                        function_id=handle.metadata.function_id, function_call_id=call["call_id"]), timeout=20, retry=None),
                    client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=handle.metadata.app_id), timeout=20, retry=None))
                proof = {"identity": item["identity"], "call_id": call["call_id"],
                         "app_id": handle.metadata.app_id, "function_id": handle.metadata.function_id,
                         "sandbox_starts": 0, "retries": 0,
                         "lifecycle": {"app_id": handle.metadata.app_id,
                             "state": api_pb2.AppState.Name(life.lifecycle.app_state),
                             "stopped_at": life.lifecycle.stopped_at},
                         "call_info": MessageToDict(info.info, preserving_proto_field_name=True,
                             always_print_fields_with_no_presence=True)}
                evidence[key] = proof
                receipt = reconcile.completed_call_bound(item, proof, config["rates"], kind)
                if Decimal(receipt["estimated_compute_usd"]) < Decimal(item["maximum"]):
                    candidates[key] = receipt
            except Exception as exc:
                excluded[key] = {"type": type(exc).__name__, "detail": str(exc)[:300]}

    await asyncio.gather(*(one(k, v) for k, v in saved["ledger"]["items"].items()))
    billing = modal.Workspace.from_context().billing
    provider = {}
    for month in ("2026-09", "2026-10"):
        try:
            provider[month] = serial(await billing.summary.aio(month))
        except Exception as exc:
            provider[month] = {"unavailable": type(exc).__name__}
    proposals = {"before_hash": fingerprint(saved), "rates": config["rates"], "items": candidates}
    release = sum((Decimal(saved["ledger"]["items"][k]["maximum"]) - Decimal(v["estimated_compute_usd"])
                   for k, v in candidates.items()), Decimal(0))
    completed = len(scheduler.get("batches", {}))
    # Reused old batches are separate from the parallel scheduler's 433 items.
    report = {"snapshot_utc": captured_utc, "checked_utc": datetime.now(timezone.utc).isoformat(), "accounting": reconcile.describe(saved["ledger"]),
              "provider_workspace_monthly_billing": provider,
              "provider_scope_note": "Workspace-wide, including storage and other apps; monthly billing is not study-only compute.",
              "completed_reconciliation": None if successor_ready is None else {
                  "verified_hold_reduction_usd":successor_ready["accounting"]["verified_hold_reduction_usd"],
                  "transaction_hash":successor_ready["accounting"]["transaction_hash"]},
              "additional_reconciliation_proposal": {"eligible_reservations":len(candidates),
                  "proposed_hold_reduction_usd":str(release), "applied_by_this_command":False,
                  "reason":"read-only report; only the exclusive accounting owner may apply corrections"},
              "evaluation": {"namespace":evaluation_prefix, "completed_batches":retained+completed, "total_batches":480,
                  "retained_batches":retained, "new_completed_batches":completed, "scheduler_total":len(scheduler.get("allowed", {})),
                  "unknown": scheduler.get("unknown"), "stop": scheduler.get("stop")}}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path("runs") / RUN / "budget-reconciliation-001" / stamp
    persist(directory, {"report": report, "ledger_snapshot": saved, "provider_call_evidence": evidence,
                        "proposal": proposals, "excluded": excluded})
    print(json.dumps(report, indent=2))
    print("Evidence saved:", directory)
    return report


if __name__ == "__main__":
    synchronizer.create_blocking(inspect)()
