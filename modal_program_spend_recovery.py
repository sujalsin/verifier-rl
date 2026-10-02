"""One reviewed recovery of eight fixed rollouts after the workspace spend stop.

This administrative launcher is deliberately outside the frozen study sources.
It never generates code or trains a model. Historical evidence is archived before
eight replacement execution receipts are published; known outcomes must agree.
"""

import asyncio
from dataclasses import asdict
from decimal import Decimal
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import modal

from verifier_rl import booking_replication as study, compute_budget as accounting
from verifier_rl import grpo_recovery, program_execution as execution, program_grading as grading
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.modal_backend import Limits
from verifier_rl.panel_execution import unpack_result
from verifier_rl.program_benchmark import bounded_map
from verifier_rl.sandbox_lifecycle import confirm_terminal, validate_terminal
from verifier_rl.suites import canonical_json, digest

RECOVERY_ID = "spend-limit-20261001-001"
PREFIX = grading.NAMESPACE + "/recovery/" + RECOVERY_ID
PARENT_APP = "ap-7FLIh7MTxx45m27v5nmvkr"
SNAPSHOT_HASH = "6a962f5f44b3bb0446f6250f11683a1b09179bec7d118bd7f2143329a6a6563d"
JOBS = (("s20261013-endpoint_omission", 18), ("s20261014-endpoint_omission", 10))
OLD_SANDBOXES = (
    "sb-MYXIm9gmURcaXhIrZITGzg", "sb-Tlx3TrB0jmfJZWIG6iJMya",
    "sb-V3Dg5AbUwZ7XBz1EUQS09H", "sb-R9scLvzHxL4pcLZMbJ0Hjk",
)
OLD_COUNTS = (63, 69, 38, 72)
EXPECTED_STOP = {"reason": "cleanup_unconfirmed", "batch": "train-s20261014-endpoint_omission-10",
                 "sample_id": "train-s20261014-endpoint_omission-10-3"}
SECONDS = 3600
LAUNCHERS = ("modal_booking_study.py", "modal_booking_baseline_comparison.py",
             "modal_booking_matched_training.py", "modal_booking_replication.py")
fingerprint = grading.fingerprint

app = modal.App("verifier-rl-program-spend-recovery")
work = modal.Volume.from_name("verifier-rl-booking-replication-work", create_if_missing=False)
archive = modal.Volume.from_name("verifier-rl-booking-replication-evidence", create_if_missing=False)
store = modal.Dict.from_name("verifier-rl-booking-replication-journal", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
image = image.add_local_python_source("verifier_rl")
for name in LAUNCHERS:
    image = image.add_local_file(name, "/root/" + name)


def root(base):
    return Path(base) / study.RUN_ID / grading.RELATIVE


def validate_context(context, sources, now):
    study.validate_plan(context["plan"])
    grading.validate_binding(context["program_runtime"])
    if (fingerprint(context["source_snapshot"]) != SNAPSHOT_HASH or sources != context["source_snapshot"]
            or context["deadline"] != context["program_runtime"]["deadline"] or now >= context["deadline"]
            or context["runtime_amendment"]["id"] != grading.AMENDMENT):
        raise ValueError("recovery cannot change frozen sources, research protocol, or deadline")


async def require_idle_apps(apps, client=None):
    """App list is recent-only; require direct, positive terminal evidence."""
    from modal.client import _Client
    from modal_proto import api_pb2

    if any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("all original workers must be stopped before maintenance")
    if client is None:
        client = await _Client.from_env()
    response = await client.stub.AppGetLifecycle(
        api_pb2.AppGetLifecycleRequest(app_id=PARENT_APP), timeout=30, retry=None)
    lifecycle = response.lifecycle
    if lifecycle.app_state != api_pb2.APP_STATE_STOPPED:
        raise ReconciliationRequired("original app is not confirmed stopped")
    return {"app_id": PARENT_APP, "method": "AppGetLifecycle", "state": "stopped",
            "stopped_at": lifecycle.stopped_at, "checked_at": time.time()}


def validate_checkpoint_group(base, policy, step, context):
    directory = Path(base) / "arms" / policy
    binding = {"release_hash": fingerprint(context["plan"]), "directory": str(directory),
               "seed": int(policy[1:9]), "control": False}
    checkpoint = grpo_recovery.latest_checkpoint(directory / "trainer", fingerprint(binding))
    if checkpoint != directory / "trainer" / f"checkpoint-{step}":
        raise ValueError("unexpected latest checkpoint; do not rewind a policy")
    group = directory / "journal" / f"group-{step:02d}"
    if (group / "reward.json").exists():
        raise ValueError("pending group already consumed a reward")
    generation = grpo_recovery.read_json(group / "generation.json")
    intent = grpo_recovery.read_json(group / "intent.json")
    boundary = grpo_recovery.read_json(checkpoint / "recovery.json")
    if (generation["binding"] != intent or generation["tokens_hash"] != fingerprint(generation["output"])
            or intent["binding_hash"] != fingerprint(binding) or intent["step"] != step
            or intent["parameter_hash"] != boundary["parameter_hash"]):
        raise ValueError("saved rollout does not belong to the preserved full-state checkpoint")
    samples = grpo_recovery.read_json(group / "samples.json")
    key = f"train-{policy}-{step:02d}"
    study.validate_batch(key, samples, "training", context["plan"])
    return key, samples, {"checkpoint_receipt": boundary, "generation_hash": fingerprint(generation),
                          "samples_hash": fingerprint(samples)}


def input_signature(record, sample, case):
    """Validate protected historical outcomes; never infer a pass from exit code."""
    value = unpack_result(record)
    meta = value.metadata
    if meta.get("input_hash") != case.input_hash or meta.get("source_hash") != digest(sample["source"]):
        raise ValueError("historical input/source identity changed")
    if value.status.value == "infrastructure_error":
        if (value.detail not in ("program_not_executed", "program_transport:candidate:ConflictError")
                or value.stdout or "supervisor_report" in meta):
            raise ValueError("unreviewed ambiguous or candidate-caused failure")
        return None
    payload = canonical_json({"source": sample["source"], "input": case.arguments, "limits": asdict(Limits())})
    status, reason, stdout, _ = execution.inspect_envelope(meta["supervisor_report"], digest(payload))
    if (status != value.status or reason != value.detail or stdout != value.stdout
            or meta.get("parent_returncode") != 0 or not meta["supervisor_report"]["candidate_started"]):
        raise ValueError("historical protected result changed")
    return value.status.value, value.detail, record["stdout_base64"]


def validate_history(history, samples, cases, runtime, terminals):
    expected = {f"train-{policy}-{step:02d}-{i}" for policy, step in JOBS for i in range(4)}
    if set(history) != expected or {s["sample_id"] for s in samples} != expected:
        raise ValueError("only the eight reviewed pending rollouts can be recovered")
    known = 0
    case_map = {c.input_hash: c for c in cases}
    for sample in samples:
        sid = sample["sample_id"]
        item = history[sid]
        if item["intent"] != grading.intent_for(sample, cases, runtime):
            raise ValueError("original program intent changed")
        if set(item["records"]) - case_map.keys():
            raise ValueError("unexpected historical test")
        if sid.startswith("train-s20261014-"):
            final = item["final"]
            if final is None:
                raise ValueError("explicit spend-limit rejection receipt required")
            grading.assemble(final, item["records"])
            m = final["metadata"]
            failure = m.get("failure", {})
            if (m.get("candidate_submission_attempted") is not False or "sandbox_id" in m
                    or failure.get("stage") != "create" or failure.get("type") != "ResourceExhaustedError"
                    or "has exceeded its spend limit" not in failure.get("detail", "")
                    or len(item["records"]) != len(cases)
                    or any(r["detail"] != "program_not_executed" for r in item["records"].values())):
                raise ValueError("failure is not the reviewed pre-candidate spending rejection")
        else:
            index = int(sid.rsplit("-", 1)[1])
            sandbox_id = OLD_SANDBOXES[index]
            validate_terminal(terminals[sandbox_id], sandbox_id)
            if item["final"] is not None or len(item["records"]) != OLD_COUNTS[index]:
                raise ValueError("interrupted program evidence changed")
            ids = {r["metadata"].get("sandbox_id") for r in item["records"].values()} - {None}
            if ids != {sandbox_id}:
                raise ValueError("interrupted sandbox identity changed")
        for h, record in item["records"].items():
            known += input_signature(record, sample, case_map[h]) is not None
    return known


def validate_replacement(document, previous, sample, cases, image_id):
    outcomes = execution.validate_program(document, sample["source"], cases, image_id)
    if document["metadata"]["cleanup"] != "terminated" or any(v["passed"] is None for v in outcomes.values()):
        raise ReconciliationRequired("recovery execution unresolved; keep stop and all original evidence")
    matched = 0
    by_hash = {c.input_hash: c for c in cases}
    for h, first in previous["records"].items():
        signature = input_signature(first, sample, by_hash[h])
        if signature is not None:
            if input_signature(document["records"][h], sample, by_hash[h]) != signature:
                raise ReconciliationRequired("recovery differs from a known outcome; do not choose the better result")
            matched += 1
    return matched


def archived_publish(directory, archive_directory, documents):
    """Recoverable move, then immutable creation; no deletion of prior files."""
    directory, archive_directory = Path(directory), Path(archive_directory)
    if archive_directory.exists():
        raise ReconciliationRequired("publication already started; inspect before continuing")
    archive_directory.parent.mkdir(parents=True, exist_ok=True)
    directory.rename(archive_directory)
    for relative, values in documents.items():
        persist(directory / relative, values)


async def put_once(key, value):
    if not await store.put.aio(key, value, skip_if_exists=True) and await store.get.aio(key) != value:
        raise ValueError("immutable recovery receipt changed: " + key)


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": work, "/evidence": archive})
def recover(context, recovery_identity):
    started = time.monotonic()
    if not store.put(PREFIX + "/owner", {"started": time.time()}, skip_if_exists=True):
        raise ReconciliationRequired("recovery worker restarted; no automatic resubmission")
    directory = root("/artifacts")
    sources = {name: (Path("/root") / name).read_text() for name in context["source_snapshot"]}
    validate_context(context, sources, time.time())
    if store.get(grading.NAMESPACE + "/stop", None) != EXPECTED_STOP:
        raise ValueError("unexpected stop condition")
    work.reload()
    target = directory / "recoveries" / RECOVERY_ID
    durable = root("/evidence") / "recoveries" / RECOVERY_ID

    async def perform():
        cases = study.pilot.cases_for("training")
        jobs, history, terminals, checkpoints, completed = {}, {}, {}, {}, {}
        for seed in study.SEEDS:
            for arm in study.ARMS:
                policy = study.label(seed, arm)
                value = await store.get.aio(grading.NAMESPACE + "/finished/training/" + policy, None)
                if (policy, 18) not in JOBS and (policy, 10) not in JOBS:
                    if value is None:
                        raise ValueError("a previously completed policy is missing")
                    study.validate_arm(value["payload"], context["plan"])
                    completed[policy] = fingerprint(value)
                elif value is not None:
                    raise ValueError("interrupted policy unexpectedly completed")
        for policy, step in JOBS:
            key, samples, checkpoint = validate_checkpoint_group(directory, policy, step, context)
            jobs[key], checkpoints[policy] = samples, checkpoint
            if ((directory / "grading" / key / "result.json").exists()
                    or await store.get.aio(grading.NAMESPACE + "/finished/grading/" + key, None) is not None):
                raise ValueError("a completed grading batch must not be retried")
            for sample in samples:
                sid = sample["sample_id"]
                job = grading.NAMESPACE + "/grading/" + key + "/" + sid
                records = {}
                for case in cases:
                    value = await store.get.aio(job + "/input/" + case.input_hash, None)
                    if value is not None:
                        records[case.input_hash] = value
                history[sid] = {"intent": await store.get.aio(job + "/intent"),
                                "final": await store.get.aio(job + "/final", None), "records": records}
        for sandbox_id in OLD_SANDBOXES:
            sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
            terminals[sandbox_id] = await confirm_terminal(sandbox)
        samples = [sample for group in jobs.values() for sample in group]
        known = validate_history(history, samples, cases, context["program_runtime"], terminals)
        original = {"version": RECOVERY_ID, "context_hash": fingerprint(context), "history": history,
                    "jobs": jobs, "terminal_evidence": terminals, "checkpoints": checkpoints,
                    "completed_policy_hashes": completed, "stop": EXPECTED_STOP,
                    "known_outcomes": known, "recovery_identity": recovery_identity,
                    "policy": "one explicit re-execution of each fixed candidate; preserve every known outcome"}
        persist(target, {"original": original})
        persist(durable, {"original": original})
        await work.commit.aio()
        await archive.commit.aio()
        await put_once(PREFIX + "/original", original)
        print("RECOVERY ARCHIVED", len(samples), "programs; known outcomes", known, flush=True)
        gate = execution.CreationGate(.30)
        backend = execution.ProgramBackend(context["setup"]["app_name"], context["program_runtime"]["image_id"], start_gate=gate)
        async def rerun(sample):
            sid = sample["sample_id"]
            if time.time() >= context["deadline"]:
                raise ReconciliationRequired("original deadline reached")
            intent = {"sample_hash": fingerprint(sample), "original_hash": fingerprint(history[sid]),
                      "additional_attempt": 1, "recovery_id": RECOVERY_ID}
            if not await store.put.aio(PREFIX + "/rerun/" + sid + "/intent", intent, skip_if_exists=True):
                raise ReconciliationRequired("recovery program already submitted; no second replay")
            persist(target / "reruns" / sid, {"intent": intent})
            await work.commit.aio()
            async def record(h, value):
                await put_once(PREFIX + "/rerun/" + sid + "/input/" + h, value)
                persist(target / "reruns" / sid / "inputs", {h: value})
            document = await backend.execute_program(sample["source"], cases, on_record=record)
            document["metadata"]["reviewed_recovery"] = intent
            persist(target / "reruns" / sid, {"result": document})
            persist(durable / "reruns" / sid, {"result": document})
            await put_once(PREFIX + "/rerun/" + sid + "/final", grading.metadata_for(document))
            matched = validate_replacement(document, history[sid], sample, cases, context["program_runtime"]["image_id"])
            print("RECOVERY PROGRAM VALIDATED", sid, "known outcomes matched", matched, flush=True)
            return sid, document
        replacements = dict(await bounded_map(samples, rerun, 4))
        await work.commit.aio()
        await archive.commit.aio()
        if len({d["metadata"]["sandbox_id"] for d in replacements.values()}) != 8:
            raise ValueError("recovery programs reused a sandbox")
        # Stop is retained through all replay and validation, not cleared on a
        # create success. No user/model activity can read partially published data.
        if await store.get.aio(grading.NAMESPACE + "/stop") != EXPECTED_STOP:
            raise ValueError("stop changed during recovery")
        for policy, expected in completed.items():
            if fingerprint(await store.get.aio(grading.NAMESPACE + "/finished/training/" + policy)) != expected:
                raise ValueError("completed policy changed during recovery")
        for key, group in jobs.items():
            documents = {}
            for sample in group:
                sid = sample["sample_id"]
                document = replacements[sid]
                documents["programs/" + sid] = {"intent": history[sid]["intent"], "result": document}
                documents["programs/" + sid + "/inputs"] = document["records"]
            archived_publish(directory / "grading" / key, target / "original_files" / key, documents)
            await work.commit.aio()
            for sample in group:
                sid = sample["sample_id"]
                job = grading.NAMESPACE + "/grading/" + key + "/" + sid
                previous, document = history[sid], replacements[sid]
                if await store.get.aio(job + "/final", None) != previous["final"]:
                    raise ValueError("canonical final changed during recovery")
                for case in cases:
                    name = job + "/input/" + case.input_hash
                    if await store.get.aio(name, None) != previous["records"].get(case.input_hash):
                        raise ValueError("canonical input changed during recovery")
                    await store.put.aio(name, document["records"][case.input_hash])
                await store.put.aio(job + "/final", grading.metadata_for(document))
        # Freshly read and validate exactly what the unchanged grader will use.
        for key, group in jobs.items():
            for sample in group:
                sid = sample["sample_id"]
                job = grading.NAMESPACE + "/grading/" + key + "/" + sid
                final = await store.get.aio(job + "/final")
                records = {c.input_hash: await store.get.aio(job + "/input/" + c.input_hash) for c in cases}
                document = grading.assemble(final, records)
                if document != replacements[sid]:
                    raise ValueError("published recovery differs from validated receipt")
                validate_replacement(document, history[sid], sample, cases, context["program_runtime"]["image_id"])
        receipt = {"recovery_id": RECOVERY_ID, "original_hash": fingerprint(original),
                   "recovered_programs": sorted(replacements), "known_outcomes_preserved": known,
                   "replacement_hashes": {s: fingerprint(v) for s, v in replacements.items()},
                   "completed_policy_hashes": completed, "plan_unchanged": True, "deadline_unchanged": True,
                   "stop_released": EXPECTED_STOP, "sandbox_seconds": str(sum(
                       min(execution.lifetime_for(96), math.ceil(d["metadata"]["total_seconds"]))
                       for d in replacements.values()))}
        persist(target, {"receipt": receipt})
        persist(durable, {"receipt": receipt})
        await work.commit.aio()
        await archive.commit.aio()
        await put_once(PREFIX + "/receipt", receipt)
        if await store.get.aio(grading.NAMESPACE + "/stop") != EXPECTED_STOP:
            raise ValueError("stop changed before release")
        await store.pop.aio(grading.NAMESPACE + "/stop")
        print("RECOVERY READY TO RESUME", receipt["recovered_programs"], flush=True)
        return receipt
    receipt = asyncio.run(perform())
    return {"receipt": receipt, "seconds": time.monotonic() - started}


@app.local_entrypoint()
def main(apply: bool = False):
    if not apply:
        raise ValueError("--apply required for this single reviewed recovery")
    context = grpo_recovery.read_json(root("runs") / "context.json")
    sources = {name: Path(name).read_text() for name in context["source_snapshot"]}
    validate_context(context, sources, time.time())
    apps = json.loads(subprocess.check_output([sys.executable, "-m", "modal", "app", "list", "--json"], text=True))
    # The CLI already owns the SDK client's event loop. Use its bridge rather
    # than moving the cached gRPC client into a fresh asyncio.run loop.
    from modal._utils.async_utils import synchronizer
    parent_terminal = synchronizer.create_blocking(require_idle_apps)(apps)
    owner = modal.App.lookup(context["setup"]["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("active candidate sandbox prevents recovery")
    if store.get(grading.NAMESPACE + "/stop", None) != EXPECTED_STOP:
        raise ValueError("unexpected stop flag")
    authorization = store.get(grading.NAMESPACE + "/authorization")
    if authorization["context_hash"] != fingerprint(context):
        raise ValueError("frozen study authorization changed")
    rates = json.loads(subprocess.check_output([sys.executable, "-m", "modal", "billing", "rates", "--json"], text=True))
    if any(accounting.hourly(rates, k) != accounting.hourly(context["rates"], k) for k in ("cpu", "gpu", "sandbox")):
        raise ValueError("resource rates changed")
    identity = fingerprint({"context": fingerprint(context), "recovery_id": RECOVERY_ID,
                            "source_hash": digest(Path(__file__).read_text()), "seconds": SECONDS})
    if not store.put(PREFIX + "/authorization", {"identity": identity, "stop": EXPECTED_STOP,
                                                "parent_terminal": parent_terminal}, skip_if_exists=True):
        raise ReconciliationRequired("recovery already submitted; inspect its receipts")
    budget_key = study.RUN_ID + "/budget"
    ledger = store.get(budget_key)
    if ledger["binding"] != context["runtime_amendment"]["original_budget_binding"] or ledger["ledger"]["ceiling"] != "250":
        raise ValueError("existing budget changed; never reset it")
    maximum = accounting.cost(rates, "cpu", SECONDS + 60) + accounting.cost(rates, "sandbox", 8 * execution.lifetime_for(96))
    ticket = grading.AMENDMENT + "/recovery/" + RECOVERY_ID
    reserved = dict(ledger, ledger=accounting.reserve(ledger["ledger"], ticket, maximum, identity))
    store.put(budget_key, reserved)
    directory = root("runs") / "recoveries" / RECOVERY_ID
    persist(directory, {"authorization": {"identity": identity, "maximum_usd": str(maximum),
                                         "prior_budget_hash": fingerprint(ledger), "stop": EXPECTED_STOP,
                                         "parent_terminal": parent_terminal}})
    call = recover.spawn(context, identity)
    persist(directory, {"launch": {"call_id": call.object_id}})
    print("RECOVERY CALL", call.object_id, "maximum USD", maximum, flush=True)
    result = call.get()
    current = store.get(budget_key)
    if current != reserved:
        raise ValueError("budget changed during exclusive maintenance; preserve hold")
    actual = accounting.cost(rates, "cpu", math.ceil(result["seconds"]) + 60)
    actual += accounting.cost(rates, "sandbox", Decimal(result["receipt"]["sandbox_seconds"]))
    settled = dict(current, ledger=accounting.settle(current["ledger"], ticket, actual, identity))
    store.put(budget_key, settled)
    persist(directory, {"result": result, "budget_after": settled})
    print("RECOVERY COMPLETE", "estimated USD", actual, "original study may now resume", flush=True)
