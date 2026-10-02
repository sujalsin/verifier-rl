"""Reviewed accounting-only handoff; all candidate execution stays frozen.

An explicit temporary admission escrow uses the existing serialized writer to
drain new submissions. It allocates no compute. No active input is cancelled.
The successor writes the ledger only after the old application has stopped.
"""
import asyncio
from copy import deepcopy
from decimal import Decimal
import json
import math
from pathlib import Path
import time

import modal
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal._functions import _Function
from modal_proto import api_pb2

import modal_parallel_evaluation as cloud
from verifier_rl import compute_budget as accounting, budget_reconciliation as audit
from verifier_rl import parallel_evaluation as parallel, parallel_handoff as handoff
from verifier_rl.evaluation_journal import persist, ReconciliationRequired
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import digest

app = cloud.app
image = cloud.image.add_local_file(__file__, "/root/modal_budget_handoff.py")
store, work, archive = cloud.store, cloud.work, cloud.archive
RUN = cloud.study.RUN_ID
OLD_APP = "ap-mDjiaxflNVbgsxtuzGoo4X"
OLD_NAME, NAME = "research-001", "research-002"
OLD_PREFIX = cloud.PREFIX + "/" + OLD_NAME
PREFIX = cloud.PREFIX + "/" + NAME
ADMIN_TICKET = "budget-reconciliation-001/handoff-admin"
ESCROW_TICKET = "budget-reconciliation-001/admission-escrow"
EVIDENCE_PATH = (Path("runs") / RUN / "budget-reconciliation-001/20261001T212501793362Z/provider_call_evidence.json")
SOURCES = ("modal_budget_handoff.py", "verifier_rl/budget_reconciliation.py", "verifier_rl/compute_budget.py",
           "verifier_rl/parallel_handoff.py")
RESUME_NAME = "research-003"
RESUME_APP = "ap-0qULKjR1R8qQ2Sm153ec38"
IDENTITY_REVISION = "local-app-identity-001"


async def lifecycle(app_id):
    client = await _Client.from_env()
    response = await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id), timeout=20, retry=None)
    return {"app_id": app_id, "state": api_pb2.AppState.Name(response.lifecycle.app_state),
            "stopped_at": response.lifecycle.stopped_at}


def current_app_id():
    """Use container-hydrated identity, not a transient invocation lookup."""
    value = app.app_id
    if not isinstance(value, str) or not value.startswith("ap-") or len(value) <= 3:
        raise ReconciliationRequired("running app identity is unavailable")
    return value


def is_identity_resume(config):
    return config.get("accounting_revision") == IDENTITY_REVISION


def namespace(config):
    return cloud.PREFIX + "/" + config.get("name", NAME)


def predecessor_app(config):
    return RESUME_APP if is_identity_resume(config) else OLD_APP


async def old_budget(config, action, ticket, amount, identity):
    client = await _Client.from_env()
    layout = (await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=OLD_APP), timeout=20, retry=None)).app_layout
    fid = layout.function_ids["handed_budget"]
    owner = await store.get.aio(OLD_PREFIX + "/budget-owner")
    if owner != {"function_id": fid, "config_hash": parallel.fingerprint(config["previous_config"])}:
        raise ValueError("old budget ownership differs")
    meta = [o.function_handle_metadata for o in layout.objects if o.object_id == fid and o.HasField("function_handle_metadata")]
    if len(meta) != 1:
        raise ValueError("unavailable sole budget writer")
    function = _Function._new_hydrated(fid, client, meta[0])
    return await function.remote(config["previous_config"], action, ticket, amount, identity)


def qualify(config):
    cloud.validate_config(config)
    old = config["previous_config"]
    names = (RESUME_NAME, NAME) if is_identity_resume(config) else (NAME, OLD_NAME)
    if (config["name"] != names[0] or old["name"] != names[1]
            or config["runtime"] != old["runtime"] or config["plan"] != old["plan"]
            or config["setup"] != old["setup"] or config["rates"] != old["rates"]
            or config["budget_binding"] != old["budget_binding"]
            or config["scientific_sources"] != old["scientific_sources"]
            or config["scheduler_sources"] != old["scheduler_sources"]):
        raise ValueError("accounting-only handoff changed scientific settings")
    if set(config["accounting_sources"]) != set(SOURCES):
        raise ValueError("complete accounting source binding required")
    for name, expected in config["accounting_sources"].items():
        if digest((Path(__file__).parent / name).read_text()) != expected:
            raise ValueError("unreviewed accounting source: " + name)


def local_path():
    return Path("runs") / cloud.RELATIVE / NAME


def save_both(config, relative, documents):
    persist(cloud.paths(config) / relative, documents)
    work.commit()
    persist(cloud.paths(config, "/evidence") / relative, documents)
    archive.commit()


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=300, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts":work, "/evidence":archive})
def reconciled_budget(config, action, payload):
    """Sole new writer, with durable before/after reconciliation and clear reports."""
    qualify(config)
    if synchronizer.create_blocking(lifecycle)(predecessor_app(config))["state"] != "APP_STATE_STOPPED":
        raise ReconciliationRequired("previous writer is still active")
    prefix = namespace(config)
    ownership = {"function_id":reconciled_budget.object_id, "config_hash":parallel.fingerprint(config),
                 "app_id":current_app_id()}
    if not store.put(prefix + "/budget-owner", ownership, skip_if_exists=True):
        previous = store.get(prefix + "/budget-owner")
        if previous != ownership:
            if (previous["config_hash"] != ownership["config_hash"]
                    or synchronizer.create_blocking(lifecycle)(previous["app_id"])["state"] != "APP_STATE_STOPPED"):
                raise ReconciliationRequired("another live successor owns this ledger")
            # Separate prepare/run apps may transfer ownership only after the
            # entire previous owner application has terminated.
            store.put(prefix + "/budget-owner", ownership)
    saved = store.get(RUN + "/budget")
    if saved["binding"] != config["budget_binding"] or saved["ledger"]["ceiling"] != "250":
        raise ValueError("original budget binding or ceiling changed")
    ready = store.get(prefix + "/accounting-ready", None)
    if action == "initialize":
        if ready is not None:
            if ready["config_hash"] != parallel.fingerprint(config):
                raise ValueError("accounting initialization belongs to another configuration")
            return ready
        work.reload()
        directory = cloud.paths(config) / "accounting"
        if (directory / "transaction.json").exists():
            transaction = handoff.read(directory / "transaction.json")
            if transaction["authorization"] != payload:
                raise ValueError("reconciliation authorization changed")
            if saved not in (transaction["before"], transaction["after"]):
                raise ReconciliationRequired("ledger differs from both durable transaction states")
        elif is_identity_resume(config):
            old_config = config["previous_config"]
            inherited = store.get(namespace(old_config) + "/accounting-ready")
            old_owner = store.get(namespace(old_config) + "/budget-owner")
            expected = {"revision":IDENTITY_REVISION, "previous_config_hash":parallel.fingerprint(old_config)}
            if (payload != expected or inherited["config_hash"] != expected["previous_config_hash"]
                    or old_owner["config_hash"] != expected["previous_config_hash"]
                    or old_owner["app_id"] != RESUME_APP):
                raise ReconciliationRequired("accounting predecessor identity differs")
            # Preserve the complete reconciled ledger. No release, reset,
            # second reconciliation or old reservation settlement is allowed.
            transaction = {"before":saved, "after":deepcopy(saved), "authorization":payload,
                           "inherited_transaction_hash":inherited["transaction_hash"],
                           "verified_hold_reduction_usd":"0"}
            save_both(config, "accounting", {"transaction":transaction})
        else:
            items, selected = {}, {}
            for key, evidence in payload["evidence"].items():
                item = saved["ledger"]["items"].get(key)
                if item is None or item["actual"] is not None:
                    continue
                kind = "gpu" if "/training/" in key else "cpu"
                try:
                    receipt = audit.completed_call_bound(item, evidence, config["rates"], kind)
                except (KeyError, ValueError):
                    continue  # Incomplete evidence retains its full hold.
                if Decimal(receipt["estimated_compute_usd"]) < Decimal(item["maximum"]):
                    items[key], selected[key] = receipt, evidence
            proposal = {"before_hash":parallel.fingerprint(saved), "rates":config["rates"], "items":items}
            after = audit.reconcile_offline(saved, proposal, selected, writers_stopped=True)
            escrow = payload["escrow"]
            escrow_key = parallel.AMENDMENT + "/" + OLD_NAME + "/" + ESCROW_TICKET
            if (escrow["key"] != escrow_key or escrow["compute_allocated"] is not False
                    or escrow["purpose"] != "reviewed_admission_only_drain"
                    or saved["ledger"]["items"][escrow_key] != escrow["item"]):
                raise ValueError("admission escrow receipt differs")
            after["ledger"] = accounting.settle(after["ledger"], escrow_key, "0", escrow["item"]["identity"])
            transaction = {"before":saved, "after":after, "proposal":proposal, "authorization":payload,
                           "verified_hold_reduction_usd":str(accounting.committed(saved["ledger"])
                               - accounting.committed(after["ledger"]) - Decimal(escrow["item"]["maximum"]))}
            save_both(config, "accounting", {"transaction":transaction})
        # Durable evidence precedes publication; recovery accepts before or after
        # but never writes over unrelated reservations. No other writer is live.
        store.put(RUN + "/budget", transaction["after"])
        if store.get(RUN + "/budget") != transaction["after"]:
            raise ReconciliationRequired("budget publication not confirmed")
        ready = {"config_hash":parallel.fingerprint(config), "transaction_hash":parallel.fingerprint(transaction),
                 "verified_hold_reduction_usd":transaction["verified_hold_reduction_usd"],
                 "report":audit.describe(transaction["after"]["ledger"])}
        store.put(prefix + "/accounting-ready", ready, skip_if_exists=True)
        return ready
    if ready is None or ready["config_hash"] != parallel.fingerprint(config):
        raise ReconciliationRequired("reconcile before reserving successor work")
    if action == "reserve":
        saved["ledger"] = accounting.reserve(saved["ledger"], parallel.AMENDMENT + "/" + config.get("name", NAME) + "/" + payload["ticket"],
                                               payload["amount"], payload["identity"])
    elif action == "settle":
        saved["ledger"] = accounting.settle(saved["ledger"], parallel.AMENDMENT + "/" + config.get("name", NAME) + "/" + payload["ticket"],
                                              payload["amount"], payload["identity"])
    elif action == "drain":
        store.put(prefix + "/drain", {"reason":"reviewed_accounting_maintenance"}, skip_if_exists=True)
    elif action != "status":
        raise ValueError("unknown budget action")
    if action in ("reserve", "settle"):
        store.put(RUN + "/budget", saved)
    return audit.describe(saved["ledger"])


@app.function(image=image, cpu=(1,1), memory=(256,256), timeout=60, retries=0, max_containers=1)
def accounting_control(config):
    """Cloud-only bookkeeping control; no research state or candidate execution."""
    qualify(config)
    ledger = accounting.initialize()
    for i in range(4):
        ledger = accounting.reserve(ledger, str(i), ".72", str(i))
    hold = str(Decimal("250") - accounting.committed(ledger))
    ledger = accounting.reserve(ledger, "escrow", hold, "control")
    ledger = accounting.settle(ledger, "0", ".12", "0")
    try:
        accounting.reserve(ledger, "next", ".72", "next")
    except accounting.BudgetReached:
        pass
    else:
        raise AssertionError("admission hold did not block new work")
    for i in range(1,4):
        ledger = accounting.settle(ledger, str(i), ".12", str(i))
    ledger = accounting.settle(ledger, "escrow", "0", "control")
    if accounting.committed(ledger) != Decimal("20.48"):
        raise AssertionError("drain lost a completed settlement")
    receipt = {"passed":True, "completed_settlements_preserved":4, "candidate_executions":0,
               "real_ledger_edits":0, "config_hash":parallel.fingerprint(config)}
    store.put(PREFIX + "/control", receipt, skip_if_exists=True)
    print("ACCOUNTING ADMISSION CONTROL PASSED", receipt, flush=True)
    return receipt


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=3600, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts":work, "/evidence":archive})
def prepare(config, authorization):
    qualify(config)
    control = store.get(PREFIX + "/control")
    if not control["passed"] or control["config_hash"] != parallel.fingerprint(config):
        raise ValueError("passing cloud control required")
    with ProgressLog(NAME, label="BUDGET_HANDOFF") as progress:
        progress.stage("wait_for_all_active_batches_to_finish")
        end = time.time() + 1800
        while synchronizer.create_blocking(lifecycle)(OLD_APP)["state"] != "APP_STATE_STOPPED":
            if time.time() > end:
                raise ReconciliationRequired("old app has not drained; no forced termination")
            time.sleep(15)
        progress.stage("validate_completed_and_never_started_batches", total=480, unit="batches")
        work.reload()
        old_config = config["previous_config"]
        snapshot = handoff.read(cloud.paths(old_config) / "inventory.json")
        state = store.get(OLD_PREFIX + "/state")
        if state["unknown"] or state["stop"] or set(state["owners"]) != set(state["batches"]):
            raise ReconciliationRequired("partial or unknown old work; do not resubmit candidates")
        completed = {}
        for key, value in snapshot["completed"].items():
            result = handoff.read(value["legacy_path"])
            if parallel.fingerprint(result) != value["result_hash"]:
                raise ValueError("legacy completed evidence changed")
            completed[key] = {"rows":value["rows"], "sandbox_ids":value["sandbox_ids"],
                              "source_path":value["legacy_path"], "source_hash":value["result_hash"]}
            progress.advance()
        missing = []
        for manifest in snapshot["missing"]:
            key = manifest["key"]
            directory = cloud.paths(old_config) / "grading" / key
            if key in state["batches"]:
                raw, receipt = handoff.read(directory / "raw.json"), handoff.read(directory / "receipt.json")
                checked = handoff.check_new(manifest, raw, receipt)
                if state["batches"][key] != {k:receipt[k] for k in ("raw_hash", "job_hash", "outcome_hashes")}:
                    raise ValueError("published old batch differs from durable receipt")
                completed[key] = {**checked, "source_path":str(directory / "raw.json"),
                                  "source_hash":parallel.fingerprint(raw)}
            else:
                if directory.exists() or any(k.startswith(key + "/") for k in state["permits"]):
                    raise ReconciliationRequired("unpublished old execution may exist")
                missing.append(manifest)
            progress.advance()
        if len(completed) + len(missing) != 480:
            raise ValueError("incomplete fixed evaluation partition")
        inherited = {"completed":completed, "missing":missing, "baseline":snapshot["baseline"],
                     "saved_arms":snapshot["saved_arms"], "config_hash":parallel.fingerprint(config)}
        save_both(config, "", {"inventory":inherited, "config":config})
        progress.stage("reconcile_verified_budget_holds")
        accounting_result = reconciled_budget.remote(config, "initialize", authorization)
        ready = {"completed":len(completed), "remaining":len(missing),
                 "inventory_hash":parallel.fingerprint(inherited), "config_hash":parallel.fingerprint(config),
                 "accounting":accounting_result}
        store.put(PREFIX + "/ready", ready, skip_if_exists=True)
        print("BUDGET HANDOFF READY", json.dumps(ready), flush=True)
        return ready


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=60000, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts":work, "/evidence":archive})
def evaluate(config):
    began = time.time()
    qualify(config)
    prefix = namespace(config)
    ready = store.get(prefix + "/ready")
    if ready["config_hash"] != parallel.fingerprint(config):
        raise ValueError("resume readiness belongs to another configuration")
    if not store.put(prefix + "/controller", {"started":time.time()}, skip_if_exists=True):
        raise ReconciliationRequired("successor already started; no blind retry")
    work.reload()
    snapshot = handoff.read(cloud.paths(config) / "inventory.json")
    if parallel.fingerprint(snapshot) != ready["inventory_hash"]:
        raise ValueError("handoff inventory changed")
    if snapshot["missing"]:
        cloud.coordinate.remote(config, "initialize", {"jobs":snapshot["missing"]})
    with ProgressLog(config["name"], label="PARALLEL_STUDY") as progress:
        progress.stage("evaluate_fixed_remaining_samples", total=480, unit="batches", workers=4, program_limit=16)
        progress.advance(len(snapshot["completed"]))
        async def run():
            async def one(manifest):
                if time.time() >= config["runtime"]["deadline"] or await store.get.aio(prefix + "/drain", None):
                    raise ReconciliationRequired("graceful drain or evaluation deadline")
                key, identity = manifest["key"], parallel.fingerprint(manifest)
                maximum = accounting.cost(config["rates"], "cpu", parallel.MAX_SECONDS+60)
                maximum += accounting.cost(config["rates"], "sandbox",
                    cloud.previous.maximum_sandbox_seconds(manifest["samples"], "evaluation"))
                broker_hold = accounting.cost(config["rates"], "cpu", 1800)
                await reconciled_budget.remote.aio(config, "reserve", {"ticket":key, "amount":str(maximum+broker_hold), "identity":identity})
                result = await cloud.grade_parallel.remote.aio(config, manifest)
                actual = accounting.cost(config["rates"], "cpu", math.ceil(result["ended"]-result["started"])+60)
                actual += accounting.cost(config["rates"], "sandbox", result["sandbox_seconds"])
                report = await reconciled_budget.remote.aio(config, "settle", {"ticket":key,
                    "amount":str(actual+broker_hold), "identity":identity})
                progress.advance()
                print("PARALLEL STUDY COMPLETED BATCH", key, "ACCOUNTING_NOT_BILLING", json.dumps(report), flush=True)
                return key
            await parallel.bounded_map(snapshot["missing"], one, parallel.WORKERS)
        asyncio.run(run())
        progress.stage("verify_and_compute_original_paired_metrics", total=480, unit="batches")
        work.reload()
        rows = {}
        for key, value in snapshot["completed"].items():
            if parallel.fingerprint(handoff.read(value["source_path"])) != value["source_hash"]:
                raise ValueError("inherited raw evidence changed")
            rows[key] = {k:value[k] for k in ("rows", "sandbox_ids")}
            progress.advance()
        for manifest in snapshot["missing"]:
            directory = cloud.paths(config) / "grading" / manifest["key"]
            rows[manifest["key"]] = handoff.check_new(manifest, handoff.read(directory / "raw.json"), handoff.read(directory / "receipt.json"))
            progress.advance()
        result = handoff.assemble(snapshot["baseline"], rows, snapshot["saved_arms"])
        result["accounting_not_provider_billing"] = reconciled_budget.remote(config, "status", {})
        result["provenance"] = {"accounting_handoff":config["name"], "previous_app":predecessor_app(config),
            "config_hash":parallel.fingerprint(config), "retained_batches":len(snapshot["completed"]),
            "new_batches":len(snapshot["missing"]), "training_rewards_and_execution_unchanged":True}
        save_both(config, "", {"result":result})
        store.put(prefix + "/result", result, skip_if_exists=True)
        print("PARALLEL STUDY COMPLETE", result["status"], result["analysis"]["contrasts"], flush=True)
        return {"status":result["status"], "analysis":result["analysis"], "provenance":result["provenance"],
                "controller_seconds":time.time()-began}


def resume_inventory(snapshot, state, directory, *, total=480, advance=lambda: None):
    """Read-only reconciliation: completed evidence or provably never started."""
    manifests = {m["key"]:m for m in snapshot["missing"]}
    if (len(manifests) != len(snapshot["missing"])
            or set(manifests) & set(snapshot["completed"])
            or len(manifests) + len(snapshot["completed"]) != total):
        raise ValueError("incomplete or duplicate original batch partition")
    if set(state["allowed"]) != set(manifests):
        raise ValueError("scheduler does not match saved evaluation population")
    if state["unknown"] or state["stop"] or set(state["owners"]) != set(state["batches"]):
        raise ReconciliationRequired("partial or unknown old work; do not resubmit candidates")
    if not set(state["batches"]) <= set(manifests):
        raise ValueError("unexpected completed batch")
    completed, missing = {}, []
    for key, value in snapshot["completed"].items():
        if parallel.fingerprint(handoff.read(value["source_path"])) != value["source_hash"]:
            raise ValueError("inherited raw evidence changed")
        completed[key] = deepcopy(value)
        advance()
    for key, manifest in manifests.items():
        if state["allowed"][key]["identity"] != parallel.fingerprint(manifest):
            raise ValueError("saved program manifest changed")
        target = Path(directory) / "grading" / key
        if key in state["batches"]:
            raw = handoff.read(target / "raw.json")
            receipt = handoff.read(target / "receipt.json")
            checked = handoff.check_new(manifest, raw, receipt)
            if state["batches"][key] != {k:receipt[k] for k in ("raw_hash", "job_hash", "outcome_hashes")}:
                raise ValueError("published batch differs from durable receipt")
            completed[key] = {**checked, "source_path":str(target / "raw.json"),
                              "source_hash":parallel.fingerprint(raw)}
        else:
            tokens = list(state["permits"]) + list(state["programs"])
            if target.exists() or any(t.startswith(key + "/") for t in tokens):
                raise ReconciliationRequired("unpublished old execution may exist")
            missing.append(manifest)
        advance()
    return {"completed":completed, "missing":missing,
            "baseline":snapshot["baseline"], "saved_arms":snapshot["saved_arms"]}


@app.function(image=image, cpu=(1,1), memory=(256,256), nonpreemptible=True,
              timeout=120, retries=0, max_containers=1, scaledown_window=2)
def identity_control(config, expected_app, ordinal):
    """Real hydrated identity and real ownership path; no candidate execution."""
    from unittest.mock import patch
    qualify(config)
    if current_app_id() != expected_app:
        raise AssertionError("container app identity differs from launch identity")
    # Exercise the exact helper with the former dependency unavailable.
    with patch.object(_Client, "from_env", side_effect=modal.exception.NotFoundError("injected missing call metadata")):
        if current_app_id() != expected_app:
            raise AssertionError("identity unexpectedly depends on metadata lookup")
    before = store.get(RUN + "/budget")
    reconciled_budget.remote(config, "status", {})
    owner = store.get(namespace(config) + "/budget-owner")
    if owner != {"function_id":reconciled_budget.object_id, "app_id":expected_app,
                 "config_hash":parallel.fingerprint(config)}:
        raise AssertionError("real budget ownership differs from stable identity")
    if store.get(RUN + "/budget") != before:
        raise AssertionError("identity/status control altered the ledger")
    result = {"passed":True, "app_id":expected_app, "ordinal":ordinal,
              "input_id":modal.current_input_id(), "lookup_failure_injected":True,
              "ledger_unchanged":True, "candidate_executions":0}
    print("APP IDENTITY CONTROL PASSED", json.dumps(result), flush=True)
    return result


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts":work, "/evidence":archive})
def prepare_identity_resume(config):
    qualify(config)
    control = store.get(namespace(config) + "/identity-control")
    if not control["passed"] or control["config_hash"] != parallel.fingerprint(config):
        raise ReconciliationRequired("passing real identity/ownership control required")
    if synchronizer.create_blocking(lifecycle)(RESUME_APP)["state"] != "APP_STATE_STOPPED":
        raise ReconciliationRequired("failed evaluator is still active")
    with ProgressLog(config["name"], label="IDENTITY_RESUME") as progress:
        progress.stage("verify_saved_results_and_never_started_batches", total=480, unit="batches")
        work.reload()
        old = config["previous_config"]
        ready = store.get(namespace(old) + "/ready")
        snapshot = handoff.read(cloud.paths(old) / "inventory.json")
        if (ready["config_hash"] != parallel.fingerprint(old)
                or ready["inventory_hash"] != parallel.fingerprint(snapshot)):
            raise ValueError("predecessor inventory binding changed")
        state = store.get(namespace(old) + "/state")
        inherited = resume_inventory(snapshot, state, cloud.paths(old), advance=progress.advance)
        inherited["config_hash"] = parallel.fingerprint(config)
        save_both(config, "", {"inventory":inherited, "config":config})
        accounting_result = reconciled_budget.remote(config, "status", {})
        result = {"completed":len(inherited["completed"]), "remaining":len(inherited["missing"]),
                  "inventory_hash":parallel.fingerprint(inherited), "config_hash":parallel.fingerprint(config),
                  "previous_app":RESUME_APP, "accounting_not_billing":accounting_result}
        key = namespace(config) + "/ready"
        if not store.put(key, result, skip_if_exists=True) and store.get(key) != result:
            raise ReconciliationRequired("resume readiness already differs")
        print("IDENTITY RESUME READY", json.dumps(result), flush=True)
        return result


@app.local_entrypoint()
def identity_resume_main(phase: str = "test"):
    if phase not in ("test", "run"):
        raise ValueError("explicit test/run phase required")
    directory = Path("runs") / cloud.RELATIVE / RESUME_NAME
    if phase == "test":
        old = handoff.read(local_path() / "config.json")
        config = deepcopy(old)
        config.update(name=RESUME_NAME, previous_config=old, accounting_revision=IDENTITY_REVISION,
                      accounting_sources={n:digest(Path(n).read_text()) for n in SOURCES})
        qualify(config)
        prefix = namespace(config)
        if store.get(prefix + "/identity-control", None) is not None:
            raise ReconciliationRequired("identity control already completed; inspect its receipt")
        persist(directory, {"config":config})
        reconciled_budget.remote(config, "initialize", {"revision":IDENTITY_REVISION,
            "previous_config_hash":parallel.fingerprint(old)})
        reconciled_budget.remote(config, "reserve", {"ticket":"identity-resume-admin", "amount":"2",
                                                     "identity":parallel.fingerprint(config)})
        results = []
        for ordinal in range(3):
            call = identity_control.spawn(config, current_app_id(), ordinal)
            persist(directory, {f"identity-control-launch-{ordinal}":{"call_id":call.object_id}})
            results.append(call.get())
        result = {"passed":all(r["passed"] for r in results), "config_hash":parallel.fingerprint(config),
                  "checks":results, "candidate_executions":0}
        store.put(prefix + "/identity-control", result, skip_if_exists=True)
        persist(directory, {"identity-control":result})
        print("IDENTITY AND OWNERSHIP CLOUD CONTROLS PASSED", json.dumps(result), flush=True)
        return
    config = handoff.read(directory / "config.json")
    qualify(config)
    prefix = namespace(config)
    if not store.put(prefix + "/resume-claim", {"config_hash":parallel.fingerprint(config)}, skip_if_exists=True):
        raise ReconciliationRequired("resume already submitted; inspect saved call, do not repeat")
    call = prepare_identity_resume.spawn(config)
    persist(directory, {"prepare-launch":{"call_id":call.object_id, "app_id":current_app_id()}})
    print("VERIFYING IDENTITY RESUME", call.object_id, flush=True)
    ready = call.get()
    persist(directory, {"ready":ready})
    maximum = accounting.cost(config["rates"], "cpu", 60060)
    report = reconciled_budget.remote(config, "reserve", {"ticket":"controller", "amount":str(maximum),
                                                          "identity":parallel.fingerprint(config)})
    print("ACCOUNTING NOT PROVIDER BILLING", json.dumps(report), flush=True)
    call = evaluate.spawn(config)
    persist(directory, {"launch":{"call_id":call.object_id, "app_id":current_app_id()}})
    print("IDENTITY FIX PARALLEL EVALUATION", call.object_id, flush=True)
    result = call.get()
    cost = accounting.cost(config["rates"], "cpu", math.ceil(result["controller_seconds"])+60)
    report = reconciled_budget.remote(config, "settle", {"ticket":"controller", "amount":str(cost),
                                                       "identity":parallel.fingerprint(config)})
    persist(directory, {"result":result, "final_accounting":report})
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def budget_handoff_main(phase: str = "test"):
    if phase not in ("test", "handoff", "run"):
        raise ValueError("explicit test/handoff/run phase required")
    if phase == "test":
        old = handoff.read(Path("runs") / cloud.RELATIVE / OLD_NAME / "lifecycle-bridge-001/config.json")
        config = deepcopy(old)
        config.update(name=NAME, previous_config=old,
                      accounting_sources={n:digest(Path(n).read_text()) for n in SOURCES})
        qualify(config)
        identity = parallel.fingerprint(config)
        if store.get(PREFIX + "/control", None) is not None:
            raise ReconciliationRequired("control already submitted; inspect stored result")
        synchronizer.create_blocking(old_budget)(config, "reserve", ADMIN_TICKET, "2", identity)
        persist(local_path(), {"config":config, "provider_call_evidence":handoff.read(EVIDENCE_PATH)})
        call = accounting_control.spawn(config)
        persist(local_path(), {"control-launch":{"call_id":call.object_id}})
        print(json.dumps(call.get(), indent=2))
    else:
        config = handoff.read(local_path() / "config.json")
        qualify(config)
        if phase == "handoff":
            control = store.get(PREFIX + "/control")
            if not control["passed"] or control["config_hash"] != parallel.fingerprint(config):
                raise ValueError("cloud control not passed")
            if not store.put(PREFIX + "/handoff-claim", {"config_hash":parallel.fingerprint(config)}, skip_if_exists=True):
                raise ReconciliationRequired("handoff already submitted; inspect its saved call")
            escrow_key = parallel.AMENDMENT + "/" + OLD_NAME + "/" + ESCROW_TICKET
            for attempt in range(8):
                state = store.get(RUN + "/budget")
                existing = state["ledger"]["items"].get(escrow_key)
                if existing is not None:
                    matches = [handoff.read(p) for p in local_path().glob("escrow-intent-*.json")
                               if handoff.read(p)["item"] == existing]
                    if len(matches) != 1:
                        raise ValueError("unexpected admission hold; inspect")
                    intent = matches[0]
                    break
                amount = str(Decimal("250") - accounting.committed(state["ledger"]))
                identity = parallel.fingerprint({"purpose":"reviewed_admission_only_drain", "config":config, "amount":amount})
                existing = {"maximum":amount, "identity":identity, "actual":None}
                intent = {"key":escrow_key, "item":existing, "compute_allocated":False, "purpose":"reviewed_admission_only_drain"}
                persist(local_path(), {f"escrow-intent-{attempt}":intent})
                try:
                    status = synchronizer.create_blocking(old_budget)(config, "reserve", ESCROW_TICKET, amount, identity)
                    print("TEMPORARY ADMISSION HOLD; NOT COMPUTE SPEND", status, flush=True)
                    break
                except Exception:
                    fresh = store.get(RUN + "/budget")["ledger"]["items"].get(escrow_key)
                    if fresh == existing:
                        break
                    if fresh is not None:
                        raise
                    if attempt == 7:
                        raise
            authorization = {"evidence":handoff.read(local_path() / "provider_call_evidence.json"), "escrow":intent}
            call = prepare.spawn(config, authorization)
            persist(local_path(), {"handoff-launch":{"call_id":call.object_id}, "authorization":authorization})
            print("GRACEFUL ACCOUNTING HANDOFF", call.object_id, flush=True)
            result = call.get()
            persist(local_path(), {"ready":result})
            print(json.dumps(result, indent=2))
        else:
            ready = store.get(PREFIX + "/ready")
            if ready["config_hash"] != parallel.fingerprint(config):
                raise ValueError("successor is not ready")
            maximum = accounting.cost(config["rates"], "cpu", 60060)
            report = reconciled_budget.remote(config, "reserve", {"ticket":"controller", "amount":str(maximum),
                                                                  "identity":parallel.fingerprint(config)})
            print("ACCOUNTING NOT PROVIDER BILLING", json.dumps(report), flush=True)
            call = evaluate.spawn(config)
            persist(local_path(), {"launch":{"call_id":call.object_id}})
            print("RECONCILED PARALLEL EVALUATION", call.object_id, flush=True)
            result = call.get()
            cost = accounting.cost(config["rates"], "cpu", math.ceil(result["controller_seconds"])+60)
            report = reconciled_budget.remote(config, "settle", {"ticket":"controller", "amount":str(cost),
                                                                 "identity":parallel.fingerprint(config)})
            persist(local_path(), {"result":result, "final_accounting":report})
            print(json.dumps(result, indent=2))
