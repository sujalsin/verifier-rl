"""Reviewed drain then evaluation-only handoff. Never terminates a live candidate."""
import asyncio
import json
import math
from pathlib import Path
import time

import modal
import modal_parallel_evaluation as cloud
from verifier_rl import parallel_evaluation as parallel, parallel_handoff as handoff
from verifier_rl import program_grading as old, program_storage as storage
from verifier_rl import booking_replication as study, compute_budget as accounting
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import digest

app = cloud.app
image = cloud.image.add_local_file(__file__, "/root/modal_parallel_handoff.py")
work, archive, store = cloud.work, cloud.archive, cloud.store
NAME = "research-001"
ROOT = Path("/artifacts")/study.RUN_ID/old.RELATIVE
ADMIN_SECONDS = 1800
CONTROLLER_SECONDS = 60000
CONTROL_REVISION = "lifecycle-bridge-001"


async def parent_lifecycle():
    from modal.client import _Client
    from modal_proto import api_pb2
    client = await _Client.from_env()
    response = await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=cloud.PARENT_APP),timeout=30,retry=None)
    return {"app_id": cloud.PARENT_APP, "state": api_pb2.AppState.Name(response.lifecycle.app_state),
            "stopped_at":response.lifecycle.stopped_at,"checked_at":time.time()}


def lifecycle_sync():
    # Private SDK RPCs must execute on the SDK's persistent loop. Public .aio
    # methods already bridge it; asyncio.run(private_rpc()) does not.
    from modal._utils.async_utils import synchronizer
    return synchronizer.create_blocking(parent_lifecycle)()


def qualify(config):
    cloud.validate_config(config)
    if config["name"] != NAME or config["runtime"]["mode"] != "research":
        raise ValueError("wrong handoff namespace")
    for name, expected in config["handoff_sources"].items():
        if digest((Path(__file__).parent/name).read_text()) != expected:
            raise ValueError("unreviewed handoff source")
    canary_config = store.get(cloud.PREFIX+"/canary-001/approved_config",None)
    result = store.get(cloud.PREFIX+"/canary-001/result",None)
    if (result is None or not result["passed"] or result["recovery_candidate_reexecutions"] != 0
            or result["overlapping_batches"] < 2 or canary_config is None
            or result["config_hash"] != parallel.fingerprint(canary_config)
            or config["scheduler_sources"] != canary_config["scheduler_sources"]
            or config["scientific_sources"] != canary_config["scientific_sources"]):
        raise ValueError("matching successful cloud canary required")


async def inventory(parent, config):
    saved = {}
    for seed in study.SEEDS:
        for arm in study.ARMS:
            policy = study.label(seed,arm)
            entry = await storage.get(store,old.NAMESPACE+"/finished/training/"+policy)
            if entry is None: raise ValueError("missing trained policy")
            result = entry["payload"]
            study.validate_arm(result,parent["plan"])
            if handoff.read(ROOT/"arms"/policy/"result.json") != result:
                raise ValueError("training receipt changed")
            saved[policy] = result
    jobs = parallel.remaining_jobs(saved,parent["plan"],config["runtime"],set())
    completed, missing, unresolved = {}, [], []
    with ProgressLog(NAME,label="HANDOFF_INVENTORY") as progress:
        progress.stage("classify_fixed_batches",total=480,unit="batches")
        for manifest in jobs:
            key = manifest["key"]
            if (ROOT/"grading"/key/"result.json").exists():
                intents = {s["sample_id"]:None for s in manifest["samples"]}
            else:
                async def intent(s):
                    return s["sample_id"],await storage.get(store,old.NAMESPACE+"/grading/"+key+"/"+s["sample_id"]+"/intent")
                intents = dict(await asyncio.gather(*(intent(s) for s in manifest["samples"])))
            try:
                classified = handoff.classify_old(manifest,ROOT,parent,intents)
            except ReconciliationRequired as exc:
                unresolved.append({"key":key,"detail":str(exc)})
            else:
                if classified["status"] == "completed": completed[key] = classified
                else: missing.append(manifest)
            progress.advance()
    unknown = await storage.get(store,old.NAMESPACE+"/unknown/"+storage.AMENDMENT)
    if unknown is None or unknown["identities"]:
        raise ReconciliationRequired("nonempty or missing inherited unknown ledger requires explicit review")
    return {"saved_arms":saved,"completed":completed,"missing":missing,"unresolved":unresolved,
            "baseline":handoff.read(ROOT/"policies/baseline-0/result.json"),"unknown_ledger":unknown}


@app.function(image=image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,
              timeout=ADMIN_SECONDS,retries=0,max_containers=1,scaledown_window=2,
              volumes={"/artifacts":work,"/evidence":archive})
def inspect_handoff(parent,config,drain=False):
    qualify(config)
    started = time.time()
    stop = {"reason":"reviewed_parallel_evaluation_handoff","config_hash":parallel.fingerprint(config),
            "successor":NAME}
    if drain:
        current = store.get(old.NAMESPACE+"/stop",None)
        if current not in (None,stop): raise ReconciliationRequired("different old stop reason; do not override")
        if current is None and not store.put(old.NAMESPACE+"/stop",stop,skip_if_exists=True):
            raise ReconciliationRequired("stop raced; inspect")
        # This only blocks new submissions. Old program cleanups finish normally.
        with ProgressLog(NAME,label="HANDOFF_DRAIN") as progress:
            progress.stage("wait_for_old_app_to_finish",unit="seconds")
            while True:
                lifecycle = lifecycle_sync()
                if lifecycle["state"] == "APP_STATE_STOPPED": break
                if time.time()-started > 900:
                    raise ReconciliationRequired("old app has not drained; do not start replacement")
                time.sleep(15)
    else:
        lifecycle = lifecycle_sync()
    work.reload()
    result = asyncio.run(inventory(parent,config))
    result["lifecycle"] = lifecycle
    result["config_hash"] = parallel.fingerprint(config)
    summary = {"completed_batches":len(result["completed"]),"missing_batches":len(result["missing"]),
               "unresolved":result["unresolved"],"old_app_state":lifecycle["state"],"snapshot_only":not drain}
    if drain:
        if result["unresolved"] or len(result["completed"])+len(result["missing"]) != 480:
            raise ReconciliationRequired("partial legacy work must be reconciled without execution retries")
        budget = store.get(study.RUN_ID+"/budget")
        result["budget_before"] = budget
        persist(cloud.paths(config),{"inventory":result,"config":config})
        work.commit()
        persist(cloud.paths(config,"/evidence"),{"inventory":result,"config":config})
        archive.commit()
        store.put(cloud.PREFIX+"/"+NAME+"/ready",{"inventory_hash":parallel.fingerprint(result),
                   "config_hash":parallel.fingerprint(config),"lifecycle":lifecycle},skip_if_exists=True)
    print("HANDOFF INSPECTION",json.dumps(summary),flush=True)
    return summary


@app.function(image=image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,
              timeout=60,retries=0,max_containers=1,scaledown_window=2)
def handed_budget(config,action,ticket,amount,identity):
    qualify(config)
    ready = store.get(cloud.PREFIX+"/"+NAME+"/ready",None)
    if ready is None or ready["config_hash"] != parallel.fingerprint(config):
        raise ReconciliationRequired("budget writer cannot transfer before reviewed drain")
    if lifecycle_sync()["state"] != "APP_STATE_STOPPED":
        raise ReconciliationRequired("old budget writer may still be active")
    writer = {"function_id":handed_budget.object_id,"config_hash":parallel.fingerprint(config)}
    owner_key = cloud.PREFIX+"/"+NAME+"/budget-owner"
    if not store.put(owner_key,writer,skip_if_exists=True) and store.get(owner_key) != writer:
        raise ReconciliationRequired("another successor owns the budget")
    state = store.get(study.RUN_ID+"/budget")
    if state["binding"] != config["budget_binding"] or state["ledger"]["ceiling"] != "250":
        raise ValueError("original cumulative budget changed")
    key = parallel.AMENDMENT+"/"+NAME+"/"+ticket
    if action == "reserve": state["ledger"] = accounting.reserve(state["ledger"],key,amount,identity)
    elif action == "settle": state["ledger"] = accounting.settle(state["ledger"],key,amount,identity)
    elif action != "status": raise ValueError("invalid accounting action")
    store.put(study.RUN_ID+"/budget",state)
    return {"committed_usd":str(accounting.committed(state["ledger"]))}


@app.function(image=image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,
              timeout=CONTROLLER_SECONDS,retries=0,max_containers=1,scaledown_window=2,
              volumes={"/artifacts":work,"/evidence":archive})
def evaluate_remaining(config):
    qualify(config)
    work.reload()
    snapshot = handoff.read(cloud.paths(config)/"inventory.json")
    ready = store.get(cloud.PREFIX+"/"+NAME+"/ready")
    if parallel.fingerprint(snapshot) != ready["inventory_hash"]:
        raise ValueError("handoff manifest changed")
    owner = cloud.PREFIX+"/"+NAME+"/controller"
    if not store.put(owner,{"started":time.time()},skip_if_exists=True):
        raise ReconciliationRequired("controller already entered; no blind restart")
    if snapshot["missing"]:
        cloud.coordinate.remote(config,"initialize",{"jobs":snapshot["missing"]})
    with ProgressLog(NAME,label="PARALLEL_STUDY") as progress:
        progress.stage("evaluate_fixed_remaining_samples",total=480,unit="batches",workers=4,program_limit=16)
        progress.advance(len(snapshot["completed"]))
        async def run():
            async def one(manifest):
                identity = parallel.fingerprint(manifest)
                key = manifest["key"]
                if time.time() >= config["runtime"]["deadline"]:
                    raise ReconciliationRequired("evaluation deadline; preserve remaining work")
                maximum = accounting.cost(config["rates"],"cpu",parallel.MAX_SECONDS+60)
                maximum += accounting.cost(config["rates"],"sandbox",old.maximum_sandbox_seconds(manifest["samples"],"evaluation"))
                # Ten broker calls plus bounded immutable-publication retries, retained as an overhead hold.
                broker_hold = accounting.cost(config["rates"],"cpu",30*60)
                await handed_budget.remote.aio(config,"reserve",key,str(maximum+broker_hold),identity)
                result = await cloud.grade_parallel.remote.aio(config,manifest)
                actual = accounting.cost(config["rates"],"cpu",math.ceil(result["ended"]-result["started"])+60)
                actual += accounting.cost(config["rates"],"sandbox",result["sandbox_seconds"])
                await handed_budget.remote.aio(config,"settle",key,str(actual+broker_hold),identity)
                progress.advance()
                print("PARALLEL STUDY COMPLETED BATCH",key,flush=True)
                return key
            return await parallel.bounded_map(snapshot["missing"],one,parallel.WORKERS)
        asyncio.run(run())
        progress.stage("verify_and_compute_original_paired_metrics",total=480,unit="batches")
        work.reload()
        rows = {}
        for key,previous in snapshot["completed"].items():
            result = handoff.read(previous["legacy_path"])
            if parallel.fingerprint(result) != previous["result_hash"]:
                raise ValueError("retained old result changed")
            rows[key] = {k:previous[k] for k in ("rows","sandbox_ids")}
            progress.advance()
        for manifest in snapshot["missing"]:
            target = cloud.paths(config)/"grading"/manifest["key"]
            rows[manifest["key"]] = handoff.check_new(manifest,handoff.read(target/"raw.json"),handoff.read(target/"receipt.json"))
            progress.advance()
        result = handoff.assemble(snapshot["baseline"],rows,snapshot["saved_arms"])
        result["budget"] = handed_budget.remote(config,"status","",None,None)
        result["provenance"] = {"amendment":parallel.AMENDMENT,"config_hash":parallel.fingerprint(config),
            "legacy_batches_reused":len(snapshot["completed"]),"new_parallel_batches":len(snapshot["missing"])}
        persist(cloud.paths(config),{"result":result})
        work.commit()
        persist(cloud.paths(config,"/evidence"),{"result":result})
        archive.commit()
        store.put(cloud.PREFIX+"/"+NAME+"/result",result,skip_if_exists=True)
        print("PARALLEL STUDY COMPLETE",result["status"],result["analysis"]["contrasts"],flush=True)
        return {"status":result["status"],"analysis":result["analysis"],"provenance":result["provenance"]}


@app.local_entrypoint()
def handoff_main(phase: str = "inspect"):
    if phase not in ("inspect","drain","run"): raise ValueError("explicit inspect/drain/run phase required")
    parent = handoff.read(Path("runs")/cloud.PARENT_CONTEXT)
    canary_config = handoff.read(Path("runs")/cloud.RELATIVE/"canary-001/config.json")
    canary_result = store.get(cloud.PREFIX+"/canary-001/result",None)
    if canary_result is None or not canary_result["passed"]:
        raise ReconciliationRequired("cloud canary not passed; current grader stays untouched")
    if not store.put(cloud.PREFIX+"/canary-001/approved_config",canary_config,skip_if_exists=True):
        if store.get(cloud.PREFIX+"/canary-001/approved_config") != canary_config:
            raise ValueError("canary release differs")
    config = cloud.config_for(parent,NAME,"research")
    config["handoff_sources"] = {n:digest(Path(n).read_text()) for n in
        ("modal_parallel_handoff.py","verifier_rl/parallel_handoff.py")}
    original = parent["budget_context"]
    config["budget_binding"] = parallel.fingerprint({k:original[k] for k in ("plan","rates","billing_before","deadline")})
    config["control_revision"] = CONTROL_REVISION
    admin_key = old.AMENDMENT+"/"+parallel.AMENDMENT+"/handoff-admin"
    ledger = store.get(study.RUN_ID+"/budget")
    held_admin = ledger["ledger"]["items"].get(admin_key)
    if ledger["binding"] != config["budget_binding"]:
        raise ValueError("admin budget belongs to another study")
    if held_admin is not None and (held_admin["maximum"] != "5" or held_admin["actual"] is not None):
        raise ValueError("admin hold was changed or already settled")
    # The cancelled read-only inspection did not execute candidates. Continue
    # this bounded admin allocation, retaining its original identity and $5
    # hold, rather than silently releasing it or granting a fresh allowance.
    config["admin_reservation_identity"] = held_admin["identity"] if held_admin else parallel.fingerprint(config)
    qualify(config)
    if phase in ("inspect","drain"):
        # Keep this reservation conservative: admin work shares a bounded $5
        # hold; never free it without a trusted complete accounting receipt.
        if phase == "inspect":
            if held_admin is None:
                from modal._utils.async_utils import synchronizer
                synchronizer.create_blocking(cloud.reserve_canary)(parent,
                    parallel.AMENDMENT+"/handoff-admin",config["admin_reservation_identity"])
        elif store.get(study.RUN_ID+"/budget")["ledger"]["items"].get(
                admin_key,{}).get("identity") != config["admin_reservation_identity"]:
            raise ReconciliationRequired("inspect and reserve before drain")
        call = inspect_handoff.spawn(parent,config,drain=phase=="drain")
        persist(Path("runs")/cloud.RELATIVE/NAME/CONTROL_REVISION,
                {phase+"-launch":{"call_id":call.object_id},"config":config})
        print("HANDOFF ADMIN CALL",call.object_id,flush=True)
        result = call.get()
        persist(Path("runs")/cloud.RELATIVE/NAME/CONTROL_REVISION,{phase:result})
        print(json.dumps(result,indent=2))
    else:
        identity = parallel.fingerprint(config)
        maximum = accounting.cost(config["rates"],"cpu",CONTROLLER_SECONDS+60)
        status = handed_budget.remote(config,"reserve","controller",str(maximum),identity)
        print("PARALLEL RESEARCH BUDGET",status,flush=True)
        call = evaluate_remaining.spawn(config)
        persist(Path("runs")/cloud.RELATIVE/NAME,{"launch":{"call_id":call.object_id}})
        print("PARALLEL EVALUATION CALL",call.object_id,flush=True)
        print(json.dumps(call.get(),indent=2))
