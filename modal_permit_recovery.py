"""Reviewed permit-clock repair and evaluation-only continuation."""
import asyncio
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import uuid

import modal
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal_proto import api_pb2

import modal_parallel_evaluation as cloud
from verifier_rl import permit_recovery as repair, parallel_evaluation as parallel
from verifier_rl import parallel_handoff as handoff
from verifier_rl import compute_budget as accounting, budget_reconciliation as audit
from verifier_rl.evaluation_journal import persist, ReconciliationRequired
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import digest

app = modal.App("verifier-rl-permit-recovery")
work, archive, store = cloud.work, cloud.archive, cloud.store
image = (cloud.image.add_local_file(__file__, "/root/modal_permit_recovery.py")
         .add_local_file("modal_budget_handoff.py", "/root/modal_budget_handoff.py"))
RUN = cloud.study.RUN_ID
NAME, OLD_NAME = "research-004", "research-003"
OLD_APP = "ap-F0GxXiD785Qt8XJTTr6ON6"
OLD_HASH = "a43023f438c69fdfcf8fdd6a44e3ec49c4f871d99b3133cf1f0b7b517583094d"
SOURCES = ("modal_permit_recovery.py", "verifier_rl/permit_recovery.py")
INTERRUPTED = {f"eval-s20261013-reference-24-{i:02d}" for i in range(17, 21)}
BOOT = uuid.uuid4().hex


def prefix(config):
    return cloud.PREFIX + "/" + config["name"]


def validate(config):
    cloud.validate_config(config)
    old = config["previous_config"]
    if parallel.fingerprint(old) != OLD_HASH or old["name"] != OLD_NAME:
        raise ValueError("wrong stopped predecessor")
    for key in ("plan", "setup", "rates", "scientific_sources", "scheduler_sources", "budget_binding", "accounting_sources"):
        if config[key] != old[key]:
            raise ValueError("permit repair changed " + key)
    expected = (parallel.runtime(old["runtime"], "canary", old["runtime"]["deadline"])
                if config["name"] == "permit-canary-001" else old["runtime"])
    if (config["name"] not in (NAME, "permit-canary-001") or config["runtime"] != expected
            or config["permit_revision"] != repair.REVISION or set(config["permit_sources"]) != set(SOURCES)):
        raise ValueError("unreviewed permit configuration")
    for name, wanted in {**config["accounting_sources"], **config["permit_sources"]}.items():
        if digest((Path(__file__).parent / name).read_text()) != wanted:
            raise ValueError("permit source changed: " + name)


async def lifecycle(app_id):
    client = await _Client.from_env()
    value = await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id), timeout=20, retry=None)
    return {"app_id": app_id, "state": api_pb2.AppState.Name(value.lifecycle.app_state),
            "stopped_at": value.lifecycle.stopped_at}


def require_stopped(app_id):
    value = synchronizer.create_blocking(lifecycle)(app_id)
    if value["state"] != "APP_STATE_STOPPED":
        raise ReconciliationRequired("previous application is still active")
    return value


def save_both(config, relative, documents):
    persist(cloud.paths(config) / relative, documents)
    work.commit()
    persist(cloud.paths(config, "/evidence") / relative, documents)
    archive.commit()


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=300, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": work, "/evidence": archive})
def permit_budget(config, action, payload):
    """One serialized successor; all previous spending and holds are retained."""
    validate(config)
    if config["name"] != NAME:
        raise ValueError("only the research configuration owns the cumulative ledger")
    require_stopped(OLD_APP)
    old_owner = store.get(cloud.PREFIX + "/" + OLD_NAME + "/budget-owner")
    if old_owner["app_id"] != OLD_APP or old_owner["config_hash"] != OLD_HASH:
        raise ValueError("unexpected predecessor accounting owner")
    owner = {"app_id": app.app_id, "function_id": permit_budget.object_id,
             "config_hash": parallel.fingerprint(config)}
    if not isinstance(owner["app_id"], str) or not owner["app_id"].startswith("ap-"):
        raise ReconciliationRequired("missing hydrated application identity")
    key = prefix(config) + "/budget-owner"
    if not store.put(key, owner, skip_if_exists=True):
        previous = store.get(key)
        if previous != owner:
            if previous["config_hash"] != owner["config_hash"]:
                raise ReconciliationRequired("different successor budget configuration")
            require_stopped(previous["app_id"])
            store.put(key, owner)
    saved = store.get(RUN + "/budget")
    if saved["binding"] != config["budget_binding"] or saved["ledger"]["ceiling"] != "250":
        raise ValueError("original budget binding/ceiling changed")
    ready = store.get(prefix(config) + "/accounting-ready", None)
    if action == "initialize":
        if ready is not None:
            if ready["config_hash"] != owner["config_hash"]:
                raise ValueError("different accounting initialization")
            return ready
        work.reload()
        target = cloud.paths(config) / "accounting" / "inherited.json"
        inherited = handoff.read(target) if target.exists() else {
            "ledger": saved, "previous_owner": old_owner, "config_hash": owner["config_hash"]}
        if inherited != {"ledger": saved, "previous_owner": old_owner, "config_hash": owner["config_hash"]}:
            raise ReconciliationRequired("ledger changed during inheritance")
        save_both(config, "accounting", {"inherited": inherited})
        ready = {"config_hash": owner["config_hash"], "inherited_hash": parallel.fingerprint(inherited),
                 "released_holds_usd": "0"}
        store.put(prefix(config) + "/accounting-ready", ready, skip_if_exists=True)
        return ready
    if ready is None or ready["config_hash"] != owner["config_hash"]:
        raise ReconciliationRequired("initialize inherited accounting first")
    ticket = parallel.AMENDMENT + "/" + NAME + "/" + payload.get("ticket", "")
    if action == "reserve":
        saved["ledger"] = accounting.reserve(saved["ledger"], ticket, payload["amount"], payload["identity"])
    elif action == "settle":
        saved["ledger"] = accounting.settle(saved["ledger"], ticket, payload["amount"], payload["identity"])
    elif action != "status":
        raise ValueError("invalid budget action")
    if action != "status":
        store.put(RUN + "/budget", saved)
    return audit.describe(saved["ledger"])


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=60, retries=0, max_containers=1, scaledown_window=300)
def permit_coordinate(config, action, payload):
    validate(config)
    key = prefix(config) + "/state"
    state = store.get(key, None)
    if action == "initialize":
        value = parallel.initialize(config["runtime"], payload["jobs"])
        if state is not None or not store.put(key, value, skip_if_exists=True):
            raise ReconciliationRequired("coordinator already initialized")
        return True
    if state is None:
        raise ValueError("missing fixed job population")
    if action == "try_permit":
        # Wall time only enforces the already-approved study deadline.
        # Rate spacing and worker freshness use independent monotonic clocks.
        state, result = repair.permit_transition(state, payload, now=time.monotonic(), wall=time.time(), boot=BOOT)
    else:
        old_stop = state["stop"]
        state, result = parallel.transition(state, action, payload, time.time())
        if old_stop is not None:
            state["stop"] = old_stop  # Keep the original cause when peers drain.
            if isinstance(result, dict) and "stop" in result:
                result["stop"] = old_stop
    if action != "status":
        store.put(key, state)
    return result


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=parallel.MAX_SECONDS, retries=0, max_containers=4, scaledown_window=2,
              volumes={"/artifacts": work, "/evidence": archive})
def permit_grade(config, manifest, *, clock_offset=0.0, delay_first_grant=False):
    validate(config)
    if config["name"] != "permit-canary-001" and (clock_offset or delay_first_grant):
        raise ValueError("fault injection only allowed for authored controls")
    if manifest["runtime"] != config["runtime"]:
        raise ValueError("runtime differs from configuration")
    work.reload()
    began = time.monotonic()
    directory = cloud.paths(config) / "grading" / manifest["key"]
    owner = parallel.fingerprint({"config": config, "manifest": manifest})
    delayed = False
    clock = lambda: time.monotonic() + clock_offset
    async def rpc(action, payload):
        nonlocal delayed
        result = await permit_coordinate.remote.aio(config, action, payload)
        if action == "try_permit" and "grant" in result and delay_first_grant and not delayed:
            delayed = True
            await asyncio.sleep(parallel.PERMIT_TTL + 1)
        return result
    permits = repair.PermitRPC(rpc, clock=clock)
    async def backup(sid, intent, document):
        persist(cloud.paths(config, "/evidence") / "grading" / manifest["key"] / sid,
                {"intent": intent, "result": document})
        await archive.commit.aio()
    def backend(gate):
        return cloud.execution.ProgramBackend(config["setup"]["app_name"], config["runtime"]["image_id"], start_gate=gate)
    with ProgressLog(manifest["key"], label="PARALLEL_GRADING") as progress:
        progress.stage("grade_fixed_programs", total=4*len(manifest["case_hashes"]), unit="inputs")
        last = 0
        def progressed(stats):
            nonlocal last
            progress.advance(stats["inputs"] - last)
            last = stats["inputs"]
        result = asyncio.run(parallel.execute(manifest, cloud.study.pilot.cases_for("evaluation"), directory,
            backend, permits, work.commit.aio, backup, owner=owner, clock=clock, progress=progressed))
    result.update(elapsed_seconds=time.monotonic()-began, expired_unused_grants=permits.expired_grants)
    print("CLOCK SAFE BATCH COMPLETE", manifest["key"], result["stats"],
          "expired_unused_grants", permits.expired_grants, flush=True)
    return result


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": work, "/evidence": archive})
def permit_control(config):
    validate(config)
    test = deepcopy(config)
    test.update(name="permit-canary-001", runtime=parallel.runtime(config["runtime"], "canary", config["runtime"]["deadline"]))
    jobs = cloud.control_jobs(test)[1:5]
    permit_coordinate.remote(test, "initialize", {"jobs": jobs})
    with ProgressLog("permit-clock-controls", label="PERMIT_CONTROL") as progress:
        progress.stage("parallel_clock_offsets_and_delayed_grant", total=4, unit="batches")
        async def run():
            async def one(pair):
                i, job = pair
                result = await permit_grade.remote.aio(test, job, clock_offset=(-3600,3600,-120,120)[i], delay_first_grant=i==0)
                progress.advance()
                return result
            return await parallel.bounded_map(list(enumerate(jobs)), one, 4)
        results = asyncio.run(run())
        status = permit_coordinate.remote(test, "status", {})
        if status != {"batches":4, "total":4, "permits":16, "unknown":0, "stop":None}:
            raise ValueError("cloud permit controls had missing or duplicate execution")
        if (any(r["outcome_hashes"] != results[0]["outcome_hashes"] for r in results)
                or sum(r["stats"]["new_programs"] for r in results) != 16
                or results[0]["expired_unused_grants"] < 1):
            raise ValueError("clock/delay controls changed outcomes or did not exercise renewal")
        value = {"passed":True, "config_hash":parallel.fingerprint(config), "clock_offsets_seconds":[-3600,3600,-120,120],
                 "delay_seconds":6, "expired_unused_grants":sum(r["expired_unused_grants"] for r in results),
                 "candidate_program_executions":16, "outcomes_equal":True, "coordinator":status}
        save_both(config, "", {"permit-control":value})
        store.put(prefix(config)+"/permit-control", value, skip_if_exists=True)
        print("PERMIT CLOCK CLOUD CONTROLS PASSED", json.dumps(value), flush=True)
        return value


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": work, "/evidence": archive})
def permit_prepare(config):
    validate(config)
    stopped = require_stopped(OLD_APP)
    control = store.get(prefix(config)+"/permit-control")
    if not control["passed"] or control["config_hash"] != parallel.fingerprint(config):
        raise ValueError("passing cloud controls required")
    work.reload(); archive.reload()
    old = config["previous_config"]
    state = store.get(prefix(old)+"/state")
    snapshot = handoff.read(cloud.paths(old)/"inventory.json")
    ready = store.get(prefix(old)+"/ready")
    if ready["inventory_hash"] != parallel.fingerprint(snapshot) or ready["config_hash"] != OLD_HASH:
        raise ValueError("old inventory changed")
    manifests = {m["key"]:m for m in snapshot["missing"]}
    if (len(manifests) != len(snapshot["missing"]) or set(manifests)&set(snapshot["completed"])
            or set(manifests) != set(state["allowed"]) or len(manifests)+len(snapshot["completed"]) != 480
            or set(state["owners"])-set(state["batches"]) != INTERRUPTED):
        raise ReconciliationRequired("interrupted population differs from reviewed failure")
    completed, missing, recovered, never_started, all_unknown = {}, [], [], [], []
    with ProgressLog(NAME, label="PERMIT_RECOVERY") as progress:
        progress.stage("verify_saved_population_and_recover_receipts", total=480, unit="batches")
        for key, value in snapshot["completed"].items():
            if parallel.fingerprint(handoff.read(value["source_path"])) != value["source_hash"]:
                raise ValueError("inherited raw evidence changed")
            completed[key] = deepcopy(value)
            progress.advance()
        for key, manifest in manifests.items():
            if state["allowed"][key]["identity"] != parallel.fingerprint(manifest):
                raise ValueError("fixed manifest changed")
            target = cloud.paths(old)/"grading"/key
            if key in state["batches"]:
                raw, receipt = handoff.read(target/"raw.json"), handoff.read(target/"receipt.json")
                checked = handoff.check_new(manifest, raw, receipt)
                if state["batches"][key] != {k:receipt[k] for k in ("raw_hash","job_hash","outcome_hashes")}:
                    raise ValueError("published batch differs from durable evidence")
                completed[key] = {**checked,"source_path":str(target/"raw.json"),"source_hash":parallel.fingerprint(raw)}
            elif key in INTERRUPTED:
                if handoff.read(target/"intent.json") != manifest:
                    raise ValueError("interrupted batch intent changed")
                documents, published = {}, {}
                for sample in manifest["samples"]:
                    sid = sample["sample_id"]
                    doc = handoff.read(target/"programs"/sid/"result.json")
                    intent = {"job_hash":parallel.fingerprint(manifest),"sample_hash":parallel.fingerprint(sample),"sample_id":sid}
                    backup = cloud.paths(old,"/evidence")/"grading"/key/sid
                    if (handoff.read(target/"programs"/sid/"intent.json") != intent
                            or handoff.read(backup/"intent.json") != intent or handoff.read(backup/"result.json") != doc
                            or key+"/"+sid not in state["permits"]):
                        raise ValueError("interrupted work and archive evidence differ")
                    documents[sid], published[sid] = doc, state["programs"][key+"/"+sid]
                classified = repair.classify_interrupted(manifest, documents, published)
                if classified["status"] == "completed":
                    save_both(config, "recovered/"+key, {"raw":classified["raw"],"receipt":classified["receipt"]})
                    raw_path = cloud.paths(config)/"recovered"/key/"raw.json"
                    completed[key] = {k:classified[k] for k in ("rows","sandbox_ids")}
                    completed[key].update(source_path=str(raw_path),source_hash=classified["receipt"]["raw_hash"])
                    recovered.append(key)
                else:
                    all_unknown.extend(classified["unknown"])
                    never_started.append({"key":key,"evidence":classified})
                    missing.append(manifest)
            else:
                if (target.exists() or key in state["owners"]
                        or any(t.startswith(key+"/") for t in list(state["permits"])+list(state["programs"]))):
                    raise ReconciliationRequired("unreviewed partial work; do not replay")
                missing.append(manifest)
            progress.advance()
        actual_unknown = {u for v in state["programs"].values() for u in v["unknown"]}
        if (len(completed) != 330 or len(missing) != 150 or len(recovered) != 2 or len(never_started) != 2
                or actual_unknown != set(all_unknown) or len(actual_unknown) != 2296
                or set(state["unknown"]) != set(sorted(actual_unknown)[:64])
                or state["stop"]["reason"] not in ("unknown_circuit","cleanup_unconfirmed")):
            raise ReconciliationRequired("unexpected unknown population or recovery counts")
        receipt = {"stopped_predecessor":stopped,"old_state_hash":parallel.fingerprint(state),
                   "recovered_batches":recovered,"never_started_batches":never_started,
                   "candidate_reexecutions":0,"old_unknown_records_retained":2296}
        inventory = {"completed":completed,"missing":missing,"baseline":snapshot["baseline"],
                     "saved_arms":snapshot["saved_arms"],"config_hash":parallel.fingerprint(config)}
        save_both(config,"",{"inventory":inventory,"config":config,"recovery":receipt,"predecessor-state":state})
        result = {"completed":330,"remaining":150,"config_hash":parallel.fingerprint(config),
                  "inventory_hash":parallel.fingerprint(inventory),"recovery_hash":parallel.fingerprint(receipt)}
        store.put(prefix(config)+"/ready",result,skip_if_exists=True)
        print("PERMIT RECOVERY READY",json.dumps(result),flush=True)
        return result


@app.function(image=image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,
              timeout=60000,retries=0,max_containers=1,scaledown_window=2,
              volumes={"/artifacts":work,"/evidence":archive})
def permit_evaluate(config):
    validate(config)
    began = time.monotonic()
    ready = store.get(prefix(config)+"/ready")
    if not store.put(prefix(config)+"/controller",{"started":time.time(),"app_id":app.app_id},skip_if_exists=True):
        raise ReconciliationRequired("evaluation already submitted; inspect instead of restarting")
    work.reload()
    snapshot = handoff.read(cloud.paths(config)/"inventory.json")
    if (parallel.fingerprint(snapshot) != ready["inventory_hash"]
            or ready["config_hash"] != parallel.fingerprint(config)):
        raise ValueError("resume inventory changed")
    permit_coordinate.remote(config,"initialize",{"jobs":snapshot["missing"]})
    with ProgressLog(NAME,label="PARALLEL_STUDY") as progress:
        progress.stage("evaluate_fixed_remaining_samples",total=480,unit="batches",workers=4,program_limit=16)
        progress.advance(len(snapshot["completed"]))
        async def run():
            async def one(manifest):
                if time.time() >= config["runtime"]["deadline"]:
                    raise ReconciliationRequired("approved evaluation deadline reached")
                status = await permit_coordinate.remote.aio(config,"status",{})
                if status["stop"]:
                    raise ReconciliationRequired("coordinator stopped before new admission")
                key,identity = manifest["key"],parallel.fingerprint(manifest)
                broker = accounting.cost(config["rates"],"cpu",1800)
                maximum = broker + accounting.cost(config["rates"],"cpu",parallel.MAX_SECONDS+60)
                maximum += accounting.cost(config["rates"],"sandbox",cloud.previous.maximum_sandbox_seconds(manifest["samples"],"evaluation"))
                await permit_budget.remote.aio(config,"reserve",{"ticket":key,"amount":str(maximum),"identity":identity})
                result = await permit_grade.remote.aio(config,manifest)
                actual = broker + accounting.cost(config["rates"],"cpu",math.ceil(result["elapsed_seconds"])+60)
                actual += accounting.cost(config["rates"],"sandbox",result["sandbox_seconds"])
                report = await permit_budget.remote.aio(config,"settle",{"ticket":key,"amount":str(actual),"identity":identity})
                progress.advance()
                print("PARALLEL STUDY COMPLETED BATCH",key,"ACCOUNTING_NOT_BILLING",json.dumps(report),flush=True)
            await parallel.bounded_map(snapshot["missing"],one,4)
        asyncio.run(run())
        progress.stage("verify_and_compute_original_paired_metrics",total=480,unit="batches")
        work.reload()
        rows = {}
        for key,value in snapshot["completed"].items():
            if parallel.fingerprint(handoff.read(value["source_path"])) != value["source_hash"]:
                raise ValueError("inherited evidence changed")
            rows[key] = {k:value[k] for k in ("rows","sandbox_ids")}
            progress.advance()
        for manifest in snapshot["missing"]:
            target = cloud.paths(config)/"grading"/manifest["key"]
            rows[manifest["key"]] = handoff.check_new(manifest,handoff.read(target/"raw.json"),handoff.read(target/"receipt.json"))
            progress.advance()
        result = handoff.assemble(snapshot["baseline"],rows,snapshot["saved_arms"])
        result["provenance"] = {"permit_revision":repair.REVISION,"config_hash":parallel.fingerprint(config),
            "retained_batches":330,"new_batches":150,"recovery_hash":ready["recovery_hash"],"training_unchanged":True}
        result["accounting_not_provider_billing"] = permit_budget.remote(config,"status",{})
        save_both(config,"",{"result":result})
        store.put(prefix(config)+"/result",result,skip_if_exists=True)
        print("PARALLEL STUDY COMPLETE",result["status"],result["analysis"]["contrasts"],flush=True)
        return {"status":result["status"],"analysis":result["analysis"],"elapsed_seconds":time.monotonic()-began}


@app.local_entrypoint()
def main():
    directory = Path("runs")/cloud.RELATIVE/NAME
    old = handoff.read(Path("runs")/cloud.RELATIVE/OLD_NAME/"config.json")
    config = deepcopy(old)
    config.update(name=NAME,previous_config=old,permit_revision=repair.REVISION,
                  permit_sources={name:digest(Path(name).read_text()) for name in SOURCES})
    validate(config)
    if not store.put(prefix(config)+"/launch-claim",{"config_hash":parallel.fingerprint(config)},skip_if_exists=True):
        raise ReconciliationRequired("resume already launched; inspect the saved call")
    persist(directory,{"config":config,"application":{"app_id":app.app_id}})
    permit_budget.remote(config,"initialize",{})
    permit_budget.remote(config,"reserve",{"ticket":"controls-and-recovery","amount":"5","identity":parallel.fingerprint(config)})
    for name,function in (("control",permit_control),("prepare",permit_prepare)):
        call = function.spawn(config)
        persist(directory,{name+"-launch":{"call_id":call.object_id,"app_id":app.app_id}})
        print("PERMIT RECOVERY",name,call.object_id,flush=True)
        persist(directory,{name:call.get()})
    maximum = accounting.cost(config["rates"],"cpu",60060)
    report = permit_budget.remote(config,"reserve",{"ticket":"controller","amount":str(maximum),"identity":parallel.fingerprint(config)})
    print("ACCOUNTING NOT PROVIDER BILLING",json.dumps(report),flush=True)
    call = permit_evaluate.spawn(config)
    persist(directory,{"launch":{"call_id":call.object_id,"app_id":app.app_id}})
    print("CLOCK SAFE EVALUATION STARTED",call.object_id,flush=True)
    result = call.get()
    cost = accounting.cost(config["rates"],"cpu",math.ceil(result["elapsed_seconds"])+60)
    permit_budget.remote(config,"settle",{"ticket":"controller","amount":str(cost),"identity":parallel.fingerprint(config)})
    persist(directory,{"result":result})
