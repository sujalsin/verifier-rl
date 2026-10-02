"""Reviewed evidence-only recovery; 25 old inputs retained, 71 new inputs only.

No model generation/training, no replay of executed candidate inputs, and no
change to the scientific plan. The original stop/counters/receipts are archived.
"""

import asyncio
from decimal import Decimal
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import modal

from modal_program_spend_recovery import validate_checkpoint_group
from verifier_rl import booking_replication as study, compute_budget as accounting
from verifier_rl import program_execution as execution, program_grading as grading, program_storage as storage
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.grpo_recovery import read_json
from verifier_rl.sandbox_lifecycle import confirm_terminal, validate_terminal
from verifier_rl.suites import digest

ID=storage.AMENDMENT
PREFIX=grading.NAMESPACE+"/storage-amendments/"+ID
PARENT_APP="ap-WvJ39J8eYLBbDnfWtZjZxW"
JOBS=(("s20261013-endpoint_omission",20),("s20261014-endpoint_omission",11))
BATCH="train-s20261014-endpoint_omission-11"
FAILED_SAMPLE=BATCH+"-2"
FAILED_SANDBOX="sb-BvXlur3ZFuuefwIaz8ZgCF"
EXPECTED_STOP={"batch":BATCH,"reason":"unknown_circuit","sample_id":FAILED_SAMPLE}
SECONDS=3600
LAUNCHERS=("modal_booking_study.py","modal_booking_baseline_comparison.py",
           "modal_booking_matched_training.py","modal_booking_replication.py")
fingerprint=grading.fingerprint

app=modal.App("verifier-rl-program-storage-recovery")
work=modal.Volume.from_name("verifier-rl-booking-replication-work",create_if_missing=False)
archive=modal.Volume.from_name("verifier-rl-booking-replication-evidence",create_if_missing=False)
store=modal.Dict.from_name("verifier-rl-booking-replication-journal",create_if_missing=False)
image=modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5").add_local_python_source("verifier_rl")
for name in (*LAUNCHERS,"modal_program_spend_recovery.py"):
    image=image.add_local_file(name,"/root/"+name)


def root(base):
    return Path(base)/study.RUN_ID/grading.RELATIVE


async def require_idle_apps(apps,client=None,*,parent_app=PARENT_APP):
    from modal.client import _Client
    from modal_proto import api_pb2
    if any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("active workers prohibit exclusive storage maintenance")
    client=client or await _Client.from_env()
    response=await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=parent_app),timeout=30,retry=None)
    if response.lifecycle.app_state!=api_pb2.APP_STATE_STOPPED:
        raise ReconciliationRequired("failed controller is not positively stopped")
    return {"app_id":parent_app,"state":"stopped","method":"AppGetLifecycle",
            "stopped_at":response.lifecycle.stopped_at,"checked_at":time.time()}


def validate_history(history,samples,cases,image_id,runtime):
    if set(history)!={BATCH+f"-{i}" for i in range(4)} or {s["sample_id"] for s in samples}!=set(history):
        raise ValueError("only the four saved group-11 programs are in recovery scope")
    for sample in samples:
        sid=sample["sample_id"]
        entry=history[sid]
        if entry["intent"]!=grading.intent_for(sample,cases,runtime) or grading.metadata_for(entry["result"])!=entry["final"]:
            raise ValueError("original program and provider evidence disagree")
        document=entry["result"]
        outcomes=execution.validate_program(document,sample["source"],cases,image_id)
        if sid==FAILED_SAMPLE:
            missing=storage.partial_cases(document,sample["source"],cases,image_id)
            if len(missing)!=71 or len(cases)!=96 or document["metadata"]["sandbox_id"]!=FAILED_SANDBOX:
                raise ValueError("reviewed 25/71 split or sandbox changed")
        elif any(o["passed"] is None for o in outcomes.values()):
            raise ValueError("one of the three completed peers has missing execution evidence")
    return missing


class FaultStore:
    """Isolated real Dict namespace; inject only FINAL-index publication errors."""
    def __init__(self,target):
        self.target=Path(target)
        self.fail=True
        self.failures=0
        self.get=SimpleNamespace(aio=self.read)
        self.put=SimpleNamespace(aio=self.write)

    async def read(self,key,default=None):
        return await store.get.aio(PREFIX+"/fault-control/"+key,default)

    async def write(self,key,value,**kwargs):
        if "/input/" in key:
            raise AssertionError("per-test Dict write was not removed")
        if key.endswith("/final") and self.fail:
            if not (self.target/"programs/storage-control/result.json").exists():
                raise AssertionError("publishing before durable evidence exists")
            self.failures+=1
            raise OSError("injected result-index storage outage")
        return await store.put.aio(PREFIX+"/fault-control/"+key,value,**kwargs)


async def live_fault_control(context,target):
    """Exercise the exact service with an authored wrong program, never a model draw."""
    source="def required_capacity(bookings): return 0"
    sample=study.pilot.original.sample_from_text(source,sid="storage-control",seed=None,
        plan=context["plan"]["experiment"],tokens=0,eos=True)
    cases=study.pilot.cases_for("training")[:2]
    backend=execution.ProgramBackend(context["setup"]["app_name"],context["program_runtime"]["image_id"],
                                     start_gate=execution.CreationGate(.30))
    proxy=FaultStore(target)
    async def backup(sid,intent,document):
        persist(root("/evidence")/"storage-amendments"/ID/"fault-control"/sid,{"intent":intent,"result":document})
        await archive.commit.aio()
    try:
        await grading.execute_programs([sample],cases,target,"storage-control",context["plan"],
            context["program_runtime"],backend,proxy,work.commit.aio,backup_program=backup)
    except storage.StorageFailure:
        pass
    else:
        raise ValueError("fault injection did not fail closed after bounded retries")
    if proxy.failures!=storage.ATTEMPTS:
        raise ValueError("storage retry limit changed")
    before=read_json(target/"programs/storage-control/result.json")
    await work.reload.aio()  # Re-read committed evidence, not the in-memory object.
    if read_json(target/"programs/storage-control/result.json")!=before:
        raise ValueError("Volume reload changed committed execution results")
    class NoExecution:
        async def execute_program(self,*args,**kwargs):
            raise AssertionError("publication recovery reran candidate code")
    proxy.fail=False
    entries,progress,seconds=await grading.execute_programs([sample],cases,target,"storage-control",context["plan"],
        context["program_runtime"],NoExecution(),proxy,work.commit.aio,backup_program=backup)
    if entries[sample["sample_id"]]["result"]!=before or progress["new_program_starts"]!=0 or seconds!="0":
        raise ValueError("storage recovery failed to preserve exact results")
    outcomes=execution.validate_program(before,source,cases,context["program_runtime"]["image_id"])
    if any(o["passed"] is None for o in outcomes.values()) or not any(o["passed"] is False for o in outcomes.values()):
        raise ValueError("control must preserve known wrong answers too")
    receipt={"passed":True,"publication_failures_injected":proxy.failures,"candidate_executions":len(cases),
             "candidate_reexecutions":0,"new_sandboxes_on_recovery":0,"document_hash":fingerprint(before),
             "sandbox_seconds":min(execution.lifetime_for(len(cases)),math.ceil(before["metadata"]["total_seconds"]))}
    persist(target,{"control":receipt})
    await work.commit.aio()
    print("STORAGE FAULT CONTROL PASSED",receipt,flush=True)
    return receipt


@app.function(image=image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,timeout=SECONDS,retries=0,
              max_containers=1,scaledown_window=2,volumes={"/artifacts":work,"/evidence":archive})
def recover(previous,context,identity):
    started=time.monotonic()
    if not store.put(PREFIX+"/owner",{"identity":identity,"started":time.time()},skip_if_exists=True):
        raise ReconciliationRequired("maintenance already entered; never blindly replay its execution")
    sources={name:(Path("/root")/name).read_text() for name in context["source_snapshot"]}
    if context!=storage.context_for(previous,sources) or time.time()>=context["deadline"]:
        raise ValueError("reviewed source context or original deadline changed")
    work.reload()
    base=root("/artifacts")
    if read_json(base/"context.json")!=previous:
        raise ValueError("original immutable context changed")
    target=base/"storage-amendments"/ID/"recovery"
    durable=root("/evidence")/"storage-amendments"/ID/"recovery"

    async def perform():
        if await storage.get(store,grading.NAMESPACE+"/stop")!=EXPECTED_STOP:
            raise ValueError("unexpected runtime stop")
        cases=study.pilot.cases_for("training")
        completed,jobs,checkpoints,history={},{},{},{}
        for seed in study.SEEDS:
            for arm in study.ARMS:
                policy=study.label(seed,arm)
                value=await storage.get(store,grading.NAMESPACE+"/finished/training/"+policy)
                if policy not in {p for p,_ in JOBS}:
                    if value is None: raise ValueError("completed policy missing")
                    study.validate_arm(value["payload"],context["plan"])
                    completed[policy]=fingerprint(value)
                elif value is not None:
                    raise ValueError("interrupted policy already completed; do not rewind")
        for policy,step in JOBS:
            print("REVIEWING FULL CHECKPOINT",policy,step,flush=True)
            key,samples,checkpoint=validate_checkpoint_group(base,policy,step,context)
            jobs[key],checkpoints[policy]=samples,checkpoint
            if (base/"grading"/key/"result.json").exists() or await storage.get(store,grading.NAMESPACE+"/finished/grading/"+key) is not None:
                raise ValueError("pending grading was already completed")
            for sample in samples:
                sid=sample["sample_id"]
                job=grading.NAMESPACE+"/grading/"+key+"/"+sid
                intent=await storage.get(store,job+"/intent")
                final=await storage.get(store,job+"/final")
                if key!=BATCH:
                    if intent is not None or final is not None or (base/"grading"/key/"programs"/sid).exists():
                        raise ValueError("seed13 next group was expected to be completely unsubmitted")
                else:
                    history[sid]={"intent":intent,"final":final,
                                  "result":read_json(base/"grading"/key/"programs"/sid/"result.json")}
        missing=validate_history(history,jobs[BATCH],cases,context["program_runtime"]["image_id"],context["program_runtime"])
        sample=next(s for s in jobs[BATCH] if s["sample_id"]==FAILED_SAMPLE)
        sandbox=await modal.Sandbox.from_id.aio(FAILED_SANDBOX)
        terminal=await confirm_terminal(sandbox)
        validate_terminal(terminal,FAILED_SANDBOX)
        old_slots={str(i):await storage.get(store,grading.NAMESPACE+f"/unknown/slot/{i}") for i in range(context["plan"]["unknown_circuit"])}
        allowed={grading.NAMESPACE+"/unknown/input/"+FAILED_SAMPLE+"/"+c.input_hash for c in missing}
        if len(set(old_slots.values()))!=64 or not set(old_slots.values())<=allowed:
            raise ValueError("circuit contains unreviewed unknown evidence")
        old_unknowns={key:await storage.get(store,key) for key in old_slots.values()}
        if any(value!={"reason":"program_not_executed"} for value in old_unknowns.values()):
            raise ValueError("unknown counter reasons changed")
        if await storage.get(store,grading.NAMESPACE+"/unknown/"+ID) is not None:
            raise ValueError("storage counter already migrated; inspect rather than reapply")
        original={"amendment":ID,"identity":identity,"context_hash":fingerprint(previous),"history":history,
                  "stop":EXPECTED_STOP,"old_slots":old_slots,"old_unknowns":old_unknowns,"terminal":terminal,
                  "checkpoints":checkpoints,"completed_policy_hashes":completed,"jobs":jobs,
                  "known_inputs_retained":25,"never_executed_inputs":71,"three_peer_programs_retained":True}
        persist(target,{"original":original})
        persist(durable,{"original":original})
        await storage.retry(work.commit.aio,label="archive original work evidence")
        await storage.retry(archive.commit.aio,label="archive original independent evidence")
        await storage.put_once(store,PREFIX+"/original",{"hash":fingerprint(original),"relative_path":str(target.relative_to('/artifacts'))})
        print("STORAGE RECOVERY REVIEW PASSED; 10 policies untouched; checkpoints 20/11; 25 inputs retained; 71 unsubmitted",flush=True)
        control=await live_fault_control(context,target/"fault-control")

        intent={"sample_hash":fingerprint(sample),"original_hash":fingerprint(history[FAILED_SAMPLE]["result"]),
                "input_hashes":[c.input_hash for c in missing],"candidate_reexecutions":0,"amendment":ID}
        persist(target/"continuation",{"intent":intent})
        await storage.retry(work.commit.aio,label="continuation intent commit")
        if time.time()>=context["deadline"]:
            raise ReconciliationRequired("original deadline reached before new input execution")
        if not await storage.put_once(store,PREFIX+"/continuation/intent",intent):
            raise ReconciliationRequired("continuation already claimed; no replay")
        backend=execution.ProgramBackend(context["setup"]["app_name"],context["program_runtime"]["image_id"],start_gate=execution.CreationGate(.30))
        count=0
        async def record(h,value):
            nonlocal count
            await storage.save(target/"continuation/inputs"/(h+".json"),value)
            count+=1
            if count%storage.CHUNK_SIZE==0:
                await storage.retry(work.commit.aio,label="continuation evidence commit")
                print("STORAGE RECOVERY INPUTS",count,"/71; original25 not replayed",flush=True)
        suffix=await backend.execute_program(sample["source"],missing,on_record=record)
        await storage.save(target/"continuation/result.json",suffix)
        await storage.retry(work.commit.aio,label="continuation final commit")
        persist(durable/"continuation",{"intent":intent,"result":suffix})
        await storage.retry(archive.commit.aio,label="continuation backup commit")
        if suffix["metadata"].get("failure",{}).get("stage")=="evidence_storage" or suffix["metadata"].get("storage_failure"):
            raise storage.StorageFailure("continuation storage failed; keep stop and new evidence")
        merged=storage.continue_document(history[FAILED_SAMPLE]["result"],suffix,sample["source"],cases,context["program_runtime"]["image_id"])
        outcomes=execution.validate_program(merged,sample["source"],cases,context["program_runtime"]["image_id"])
        persist(target,{"merged":merged})
        persist(durable,{"merged":merged})
        await storage.retry(work.commit.aio,label="merged evidence commit")
        await storage.retry(archive.commit.aio,label="merged evidence backup")
        for policy,expected in completed.items():
            if fingerprint(await storage.get(store,grading.NAMESPACE+"/finished/training/"+policy))!=expected:
                raise ValueError("completed policy changed during maintenance")
        if await storage.get(store,grading.NAMESPACE+"/stop")!=EXPECTED_STOP:
            raise ValueError("stop changed during maintenance")
        canonical=base/"grading"/BATCH/"programs"/FAILED_SAMPLE
        old_location=target/"original_files"/FAILED_SAMPLE
        if old_location.exists():
            raise ReconciliationRequired("publication already attempted; do not repeat execution")
        old_location.parent.mkdir(parents=True,exist_ok=True)
        canonical.rename(old_location)  # Recoverable archive, never deletion.
        await storage.save(canonical/"intent.json",history[FAILED_SAMPLE]["intent"])
        await storage.save(canonical/"result.json",merged)
        for h,value in merged["records"].items():
            await storage.save(canonical/"inputs"/(h+".json"),value)
        await storage.retry(work.commit.aio,label="canonical program commit")
        archive_target=root("/evidence")/"grading"/BATCH/"programs"/FAILED_SAMPLE
        persist(archive_target,{"intent":history[FAILED_SAMPLE]["intent"],"result":merged})
        await storage.retry(archive.commit.aio,label="canonical independent backup")
        final_key=grading.NAMESPACE+"/grading/"+BATCH+"/"+FAILED_SAMPLE+"/final"
        async def publish():
            old=await store.get.aio(final_key)
            if old not in (history[FAILED_SAMPLE]["final"],grading.metadata_for(merged)):
                raise ValueError("canonical final index changed during recovery")
            await store.put.aio(final_key,grading.metadata_for(merged))
        await storage.retry(publish,label="reviewed final index replacement")
        if await storage.get(store,final_key)!=grading.metadata_for(read_json(canonical/"result.json")):
            raise ValueError("published index does not match durable execution evidence")
        receipt={"amendment":ID,"context_hash":fingerprint(context),"original_hash":fingerprint(original),
                 "merged_hash":fingerprint(merged),"storage_fault_control":control,"known_inputs_retained":25,
                 "new_inputs_executed":71,"candidate_inputs_reexecuted":0,"complete_peer_programs_reexecuted":0,
                 "unknown_inputs":sum(o["passed"] is None for o in outcomes.values()),
                 "resume_checkpoints":dict(JOBS),"completed_policy_hashes":completed,"old_circuit_retained":True,
                 "sandbox_seconds":str(control["sandbox_seconds"]+min(execution.lifetime_for(71),math.ceil(suffix["metadata"]["total_seconds"]))),
                 "new_budget":False,"deadline_unchanged":True,"grpo_and_rewards_unchanged":True}
        persist(target,{"receipt":receipt})
        persist(durable,{"receipt":receipt})
        persist(base/"storage-amendments"/ID,{"context":context})
        persist(root("/evidence")/"storage-amendments"/ID,{"context":context})
        await storage.retry(work.commit.aio,label="recovery receipt commit")
        await storage.retry(archive.commit.aio,label="recovery receipt backup")
        # Keep every legacy counter and individual unknown record unchanged.
        # All 64 indexed unknowns belong to the now-complete reviewed suffix.
        await storage.put_once(store,grading.NAMESPACE+"/unknown/"+ID,{"identities":[]})
        await storage.put_once(store,PREFIX+"/ready",receipt)
        async def release_stop():
            current=await store.get.aio(grading.NAMESPACE+"/stop",None)
            if current is None: return  # A lost ACK is safe after the ready receipt.
            if current!=EXPECTED_STOP: raise ValueError("stop changed before release")
            await store.pop.aio(grading.NAMESPACE+"/stop")
        await storage.retry(release_stop,label="reviewed stop release")
        print("STORAGE RECOVERY READY",receipt,flush=True)
        return receipt
    receipt=asyncio.run(perform())
    return {"receipt":receipt,"seconds":time.monotonic()-started}


@app.function(image=image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,timeout=600,retries=0,
              max_containers=1,scaledown_window=2,volumes={"/artifacts":work,"/evidence":archive})
def review_controls(previous,context):
    """Metadata-only CPU preflight of the ACTUAL GPU entry guard, no candidates."""
    import ast
    import modal_booking_replication as launch
    started=time.monotonic()
    prefix=grading.NAMESPACE+"/"+storage.metadata_relative(context)
    if not store.put(prefix+"/owner",{"context_hash":fingerprint(context)},skip_if_exists=True):
        raise ReconciliationRequired("guard review already entered")
    work.reload()
    base=root("/artifacts")
    original=read_json(base/"context.json")
    sources={name:(Path("/root")/name).read_text() for name in context["source_snapshot"]}
    if (storage.context_for(original,sources)!=context or time.time()>=context["deadline"]
            or read_json(base/"storage-amendments"/ID/"context.json")!=previous):
        raise ValueError("guard review context changed")
    for name in ("verifier_rl/program_grading.py","verifier_rl/program_execution.py"):
        if previous["source_snapshot"][name]!=sources[name]:
            raise ValueError("live-tested storage/execution implementation changed")
    def functions(source):
        return {n.name:ast.dump(n) for n in ast.parse(source).body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    before,after=(functions(s["verifier_rl/program_storage.py"]) for s in (previous["source_snapshot"],sources))
    for name in set(before)-{"validate_sources","context_for","parent_context"}:
        if before[name]!=after.get(name):
            raise ValueError("live-tested storage helper changed: "+name)
    prior=store.get(PREFIX+"/ready")
    if (prior["context_hash"]!=fingerprint(previous) or prior["unknown_inputs"]!=0
            or prior["candidate_inputs_reexecuted"]!=0 or not prior["storage_fault_control"]["passed"]
            or store.get(grading.NAMESPACE+"/stop",None) is not None):
        raise ValueError("successful storage recovery prerequisite changed")
    checkpoints={}
    for policy,step in JOBS:
        print("CONTROL-GUARD REVIEW CHECKPOINT",policy,step,flush=True)
        _,_,checkpoints[policy]=validate_checkpoint_group(base,policy,step,context)
    # Re-run the exact entry guard that blocked attempt-2, on the actual saved
    # controls and new source snapshot. It does not allocate a GPU or a sandbox.
    launch.require_controls(context)
    for policy,expected in prior["completed_policy_hashes"].items():
        if fingerprint(store.get(grading.NAMESPACE+"/finished/training/"+policy))!=expected:
            raise ValueError("completed policy changed")
    receipt={"context_hash":fingerprint(context),"original_recovery_hash":fingerprint(prior),
             "actual_training_entry_guard_passed":True,"checkpoints":checkpoints,
             "candidate_executions":0,"research_updates":0,"budget_ceiling_usd":"250",
             "old_failed_call_holds_retained":True,"resume_revision":storage.RESUME_REVISION}
    relative=storage.metadata_relative(context)
    persist(base/relative,{"context":context,"guard_review":receipt})
    work.commit()
    persist(root("/evidence")/relative,{"context":context,"guard_review":receipt})
    archive.commit()
    if not store.put(prefix+"/ready",receipt,skip_if_exists=True):
        raise ReconciliationRequired("guard ready receipt already published")
    print("ACTUAL TRAINING ENTRY GUARD PASSED; no candidate execution; no training; checkpoints 20/11 intact",flush=True)
    return {"receipt":receipt,"seconds":time.monotonic()-started}


def launch_guard_review():
    original=read_json(root("runs")/"context.json")
    previous=read_json(root("runs")/"storage-amendments"/ID/"context.json")
    paths=sorted(Path("verifier_rl").glob("*.py"))+[Path(n) for n in LAUNCHERS]
    context=storage.context_for(original,{p.as_posix():p.read_text() for p in paths})
    apps=json.loads(subprocess.check_output([sys.executable,"-m","modal","app","list","--json"],text=True))
    from modal._utils.async_utils import synchronizer
    terminal=synchronizer.create_blocking(require_idle_apps)(apps,parent_app="ap-W9Jg2ysAU04axoHEIJoIJe")
    owner=modal.App.lookup(context["setup"]["app_name"],create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("active sandboxes prevent guard review")
    if time.time()>=context["deadline"] or store.get(grading.NAMESPACE+"/stop",None) is not None:
        raise ReconciliationRequired("deadline or new stop prevents correction")
    prefix=grading.NAMESPACE+"/"+storage.metadata_relative(context)
    identity=fingerprint({"context_hash":fingerprint(context),"launcher_hash":digest(Path(__file__).read_text()),"seconds":600})
    if not store.put(prefix+"/authorization",{"identity":identity,"parent_terminal":terminal},skip_if_exists=True):
        raise ReconciliationRequired("guard review already authorized")
    key=study.RUN_ID+"/budget"
    ledger=store.get(key)
    if ledger["binding"]!=context["runtime_amendment"]["original_budget_binding"] or ledger["ledger"]["ceiling"]!="250":
        raise ValueError("original budget changed")
    maximum=accounting.cost(context["rates"],"cpu",660)
    ticket=grading.AMENDMENT+"/storage-recovery/"+ID+"/"+storage.RESUME_REVISION
    reserved=dict(ledger,ledger=accounting.reserve(ledger["ledger"],ticket,maximum,identity))
    store.put(key,reserved)
    directory=root("runs")/storage.metadata_relative(context)
    persist(directory,{"context":context,"authorization":{"identity":identity,"maximum_usd":str(maximum),"parent_terminal":terminal}})
    call=review_controls.spawn(previous,context)
    persist(directory,{"launch":{"call_id":call.object_id}})
    print("CONTROL-GUARD REVIEW CALL",call.object_id,"maximum USD",maximum,flush=True)
    result=call.get()
    current=store.get(key)
    if current!=reserved: raise ValueError("budget changed during exclusive guard review")
    actual=accounting.cost(context["rates"],"cpu",math.ceil(result["seconds"])+60)
    settled=dict(current,ledger=accounting.settle(current["ledger"],ticket,actual,identity))
    store.put(key,settled)
    persist(directory,{"result":result,"budget_after":settled})
    print("CONTROL-GUARD REVIEW COMPLETE",actual,flush=True)


@app.local_entrypoint()
def main(apply:bool=False,review_resume_guard:bool=False):
    if review_resume_guard:
        if apply: raise ValueError("guard review must not reapply execution recovery")
        return launch_guard_review()
    if not apply: raise ValueError("--apply required for this explicitly reviewed recovery")
    previous=read_json(root("runs")/"context.json")
    names=sorted(Path("verifier_rl").glob("*.py"))+[Path(n) for n in LAUNCHERS]
    context=storage.context_for(previous,{p.as_posix():p.read_text() for p in names})
    if time.time()>=context["deadline"]: raise ReconciliationRequired("original study deadline reached")
    apps=json.loads(subprocess.check_output([sys.executable,"-m","modal","app","list","--json"],text=True))
    from modal._utils.async_utils import synchronizer
    terminal=synchronizer.create_blocking(require_idle_apps)(apps)
    owner=modal.App.lookup(context["setup"]["app_name"],create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("active candidate sandboxes prevent maintenance")
    if store.get(grading.NAMESPACE+"/authorization")["context_hash"]!=fingerprint(previous):
        raise ValueError("original runtime authorization changed")
    if store.get(grading.NAMESPACE+"/stop",None)!=EXPECTED_STOP:
        raise ValueError("unexpected runtime stop")
    rates=json.loads(subprocess.check_output([sys.executable,"-m","modal","billing","rates","--json"],text=True))
    if any(accounting.hourly(rates,k)!=accounting.hourly(context["rates"],k) for k in ("cpu","gpu","sandbox")):
        raise ValueError("resource rates changed")
    identity=fingerprint({"context_hash":fingerprint(context),"launcher_hash":digest(Path(__file__).read_text()),"seconds":SECONDS})
    if not store.put(PREFIX+"/authorization",{"identity":identity,"parent_terminal":terminal,"context_hash":fingerprint(context)},skip_if_exists=True):
        raise ReconciliationRequired("recovery already authorized; inspect saved receipts")
    budget_key=study.RUN_ID+"/budget"
    ledger=store.get(budget_key)
    if ledger["binding"]!=context["runtime_amendment"]["original_budget_binding"] or ledger["ledger"]["ceiling"]!="250":
        raise ValueError("original budget binding/ceiling changed")
    maximum=accounting.cost(rates,"cpu",SECONDS+60)+accounting.cost(rates,"sandbox",execution.lifetime_for(71)+execution.lifetime_for(2))
    ticket=grading.AMENDMENT+"/storage-recovery/"+ID
    reserved=dict(ledger,ledger=accounting.reserve(ledger["ledger"],ticket,maximum,identity))
    store.put(budget_key,reserved)
    directory=root("runs")/"storage-amendments"/ID
    persist(directory,{"context":context,"authorization":{"identity":identity,"maximum_usd":str(maximum),
                          "old_budget_hash":fingerprint(ledger),"parent_terminal":terminal}})
    call=recover.spawn(previous,context,identity)
    persist(directory,{"launch":{"call_id":call.object_id}})
    print("STORAGE RECOVERY CALL",call.object_id,"maximum USD",maximum,flush=True)
    result=call.get()
    current=store.get(budget_key)
    if current!=reserved: raise ValueError("budget changed during exclusive maintenance; retain hold")
    actual=accounting.cost(rates,"cpu",math.ceil(result["seconds"])+60)+accounting.cost(rates,"sandbox",Decimal(result["receipt"]["sandbox_seconds"]))
    settled=dict(current,ledger=accounting.settle(current["ledger"],ticket,actual,identity))
    store.put(budget_key,settled)
    persist(directory,{"result":result,"budget_after":settled})
    print("STORAGE RECOVERY COMPLETE",result["receipt"]["resume_checkpoints"],"estimated USD",actual,flush=True)
