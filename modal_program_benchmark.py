"""Explicit CPU-only program-sandbox benchmark, charged to the existing $250 ledger.

No research launch, GRPO, generation, historical retries or stop-flag clearing.
One Function owns every sandbox start, so one shared pacing gate is sufficient.
"""

import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import modal

from verifier_rl import booking_replication as study, compute_budget as accounting
from verifier_rl import program_benchmark as bench, program_execution as execution
from verifier_rl import supervised_execution as legacy
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.panel_execution import pack_result, request_for
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import BOOKING

app = modal.App("verifier-rl-program-sandbox-benchmark")
volume = modal.Volume.from_name("verifier-rl-program-benchmark",create_if_missing=True)
claims = modal.Dict.from_name("verifier-rl-booking-replication-journal",create_if_missing=False)
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
             .add_local_python_source("verifier_rl")
             .add_local_file("modal_program_benchmark.py","/root/modal_program_benchmark.py"))


def fingerprint(value):
    return digest(canonical_json(value))


def read_modal(*args):
    return json.loads(subprocess.check_output([sys.executable,"-m","modal",*args,"--json"],text=True))


def budget_binding(context):
    return fingerprint({k:context[k] for k in ("plan","rates","billing_before","deadline")})


def reserve_benchmark(store,context,manifest,snapshot):
    """One explicit local writer after checking no apps are active. Never reset."""
    study.validate_plan(context["plan"])
    state_key=study.RUN_ID+"/budget"
    saved=store.get(state_key,None)
    if saved is None or saved["binding"]!=budget_binding(context):
        raise ValueError("original budget missing or changed")
    identity=fingerprint({"manifest":manifest,"snapshot":snapshot,"original_budget":saved["binding"]})
    lock_key=study.RUN_ID+"/maintenance-owner"
    if not store.put(lock_key,{"benchmark":bench.RUN_ID,"identity":identity},skip_if_exists=True):
        raise ReconciliationRequired("maintenance already owned; do not duplicate benchmark")
    maximum=accounting.cost(context["rates"],"cpu",bench.FUNCTION_SECONDS+120)
    maximum+=accounting.cost(context["rates"],"sandbox",bench.maximum_sandbox_seconds())
    if maximum>accounting.number(bench.RESERVATION_USD):
        raise ValueError("benchmark resource bound exceeds its reservation")
    saved["ledger"]=accounting.reserve(saved["ledger"],bench.RUN_ID,bench.RESERVATION_USD,identity)
    store.put(state_key,saved)
    return {"identity":identity,"maximum_resource_cost_usd":str(maximum),
            "committed_usd":str(accounting.committed(saved["ledger"])),"reservation_usd":bench.RESERVATION_USD}


async def exercise(context,directory):
    manifest,setup=context["manifest"],context["setup"]
    gate=execution.CreationGate(.30)
    backend=execution.ProgramBackend(setup["app_name"],setup["program_image_id"],start_gate=gate)
    total_sandbox_seconds=0.0
    ids=set()
    commit_lock=asyncio.Lock()
    async def commit():
        async with commit_lock:
            await volume.commit.aio()
    async def run_program(key,source,cases):
        nonlocal total_sandbox_seconds
        prefix=bench.RUN_ID+"/program/"+key
        intent={"source_hash":digest(source),"inputs":[c.input_hash for c in cases],
                "manifest_hash":fingerprint(manifest)}
        if not await claims.put.aio(prefix+"/intent",intent,skip_if_exists=True):
            raise ReconciliationRequired("benchmark program already submitted: "+key)
        target=directory/"programs"/key
        persist(target,{"intent":intent,"source":{"source":source}})
        await commit()
        async def record(input_hash,result):
            # Each input is persisted immediately. The parent program is not
            # deemed reusable/gradable until final cleanup is confirmed.
            persist(target/"inputs",{input_hash:result})
        with ProgressLog(key,label="PROGRAM_BENCHMARK") as progress:
            progress.stage("execute_inputs",total=len(cases),unit="tests")
            async def recorded(input_hash,result):
                await record(input_hash,result)
                progress.advance()
            document=await backend.execute_program(source,cases,on_record=recorded)
            persist(target,{"result":document})
            await commit()
            progress.stage("verify_and_score")
            outcomes=execution.validate_program(document,source,cases,setup["program_image_id"])
            persist(target,{"outcomes":outcomes})
            await commit()
        m=document["metadata"]
        total_sandbox_seconds+=min(execution.lifetime_for(len(cases)),math.ceil(m["total_seconds"]))
        if m["cleanup"]!="terminated":
            raise ReconciliationRequired("program cleanup unconfirmed")
        if m["sandbox_id"] in ids:
            raise ValueError("programs reused a sandbox")
        ids.add(m["sandbox_id"])
        print("PROGRAM COMPLETE",key,"inputs",len(cases),"seconds",round(m["total_seconds"],2),flush=True)
        return {"outcomes":outcomes,"metadata":m}

    controls={}
    for name,(source,_,_) in bench.controls().items():
        controls[name]=await run_program("control-"+name,source,bench.control_cases())
        bench.validate_control(name,controls[name]["outcomes"])
        print("PROGRAM CONTROL PASSED",name,flush=True)
    persist(directory,{"controls_passed":{"count":len(controls)}})
    await volume.commit.aio()

    pool=bench.programs(context["saved_pool"])
    for program,expected in zip(pool[:3],([96,57,57],[64,57,49],[1,1,1])):
        result=await run_program("grader-"+program["id"],program["source"],study.pilot.cases_for("training"))
        observed=bench.grader_counts(result["outcomes"])
        if observed!=expected or any(v["passed"] is None for v in result["outcomes"].values()):
            raise ValueError("reference/weak/repair grader control failed")
        print("PROGRAM GRADER PASSED",program["id"],observed,flush=True)

    comparison=bench.comparison_cases()
    old=legacy.SupervisedPanelBackend(setup["app_name"],setup["sandbox_image_id"],BOOKING,
                                     creation_interval_seconds=.26)
    # Old and new phases do not overlap. Old starts have ONE owning backend.
    parity={}
    for program in pool:
        started=time.monotonic()
        old_records,old_outputs={},{}
        for case in comparison:
            key=bench.RUN_ID+"/legacy/"+program["id"]+"/"+case.input_hash
            if not await claims.put.aio(key,{"source_hash":digest(program["source"])},skip_if_exists=True):
                raise ReconciliationRequired("legacy comparison already submitted")
            result=await old.execute(request_for(program["source"],case))
            old_records[case.input_hash]=pack_result(result)
            persist(directory/"legacy"/program["id"],{case.input_hash:old_records[case.input_hash]})
            await volume.commit.aio()
            legacy.require_evidence(result,BOOKING,setup["sandbox_image_id"],source=program["source"],case=case)
            old_outputs[case.input_hash]={"status":result.status.value,"stdout_base64":old_records[case.input_hash]["stdout_base64"]}
            total_sandbox_seconds+=min(120,math.ceil(result.metadata["total_seconds"]))
            if result.metadata["sandbox_id"] in ids:
                raise ValueError("legacy sandbox reused")
            ids.add(result.metadata["sandbox_id"])
        old_seconds=time.monotonic()-started
        started=time.monotonic()
        new=await run_program("parity-"+program["id"],program["source"],comparison)
        new_document=json.loads((directory/"programs"/("parity-"+program["id"])/"result.json").read_text())
        new_outputs={h:{k:r[k] for k in ("status","stdout_base64")} for h,r in new_document["records"].items()}
        if new_outputs!=old_outputs:
            raise ValueError("old/new observable behavior differs: "+program["id"])
        parity[program["id"]]={"matched_inputs":len(comparison),"legacy_seconds":old_seconds,
            "program_seconds":time.monotonic()-started,"legacy_sandbox_starts":len(comparison),"program_sandbox_starts":1}
        print("PROGRAM PARITY PASSED",program["id"],parity[program["id"]],flush=True)
    persist(directory,{"parity":parity})
    await volume.commit.aio()

    timings,results={},{}
    cases=study.pilot.cases_for("evaluation")
    for mode in ("sequential","parallel"):
        async def worker(program):
            value=await run_program(mode+"-"+program["id"],program["source"],cases)
            if any(o["passed"] is None for o in value["outcomes"].values()):
                raise ReconciliationRequired("unknown result in throughput benchmark")
            return program["id"],value["outcomes"]
        started=time.monotonic()
        # Wait for in-flight programs' cleanup even if one fails; never cancel
        # an untrusted execution just to make the benchmark appear faster.
        rows=await bench.bounded_map(pool,worker,1 if mode=="sequential" else 4)
        timings[mode]=bench.summarize_timing(time.monotonic()-started,len(pool),len(cases))
        results[mode]=dict(rows)
        persist(directory,{mode:{"timing":timings[mode],"outcomes":results[mode]}})
        await volume.commit.aio()
        print("PROGRAM THROUGHPUT",mode,timings[mode],flush=True)
    if results["sequential"]!=results["parallel"]:
        raise ValueError("sequential/parallel outcomes differ")
    return {"status":"passed","controls":len(controls),"parity":parity,"timings":timings,
        "unique_sandboxes":len(ids),"new_program_sandbox_starts":gate.starts,
        "sandbox_seconds":total_sandbox_seconds,"manifest":manifest,
        "full_study_started":False,"research_updates":0,"model_samples":0}


@app.function(image=cpu_image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,
              timeout=bench.FUNCTION_SECONDS,retries=0,max_containers=1,scaledown_window=2,
              volumes={"/artifacts":volume})
def run_benchmark(context):
    started=time.monotonic()
    if context["manifest"]!=bench.manifest(context["saved_pool"]):
        raise ValueError("benchmark manifest changed")
    for name,source in context["snapshot"].items():
        if (Path("/root")/name).read_text()!=source:
            raise ValueError("benchmark source changed: "+name)
    state=claims.get(study.RUN_ID+"/budget")
    item=state["ledger"]["items"][bench.RUN_ID]
    if item["identity"]!=context["reservation"]["identity"] or item["actual"] is not None:
        raise ValueError("missing fresh benchmark reservation")
    if not claims.put(bench.RUN_ID+"/owner",{"utc":datetime.now(timezone.utc).isoformat()},skip_if_exists=True):
        raise ReconciliationRequired("benchmark restarted; preserve evidence, no automatic repeat")
    directory=Path("/artifacts")/bench.RUN_ID
    persist(directory,{"context":context})
    volume.commit()
    try:
        result=asyncio.run(exercise(context,directory))
    except Exception as exc:
        persist(directory,{"failure":{"type":type(exc).__name__,"detail":str(exc)[:2000]}})
        volume.commit()
        raise
    result["function_seconds"]=time.monotonic()-started
    persist(directory,{"result":result})
    volume.commit()
    print("PROGRAM BENCHMARK PASSED",result["timings"],flush=True)
    return result


@app.local_entrypoint()
def main(allow_cloud:bool=False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required; CPU benchmark only")
    directory=Path("runs")/bench.RUN_ID
    if (directory/"launch_intent.json").exists():
        raise ReconciliationRequired("benchmark already submitted; do not rerun")
    apps=read_modal("app","list")
    if any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("other jobs active; cannot own the study budget exclusively")
    old=json.loads((Path("runs")/study.RUN_ID/"context.json").read_text())
    if time.time()>=old["deadline"]:
        raise ReconciliationRequired("original spending window elapsed; review before new compute")
    owner=modal.App.lookup(old["setup"]["app_name"],create_if_missing=False)
    if not owner.app_id:
        raise ValueError("sandbox owner has no resolved app ID")
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("active candidate sandboxes prohibit benchmark")
    pool=json.loads(Path(bench.SAVED_POOL).read_text())
    manifest=bench.manifest(pool)
    names=sorted(Path("verifier_rl").glob("*.py"))+[Path("modal_program_benchmark.py")]
    snapshot={p.as_posix():p.read_text() for p in names}
    reservation=reserve_benchmark(claims,old,manifest,snapshot)
    persist(directory,{"launch_intent":{"apps":apps,"reservation":reservation},"manifest":manifest})
    print("PROGRAM BENCHMARK BUDGET",reservation,flush=True)
    # Derived image has the SAME pinned Python base, one extra system library,
    # no project source, expected answers, model dependencies or credentials.
    candidate_image=modal.Image.from_id(old["setup"]["sandbox_image_id"]).apt_install("libseccomp2")
    candidate_image.build(owner)
    setup=dict(old["setup"],program_image_id=candidate_image.object_id)
    context={"manifest":manifest,"snapshot":snapshot,"setup":setup,"saved_pool":pool,
             "reservation":reservation,"original_budget_binding":budget_binding(old)}
    persist(directory,{"context":context})
    call=run_benchmark.spawn(context)
    persist(directory,{"launch":{"call_id":call.object_id}})
    print("PROGRAM BENCHMARK CALL",call.object_id,flush=True)
    result=call.get()
    persist(directory,{"result":result})
    # No other writer can run: the study remains stopped and maintenance lock
    # remains owned. Lost/failed calls retain the full $5 reservation.
    state=claims.get(study.RUN_ID+"/budget")
    if state["binding"]!=budget_binding(old):
        raise ValueError("original budget changed during benchmark")
    actual=accounting.cost(old["rates"],"cpu",math.ceil(result["function_seconds"])+120)
    actual+=accounting.cost(old["rates"],"sandbox",result["sandbox_seconds"])
    state["ledger"]=accounting.settle(state["ledger"],bench.RUN_ID,str(actual),reservation["identity"])
    claims.put(study.RUN_ID+"/budget",state)
    persist(directory,{"cost_receipt":{"estimated_resource_cost_usd":str(actual),
        "ledger_committed_usd":str(accounting.committed(state["ledger"])),"not_an_invoice":True}})
    print("PROGRAM BENCHMARK FINISHED; full study remains stopped",flush=True)
