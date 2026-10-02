"""Durable program-level grading; original research scores are unchanged.

The remote backend alone executes candidates. A claimed program with no final
lifecycle receipt is never automatically resubmitted. Completed programs may
be reconstructed from the provider journal after a controller interruption.
"""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import time

from . import booking_replication as study, program_execution as execution
from . import program_storage
from .evaluation_journal import ReconciliationRequired, persist
from .parallel_finalization import read_record
from .program_benchmark import bounded_map
from .suites import canonical_json, digest

VERSION = "durable-program-grading-0.1"
AMENDMENT = "program-sandbox-001"
NAMESPACE = study.RUN_ID + "/" + AMENDMENT
RELATIVE = "amendments/" + AMENDMENT
GRADE_SECONDS = 3600
PARALLEL_PROGRAMS = 4


def fingerprint(value):
    return digest(canonical_json(value))


def binding(image_id, deadline):
    if not isinstance(image_id,str) or not image_id.startswith("im-") or not math.isfinite(deadline):
        raise ValueError("invalid program runtime binding")
    return {"version":VERSION,"namespace":NAMESPACE,"image_id":image_id,"deadline":deadline,
            "execution_version":execution.VERSION,"runner_hash":digest(execution.runner()),
            "profile":execution.PROFILE,"program_concurrency":PARALLEL_PROGRAMS,
            "grading_functions":1,"grading_timeout_seconds":GRADE_SECONDS,
            "creation_interval_seconds":.30,"automatic_candidate_retries":0}


def validate_binding(runtime):
    if runtime != binding(runtime["image_id"],runtime["deadline"]):
        raise ValueError("program runtime contract changed")


def intent_for(sample, cases, runtime):
    validate_binding(runtime)
    return {"sample_id":sample["sample_id"],"sample_hash":fingerprint(sample),
            "source_hash":digest(sample["source"]),"input_order":[c.input_hash for c in cases],
            "runtime":runtime,"attempt":1}


def metadata_for(document):
    result = {"metadata":document["metadata"],"input_order":document["input_order"],
              "record_hashes":{h:fingerprint(r) for h,r in document["records"].items()}}
    if "continuation" in document:
        result["continuation_hash"] = fingerprint(document["continuation"])
    return result


def assemble(metadata, records):
    if "continuation_hash" in metadata:
        raise ReconciliationRequired("continuation requires its durable Volume provenance, not input-only reconstruction")
    if set(metadata["record_hashes"])!=set(records) or any(
            metadata["record_hashes"][h]!=fingerprint(r) for h,r in records.items()):
        raise ValueError("program journal records changed")
    return {"metadata":metadata["metadata"],"input_order":metadata["input_order"],"records":records}


async def execute_programs(samples, cases, directory, key, plan, runtime, backend, store, commit,
                           *, clock=time.time, on_program=None, backup_program=None):
    """Called only by one non-concurrent Modal Function, with four child jobs."""
    validate_binding(runtime)
    if len({s["sample_id"] for s in samples})!=len(samples):
        raise ValueError("duplicate program identity")
    for name in [key]+[s["sample_id"] for s in samples]:
        if not name or name in (".","..") or "/" in name or "\\" in name:
            raise ValueError("unsafe program journal path")
    directory=Path(directory)
    prefix=runtime["namespace"]+"/grading/"+key
    unknown_prefix=runtime["namespace"]+"/unknown"
    async def get(name, default=None):
        return await program_storage.get(store, name, default)
    async def put(name, value):
        return await program_storage.put_once(store, name, value)
    ledger_key=unknown_prefix+"/"+program_storage.AMENDMENT
    unknown_ledger=await get(ledger_key)
    if unknown_ledger is None:
        # One-time migration of legacy counters, never silently reset unknowns.
        identities=[await get(unknown_prefix+f"/slot/{i}") for i in range(plan["unknown_circuit"])]
        unknown_ledger={"identities":sorted(i for i in identities if i is not None)}
        await put(ledger_key,unknown_ledger)
    if len(unknown_ledger["identities"])>=plan["unknown_circuit"]:
        raise ReconciliationRequired("unknown-outcome circuit already exhausted")
    batch={"key":key,"samples_hash":fingerprint(samples),"case_hashes":[c.input_hash for c in cases],
           "plan_hash":fingerprint(plan),"runtime":runtime}
    persist(directory,{"intent":batch})
    await program_storage.retry(commit,label="Volume commit")
    await put(prefix+"/batch",batch)
    # The single grading Function also waits at batch boundaries. A new local
    # CreationGate cannot otherwise remember the last start of the prior batch.
    await asyncio.sleep(runtime["creation_interval_seconds"])
    progress={"new_program_starts":0,"reused_programs":0,"inputs":0,"unknown_inputs":0}
    charged_seconds=Decimal(0)
    commit_lock=asyncio.Lock()
    unknown_lock=asyncio.Lock()
    async def committed():
        async with commit_lock:
            await program_storage.retry(commit,label="Volume commit")
    async def worker(sample):
        nonlocal charged_seconds
        sid=sample["sample_id"]
        if sample["extraction_status"].startswith("rejected_"):
            return sid,{"rejected":True,"sample_hash":fingerprint(sample)}
        job=prefix+"/"+sid
        target=directory/"programs"/sid
        intent=intent_for(sample,cases,runtime)
        previous=await get(job+"/intent")
        if previous is not None and previous!=intent:
            raise ValueError("program intent changed")
        final=await get(job+"/final")
        if previous is None and (target/"result.json").exists():
            raise ReconciliationRequired("durable result without provider intent; inspect, never reexecute")
        recovered_publication=previous is not None and (target/"result.json").exists()
        if final is not None or recovered_publication:
            if previous is None:
                raise ValueError("result without program intent")
            if (target/"result.json").exists():
                document=json.loads((target/"result.json").read_text())
                if final is not None and metadata_for(document)!=final:
                    raise ValueError("saved program differs from provider receipt")
            else:
                # Every input was already submitted; this only restores JSON.
                records={}
                for case in cases:
                    record=await get(job+"/input/"+case.input_hash)
                    if record is None:
                        raise ReconciliationRequired("completed program missing durable input evidence")
                    records[case.input_hash]=record
                document=assemble(final,records)
            progress["reused_programs"]+=1
        else:
            if previous is not None:
                raise ReconciliationRequired("program submitted without final lifecycle evidence; no resubmission")
            if clock()>=runtime["deadline"]:
                raise ReconciliationRequired("program grading deadline before submission")
            if await get(runtime["namespace"]+"/stop") is not None:
                raise ReconciliationRequired("program scheduler stopped before submission")
            if not await put(job+"/intent",intent):
                raise ReconciliationRequired("program already claimed; no duplicate sandbox")
            persist(target,{"intent":intent})
            await committed()
            progress["new_program_starts"]+=1
            recorded={}
            async def record(h,value):
                await program_storage.save(target/"inputs"/(h+".json"),value)
                recorded[h]=value
                if len(recorded)%program_storage.CHUNK_SIZE==0:
                    await committed()
                progress["inputs"]+=1
                if progress["inputs"]%100==0:
                    print("PROGRAM INPUT PROGRESS",key,progress,flush=True)
            document=await backend.execute_program(sample["source"],cases,on_record=record)
            await program_storage.save(target/"result.json",document)
            await committed()
            # Fill explicit not-executed outcomes too; never shrink denominator.
            for h,value in document["records"].items():
                if h not in recorded:
                    await record(h,value)
                elif recorded[h]!=value:
                    raise ValueError("input callback differs from final report")
            charged_seconds+=min(Decimal(execution.lifetime_for(len(cases))),
                                 Decimal(math.ceil(document["metadata"]["total_seconds"])))
        checked=execution.validate_program(document,sample["source"],cases,runtime["image_id"])
        await program_storage.save(target/"intent.json",intent)
        await program_storage.save(target/"result.json",document)
        for h,record in document["records"].items():
            await program_storage.save(target/"inputs"/(h+".json"),record)
        await committed()
        if backup_program is not None:
            # Compact, independent archive copy. Dict is only a discovery index.
            async with commit_lock:
                await program_storage.retry(lambda: backup_program(sid,intent,document),label="archive commit")
        if document["metadata"].get("failure",{}).get("stage")=="evidence_storage" or document["metadata"].get("storage_failure"):
            raise program_storage.StorageFailure("partial program preserved; review never-executed suffix before continuation")
        await put(job+"/final",metadata_for(document))
        if document["metadata"]["cleanup"]!="terminated":
            await put(runtime["namespace"]+"/stop",
                {"reason":"cleanup_unconfirmed","batch":key,"sample_id":sid})
            raise ReconciliationRequired("cleanup unconfirmed; stop new work")
        progress["unknown_inputs"]+=sum(o["passed"] is None for o in checked.values())
        unknown=[unknown_prefix+"/input/"+sid+"/"+h for h,o in checked.items() if o["passed"] is None]
        if unknown:
            async with unknown_lock:
                identities=sorted(set(unknown_ledger["identities"])|set(unknown))
                # Single service + lock: one bounded index update per program,
                # not the old quadratic per-input slot probing. Full details are
                # already immutable on the Volumes.
                identities=identities[:plan["unknown_circuit"]]
                await program_storage.save(directory/"unknown"/(sid+".json"),{"identities":unknown})
                await committed()
                updated={"identities":identities}
                await program_storage.retry(lambda: store.put.aio(ledger_key,updated),label="unknown index")
                unknown_ledger.update(updated)
                if len(identities)>=plan["unknown_circuit"]:
                    await put(runtime["namespace"]+"/stop",{"reason":"unknown_circuit","batch":key,"sample_id":sid})
                    raise ReconciliationRequired("unknown-outcome circuit reached")
        print("DURABLE PROGRAM COMPLETE",key,sid,"reused",final is not None,
              "unknown",sum(o["passed"] is None for o in checked.values()),flush=True)
        return sid,{"intent":intent,"result":document}
    async def tracked(sample):
        result=await worker(sample)
        if on_program is not None:
            on_program(sample["sample_id"])
        return result
    try:
        entries=dict(await bounded_map(samples,tracked,PARALLEL_PROGRAMS))
        return entries,progress,str(charged_seconds)
    finally:
        await committed()


def verify_raw(raw, samples, key, role, plan, runtime):
    validate_binding(runtime)
    study.validate_batch(key,samples,role,plan)
    if (raw.get("version")!=VERSION or raw["samples"]!=samples or raw["key"]!=key or raw["role"]!=role
            or raw["plan_hash"]!=fingerprint(plan) or raw["runtime"]!=runtime
            or set(raw["programs"])!={s["sample_id"] for s in samples}):
        raise ValueError("program batch provenance mismatch")
    cases=study.pilot.cases_for(role)
    rows,ids=[],[]
    for sample in samples:
        entry=raw["programs"][sample["sample_id"]]
        if sample["extraction_status"].startswith("rejected_"):
            if entry!={"rejected":True,"sample_hash":fingerprint(sample)}:
                raise ValueError("rejected source execution invented")
            outcomes={c.input_hash:{"passed":False,"inclusive_match":False,"reason":"extraction_rejected"}
                      for c in cases}
        else:
            if set(entry)!={"intent","result"} or entry["intent"]!=intent_for(sample,cases,runtime):
                raise ValueError("program intent differs from fixed request")
            document=entry["result"]
            outcomes=execution.validate_program(document,sample["source"],cases,runtime["image_id"])
            for case in cases:
                outcome=outcomes[case.input_hash]
                outcome["inclusive_match"]=(None if outcome["passed"] is None else
                    type(outcome["actual"]) is int and outcome["actual"]==study.pilot.contrast.inclusive_answer(case.arguments_json))
            ids.extend(program_storage.sandbox_ids(document))
        rows.append(study.program_summary(sample,outcomes,role))
    if len(ids)!=len(set(ids)):
        raise ValueError("sandbox reused across different programs")
    return {"rows":rows,"sandbox_ids":ids,
            "submitted_attempts":sum(not s["extraction_status"].startswith("rejected_") for s in samples),
            "unknown_inputs":sum(r["unknown_inputs"] for r in rows)}


def rewards_from_raw(raw,samples,key,seed,arm,plan,runtime):
    if not key.startswith(f"train-{study.label(seed,arm)}-"):
        raise ValueError("reward belongs to different policy")
    checked=verify_raw(raw,samples,key,"training",plan,runtime)
    if checked["unknown_inputs"]:
        raise ReconciliationRequired("unknown training outcome: keep rollout/checkpoint; do not update")
    return [r[arm]["reward_bounds"][0] for r in checked["rows"]]


def journal_records(key,raw):
    for sid,entry in sorted(raw["programs"].items()):
        for name in (key,sid):
            if not name or name in (".","..") or "/" in name or "\\" in name:
                raise ValueError("unsafe journal path")
        if "rejected" in entry:
            continue
        base=f"grading/{key}/programs/{sid}"
        yield base+"/intent.json",entry["intent"]
        yield base+"/result.json",entry["result"]
        for h,record in sorted(entry["result"]["records"].items()):
            if len(h)!=64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("unsafe input hash")
            yield base+"/inputs/"+h+".json",record


def verify_journals(directory,key,raw):
    """Read every individual file with bounded prefetch, including after reuse."""
    started=time.monotonic()
    directory=Path(directory)
    if json.loads((directory/"grading"/key/"raw.json").read_text())!=raw:
        raise ValueError("program raw file changed")
    jobs=iter(journal_records(key,raw))
    count,total,transcript=0,0,hashlib.sha256()
    with ThreadPoolExecutor(max_workers=16) as pool:
        pending=deque()
        def fill():
            while len(pending)<32:
                item=next(jobs,None)
                if item is None: break
                name,expected=item
                pending.append((name,expected,pool.submit(read_record,directory/name)))
        fill()
        while pending:
            name,expected,future=pending.popleft()
            value,size,sha=future.result()
            if value!=expected:
                raise ValueError("individual program journal changed: "+name)
            transcript.update(canonical_json([name,sha]).encode()+b"\n")
            count+=1
            total+=size
            fill()
    return {"version":VERSION,"key":key,"journal_files":count,"bytes":total,
            "read_order_sha256":transcript.hexdigest(),"seconds":time.monotonic()-started}


def maximum_sandbox_seconds(samples,role):
    return sum(not s["extraction_status"].startswith("rejected_") for s in samples)*execution.lifetime_for(
        len(study.pilot.cases_for(role)))
