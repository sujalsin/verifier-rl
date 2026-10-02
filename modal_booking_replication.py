"""Opt-in four-seed/three-verifier study. No automatic expansion or resampling.

Each batch and policy has one owner. The reviewed program-sandbox amendment
serializes grading batches and runs four restricted program sandboxes inside
each batch. Research verifiers, model initialization and GRPO are unchanged.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import time

import modal

from modal_booking_study import load_policy, parameter_hash, require_deadline
from modal_booking_baseline_comparison import check_snapshot, read_modal
from modal_booking_matched_training import ControlledInterruption, control_result
from verifier_rl import booking_replication as study, booking_matched_training as old_release
from verifier_rl import compute_budget as accounting, grpo_recovery as recovery
from verifier_rl import program_storage
from verifier_rl import supervised_execution as supervised, supervisor_controls
from verifier_rl import program_grading as programs, program_execution, program_benchmark
from verifier_rl.durable_grading import execute_batch
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.panel_execution import pack_result, request_for
from verifier_rl.parallel_finalization import PrefetchJSONReader, journal_records
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import canonical_json, digest

app = modal.App("verifier-rl-booking-replication-repair")
work = modal.Volume.from_name("verifier-rl-booking-replication-work", create_if_missing=True, version=2)
archive = modal.Volume.from_name("verifier-rl-booking-replication-evidence", create_if_missing=True, version=1)
claims = modal.Dict.from_name("verifier-rl-booking-replication-journal", create_if_missing=True)
LAUNCHERS = ("modal_booking_study.py", "modal_booking_baseline_comparison.py",
             "modal_booking_matched_training.py", "modal_booking_replication.py")


def mount_sources(image):
    image = image.add_local_python_source("verifier_rl")
    for name in LAUNCHERS:
        image = image.add_local_file(name, "/root/" + name)
    return image


cpu_image = mount_sources(modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5"))
gpu_image = mount_sources(modal.Image.debian_slim(python_version="3.12")
    .pip_install("modal==1.5.5", "torch==2.8.0", "transformers==4.57.1", "trl==0.28.0",
                 "datasets==3.5.1", "accelerate==1.12.0")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
          "CUBLAS_WORKSPACE_CONFIG": recovery.CUDA_WORKSPACE}))
VOLUMES = {"/artifacts": work, "/evidence": archive}
read_json = recovery.read_json


def fingerprint(value):
    return digest(canonical_json(value))


def root(plan):
    return Path("/artifacts") / plan["run_id"]


def program_mode(context):
    return context.get("runtime_amendment",{}).get("id")==programs.AMENDMENT


def namespace(context):
    return programs.NAMESPACE if program_mode(context) else study.RUN_ID


def run_root(context):
    return root(context["plan"])/(programs.RELATIVE if program_mode(context) else "")


def checked_raw(context,raw,samples,key,role):
    if program_mode(context):
        return programs.verify_raw(raw,samples,key,role,context["plan"],context["program_runtime"])
    return study.verify_raw(raw,samples,key,role,context["plan"],context["setup"]["sandbox_image_id"])


def execution_rewards(context,raw,samples,key,seed,arm):
    if program_mode(context):
        return programs.rewards_from_raw(raw,samples,key,seed,arm,context["plan"],context["program_runtime"])
    return study.rewards_from_raw(raw,samples,key,seed,arm,context["plan"],context["setup"]["sandbox_image_id"])


def require_program_authorization(context):
    programs.validate_binding(context["program_runtime"])
    original=context["budget_context"]
    if (context["plan"]!=original["plan"] or context["rates"]!=original["rates"]
            or context["billing_before"]!=original["billing_before"]
            or context["program_runtime"]["deadline"]!=context["deadline"]
            or context["program_runtime"]["image_id"]!=context["setup"]["sandbox_image_id"]):
        raise ValueError("runtime amendment changed research/budget identity")
    authorization=claims.get(programs.NAMESPACE+"/authorization",None)
    expected=fingerprint(context)
    if context.get("storage_amendment"):
        expected=program_storage.parent_context(context)
        storage_authorization=claims.get(programs.NAMESPACE+"/"+program_storage.metadata_relative(context)+"/ready",None)
        if storage_authorization is None or storage_authorization["context_hash"]!=fingerprint(context):
            raise ValueError("storage repair and recovery are not yet reviewed/ready")
    if authorization is None or authorization["context_hash"]!=expected:
        raise ValueError("program runtime not explicitly authorized")


def runtime_relative(context):
    amendment=context.get("runtime_amendment")
    if amendment is None:
        return ""
    if program_mode(context):
        return programs.RELATIVE
    if amendment["id"]!="journal-order-001":
        raise ValueError("unreviewed runtime amendment")
    return "amendments/journal-order-001"


def backup(context, relative, documents):
    base=Path("/evidence")/context["plan"]["run_id"]
    if program_mode(context):
        base=base/programs.RELATIVE
    persist(base / relative, documents)
    archive.commit()


def claim(key, maximum=1, *, prefix=study.RUN_ID):
    for slot in range(maximum):
        if claims.put(prefix + f"/starts/{key}/{slot}", {"utc": datetime.now(timezone.utc).isoformat()}, skip_if_exists=True):
            return slot
    raise ReconciliationRequired("bounded invocation limit reached: " + key)


@app.function(image=cpu_image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=60, retries=0, max_containers=1, scaledown_window=2)
def budget(action, context, key=None, amount=None, identity=None):
    """Single serialized writer; no @concurrent and no shared-volume balance."""
    study.validate_plan(context["plan"])
    original=context["budget_context"] if program_mode(context) else context
    binding = fingerprint({k: original[k] for k in ("plan", "rates", "billing_before", "deadline")})
    state_key = study.RUN_ID + "/budget"
    saved = claims.get(state_key, None)
    if saved is None:
        if program_mode(context):
            raise ValueError("original ledger missing; runtime amendment cannot initialize a budget")
        saved = {"binding": binding, "ledger": accounting.initialize()}
    if saved["binding"] != binding:
        raise ValueError("budget identity changed")
    if program_mode(context):
        require_program_authorization(context)
        if key is not None:
            key=programs.AMENDMENT+"/"+key
    if action == "reserve":
        saved["ledger"] = accounting.reserve(saved["ledger"], key, amount, identity)
    elif action == "settle":
        saved["ledger"] = accounting.settle(saved["ledger"], key, amount, identity)
    elif action != "status":
        raise ValueError("unknown budget action")
    claims.put(state_key, saved)
    return {"committed_usd": str(accounting.committed(saved["ledger"])), "ledger": saved["ledger"]}


def invoke(function, key, kind, seconds, context, *args, sandbox_starts=0, sandbox_seconds=0):
    """Reserve BEFORE allocating a worker; retain full holds if a call is lost."""
    require_deadline(context["deadline"])
    prefix=namespace(context)
    stopped = claims.get(prefix + "/stop", None)
    if stopped is not None:
        raise ReconciliationRequired("study scheduling stopped: " + str(stopped))
    identity_fields={"key":key, "kind":kind, "seconds":seconds, "sandbox_starts":sandbox_starts,
                     "plan":context["plan"], "arguments":args}
    if program_mode(context):
        identity_fields.update(runtime=context["program_runtime"],sandbox_seconds=sandbox_seconds)
    elif sandbox_seconds:
        raise ValueError("program lifetime reservation requires reviewed runtime")
    identity = fingerprint(identity_fields)
    cache_key = prefix + "/finished/" + key
    cached = claims.get(cache_key, None)
    if cached is not None:
        if cached["identity"] != identity:
            raise ValueError("completed invocation arguments changed")
        return cached["payload"]
    maximum_starts=context["plan"]["max_call_starts"]
    if context.get("storage_amendment"):
        require_program_authorization(context)
        if key in program_storage.EXTRA_STARTS:
            maximum_starts+=context["storage_amendment"]["additional_start_count"]
    slot = claim(key, maximum_starts,prefix=prefix)
    ticket = f"{key}/attempt-{slot}"
    maximum = accounting.cost(context["rates"],kind,seconds+60)
    maximum += accounting.cost(context["rates"],"sandbox",sandbox_starts*120+sandbox_seconds)
    try:
        status = budget.remote("reserve",context,ticket,str(maximum),identity)
    except accounting.BudgetReached:
        claims.put(prefix+"/stop",{"reason":"compute_budget","ticket":ticket},skip_if_exists=True)
        raise
    print("BUDGET RESERVED", ticket, status["committed_usd"], "/ 250 USD", flush=True)
    call = function.spawn(context, ticket, *args)
    claims.put(prefix+"/calls/"+ticket,{"call_id":call.object_id,"identity":identity},skip_if_exists=True)
    if key == "controller":
        launch_directory=Path("runs")/study.RUN_ID/runtime_relative(context)
        name="launch" if not (launch_directory/"launch.json").exists() else f"launch-{slot}"
        persist(launch_directory,{name:{"call_id":call.object_id,"ticket":ticket}})
        print("REPLICATION CONTROLLER CALL",call.object_id,flush=True)
    try:
        result = call.get()
    except ReconciliationRequired as exc:
        if "cleanup unconfirmed" in str(exc):
            claims.put(prefix+"/stop",{"reason":"cleanup_unconfirmed","ticket":ticket},skip_if_exists=True)
        raise
    actual = accounting.cost(context["rates"],kind,math.ceil(result["seconds"])+60)
    actual += accounting.cost(context["rates"],"sandbox",result.get("sandbox_seconds",0))
    budget.remote("settle",context,ticket,str(actual),identity)
    saved = {"identity":identity, "payload":result["payload"]}
    if not claims.put(cache_key,saved,skip_if_exists=True) and claims.get(cache_key) != saved:
        raise ValueError("completed invocation changed")
    return saved["payload"]


def enter(context, ticket):
    study.validate_plan(context["plan"])
    require_deadline(context["deadline"])
    check_snapshot(context["source_snapshot"], Path("/root"))
    if program_mode(context):
        require_program_authorization(context)
        programs.validate_binding(context["program_runtime"])
    # Provider preemption must not concurrently re-enter a partially completed
    # call. A new explicitly launched call can reuse durable checkpoints later.
    if not claims.put(namespace(context)+"/owners/"+ticket, {"started":time.time()}, skip_if_exists=True):
        raise ReconciliationRequired("worker restarted; preserve state and explicitly reconcile: "+ticket)
    work.reload()
    return time.monotonic(), context["plan"]


def response(payload, started, sandbox_seconds=0):
    return {"payload":payload, "seconds":time.monotonic()-started, "sandbox_seconds":str(sandbox_seconds)}


def runtime():
    import torch
    return {"gpu":torch.cuda.get_device_name(), "gpu_image_id":gpu_image.object_id,
            "packages":{p:importlib.metadata.version(p) for p in recovery.PACKAGES},
            "determinism":recovery.deterministic_training()}


def build_trainer(directory, plan, seed, reward, *, control=False, after_checkpoint=None):
    import gc
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig
    experiment = study.experiment_for(plan,seed)
    binding = {"release_hash":fingerprint(plan), "directory":str(directory), "seed":seed, "control":control}
    recovery.deterministic_training()
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(seed)
    with ProgressLog(directory.name,label="MODEL") as progress:
        progress.stage("load_original_policy")
        model,tokenizer = load_policy(experiment)
    model.config.use_cache = False
    kwargs = study.trainer_kwargs(directory/"trainer",seed)
    kwargs["logging_dir"] = str(directory/"logs")
    if control:
        kwargs["max_steps"] = plan["control_steps"]
    config = GRPOConfig(**kwargs)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=0,betas=(.9,.999),eps=1e-8,foreach=False)
    trainer = recovery.trainer_class()(model=model,args=config,reward_funcs=reward,
        train_dataset=Dataset.from_list([{"prompt":[{"role":"user","content":experiment["prompt"]}]}]*24),
        processing_class=tokenizer,optimizers=(optimizer,None),journal=directory/"journal",
        binding_hash=fingerprint(binding),initial_parameter_hash=experiment["initial_parameter_hash"],
        parameter_hash=parameter_hash,commit=work.commit,after_checkpoint=after_checkpoint)
    persist(directory,{"binding":binding,"trainer_config":json.loads(config.to_json_string()),
                       "generation_config":trainer.generation_config.to_dict()})
    work.commit()
    checkpoint = recovery.latest_checkpoint(directory/"trainer",fingerprint(binding))
    return trainer,tokenizer,checkpoint


@app.function(image=gpu_image,gpu="L40S",cpu=(2,2),memory=(32768,32768),timeout=study.CONTROL_SECONDS,
              retries=0,max_containers=1,scaledown_window=2,volumes=VOLUMES)
def control_part(context,ticket,part):
    started,plan = enter(context,ticket)
    target = run_root(context)/"recovery-control"
    if part not in plan["control_parts"]:
        raise ValueError("unknown control part")
    if (target/f"{part}.json").exists():
        return response(read_json(target/f"{part}.json"),started)
    directory = target/("uninterrupted" if part == "uninterrupted" else "interrupted")
    persist(target/"runtimes",{part:runtime()})
    def reward(completions,completion_ids=None,trainer_state=None,**kwargs):
        require_deadline(context["deadline"])
        index = trainer_state.global_step
        if len(completions)!=4 or completion_ids is None or not 0<=index<3:
            raise ValueError("control rollout schedule changed")
        group = directory/f"journal/group-{index:02d}"
        if read_json(group/"generation.json")["output"][1]!=completion_ids:
            raise ValueError("control token identity changed")
        record = {"completion_ids":completion_ids,"completions":completions,"rewards":plan["control_reward"]}
        persist(group,{"reward":record})
        work.commit()
        if part=="interrupt" and index==1:
            raise ControlledInterruption("saved pending second group before update")
        return record["rewards"]
    trainer,_,checkpoint = build_trainer(directory,plan,study.SEEDS[0],reward,control=True)
    if (part=="resume") != (checkpoint is not None):
        raise ValueError("unexpected recovery-control checkpoint")
    interrupted=False
    try:
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    except ControlledInterruption:
        interrupted=True
    result=control_result(directory,trainer,interrupted)
    persist(target,{part:result})
    work.commit()
    backup(context,"recovery-control",{part:result})
    print("REPLICATION RECOVERY CONTROL",part,result["steps"],flush=True)
    return response(result,started)


def require_controls(context):
    plan=context["plan"]
    parts={p:read_json(run_root(context)/f"recovery-control/{p}.json") for p in plan["control_parts"]}
    checked=old_release.validate_control(parts,plan)
    if read_json(run_root(context)/"recovery-control/result.json")!=checked:
        raise ValueError("live full-state recovery control missing or changed")
    control_sources=read_json(root(plan)/runtime_relative(context)/"source_snapshot.json")
    if context.get("storage_amendment"):
        # Keep the original control snapshot immutable. Reuse is valid only if
        # the independently reviewed amendment leaves every research source and
        # the protected execution contract unchanged; never waive this check.
        if program_storage.validate_sources(control_sources,context["source_snapshot"])!=context["storage_amendment"]["sources"]:
            raise ValueError("control source amendment differs from reviewed sources")
    elif control_sources!=context["source_snapshot"]:
        raise ValueError("control and training source differ")
    if read_json(run_root(context)/"preflight.json")["passed"] is not True:
        raise ValueError("grader preflight not passed")
    parallel=read_json(run_root(context)/"parallel_preflight.json")
    if parallel["passed"] is not True or parallel["batches"]!=4:
        raise ValueError("parallel grading preflight not passed")
    if program_mode(context):
        reuse=read_json(run_root(context)/"program_reuse_preflight.json")
        if (reuse["passed"] is not True or reuse["service_concurrency"]!=1
                or reuse["program_concurrency"]!=4 or reuse["new_sandbox_reservation_seconds"]!=0
                or read_json(run_root(context)/"supervisor_controls.json")["passed"] is not True):
            raise ValueError("program runtime/reuse controls missing")


@app.function(image=cpu_image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,
              timeout=study.GRADE_SECONDS,retries=0,max_containers=4,scaledown_window=2,volumes=VOLUMES)
def grade_batch(context,ticket,key,samples,role):
    if program_mode(context):
        raise ValueError("legacy per-input worker is disabled for program runtime")
    started,plan=enter(context,ticket)
    study.validate_batch(key,samples,role,plan)
    directory=run_root(context)/"grading"/key
    setup=context["setup"]
    if (directory/"result.json").exists():
        raw,checked,_=checked_saved(context,key,samples,role)
        result=read_json(directory/"result.json")
        backup(context,"grading/"+key,{"result":result})
        print("REPLICATION REUSED GRADING",key,flush=True)
        return response(result,started)  # no sandbox execution in this invocation
    backend=supervised.SupervisedPanelBackend(setup["app_name"],setup["sandbox_image_id"],study.pilot.BOOKING,
                                            creation_interval_seconds=.26)
    try:
        entries,progress=asyncio.run(execute_batch(samples,study.pilot.cases_for(role),directory,key,
            study.execution_plan(plan),setup["sandbox_image_id"],context["deadline"],backend,claims,work.commit.aio))
        raw={"key":key,"role":role,"samples":samples,"entries":entries,"plan_hash":fingerprint(plan)}
        persist(directory,{"raw":raw})
        work.commit()
        checked=study.verify_raw(raw,samples,key,role,plan,setup["sandbox_image_id"])
        # Verify every individual journal file while this batch is still small.
        # Final aggregation verifies these raw records again, plus this receipt.
        receipt={"raw_hash":fingerprint(raw),"checked_hash":fingerprint(checked),
                 "reader":verify_batch_journals(root(plan),key,raw)}
        result={"raw":raw,"checked":checked,"receipt":receipt}
        persist(directory,{"result":result,"execution_progress":progress})
        work.commit()
        backup(context,"grading/"+key,{"result":result})
        print("REPLICATION GRADED",key,[(r["reference"]["passed_bounds"],r["endpoint_omission"]["passed_bounds"],
            r["repaired"]["passed_bounds"],r.get("audit",{}).get("passed_bounds")) for r in checked["rows"]],flush=True)
        print("REPLICATION EXECUTION COUNTS",key,progress,flush=True)
        return response(result,started,accounting.sandbox_seconds(raw) if progress["new_starts"] else 0)
    except Exception as exc:
        persist(directory,{f"interrupted-{time.time_ns()}":{"type":type(exc).__name__,"detail":str(exc)[:2000]}})
        work.commit()
        raise


@app.function(image=cpu_image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,
              timeout=programs.GRADE_SECONDS,retries=0,max_containers=1,scaledown_window=2,volumes=VOLUMES)
def grade_program_batch(context,ticket,key,samples,role,reuse_only=False):
    """One serial service owns ALL starts; four program sandboxes inside a batch."""
    if not program_mode(context):
        raise ValueError("reviewed program runtime required")
    started,plan=enter(context,ticket)
    study.validate_batch(key,samples,role,plan)
    directory=run_root(context)/"grading"/key
    if (directory/"result.json").exists():
        raw,checked,_=checked_saved(context,key,samples,role)
        programs.verify_journals(run_root(context),key,raw)
        result=read_json(directory/"result.json")
        backup(context,"grading/"+key,{"result":result})
        return response(result,started)
    if reuse_only:
        raise ReconciliationRequired("reuse-only check cannot create missing program evidence")
    gate=program_execution.CreationGate(.30)
    backend=program_execution.ProgramBackend(context["setup"]["app_name"],
        context["program_runtime"]["image_id"],start_gate=gate)
    try:
        with ProgressLog(key,label="PROGRAM_GRADING") as log:
            log.stage("durable_program_execution",total=len(samples),unit="programs")
            def completed(sid):
                log.item(sid)
                log.advance()
            async def archive_program(sid,intent,document):
                target=Path("/evidence")/plan["run_id"]/programs.RELATIVE/"grading"/key/"programs"/sid
                persist(target,{"intent":intent,"result":document})
                await archive.commit.aio()
            entries,progress,charged=asyncio.run(programs.execute_programs(
                samples,study.pilot.cases_for(role),directory,key,plan,context["program_runtime"],
                backend,claims,work.commit.aio,on_program=completed,backup_program=archive_program))
            log.stage("verify_input_evidence")
            raw={"version":programs.VERSION,"key":key,"role":role,"samples":samples,"programs":entries,
                 "plan_hash":fingerprint(plan),"runtime":context["program_runtime"]}
            persist(directory,{"raw":raw})
            work.commit()
            checked=checked_raw(context,raw,samples,key,role)
            receipt={"raw_hash":fingerprint(raw),"checked_hash":fingerprint(checked),
                     "reader":programs.verify_journals(run_root(context),key,raw)}
            result={"raw":raw,"checked":checked,"receipt":receipt}
            persist(directory,{"result":result,"execution_progress":progress})
            work.commit()
            backup(context,"grading/"+key,{"result":result})
        print("PROGRAM REPLICATION GRADED",key,
              [(r["reference"]["passed_bounds"],r["endpoint_omission"]["passed_bounds"],r["repaired"]["passed_bounds"])
               for r in checked["rows"]],"progress",progress,flush=True)
        return response(result,started,charged)
    except Exception as exc:
        persist(directory,{f"interrupted-{time.time_ns()}":{"type":type(exc).__name__,"detail":str(exc)[:2000]}})
        work.commit()
        raise


def verify_batch_journals(directory,key,raw):
    with ProgressLog(key,label="BATCH_EVIDENCE") as log:
        log.stage("verify_individual_input_journals")
        with PrefetchJSONReader(directory) as reader:
            saved=reader(Path(directory)/"grading"/key/"raw.json")
            if saved!=raw:
                raise ValueError("raw journal changed")
            # persist() canonicalizes object keys. The read-ahead iterator uses
            # that on-disk order, not the original insertion order from workers.
            for name,expected in journal_records(key,saved):
                if reader(Path(directory)/name)!=expected:
                    raise ValueError("individual input record differs: "+name)
                log.advance()
    return reader.receipt()


def grade(context,key,samples,role):
    if program_mode(context):
        return invoke(grade_program_batch,"grading/"+key,"cpu",programs.GRADE_SECONDS,context,key,samples,role,
                      sandbox_seconds=programs.maximum_sandbox_seconds(samples,role))
    count=sum(not s["extraction_status"].startswith("rejected_") for s in samples)*len(study.pilot.cases_for(role))
    return invoke(grade_batch,"grading/"+key,"cpu",study.GRADE_SECONDS,context,key,samples,role,
                  sandbox_starts=count+context["plan"]["max_startup_retries"])


def draw(model,tokenizer,context,policy,step,directory,parameter_identity):
    import torch
    from transformers import set_seed
    experiment=context["plan"]["experiment"]
    formatted=tokenizer.apply_chat_template([{"role":"user","content":experiment["prompt"]}],
                                           tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(formatted,add_special_tokens=False,return_tensors="pt").to("cuda")
    samples=[]
    with ProgressLog(policy+f"-{step}",label="EVALUATION_GENERATION") as progress:
        progress.stage("fixed_sample_generation",total=len(study.identities(policy,step)),unit="programs")
        for sid,seed in study.identities(policy,step):
            require_deadline(context["deadline"])
            intent={"sample_id":sid,"seed":seed,"parameter_hash":parameter_identity}
            prefix=namespace(context)+"/generation/"+sid
            sample=claims.get(prefix+"/sample",None)
            if sample is None:
                if not claims.put(prefix+"/intent",intent,skip_if_exists=True):
                    raise ReconciliationRequired("sample intent without result; do not replace draw: "+sid)
                persist(directory/"intents",{sid:intent})
                work.commit()
                set_seed(seed)
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    tokens=model.generate(**inputs,max_new_tokens=512,do_sample=True,temperature=.8,top_p=.95,top_k=0,
                        repetition_penalty=experiment["repetition_penalty"],pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,use_cache=True)
                completion=tokens[0,inputs["input_ids"].shape[1]:]
                sample=study.pilot.original.sample_from_text(tokenizer.decode(completion,skip_special_tokens=True),
                    sid=sid,seed=seed,plan=experiment,tokens=len(completion),
                    eos=bool(len(completion) and completion[-1].item()==tokenizer.eos_token_id))
                if not claims.put(prefix+"/sample",sample,skip_if_exists=True):
                    raise ValueError("duplicate generated sample")
            if claims.get(prefix+"/intent")!=intent:
                raise ValueError("sample intent changed")
            persist(directory/"intents",{sid:intent})
            persist(directory/"samples",{sid:sample})
            work.commit()
            samples.append(sample)
            progress.advance()
            print("REPLICATION GENERATED",sid,"tokens",sample["tokens"],flush=True)
    if parameter_hash(model)!=parameter_identity:
        raise ValueError("generation changed policy weights")
    result={"parameter_hash":parameter_identity,"samples":samples}
    persist(directory,{"result":result})
    work.commit()
    backup(context,"generation/"+policy+f"-{step}",{"result":result})
    return result


@app.function(image=gpu_image,gpu="L40S",cpu=(2,2),memory=(32768,32768),timeout=study.BASELINE_SECONDS,
              retries=0,max_containers=1,scaledown_window=2,volumes=VOLUMES)
def baseline(context,ticket):
    started,plan=enter(context,ticket)
    require_controls(context)
    target=run_root(context)/"baseline"
    if (target/"result.json").exists():
        result=read_json(target/"result.json")
        if result["parameter_hash"]!=plan["experiment"]["initial_parameter_hash"] or len(result["samples"])!=128:
            raise ValueError("saved baseline identity changed")
        for i in range(32):
            study.validate_batch(f"eval-baseline-00-{i:02d}",result["samples"][4*i:4*i+4],"evaluation",plan)
        return response(result,started)
    model,tokenizer=load_policy(plan["experiment"])
    model.requires_grad_(False)
    model.eval()
    result=draw(model,tokenizer,context,"baseline",0,run_root(context)/"baseline",plan["experiment"]["initial_parameter_hash"])
    return response(result,started)


def evaluation_at_boundary(trainer,tokenizer,context,policy,step,directory):
    import torch
    rng,mode,cache=recovery.capture_rng(),trainer.model.training,trainer.model.config.use_cache
    deterministic=torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic=torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic=False
        trainer.model.eval()
        snapshot=directory/f"model-{step:02d}"
        receipt={"parameter_hash":trainer.boundary_hash,"step":step}
        if not (snapshot/"receipt.json").exists():
            trainer.save_model(str(snapshot))
            tokenizer.save_pretrained(snapshot)
            persist(snapshot,{"receipt":receipt})
            work.commit()
        elif read_json(snapshot/"receipt.json")!=receipt:
            raise ValueError("evaluation model snapshot changed")
        return draw(trainer.model,tokenizer,context,policy,step,directory/f"evaluation-{step:02d}",trainer.boundary_hash)
    finally:
        trainer.model.train(mode)
        trainer.model.config.use_cache=cache
        recovery.restore_rng(rng)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic=cudnn_deterministic


@app.function(image=gpu_image,gpu="L40S",cpu=(2,2),memory=(32768,32768),timeout=study.GPU_SECONDS,
              retries=0,max_containers=2,scaledown_window=2,volumes=VOLUMES)
def train_arm(context,ticket,seed,arm):
    import torch
    from transformers import TrainerCallback
    started,plan=enter(context,ticket)
    require_controls(context)
    policy=study.label(seed,arm)
    directory=run_root(context)/"arms"/policy
    if (directory/"result.json").exists():
        result=study.validate_arm(read_json(directory/"result.json"),plan)
        return response(result,started)
    persist(directory/"runtimes",{ticket.rsplit("/",1)[-1]:runtime()})
    experiment,holder=study.experiment_for(plan,seed),{}
    def reward(completions,completion_ids=None,trainer_state=None,**kwargs):
        require_deadline(context["deadline"])
        index=trainer_state.global_step
        if not 0<=index<24 or len(completions)!=4 or completion_ids is None:
            raise ValueError("unexpected research rollout schedule")
        target=directory/f"journal/group-{index:02d}"
        generated=read_json(target/"generation.json")
        if generated["output"][1]!=completion_ids:
            raise ValueError("reward token identity changed")
        if index==0:
            key=namespace(context)+f"/matched-first/{seed}"
            claims.put(key,completion_ids,skip_if_exists=True)
            if claims.get(key)!=completion_ids:
                raise ValueError("first rollout differs across seed-matched arms")
        key=f"train-{policy}-{index:02d}"
        texts=[v[0]["content"] if isinstance(v,list) else v for v in completions]
        samples=[study.pilot.original.sample_from_text(text,sid=f"{key}-{j}",plan=experiment,
            tokens=len(completion_ids[j]),eos=bool(completion_ids[j] and completion_ids[j][-1]==holder["tokenizer"].eos_token_id))
            for j,text in enumerate(texts)]
        persist(target,{"samples":samples})
        work.commit()
        with ProgressLog(f"{policy}-{index+1}",label="TRAINING_REWARD") as progress:
            progress.stage("sandbox_grading",total=384,unit="input_slots")
            graded=grade(context,key,samples,"training")
            work.reload()
            progress.stage("verify_reward")
            values=execution_rewards(context,graded["raw"],samples,key,seed,arm)
        persist(target,{"reward":{"completion_ids":completion_ids,"samples":samples,"rewards":values,
                                  "raw_hash":fingerprint(graded["raw"])}})
        work.commit()
        print("REPLICATION REWARD",policy,"update",index+1,"rewards",values,flush=True)
        return values
    def after_checkpoint(trainer,path,receipt):
        if receipt["step"] in (12,24):
            evaluation_at_boundary(trainer,holder["tokenizer"],context,policy,receipt["step"],directory)
    trainer,tokenizer,checkpoint=build_trainer(directory,plan,seed,reward,after_checkpoint=after_checkpoint)
    holder["tokenizer"]=tokenizer
    class ResumeEvaluation(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            if state.global_step in (12,24):
                evaluation_at_boundary(trainer,tokenizer,context,policy,state.global_step,directory)
            return control
    trainer.add_callback(ResumeEvaluation())
    print("REPLICATION TRAINING START",policy,"checkpoint",str(checkpoint) if checkpoint else "original/fresh optimizer",flush=True)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    if trainer.state.global_step!=24 or not all(torch.isfinite(p).all().item() for p in trainer.model.parameters()):
        raise ValueError("incomplete finite training")
    final_hash=parameter_hash(trainer.model)
    trainer._load_from_checkpoint(str(directory/"trainer/checkpoint-24"))
    result={"seed":seed,"arm":arm,
        "evaluations":{str(s):read_json(directory/f"evaluation-{s:02d}/result.json") for s in (12,24)},
        "boundaries":{str(s):read_json(directory/f"journal/boundaries/{s}.json") for s in range(1,25)},
        "first_tokens":read_json(directory/"journal/group-00/generation.json")["output"][1],
        "metrics":{"global_step":24,"log_history":trainer.state.log_history,
            "before_parameter_hash":experiment["initial_parameter_hash"],"after_parameter_hash":final_hash,
            "parameters_changed":final_hash!=experiment["initial_parameter_hash"],"finite_parameters":True,
            "final_reload_verified":parameter_hash(trainer.model)==final_hash,
            "nonzero_gradient_steps":sum(r.get("grad_norm",0)>0 for r in trainer.state.log_history),
            "generated_training_tokens":sum(len(ids) for i in range(24) for ids in
                read_json(directory/f"journal/group-{i:02d}/generation.json")["output"][1])}}
    study.validate_arm(result,plan)
    persist(directory,{"result":result})
    work.commit()
    backup(context,"arms/"+policy,{"result":result})
    print("REPLICATION TRAINING COMPLETE",policy,flush=True)
    return response(result,started)


def checked_saved(context,key,samples,role):
    plan=context["plan"]
    directory=run_root(context)/"grading"/key
    raw=read_json(directory/"raw.json")
    result=read_json(directory/"result.json")
    checked=checked_raw(context,raw,samples,key,role)
    receipt=result["receipt"]
    if program_mode(context):
        expected_files=sum(1 for _ in programs.journal_records(key,raw))
        reader_ok=(receipt["reader"]["version"]==programs.VERSION and receipt["reader"]["key"]==key)
    else:
        expected_files=sum(len(e) for inputs in raw["entries"].values() for e in inputs.values())
        reader_ok=len(receipt["reader"]["batches"])==1 and receipt["reader"]["batches"][0]["key"]==key
    if (result["raw"]!=raw or result["checked"]!=checked
            or receipt["raw_hash"]!=fingerprint(raw) or receipt["checked_hash"]!=fingerprint(checked)
            or receipt["reader"]["journal_files"]!=expected_files
            or not reader_ok):
        raise ValueError("batch differs from verified individual-file receipt")
    return raw,checked,receipt


def evaluate_policy(context,policy,step,samples,*,parallel=False):
    """Fixed batches; parallel only across disjoint immutable identities."""
    count=len(study.identities(policy,step))
    if len(samples)!=count:
        raise ValueError("incomplete fixed generation pool")
    batches=[(f"eval-{policy}-{step:02d}-{i:02d}",samples[4*i:4*i+4]) for i in range(count//4)]
    if parallel:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(grade,context,k,s,"evaluation") for k,s in batches]
            for future in as_completed(futures):
                future.result()
    else:
        for key,group in batches:
            grade(context,key,group,"evaluation")
    work.reload()
    rows,ids,receipts=[],[],{}
    with ProgressLog(f"{policy}-{step}",label="POLICY_FINALIZATION") as progress:
        progress.stage("replay_verified_batches",total=len(batches),unit="batches")
        for key,group in batches:
            progress.item(key)
            _,checked,receipt=checked_saved(context,key,group,"evaluation")
            rows.extend(checked["rows"])
            ids.extend(checked["sandbox_ids"])
            receipts[key]=receipt
            progress.advance()
    if len(ids)!=len(set(ids)):
        raise ValueError("evaluation reused sandbox identity")
    result={"policy":policy,"step":step,"programs":rows,"summary":study.policy_summary(rows,policy,step),
            "sandbox_ids":ids,"batch_receipts":receipts}
    target=run_root(context)/"policies"/f"{policy}-{step}"
    persist(target,{"result":result})
    work.commit()
    backup(context,f"policies/{policy}-{step}",{"result":result})
    return result


def verify_arm_journals(context,seed,arm,result):
    """Check every consumed reward against raw evidence and its policy boundary."""
    plan=context["plan"]
    study.validate_arm(result,plan)
    policy=study.label(seed,arm)
    directory=run_root(context)/"arms"/policy
    binding={"release_hash":fingerprint(plan),"directory":str(directory),"seed":seed,"control":False}
    if read_json(directory/"binding.json")!=binding:
        raise ValueError("training journal identity changed")
    previous=plan["experiment"]["initial_parameter_hash"]
    ids,receipts=[],{}
    with ProgressLog(policy,label="TRAINING_FINALIZATION") as progress:
        progress.stage("replay_all_training_rewards",total=24,unit="updates")
        for index in range(24):
            require_deadline(context["deadline"])
            group=directory/f"journal/group-{index:02d}"
            generated=read_json(group/"generation.json")
            intent=read_json(group/"intent.json")
            consumed=read_json(group/"reward.json")
            boundary=read_json(directory/f"journal/boundaries/{index+1}.json")
            if (intent!=generated["binding"] or intent["binding_hash"]!=fingerprint(binding)
                    or intent["parameter_hash"]!=previous or intent["step"]!=index
                    or intent["version"]!=recovery.VERSION
                    or generated["tokens_hash"]!=fingerprint(generated["output"])
                    or consumed["completion_ids"]!=generated["output"][1]
                    or read_json(group/"samples.json")!=consumed["samples"]
                    or boundary!=result["boundaries"][str(index+1)]
                    or boundary["binding_hash"]!=fingerprint(binding)
                    or boundary["step"]!=index+1 or boundary["trl_step"]!=4*(index+1)):
                raise ValueError("training token/RNG/checkpoint linkage changed")
            if index==0 and result["first_tokens"]!=generated["output"][1]:
                raise ValueError("first rollout changed")
            key=f"train-{policy}-{index:02d}"
            raw,checked,receipt=checked_saved(context,key,consumed["samples"],"training")
            rewards=execution_rewards(context,raw,consumed["samples"],key,seed,arm)
            if consumed["rewards"]!=rewards or consumed["raw_hash"]!=fingerprint(raw):
                raise ValueError("optimizer consumed different rewards")
            ids.extend(checked["sandbox_ids"])
            receipts[key]=receipt
            previous=boundary["parameter_hash"]
            progress.advance()
        progress.stage("verify_fixed_evaluation_sources",total=160,unit="programs")
        for step in (12,24):
            evaluation=result["evaluations"][str(step)]
            target=directory/f"evaluation-{step:02d}"
            if (read_json(target/"result.json")!=evaluation
                    or read_json(directory/f"model-{step:02d}/receipt.json")!={
                        "step":step,"parameter_hash":evaluation["parameter_hash"]}):
                raise ValueError("evaluation boundary changed")
            for sample in evaluation["samples"]:
                sid=sample["sample_id"]
                if (read_json(target/"samples"/f"{sid}.json")!=sample
                        or read_json(target/"intents"/f"{sid}.json")!={
                            "sample_id":sid,"seed":sample["seed"],"parameter_hash":evaluation["parameter_hash"]}):
                    raise ValueError("fixed evaluation source journal changed")
                progress.advance()
    if len(ids)!=len(set(ids)):
        raise ValueError("training reused a sandbox")
    return {"sandbox_ids":ids,"batch_receipts":receipts,"all_rewards_replayed":True}


@app.function(image=cpu_image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,
              timeout=study.CONTROLLER_SECONDS,retries=0,max_containers=4,scaledown_window=2,volumes=VOLUMES)
def seed_controller(context,ticket,seed):
    started,plan=enter(context,ticket)
    require_controls(context)
    policies,arms,ids,training_receipts={},{},[],{}
    target=run_root(context)/"seeds"/str(seed)
    try:
        for arm in plan["arm_orders"][str(seed)]:
            policy=study.label(seed,arm)
            result=invoke(train_arm,"training/"+policy,"gpu",study.GPU_SECONDS,context,seed,arm)
            work.reload()
            if arms and result["first_tokens"]!=next(iter(arms.values()))["first_tokens"]:
                raise ValueError("seed-matched first rollout differs")
            arms[arm]=result
            verified=verify_arm_journals(context,seed,arm,result)
            ids.extend(verified["sandbox_ids"])
            training_receipts[arm]=verified["batch_receipts"]
            for step in (12,24):
                measured=evaluate_policy(context,policy,step,result["evaluations"][str(step)]["samples"])
                ids.extend(measured["sandbox_ids"])
                policies[f"{policy}-{step}"]=measured
        if len(ids)!=len(set(ids)):
            raise ValueError("sandbox reused across policies")
        result={"seed":seed,"policies":policies,"training_receipts":training_receipts,
                "training_metrics":{a:r["metrics"] for a,r in arms.items()},
                "sandbox_ids":ids,"all_first_rollouts_equal":True}
        persist(target,{"result":result})
        work.commit()
        backup(context,"seeds/"+str(seed),{"result":result})
        print("REPLICATION SEED COMPLETE",seed,flush=True)
        return response(result,started)
    except Exception as exc:
        record={"seed":seed,"type":type(exc).__name__,"detail":str(exc)[:2000],
                "completed_arms":list(arms),"completed_evaluations":list(policies)}
        persist(target,{f"interrupted-{time.time_ns()}":record})
        work.commit()
        backup(context,"interruptions",{f"seed-{seed}-{time.time_ns()}":record})
        raise


async def supervisor_preflight(context):
    plan,setup=context["plan"],context["setup"]
    directory=root(plan)
    document={"saved_source":context["regression_source"],"records":{}}
    backend=supervised.SupervisedPanelBackend(setup["app_name"],setup["sandbox_image_id"],study.pilot.BOOKING,
                                            creation_interval_seconds=.26)
    for name,(source,_,_,_) in supervisor_controls.controls(context["regression_source"]).items():
        require_deadline(context["deadline"])
        prefix=study.RUN_ID+"/supervisor/"+name
        record=await claims.get.aio(prefix+"/result",None)
        if record is None:
            if not await claims.put.aio(prefix+"/intent",{"source_hash":digest(source)},skip_if_exists=True):
                raise ReconciliationRequired("supervisor control outcome lost")
            record=pack_result(await backend.execute(request_for(source,supervisor_controls.case_for_control())))
            await claims.put.aio(prefix+"/result",record,skip_if_exists=True)
        document["records"][name]=record
        persist(directory/"supervisor_controls",{name:record})
        await work.commit.aio()
        print("REPLICATION SUPERVISOR",name,record["status"],record["detail"],flush=True)
    ids=supervisor_controls.validate_controls(document,setup["sandbox_image_id"])
    persist(directory,{"supervisor_controls":document})
    await work.commit.aio()
    backup(context,"",{"supervisor_controls":document})
    return ids


async def program_supervisor_preflight(context):
    directory=run_root(context)
    gate=program_execution.CreationGate(.30)
    backend=program_execution.ProgramBackend(context["setup"]["app_name"],
        context["setup"]["sandbox_image_id"],start_gate=gate)
    cases=program_benchmark.control_cases()
    ids,records,seconds=[],{},0
    for name,(source,_,_) in program_benchmark.controls().items():
        require_deadline(context["deadline"])
        prefix=namespace(context)+"/supervisor/"+name
        document=await claims.get.aio(prefix+"/result",None)
        intent={"source_hash":digest(source),"runtime":context["program_runtime"]}
        if document is None:
            if not await claims.put.aio(prefix+"/intent",intent,skip_if_exists=True):
                raise ReconciliationRequired("supervisor program intent unresolved; no repeat")
            document=await backend.execute_program(source,cases)
            await claims.put.aio(prefix+"/result",document,skip_if_exists=True)
            seconds+=min(program_execution.lifetime_for(len(cases)),math.ceil(document["metadata"]["total_seconds"]))
        if await claims.get.aio(prefix+"/intent")!=intent:
            raise ValueError("supervisor program identity changed")
        records[name]=document
        persist(directory/"supervisor_controls",{name:document})
        await work.commit.aio()
        if document["metadata"]["cleanup"]!="terminated":
            await claims.put.aio(namespace(context)+"/stop",{"reason":"cleanup_unconfirmed","control":name},skip_if_exists=True)
            raise ReconciliationRequired("cleanup unconfirmed in supervisor control")
        outcomes=program_execution.validate_program(document,source,cases,context["setup"]["sandbox_image_id"])
        program_benchmark.validate_control(name,outcomes)
        ids.append(document["metadata"]["sandbox_id"])
        print("PROGRAM SUPERVISOR CONTROL PASSED",name,flush=True)
    if len(ids)!=len(set(ids)):
        raise ValueError("supervisor programs shared a sandbox")
    document={"version":programs.VERSION,"records":records,"passed":True}
    persist(directory,{"supervisor_controls":document})
    await work.commit.aio()
    return ids,seconds


async def program_training_phase(context,*,on_policy=None):
    """Two policies at once; no long post-training audits while a GPU is paid."""
    plan=context["plan"]
    jobs=[(seed,plan["arm_orders"][str(seed)][index]) for index in range(len(study.ARMS)) for seed in study.SEEDS]
    async def train(job):
        seed,arm=job
        policy=study.label(seed,arm)
        result=await asyncio.to_thread(invoke,train_arm,"training/"+policy,"gpu",study.GPU_SECONDS,
                                       context,seed,arm)
        study.validate_arm(result,plan)
        print("PROGRAM TRAINING PHASE FINISHED POLICY",policy,flush=True)
        if on_policy is not None:
            on_policy(policy)
        return policy
    # On failure, preserve completed/cached arms, await the other active GPU,
    # and stop queued policies. There is no automatic replay or resampling.
    return await program_benchmark.bounded_map(jobs,train,plan["gpu_concurrency"])


@app.function(image=cpu_image,cpu=(1,1),memory=(2048,2048),nonpreemptible=True,
              timeout=study.CONTROLLER_SECONDS,retries=0,max_containers=1,scaledown_window=2,volumes=VOLUMES)
def run_study(context,ticket):
    started,plan=enter(context,ticket)
    directory=run_root(context)
    relative="" if program_mode(context) else runtime_relative(context)
    if relative:
        if read_json(directory/"plan.json")!=plan or read_json(directory/"setup.json")!=context["setup"]:
            raise ValueError("runtime amendment changed the experiment")
        if any((directory/name).exists() for name in ("recovery-control","arms","baseline","preflight.json")):
            raise ReconciliationRequired("journal-order amendment is restricted to the failed preflight")
        if not (directory/"grading/controls/raw.json").exists():
            raise ValueError("saved preflight raw evidence required")
    metadata_relative=(program_storage.metadata_relative(context) if context.get("storage_amendment") else relative)
    persist(directory/metadata_relative,{"context":context,"plan":plan,"source_snapshot":context["source_snapshot"],"setup":context["setup"]})
    work.commit()
    backup(context,metadata_relative,{"context":context,"plan":plan,"source_snapshot":context["source_snapshot"],"setup":context["setup"]})
    if program_mode(context):
        ids,supervisor_seconds=asyncio.run(program_supervisor_preflight(context))
    else:
        ids=asyncio.run(supervisor_preflight(context))
        supervisor_seconds=12*120
    graded=grade(context,"controls",study.pilot.controls(plan["experiment"]),"training")
    ids.extend(graded["checked"]["sandbox_ids"])
    preflight=study.validate_controls(graded["checked"])
    persist(directory,{"preflight":preflight})
    work.commit()
    backup(context,"",{"preflight":preflight})
    print("REPLICATION GRADER PREFLIGHT PASSED",preflight,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(grade,context,f"parallel-controls-{i}",study.parallel_controls(plan,i),"training") for i in range(4)]
        parallel_results=[future.result() for future in futures]
    for graded_control in parallel_results:
        study.validate_controls(graded_control["checked"])
        ids.extend(graded_control["checked"]["sandbox_ids"])
    if len(ids)!=len(set(ids)):
        raise ValueError("parallel controls reused a sandbox")
    persist(directory,{"parallel_preflight":{"passed":True,"batches":4,"distinct_sandbox_ids":len(ids)}})
    work.commit()
    backup(context,"",{"parallel_preflight":read_json(directory/"parallel_preflight.json")})
    print("REPLICATION PARALLEL PREFLIGHT PASSED",4,flush=True)
    if program_mode(context):
        # Exercise the actual service a second time, bypassing the caller cache.
        # It must recheck evidence and spend zero new sandbox seconds.
        replay_key="parallel-controls-0"
        replay=invoke(grade_program_batch,"preflight-reuse","cpu",programs.GRADE_SECONDS,context,
                      replay_key,study.parallel_controls(plan,0),"training",True,sandbox_seconds=0)
        if replay!=parallel_results[0]:
            raise ValueError("completed program reuse changed grading")
        persist(directory,{"program_reuse_preflight":{"passed":True,"reused_batch":replay_key,
            "new_sandbox_reservation_seconds":0,"service_concurrency":1,"program_concurrency":4}})
        work.commit()
        print("PROGRAM SERVICE AND REUSE PREFLIGHT PASSED",flush=True)
    parts={}
    for part in plan["control_parts"]:
        parts[part]=invoke(control_part,"control/"+part,"gpu",study.CONTROL_SECONDS,context,part)
    control=old_release.validate_control(parts,plan)
    persist(directory/"recovery-control",{"result":control})
    work.commit()
    backup(context,"recovery-control",{"result":control})
    print("REPLICATION FULL-STATE CONTROL PASSED",control,flush=True)
    generated=invoke(baseline,"baseline","gpu",study.BASELINE_SECONDS,context)
    baseline_result=evaluate_policy(context,"baseline",0,generated["samples"],parallel=True)
    ids.extend(baseline_result["sandbox_ids"])
    if program_mode(context):
        print("PROGRAM STUDY PHASE training_all_policies",flush=True)
        with ProgressLog("all-policies",label="TRAINING_PHASE") as progress:
            progress.stage("train_before_post_training_audits",total=12,unit="policies")
            def completed_policy(policy):
                progress.item(policy)
                progress.advance()
            policies=asyncio.run(program_training_phase(context,on_policy=completed_policy))
        persist(directory,{"training_phase":{"completed_policies":policies,"no_post_training_audits_during_training":True}})
        work.commit()
        print("PROGRAM STUDY PHASE grade_saved_evaluation_samples",flush=True)
    results,failures={},{}
    # At most four seed controllers and two GPU workers. Program mode has one
    # grading service with four inner programs (legacy mode had four workers).
    # Each controller owns distinct directories; no shared-file append or reload
    # occurs in these local invocation threads.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(invoke,seed_controller,f"seed/{s}","cpu",study.CONTROLLER_SECONDS,context,s):s for s in study.SEEDS}
        for future in as_completed(futures):
            seed=futures[future]
            try:
                results[str(seed)]=future.result()
            except Exception as exc:
                failures[str(seed)]={"type":type(exc).__name__,"detail":str(exc)[:2000]}
                print("REPLICATION SEED INTERRUPTED",seed,failures[str(seed)],flush=True)
    work.reload()
    if failures:
        result={"status":"incomplete","completed_seeds":list(results),"failures":failures,
                "no_partial_seed_selection":True,"budget":budget.remote("status",context)}
    else:
        policies={}
        for seed in study.SEEDS:
            saved=read_json(directory/f"seeds/{seed}/result.json")
            if saved!=results[str(seed)]:
                raise ValueError("seed result changed")
            ids.extend(saved["sandbox_ids"])
            policies.update(saved["policies"])
        if len(ids)!=len(set(ids)):
            raise ValueError("global sandbox identity reused")
        result={"status":"completed_with_uncertainty" if any(p["summary"]["unknown_inputs"] for p in policies.values())
                    or baseline_result["summary"]["unknown_inputs"] else "completed",
                "analysis":study.analyze(baseline_result,policies),
                "training_metrics":{s:r["training_metrics"] for s,r in results.items()},
                "unique_sandboxes":len(ids),"budget":budget.remote("status",context),
                "pilot_kept_separate":True,"plan_hash":fingerprint(plan)}
    result_name=("result-"+ticket.rsplit("/",1)[-1]
                 if program_mode(context) and result["status"]=="incomplete" else "result")
    persist(directory,{result_name:result})
    work.commit()
    backup(context,"",{result_name:result})
    print("REPLICATION STUDY FINISHED",result["status"],flush=True)
    # Controls' maximum lifetime remains charged conservatively to the root.
    return response(result,started,supervisor_seconds)


def preflight_resume_context(previous,snapshot,apps):
    """One reviewed pre-training amendment; preserve protocol, deadline and holds."""
    import ast
    parent="ap-viF7pPM44WU8sRtjGICMNt"
    if not any(a["app_id"]==parent and a["state"]=="stopped" and int(a["tasks"])==0 for a in apps):
        raise ReconciliationRequired("failed parent app must be terminal before resuming")
    if any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("active jobs prohibit preflight amendment")
    original=previous["source_snapshot"]
    changed={n for n in snapshot.keys() | original.keys() if snapshot.get(n)!=original.get(n)}
    if changed!={"modal_booking_replication.py"}:
        raise ValueError("only the reviewed launcher repair may change")
    allowed={"invoke","require_controls","grade_batch","run_study","main",
             "runtime_relative","verify_batch_journals","preflight_resume_context"}
    def unchanged(source):
        return [ast.dump(n) for n in ast.parse(source).body
                if not (isinstance(n,ast.FunctionDef) and n.name in allowed)]
    if unchanged(original["modal_booking_replication.py"])!=unchanged(snapshot["modal_booking_replication.py"]):
        raise ValueError("amendment changed model, GRPO, generation, grading, or budget helpers")
    study.validate_plan(previous["plan"])
    require_deadline(previous["deadline"])
    if previous.get("runtime_amendment"):
        raise ValueError("amendment already applied")
    return dict(previous,source_snapshot=snapshot,runtime_amendment={
        "id":"journal-order-001","parent_app":parent,"parent_snapshot_hash":fingerprint(original),
        "scope":"canonical saved-JSON traversal and metadata-only resume; no research parameter changes",
        "reuse_saved_controls":True,"new_budget":False,"extend_deadline":False})


def program_context(previous,benchmark_context,benchmark_result,snapshot,apps,*,now=None):
    """New runtime namespace, same scientific plan and original budget binding."""
    now=time.time() if now is None else now
    study.validate_plan(previous["plan"])
    if any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("active jobs prohibit execution-runtime amendment")
    if (benchmark_result.get("status")!="passed" or benchmark_result.get("controls")!=13
            or benchmark_result.get("unique_sandboxes")!=121
            or benchmark_result.get("full_study_started") is not False
            or benchmark_result.get("research_updates")!=0 or benchmark_result.get("model_samples")!=0
            or benchmark_result.get("manifest")!=program_benchmark.manifest(benchmark_context["saved_pool"])
            or benchmark_context["manifest"]!=benchmark_result["manifest"]):
        raise ValueError("completed, unchanged runtime benchmark required")
    # The model/GRPO implementations and task definitions are not part of this
    # amendment. Every historical library file must be byte-for-byte identical.
    for name,source in previous["source_snapshot"].items():
        if name!="modal_booking_replication.py" and snapshot.get(name)!=source:
            raise ValueError("runtime amendment changed historical library: "+name)
    for name in ("verifier_rl/program_execution.py","verifier_rl/program_benchmark.py"):
        if snapshot.get(name)!=benchmark_context["snapshot"][name]:
            raise ValueError("benchmarked execution contract changed")
    import ast
    def functions(source):
        return {n.name:ast.dump(n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    before,after=(functions(s["modal_booking_replication.py"]) for s in (previous["source_snapshot"],snapshot))
    for name in ("build_trainer","evaluation_at_boundary","runtime"):
        if before[name]!=after[name]:
            raise ValueError("runtime amendment changed model/GRPO mechanics: "+name)
    deadline=now+study.CONTROLLER_SECONDS-300
    setup=dict(benchmark_context["setup"],sandbox_image_id=benchmark_context["setup"]["program_image_id"],
               parent_sandbox_image_id=previous["setup"]["sandbox_image_id"])
    original={k:previous[k] for k in ("plan","rates","billing_before","deadline")}
    return dict(previous,setup=setup,source_snapshot=snapshot,deadline=deadline,budget_context=original,
        program_runtime=programs.binding(setup["sandbox_image_id"],deadline),
        runtime_amendment={"id":programs.AMENDMENT,"namespace":programs.NAMESPACE,
            "parent_snapshot_hash":fingerprint(previous["source_snapshot"]),
            "benchmark_result_hash":fingerprint(benchmark_result),"benchmark_run":program_benchmark.RUN_ID,
            "original_budget_binding":fingerprint(original),"new_budget":False,
            "historical_stop_retained":True,"original_model_initialization":True,
            "new_runtime_window_seconds":study.CONTROLLER_SECONDS-300,
            "effective_grading_functions":1,"effective_program_concurrency":4,
            "phase_order":["controls","baseline","train_all_policies","grade_saved_evaluations"],
            "fixed_evaluation_draws_still_generated_at_updates":[12,24],
            "automatic_relaunch":False,"reuse_historical_grades":False})


def program_launch(directory,snapshot,apps,*,resume=False,storage_resume=False):
    directory=directory/programs.RELATIVE
    owner=None
    if resume:
        context_path=directory/("storage-amendments/"+program_storage.AMENDMENT+"/"+program_storage.RESUME_REVISION if storage_resume else "")/"context.json"
        context=read_json(context_path)
        if not program_mode(context) or context["source_snapshot"]!=snapshot:
            raise ValueError("resume requires the exact frozen runtime source")
        require_deadline(context["deadline"])
        require_program_authorization(context)
        if claims.get(programs.NAMESPACE+"/stop",None) is not None:
            raise ReconciliationRequired("runtime stop still requires explicit reconciliation")
    else:
        if (directory/"launch_intent.json").exists():
            raise ReconciliationRequired("program study already submitted; inspect or explicitly resume")
        previous=read_json(Path("runs")/study.RUN_ID/"context.json")
        benchmark_directory=Path("runs")/program_benchmark.RUN_ID
        benchmark_context=read_json(benchmark_directory/"context.json")
        benchmark_result=read_json(benchmark_directory/"result.json")
        benchmark_volume=modal.Volume.from_name("verifier-rl-program-benchmark",create_if_missing=False)
        for name,expected in (("context",benchmark_context),("result",benchmark_result)):
            if json.loads(b"".join(benchmark_volume.read_file(program_benchmark.RUN_ID+f"/{name}.json")))!=expected:
                raise ValueError("benchmark cloud evidence differs from local copy")
        context=program_context(previous,benchmark_context,benchmark_result,snapshot,apps)
        ledger=claims.get(study.RUN_ID+"/budget",None)
        item=(ledger or {}).get("ledger",{}).get("items",{}).get(program_benchmark.RUN_ID)
        if (ledger is None or ledger["binding"]!=context["runtime_amendment"]["original_budget_binding"]
                or ledger["ledger"]["ceiling"]!="250" or not item or item["actual"] is None
                or item["identity"]!=benchmark_context["reservation"]["identity"]):
            raise ValueError("existing compute ledger/settled benchmark required; never reset it")
        # Reject price drift before spending under the saved rate assumptions.
        rates=read_modal("billing","rates")
        for kind in ("cpu","sandbox","gpu"):
            if accounting.hourly(rates,kind)!=accounting.hourly(context["rates"],kind):
                raise ValueError("published resource rates changed; review reservations")
        owner=modal.App.lookup(context["setup"]["app_name"],create_if_missing=False)
        if list(modal.Sandbox.list(app_id=owner.app_id)):
            raise ReconciliationRequired("candidate sandboxes still active; cannot change runtime")
        authorization={"context_hash":fingerprint(context),"previous_budget_binding":ledger["binding"],
            "previous_committed_usd":str(accounting.committed(ledger["ledger"])),
            "previous_stop":claims.get(study.RUN_ID+"/stop",None),
            "previous_maintenance_owner":claims.get(study.RUN_ID+"/maintenance-owner",None)}
        if not claims.put(programs.NAMESPACE+"/authorization",authorization,skip_if_exists=True):
            raise ReconciliationRequired("runtime already authorized; do not submit it twice")
        persist(directory,{"context":context,"source_snapshot":snapshot,"authorization":authorization,
            "launch_intent":{"apps":apps,"deadline":context["deadline"],"compute_ceiling_usd":"250",
                             "new_budget":False,"old_stop_retained":True}})
    if owner is None:
        owner=modal.App.lookup(context["setup"]["app_name"],create_if_missing=False)
        if list(modal.Sandbox.list(app_id=owner.app_id)):
            raise ReconciliationRequired("active sandboxes prevent reviewed continuation")
    print("PROGRAM RUNTIME AMENDMENT",context["runtime_amendment"],flush=True)
    result=invoke(run_study,"controller","cpu",study.CONTROLLER_SECONDS,context,
        sandbox_seconds=len(program_benchmark.controls())*program_execution.lifetime_for(2))
    if result["status"]=="incomplete":
        persist(directory,{f"interrupted-{time.time_ns()}":{"result":result,"budget_after":budget.remote("status",context)}})
    else:
        persist(directory,{"result":result,"budget_after":budget.remote("status",context)})
    print("PROGRAM STUDY RESULT",result["status"],flush=True)


@app.local_entrypoint()
def main(allow_cloud: bool=False, resume_preflight: bool=False, program_runtime: bool=False, resume_program: bool=False,
         resume_storage: bool=False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required; this launches the approved fixed study")
    directory=Path("runs")/study.RUN_ID
    if resume_storage and not (program_runtime and resume_program):
        raise ValueError("storage resume requires --program-runtime --resume-program")
    if resume_program and not program_runtime or program_runtime and resume_preflight:
        raise ValueError("choose one reviewed runtime path")
    if (directory/"launch_intent.json").exists() and not (resume_preflight or program_runtime):
        raise ReconciliationRequired("study already submitted; inspect/resume explicitly, do not relaunch")
    apps=read_modal("app","list")
    if any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("other Modal jobs active; isolate the study's budget and resources")
    plan=study.make_plan(Path.cwd())
    setup=read_json(Path("runs/qwen-booking-matched-training-20260929-v1/setup.json"))
    path=Path("runs/booking-boundary-contrast-20260929-v1/evidence/qwen-booking-reward-shape-20260928-v1/arms/linear/rollouts/train-linear-14.json")
    regression_source=next(s["source"] for s in read_json(path) if s["sample_id"]==supervisor_controls.FAILED_SAMPLE)
    supervisor_controls.controls(regression_source)
    names=sorted(Path("verifier_rl").glob("*.py"))+[Path(n) for n in LAUNCHERS]
    snapshot={p.as_posix():p.read_text() for p in names}
    if program_runtime:
        return program_launch(directory,snapshot,apps,resume=resume_program,storage_resume=resume_storage)
    if resume_preflight:
        context=preflight_resume_context(read_json(directory/"context.json"),snapshot,apps)
        directory=directory/runtime_relative(context)
        if (directory/"launch_intent.json").exists():
            raise ReconciliationRequired("reviewed preflight resume already submitted")
    else:
        context={"plan":plan,"source_snapshot":snapshot,"setup":setup,"regression_source":regression_source,
            "rates":read_modal("billing","rates"),"billing_before":read_modal("billing","summary"),
            "deadline":time.time()+study.CONTROLLER_SECONDS-300}
    persist(directory,{"context":context,"plan":plan,"source_snapshot":snapshot,
                       "launch_intent":{"deadline":context["deadline"],"apps":apps,"authorized_compute_usd":"250"}})
    print("FROZEN REPLICATION",fingerprint(plan),study.workload(),flush=True)
    result=invoke(run_study,"controller","cpu",study.CONTROLLER_SECONDS,context,sandbox_starts=12)
    persist(directory,{"result":result,"budget_after":budget.remote("status",context)})
    print("STUDY RESULT",result["status"],flush=True)
