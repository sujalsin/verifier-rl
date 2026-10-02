"""Separate evaluation-only scheduling amendment; canary NEVER stops old work."""
import asyncio
from decimal import Decimal
import json
from pathlib import Path
import time

import modal

from verifier_rl import parallel_evaluation as parallel, program_execution as execution
from verifier_rl import program_grading as previous, program_storage as storage
from verifier_rl import booking_replication as study, compute_budget as accounting
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import digest

PARENT_APP = "ap-5UDmaoMEAolF6ZLohphRXd"
PARENT_BUDGET = "fu-H0oP0qLjHWwF0fFGkfqNFb"
PARENT_CONTEXT = (f"{study.RUN_ID}/{previous.RELATIVE}/storage-amendments/"
                  "evidence-storage-001/control-source-001/context.json")
PREFIX = previous.NAMESPACE + "/" + parallel.AMENDMENT
RELATIVE = f"{study.RUN_ID}/{previous.RELATIVE}/{parallel.AMENDMENT}"
FROZEN = ("booking_replication.py", "booking_matched_training.py", "booking_baseline_comparison.py",
          "booking_boundary_contrast.py", "booking_reward_pilot.py", "booking_verifier_v2.py",
          "program_execution.py", "program_storage.py", "program_benchmark.py", "grading.py",
          "suites.py", "supervised_execution.py", "modal_backend.py")
app = modal.App("verifier-rl-parallel-evaluation")
work = modal.Volume.from_name("verifier-rl-booking-replication-work", create_if_missing=False)
archive = modal.Volume.from_name("verifier-rl-booking-replication-evidence", create_if_missing=False)
store = modal.Dict.from_name("verifier-rl-booking-replication-journal", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
         .add_local_python_source("verifier_rl").add_local_file(__file__, "/root/modal_parallel_evaluation.py"))


def validate_config(config):
    parallel.validate_runtime(config["runtime"])
    study.validate_plan(config["plan"])
    if config["parent_app"] != PARENT_APP or config["case_hashes"] != [c.input_hash for c in study.pilot.cases_for("evaluation")]:
        raise ValueError("wrong parent or evaluation inputs")
    root = Path(__file__).parent
    for name, expected in config["scientific_sources"].items():
        if digest((root/name).read_text()) != expected:
            raise ValueError("frozen source changed: " + name)
    for name, expected in config["scheduler_sources"].items():
        if digest((root/name).read_text()) != expected:
            raise ValueError("untested scheduler source changed: " + name)


def paths(config, base="/artifacts"):
    return Path(base)/RELATIVE/config["name"]


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=60, retries=0, max_containers=1, scaledown_window=2)
def coordinate(config, action, payload):
    """The only writer of this namespace; no concurrent-input decorator."""
    validate_config(config)
    key = PREFIX + "/" + config["name"] + "/state"
    state = store.get(key, None)
    if action == "initialize":
        initial = parallel.initialize(config["runtime"], payload["jobs"])
        if state is not None:
            raise ReconciliationRequired("scheduler already initialized; inspect instead of restarting")
        if not store.put(key, initial, skip_if_exists=True):
            raise ReconciliationRequired("scheduler ownership raced")
        return True
    if state is None:
        raise ValueError("missing scheduler manifest")
    if action == "permit" and state["last_permit"] is not None:
        delay = state["last_permit"] + config["runtime"]["creation_interval"] - time.time()
        if delay > 0: time.sleep(delay)
    state, answer = parallel.transition(state, action, payload, time.time())
    if action != "status":
        store.put(key, state)  # RPC retries happen only for idempotent final publications.
    return answer


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=parallel.MAX_SECONDS, retries=0, max_containers=parallel.WORKERS,
              scaledown_window=2, volumes={"/artifacts": work, "/evidence": archive})
def grade_parallel(config, manifest, *, publication_fault=False, publish_only=False):
    validate_config(config)
    work.reload()
    began = time.time()
    directory = paths(config)/"grading"/manifest["key"]
    owner = parallel.fingerprint({"config": config, "manifest": manifest})
    fault_count = 0
    async def rpc(action, payload):
        nonlocal fault_count
        if (publication_fault and action == "program"
                and payload["sample_id"] == manifest["samples"][0]["sample_id"]):
            if not (directory/"programs"/payload["sample_id"]/"result.json").exists():
                raise AssertionError("publication preceded durable evidence")
            fault_count += 1
            raise OSError("injected isolated final-index outage")
        return await coordinate.remote.aio(config, action, payload)
    async def backup(sid, intent, document):
        persist(paths(config, "/evidence")/"grading"/manifest["key"] / sid,
                {"intent": intent, "result": document})
        await archive.commit.aio()
    class NoExecution:
        async def execute_program(self, *args, **kwargs):
            raise AssertionError("publication recovery must not reexecute candidate code")
    def backend(gate):
        return NoExecution() if publish_only else execution.ProgramBackend(
            config["setup"]["app_name"], config["runtime"]["image_id"], start_gate=gate)
    with ProgressLog(manifest["key"], label="PARALLEL_GRADING") as progress:
        progress.stage("grade_fixed_programs", total=4*len(manifest["case_hashes"]), unit="inputs")
        last = 0
        def progressed(stats):
            nonlocal last
            progress.advance(stats["inputs"]-last)
            last = stats["inputs"]
        try:
            result = asyncio.run(parallel.execute(manifest, study.pilot.cases_for("evaluation"), directory,
                backend, rpc, work.commit.aio, backup, owner=owner, publish_only=publish_only, progress=progressed))
        except storage.StorageFailure:
            if not publication_fault or fault_count != storage.ATTEMPTS:
                raise
            # Check all four peers drained and their exact protected results exist.
            hashes = {s["sample_id"]: parallel.fingerprint(json.loads(
                (directory/"programs"/s["sample_id"]/"result.json").read_text())) for s in manifest["samples"]}
            receipt = {"injected_failures": fault_count, "document_hashes": hashes,
                       "started": began, "ended": time.time(), "publication_failed_closed": True}
            persist(directory, {"fault": receipt})
            work.commit()
            print("PARALLEL CANARY INJECTED STORAGE FAILURE PRESERVED", manifest["key"], flush=True)
            return receipt
    result.update(started=began, ended=time.time())
    if publish_only:
        fault = json.loads((directory/"fault.json").read_text())
        hashes = {s["sample_id"]: parallel.fingerprint(json.loads(
            (directory/"programs"/s["sample_id"]/"result.json").read_text())) for s in manifest["samples"]}
        if hashes != fault["document_hashes"] or result["stats"]["new_programs"] or result["sandbox_seconds"] != "0":
            raise ValueError("publication recovery changed results or reran code")
    print("PARALLEL BATCH COMPLETE", manifest["key"], result["stats"], flush=True)
    return result


def control_jobs(config):
    controls = study.pilot.controls(config["plan"]["experiment"])
    sources = [s["source"] for s in controls] + ["def required_capacity(bookings): raise ValueError('authored control')"]
    cases = study.pilot.cases_for("evaluation")
    jobs = []
    for n in range(5):
        samples = [study.pilot.original.sample_from_text(source, sid=f"parallel-control-{n}-{i}", seed=None,
            plan=config["plan"]["experiment"], tokens=0, eos=True) for i, source in enumerate(sources)]
        jobs.append(parallel.job(samples, cases, f"canary-{n:02d}", config["plan"], config["runtime"], control=True))
    return jobs


@app.function(image=image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=parallel.MAX_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": work, "/evidence": archive})
def canary(config):
    validate_config(config)
    owner_key = PREFIX + "/" + config["name"] + "/controller"
    if not store.put(owner_key, {"config_hash": parallel.fingerprint(config)}, skip_if_exists=True):
        raise ReconciliationRequired("canary already started; never repeat blindly")
    work.reload()
    persist(paths(config), {"config": config})
    work.commit()
    jobs = control_jobs(config)
    coordinate.remote(config, "initialize", {"jobs": jobs})
    with ProgressLog(config["name"], label="PARALLEL_CANARY") as progress:
        progress.stage("serial_control_batch", total=1, unit="batches")
        serial = grade_parallel.remote(config, jobs[0])
        progress.advance()
        progress.stage("four_parallel_control_batches", total=4, unit="batches")
        async def run():
            async def one(j):
                result = await grade_parallel.remote.aio(config, j, publication_fault=j["key"] == "canary-01")
                progress.advance()
                return result
            return await asyncio.gather(*(one(j) for j in jobs[1:]), return_exceptions=True)
        values = asyncio.run(run())
        failures = [v for v in values if isinstance(v, BaseException)]
        if failures: raise failures[0]  # all peers already drained
        fault = values[0]
        if fault.get("injected_failures") != 3:
            raise ValueError("missing live fault test")
        progress.stage("publication_only_recovery", total=1, unit="batches")
        values[0] = grade_parallel.remote(config, jobs[1], publish_only=True)
        progress.advance()
        status = coordinate.remote(config, "status", {})
        if status != {"batches": 5, "total": 5, "permits": 20, "unknown": 0, "stop": None}:
            raise ValueError("unknown, duplicate or missing control execution")
        if any(v["outcome_hashes"] != serial["outcome_hashes"] for v in values):
            raise ValueError("parallel execution changed semantic outcomes")
        windows = [(fault["started"], fault["ended"])] + [(v["started"],v["ended"]) for v in values[1:]]
        peak = max(sum(start <= point < end for start,end in windows) for point in [s for s,e in windows])
        if peak < 2: raise ValueError("cloud did not actually overlap batches")
        result = {"passed": True, "serial_parallel_outcomes_equal": True, "overlapping_batches": peak,
                  "input_cases_per_program": len(config["case_hashes"]), "control_programs": 20,
                  "publication_failures_injected": 3, "recovery_candidate_reexecutions": 0,
                  "new_sandboxes_during_recovery": 0, "coordinator": status,
                  "serial_batch_seconds": serial["ended"]-serial["started"],
                  "parallel_four_batch_seconds": max(e for s,e in windows)-min(s for s,e in windows),
                  "legacy_app_untouched": PARENT_APP, "config_hash": parallel.fingerprint(config)}
        persist(paths(config), {"result": result})
        work.commit()
        persist(paths(config, "/evidence"), {"result": result, "config": config})
        archive.commit()
        store.put(PREFIX+"/"+config["name"]+"/result", result, skip_if_exists=True)
        print("PARALLEL CANARY PASSED", json.dumps(result, sort_keys=True), flush=True)
        return result


def config_for(parent, name, mode="canary"):
    here = Path(__file__).parent
    names = ["verifier_rl/"+n for n in FROZEN]
    scientific = {n: digest(parent["source_snapshot"][n]) for n in names}
    for n,h in scientific.items():
        if digest((here/n).read_text()) != h: raise ValueError("original scientific source differs: "+n)
    return {"name": name, "parent_app": PARENT_APP, "parent_context_hash": parallel.fingerprint(parent),
            "plan": parent["plan"], "setup": parent["setup"], "rates": parent["rates"],
            "runtime": parallel.runtime(parent["program_runtime"], mode,
                min(time.time()+parallel.MAX_SECONDS, parent["deadline"]) if mode == "canary" else parent["deadline"]+43200),
            "scientific_sources": scientific,
            "scheduler_sources": {n:digest((here/n).read_text()) for n in
                ("modal_parallel_evaluation.py", "verifier_rl/parallel_evaluation.py")},
            "case_hashes": [c.input_hash for c in study.pilot.cases_for("evaluation")]}


async def reserve_canary(parent, ticket, identity, *, client=None):
    """Pinned SDK adapter to the LIVE single writer, never a second budget writer."""
    from modal.client import _Client
    from modal._functions import _Function
    from modal_proto import api_pb2
    if modal.__version__ != "1.5.5": raise ValueError("budget adapter requires pinned SDK")
    client = client or await _Client.from_env()
    layout = (await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=PARENT_APP),
                                            timeout=30, retry=None)).app_layout
    if layout.function_ids.get("budget") != PARENT_BUDGET:
        raise ValueError("live serialized budget service changed")
    matches = [o for o in layout.objects if o.object_id == PARENT_BUDGET and o.HasField("function_handle_metadata")]
    if len(matches) != 1: raise ValueError("missing budget service metadata")
    budget = _Function._new_hydrated(PARENT_BUDGET, client, matches[0].function_handle_metadata)
    return await budget.remote("reserve", parent, ticket, str(parallel.CANARY_CEILING), identity)


@app.local_entrypoint()
def main(name: str = "canary-001"):
    if name != "canary-001": raise ValueError("new attempts require review")
    parent = json.loads((Path("runs")/PARENT_CONTEXT).read_text())
    config = config_for(parent, name)
    identity = parallel.fingerprint(config)
    ticket = parallel.AMENDMENT+"/"+name
    if store.get(PREFIX+"/"+name+"/controller",None) is not None:
        raise ReconciliationRequired("canary already started")
    held = asyncio.run(reserve_canary(parent, ticket, identity))
    persist(Path("runs")/RELATIVE/name, {"config":config,"budget":{"ticket":ticket,"identity":identity,"held_usd":held["committed_usd"]}})
    print("PARALLEL CANARY RESERVED within original ceiling", held["committed_usd"], flush=True)
    call = canary.spawn(config)
    persist(Path("runs")/RELATIVE/name, {"launch":{"call_id":call.object_id,"parent_app_untouched":PARENT_APP}})
    print("PARALLEL CANARY CALL",call.object_id,flush=True)
    result = call.get()
    persist(Path("runs")/RELATIVE/name, {"result":result})
    print(json.dumps(result, indent=2))
