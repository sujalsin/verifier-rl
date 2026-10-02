"""Bounded baseline phase of the original-policy verifier comparison. No RL launch."""

import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import modal

from modal_booking_study import (artifacts, claims, cpu_image as base_cpu, gpu_image as base_gpu,
                                load_policy, parameter_hash, require_deadline)
from verifier_rl import booking_baseline_comparison as study
from verifier_rl import supervised_execution as supervised, supervisor_controls
from verifier_rl.durable_grading import execute_batch
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.panel_execution import pack_result, request_for
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import canonical_json, digest

app = modal.App("verifier-rl-booking-original-baseline")


def with_launchers(image):
    for name in ("modal_booking_study.py", "modal_booking_baseline_comparison.py"):
        image = image.add_local_file(name, "/root/" + name)
    return image


cpu_image = with_launchers(base_cpu)
gpu_image = with_launchers(base_gpu)


def read_json(path):
    return json.loads(Path(path).read_text())


def check_snapshot(snapshot, root):
    if not {"modal_booking_baseline_comparison.py", "verifier_rl/booking_baseline_comparison.py",
            "verifier_rl/durable_grading.py", "verifier_rl/supervised_execution.py"}.issubset(snapshot):
        raise ValueError("source snapshot incomplete")
    for name, value in snapshot.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py" or (root / path).read_text() != value:
            raise ValueError("frozen source changed: " + name)


def claim_start(key, maximum):
    for slot in range(maximum):
        if claims.put(study.RUN_ID + f"/starts/{key}/{slot}", {"utc": datetime.now(timezone.utc).isoformat()},
                      skip_if_exists=True):
            return slot
    raise ReconciliationRequired("bounded invocation limit reached: " + key)


def validate_generation(saved, plan):
    if (saved["parameter_hash"] != plan["initial_parameter_hash"] or saved["parameters_unchanged"] is not True
            or saved["initial_checkpoint"] is not None or saved["optimizer_updates"] != 0
            or saved["plan_hash"] != digest(canonical_json(plan)) or len(saved["samples"]) != len(study.SEEDS)):
        raise ValueError("baseline generation identity differs")
    for i in range(len(study.SEEDS)//study.GROUP):
        study.validate_batch(f"eval-baseline-00-{i:02d}", saved["samples"][4*i:4*i+4], "evaluation", plan)


@app.function(image=cpu_image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=study.GRADE_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def grade_batch(key, samples, role, plan, setup, deadline):
    study.validate_batch(key, samples, role, plan)
    require_deadline(deadline)
    directory = Path("/artifacts") / plan["run_id"] / "grading" / key
    artifacts.reload()
    if (directory / "result.json").exists():
        result = read_json(directory / "result.json")
        if result["checked"] != study.verify_raw(result["raw"], samples, key, role, plan, setup["sandbox_image_id"]):
            raise ValueError("saved batch changed")
        print("REUSED GRADED BATCH", key, flush=True)
        return result
    claim_start("grading/" + key, plan["max_grading_starts_per_batch"])
    backend = supervised.SupervisedPanelBackend(setup["app_name"], setup["sandbox_image_id"], study.BOOKING,
                                                creation_interval_seconds=plan["creation_interval_seconds"])
    try:
        entries, progress = asyncio.run(execute_batch(samples, study.cases_for(role), directory, key, plan,
            setup["sandbox_image_id"], deadline, backend, claims, artifacts.commit.aio))
        raw = {"key": key, "role": role, "samples": samples, "entries": entries,
               "plan_hash": digest(canonical_json(plan))}
        persist(directory, {"raw": raw})
        artifacts.commit()
        checked = study.verify_raw(raw, samples, key, role, plan, setup["sandbox_image_id"])
        result = {"raw": raw, "checked": checked}
        persist(directory, {"result": result, "execution_progress": progress})
        artifacts.commit()
        print("BASELINE GRADED", key, [(r["reference"]["passed_bounds"], r["endpoint_omission"]["passed_bounds"],
            r.get("audit", {}).get("passed_bounds")) for r in checked["rows"]], flush=True)
        return result
    except Exception as exc:
        persist(directory, {f"interrupted-{time.time_ns()}": {"type": type(exc).__name__, "detail": str(exc)[:2000]}})
        artifacts.commit()
        raise


@app.function(image=gpu_image, gpu="L40S", cpu=(2,2), memory=(32768,32768),
              timeout=study.GPU_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def generate_baseline(plan, setup, deadline):
    import torch
    from transformers import set_seed
    study.validate_plan(plan)
    require_deadline(deadline)
    directory = Path("/artifacts") / plan["run_id"] / "generation"
    artifacts.reload()
    if (directory / "result.json").exists():
        result = read_json(directory / "result.json")
        validate_generation(result, plan)
        return result
    claim_start("generation", plan["max_gpu_starts"])
    if read_json(directory.parent / "preflight.json")["passed"] is not True:
        raise ValueError("live controls must pass before generation")
    supervisor_controls.validate_controls(read_json(directory.parent / "supervisor_controls.json"), setup["sandbox_image_id"])
    model, tokenizer = load_policy(plan)
    model.requires_grad_(False)
    model.eval()
    persist(directory, {"runtime": {"gpu": torch.cuda.get_device_name(), "gpu_image_id": gpu_image.object_id,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "parameter_dtype": "float32", "autocast_dtype": "bfloat16", "optimizer": None,
        "initial_checkpoint": None, "revision": plan["revision"]}})
    artifacts.commit()
    formatted = tokenizer.apply_chat_template([{"role":"user", "content":plan["prompt"]}],
                                              tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")
    samples = []
    for sid, seed in study.identities():
        require_deadline(deadline)
        intent = {"sample_id":sid, "seed":seed, "parameter_hash":plan["initial_parameter_hash"]}
        prefix = plan["run_id"] + "/generation/" + sid
        if not claims.put(prefix + "/intent", intent, skip_if_exists=True):
            raise ReconciliationRequired("generation already submitted; no replacement draw")
        persist(directory / "sample-intents", {sid:intent})
        artifacts.commit()
        set_seed(seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = model.generate(**inputs, max_new_tokens=plan["max_completion_tokens"], do_sample=True,
                temperature=plan["temperature"], top_p=plan["top_p"], top_k=plan["top_k"],
                repetition_penalty=plan["repetition_penalty"], pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True)
        completion = tokens[0, inputs["input_ids"].shape[1]:]
        sample = study.original.sample_from_text(tokenizer.decode(completion, skip_special_tokens=True),
            sid=sid, seed=seed, plan=plan, tokens=len(completion),
            eos=bool(len(completion) and completion[-1].item() == tokenizer.eos_token_id))
        if not claims.put(prefix + "/sample", sample, skip_if_exists=True):
            raise ReconciliationRequired("generated sample already recorded")
        persist(directory / "samples", {sid:sample})
        artifacts.commit()
        samples.append(sample)
        print("ORIGINAL BASELINE GENERATED", sid, "tokens", len(completion), flush=True)
    after = parameter_hash(model)
    result = {"samples":samples, "parameter_hash":after, "parameters_unchanged":after == plan["initial_parameter_hash"],
              "initial_checkpoint":None, "optimizer_updates":0, "plan_hash":digest(canonical_json(plan))}
    validate_generation(result, plan)
    persist(directory, {"result":result})
    artifacts.commit()
    return result


async def live_controls(directory, plan, setup, deadline):
    path = directory / "supervisor_controls.json"
    if path.exists():
        return supervisor_controls.validate_controls(read_json(path), setup["sandbox_image_id"])
    source_path = Path("/artifacts/qwen-booking-reward-shape-20260928-v1/arms/linear/rollouts/train-linear-14.json")
    source = next(s["source"] for s in read_json(source_path) if s["sample_id"] == supervisor_controls.FAILED_SAMPLE)
    document = {"saved_source":source, "records":{}}
    backend = supervised.SupervisedPanelBackend(setup["app_name"], setup["sandbox_image_id"], study.BOOKING,
                                                creation_interval_seconds=.26)
    for name, (program, _, _, _) in supervisor_controls.controls(source).items():
        require_deadline(deadline)
        prefix = plan["run_id"] + "/supervisor-control/" + name
        record = await claims.get.aio(prefix + "/result", None)
        if record is None:
            if not await claims.put.aio(prefix + "/intent", {"source_hash":digest(program)}, skip_if_exists=True):
                raise ReconciliationRequired("safety control outcome lost; do not launch model")
            record = pack_result(await backend.execute(request_for(program, supervisor_controls.case_for_control())))
            if not await claims.put.aio(prefix + "/result", record, skip_if_exists=True):
                raise ReconciliationRequired("duplicate control result")
        document["records"][name] = record
        persist(directory / "supervisor_controls", {name:record})
        await artifacts.commit.aio()
        print("BASELINE SUPERVISOR CONTROL", name, record["status"], record["detail"], flush=True)
    ids = supervisor_controls.validate_controls(document, setup["sandbox_image_id"])
    persist(directory, {"supervisor_controls":document})
    await artifacts.commit.aio()
    return ids


def verify_baseline(directory, *, progress=None):
    directory = Path(directory)
    if progress is None:
        with ProgressLog(directory.name, label="EVIDENCE_REPLAY") as progress:
            return verify_baseline(directory, progress=progress)
    progress.stage("validate_metadata")
    plan, setup = read_json(directory / "plan.json"), read_json(directory / "setup.json")
    study.validate_plan(plan)
    ids = supervisor_controls.validate_controls(read_json(directory / "supervisor_controls.json"), setup["sandbox_image_id"])
    generated = read_json(directory / "generation/result.json")
    validate_generation(generated, plan)
    rows, starts, slots = [], len(ids), []
    batches = [("controls", "training", study.controls(plan))]
    batches += [(f"eval-baseline-00-{i:02d}", "evaluation", generated["samples"][4*i:4*i+4]) for i in range(len(study.SEEDS)//4)]
    for index, (key, role, samples) in enumerate(batches, 1):
        position = {"batch": key, "batch_index": index, "batches_total": len(batches)}
        progress.stage("recompute_batch_scores", **position)
        raw = read_json(directory / f"grading/{key}/raw.json")
        checked = study.verify_raw(raw, samples, key, role, plan, setup["sandbox_image_id"])
        if read_json(directory / f"grading/{key}/result.json") != {"raw":raw, "checked":checked}:
            raise ValueError("batch replay disagrees")
        if key == "controls":
            if study.validate_controls(checked) != read_json(directory / "preflight.json"):
                raise ValueError("preflight changed")
        else:
            rows.extend(checked["rows"])
        ids.extend(checked["sandbox_ids"])
        starts += checked["submitted_attempts"]
        journal_files = sum(len(entry) for inputs in raw["entries"].values() for entry in inputs.values())
        progress.stage("verify_input_journals", total=journal_files, unit="files", **position)
        for sample in samples:
            for h, entry in raw["entries"][sample["sample_id"]].items():
                for name, value in entry.items():
                    progress.item(f"grading/{key}/inputs/{sample['sample_id']}/{h}/{name}.json")
                    if read_json(directory / f"grading/{key}/inputs/{sample['sample_id']}/{h}/{name}.json") != value:
                        raise ValueError("input journal differs from batch")
                    progress.advance()
                if "intent-2" in entry:
                    slot = entry["intent-2"]["retry_slot"]
                    if type(slot) is not int or not 0 <= slot < plan["max_startup_retries"]:
                        raise ValueError("invalid startup replacement reservation")
                    slots.append(slot)
    progress.stage("verify_generation_journals", total=2*len(generated["samples"]), unit="files")
    for sample in generated["samples"]:
        sid = sample["sample_id"]
        progress.item(sid)
        if (read_json(directory / f"generation/samples/{sid}.json") != sample or
                read_json(directory / f"generation/sample-intents/{sid}.json") != {
                    "sample_id":sid, "seed":sample["seed"], "parameter_hash":plan["initial_parameter_hash"]}):
            raise ValueError("generation evidence changed")
        progress.advance(2)
    progress.stage("aggregate_summary")
    if (len(ids) != len(set(ids)) or len(slots) != len(set(slots))
            or starts > plan["baseline_max_sandbox_starts"]):
        raise ValueError("resource bound violated or sandbox reused")
    summary = study.policy_summary(rows)
    return {"version":study.VERSION, "status":"completed_with_uncertainty" if summary["unknown_input_outcomes"] else "completed",
        "plan_hash":digest(canonical_json(plan)), "source_snapshot_hash":digest(canonical_json(read_json(directory / "source_snapshot.json"))),
        "generation_hash":digest(canonical_json(generated)), "summary":summary, "programs":rows,
        "submitted_sandbox_attempts":starts, "startup_replacements":len(slots), "optimizer_updates":0,
        "research_question_answered":False, "automatic_training":False,
        "next_step":"live full-state GRPO resume control, then separately released matched training"}


def finalize_baseline(directory):
    """Log replay and publication separately; complete only after Volume commit."""
    with ProgressLog(Path(directory).name) as progress:
        progress.stage("reload_artifacts")
        artifacts.reload()
        result = verify_baseline(directory, progress=progress)
        progress.stage("write_report")
        persist(directory, {"result": result})
        progress.stage("commit_artifacts")
        artifacts.commit()
        return result


@app.function(image=cpu_image, cpu=(1,1), memory=(2048,2048), nonpreemptible=True,
              timeout=study.CONTROLLER_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_baseline(plan, setup, budget, snapshot, deadline):
    study.validate_plan(plan)
    check_snapshot(snapshot, Path("/root"))
    require_deadline(deadline)
    directory = Path("/artifacts") / plan["run_id"]
    artifacts.reload()
    if (directory / "result.json").exists():
        return verify_baseline(directory)
    claim_start("controller", plan["max_controller_starts"])
    persist(directory, {"plan":plan, "setup":setup, "budget":budget, "source_snapshot":snapshot, "deadline":deadline})
    artifacts.commit()
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("candidate sandboxes active before baseline/re-entry")
    stage = "supervisor_controls"
    try:
        asyncio.run(live_controls(directory, plan, setup, deadline))
        stage = "grading_controls"
        checked = grade_batch.remote("controls", study.controls(plan), "training", plan, setup, deadline)["checked"]
        artifacts.reload()
        preflight = study.validate_controls(checked)
        persist(directory, {"preflight":preflight})
        artifacts.commit()
        print("ORIGINAL BASELINE PREFLIGHT PASSED", preflight, flush=True)
        stage = "original_policy_generation"
        generated = generate_baseline.remote(plan, setup, deadline)
        print("ORIGINAL BASELINE GENERATION COMPLETE", len(generated["samples"]), "weights unchanged", flush=True)
        stage = "independent_evaluation"
        for i in range(len(study.SEEDS)//4):
            require_deadline(deadline)
            grade_batch.remote(f"eval-baseline-00-{i:02d}", generated["samples"][4*i:4*i+4], "evaluation", plan, setup, deadline)
        stage = "evidence_replay"
        result = finalize_baseline(directory)
        print("ORIGINAL BASELINE COMPLETE", result["summary"], "NO OPTIMIZER UPDATES", flush=True)
        return result
    except Exception as exc:
        persist(directory, {f"interrupted-{time.time_ns()}": {"stage":stage, "type":type(exc).__name__, "detail":str(exc)[:2000]}})
        artifacts.commit()
        raise


def read_modal(*args):
    return json.loads(subprocess.run([sys.executable, "-m", "modal", *args, "--json"],
                                    check=True, capture_output=True, text=True, timeout=45).stdout)


@app.local_entrypoint()
def launch(allow_cloud: bool = False, resume: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required; offline plan does not spend")
    target = Path("runs") / study.RUN_ID
    if (target / "launch_intent.json").exists() and not resume:
        raise ReconciliationRequired("already launched; inspect the existing app, do not create another baseline")
    apps = read_modal("app", "list")
    if any(int(row["tasks"]) for row in apps):
        raise ReconciliationRequired("other Modal tasks active")
    if resume:
        plan, setup, budget, snapshot, deadline = [read_json(target / f"{name}.json") for name in
            ("plan", "setup", "budget", "source_snapshot", "deadline")]
        if (target / "resume_launch_intent.json").exists():
            raise ReconciliationRequired("one bounded controller continuation already launched")
    else:
        plan = study.make_plan(Path.cwd())
        setup = read_json("runs/qwen-booking-boundary-screen-20260929-v1/setup.json")
        budget = study.budget_quote(plan, read_modal("billing", "rates"), read_modal("billing", "summary"))
        names = sorted(Path("verifier_rl").glob("*.py")) + [Path("modal_booking_study.py"), Path("modal_booking_baseline_comparison.py")]
        snapshot = {p.as_posix():p.read_text() for p in names}
        deadline = time.time() + study.CONTROLLER_SECONDS - 120
    check_snapshot(snapshot, Path.cwd())
    study.validate_plan(plan)
    require_deadline(deadline)
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("other candidate sandboxes active")
    prefix = "resume_" if resume else ""
    persist(target, {"plan":plan, "setup":setup, "budget":budget, "source_snapshot":snapshot, "deadline":deadline,
        "protocol":Path("docs/booking_baseline_comparison_protocol.txt").read_text(),
        prefix+"launch_intent":{"plan_hash":digest(canonical_json(plan)), "deadline":deadline, "observed_apps":apps}})
    call = run_baseline.spawn(plan, setup, budget, snapshot, deadline)
    persist(target, {prefix+"launch":{"call_id":call.object_id, "created_utc":datetime.now(timezone.utc).isoformat()}})
    print("ORIGINAL BASELINE CALL", call.object_id, "MAX ADDITIONAL RESOURCE ENVELOPE USD",
          budget["additional_resource_envelope_usd"], "NO TRAINING", flush=True)
    result = call.get()
    persist(target, {"result":result})
