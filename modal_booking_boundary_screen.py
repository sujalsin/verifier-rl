"""Explicitly launched, generation-only screen. Never train or repair candidates."""

import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import modal

from modal_booking_study import (artifacts, claims, begin, finish, generate_samples, parameter_hash,
                                require_deadline, cpu_image as base_cpu_image, gpu_image as base_gpu_image)
from modal_booking_reward_pilot import execute_group, live_supervisor_controls, read_modal, check_snapshot
from verifier_rl import booking_boundary_screen as screen
from verifier_rl import supervised_execution as supervised, supervisor_controls
from verifier_rl.evaluation_journal import persist, ReconciliationRequired
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import BOOKING

app = modal.App("verifier-rl-booking-boundary-screen")


def with_launchers(image):
    for name in ("modal_booking_study.py", "modal_booking_reward_pilot.py", "modal_booking_boundary_screen.py"):
        image = image.add_local_file(name, "/root/" + name)
    return image


cpu_image, gpu_image = with_launchers(base_cpu_image), with_launchers(base_gpu_image)


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=screen.GRADE_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def grade_batch(key, samples, plan, setup, deadline):
    screen.validate_batch(key, samples, plan)
    directory = Path("/artifacts") / plan["run_id"] / "grading" / key
    intent = {"key": key, "samples": samples, "plan_hash": digest(canonical_json(plan)),
              "setup": setup, "deadline": deadline}
    saved = begin(directory, "grade/" + key, intent, deadline, run_id=plan["run_id"])
    if saved is not None:
        screen.verify_batch(saved, samples, key, plan, setup["sandbox_image_id"])
        return saved
    backend = supervised.SupervisedPanelBackend(setup["app_name"], setup["sandbox_image_id"], BOOKING,
                                                creation_interval_seconds=.26)
    try:
        records, retries = asyncio.run(execute_group(samples, "training", setup, directory, deadline,
            backend, claims, plan, retry_policy=plan["startup_retry_version"]))
        raw = {"key": key, "role": "training", "samples": samples, "records": records,
               "retries": retries, "plan_hash": digest(canonical_json(plan))}
        persist(directory, {"raw": raw})
        artifacts.commit()
        result = dict(raw, graded=[screen.grade(s, records[s["sample_id"]], setup["sandbox_image_id"], plan) for s in samples])
        screen.verify_batch(result, samples, key, plan, setup["sandbox_image_id"])
        print("SCREEN GRADED", key, [(g["row"]["reference"]["passed"], g["row"]["endpoint_omission"]["passed"],
                                    g["row"]["inclusive_output_signature"]) for g in result["graded"]], flush=True)
        return finish(directory, result)
    except Exception as exc:
        persist(directory, {f"stopped-{time.time_ns()}": {"type": type(exc).__name__, "detail": str(exc)[:2000]}})
        artifacts.commit()
        raise


@app.function(image=gpu_image, gpu="L40S", cpu=(2, 2), memory=(32768, 32768),
              timeout=screen.GPU_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def generate(plan, setup, deadline):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    screen.validate_plan(plan)
    directory = Path("/artifacts") / plan["run_id"] / "generation"
    intent = {"plan_hash": digest(canonical_json(plan)), "deadline": deadline}
    saved = begin(directory, "generation", intent, deadline, run_id=plan["run_id"])
    if saved is not None:
        return saved
    supervisor_controls.validate_controls(json.loads((directory.parent / "supervisor_controls.json").read_text()),
                                          setup["sandbox_image_id"])
    control_raw = json.loads((directory.parent / "grading/controls/result.json").read_text())
    checked, _, _ = screen.verify_batch(control_raw, screen.controls(plan), "controls", plan, setup["sandbox_image_id"])
    if json.loads((directory.parent / "preflight.json").read_text()) != screen.validate_controls(checked):
        raise ValueError("preflight changed before GPU generation")
    checkpoint = Path(plan["initial_checkpoint"])
    receipt = json.loads((checkpoint.parent / "checkpoints/12.json").read_text())
    if receipt != {"path": str(checkpoint), "parameter_hash": plan["initial_parameter_hash"]}:
        raise ValueError("saved checkpoint receipt differs")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=False, local_files_only=True,
        use_safetensors=True, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model.requires_grad_(False)
    if (digest(tokenizer.chat_template) != plan["chat_template_hash"]
            or sum(p.numel() for p in model.parameters()) != plan["parameter_count"]
            or parameter_hash(model) != plan["initial_parameter_hash"]):
        raise ValueError("checkpoint weights or tokenizer changed")
    persist(directory, {"runtime": {"gpu": torch.cuda.get_device_name(), "gpu_image_id": gpu_image.object_id,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "parameter_dtype": "float32", "autocast_dtype": "bfloat16", "optimizer": None,
        "checkpoint_receipt": receipt}})
    artifacts.commit()
    samples = generate_samples(model, tokenizer, plan, screen.identities(), directory, deadline)
    if parameter_hash(model) != plan["initial_parameter_hash"]:
        raise ValueError("generation changed model weights")
    for index in range(16):
        screen.validate_batch(f"screen-{index:02d}", samples[index*4:index*4+4], plan)
    return finish(directory, {"samples": samples, "parameter_hash": plan["initial_parameter_hash"],
        "initial_checkpoint": str(checkpoint), "parameters_unchanged": True, "gpu_image_id": gpu_image.object_id})


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=screen.SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_screen(plan, setup, budget, snapshot, deadline):
    required = {"modal_booking_boundary_screen.py", "verifier_rl/booking_boundary_screen.py",
                "verifier_rl/supervised_execution.py", "verifier_rl/booking_boundary_contrast.py"}
    if not required.issubset(snapshot):
        raise ValueError("screen snapshot incomplete")
    check_snapshot(snapshot, Path("/root"))
    screen.validate_plan(plan)
    require_deadline(deadline)
    if not claims.put(plan["run_id"] + "/controller", {"plan": plan, "deadline": deadline}, skip_if_exists=True):
        raise ReconciliationRequired("screen already claimed; never regenerate a different pool")
    directory = Path("/artifacts") / plan["run_id"]
    artifacts.reload()
    persist(directory, {"plan": plan, "setup": setup, "budget": budget, "source_snapshot": snapshot, "deadline": deadline})
    artifacts.commit()
    stage = "supervisor_controls"
    try:
        owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
        if list(modal.Sandbox.list(app_id=owner.app_id)):
            raise ReconciliationRequired("other candidate sandboxes active")
        asyncio.run(live_supervisor_controls(plan, setup, directory, deadline))
        stage = "grading_controls"
        raw = grade_batch.remote("controls", screen.controls(plan), plan, setup,
                                 min(deadline, time.time() + screen.GRADE_SECONDS - 60))
        preflight = screen.validate_controls(raw["graded"])
        artifacts.reload()
        persist(directory, {"preflight": preflight})
        artifacts.commit()
        print("SCREEN PREFLIGHT PASSED", preflight, flush=True)
        stage = "fresh_generation"
        generated = generate.remote(plan, setup, min(deadline, time.time() + screen.GPU_SECONDS - 60))
        print("SCREEN GENERATION COMPLETE", len(generated["samples"]), "weights unchanged", flush=True)
        stage = "screen_grading"
        for index in range(16):
            require_deadline(deadline)
            grade_batch.remote(f"screen-{index:02d}", generated["samples"][index*4:index*4+4], plan, setup,
                               min(deadline, time.time() + screen.GRADE_SECONDS - 60))
        stage = "offline_replay"
        artifacts.reload()
        result = screen.verify_run(directory)
        persist(directory, {"result": result})
        artifacts.commit()
        print("SCREEN COMPLETE", result["summary"], result["decision"], flush=True)
        return result
    except Exception as exc:
        persist(directory, {f"stopped-{time.time_ns()}": {"stage": stage, "type": type(exc).__name__, "detail": str(exc)[:2000]}})
        artifacts.commit()
        raise


@app.local_entrypoint()
def launch(allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required; no spending during offline planning")
    plan = screen.make_plan(Path.cwd())
    target = Path("runs") / plan["run_id"]
    if (target / "launch_intent.json").exists():
        raise ReconciliationRequired("launch already recorded; inspect existing job")
    apps = read_modal("app", "list")
    if any(int(a["tasks"]) for a in apps):
        raise ReconciliationRequired("other Modal tasks active")
    previous = Path("runs") / screen.parent.WARM_RUN_ID
    setup = json.loads((previous / "setup.json").read_text())
    prior = json.loads((previous / "budget.json").read_text())["cumulative_reservation_usd"]
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("other candidate sandboxes active")
    budget = screen.budget_quote(plan, read_modal("billing", "rates"), read_modal("billing", "summary"), prior)
    names = sorted(Path("verifier_rl").glob("*.py")) + [Path(name) for name in
        ("modal_booking_study.py", "modal_booking_reward_pilot.py", "modal_booking_boundary_screen.py")]
    snapshot = {p.as_posix(): p.read_text() for p in names}
    check_snapshot(snapshot, Path.cwd())
    deadline = time.time() + screen.SECONDS - 60
    persist(target, {"plan": plan, "setup": setup, "budget": budget, "source_snapshot": snapshot,
        "protocol": Path("docs/booking_boundary_screen_protocol.txt").read_text(),
        "launch_intent": {"plan_hash": digest(canonical_json(plan)), "deadline": deadline, "observed_apps": apps}})
    call = run_screen.spawn(plan, setup, budget, snapshot, deadline)
    persist(target, {"launch": {"call_id": call.object_id, "created_utc": datetime.now(timezone.utc).isoformat()}})
    print("BOUNDARY SCREEN CALL", call.object_id, "MAX ADDITIONAL RESOURCE ENVELOPE USD",
          budget["additional_reservation_usd"], "NO TRAINING", flush=True)
    persist(target, {"result": call.get()})
