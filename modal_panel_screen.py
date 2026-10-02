"""One generation-only three-task screen. Candidate source runs only in CPU sandboxes."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import modal

from verifier_rl import panel_screen as screen
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.evaluation_recovery import check_frozen_sources
from verifier_rl.panel_execution import PanelBackend, pack_result
from verifier_rl.smoke import require_current_conformance
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import TASK_IDS
from verifier_rl.verifier_quality import SCREEN_SEEDS, screen_budget_preview

app = modal.App("verifier-rl-panel-screen")
artifacts = modal.Volume.from_name("verifier-rl-cache-artifacts", create_if_missing=False)
claims = modal.Dict.from_name("verifier-rl-evaluation-claims", create_if_missing=False)
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.5.5")
             .add_local_python_source("verifier_rl"))
# Same pinned dependency recipe as the completed 1.5B pilot, to reuse image layers.
# Only inference runs here; importing this module locally does not import torch.
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("modal==1.5.5", "torch==2.8.0", "transformers==4.57.1",
                          "trl==0.28.0", "datasets==3.5.1", "accelerate==1.12.0")
             .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
             .add_local_python_source("verifier_rl"))


def require_deadline(deadline):
    if time.time() >= deadline:
        raise ReconciliationRequired("screen submission deadline reached; no new work")


def claim_start(kind):
    if not any(claims.put(f"{screen.RUN_ID}/{kind}-start-{index}", True, skip_if_exists=True)
               for index in range(screen.MAX_STARTS)):
        raise ReconciliationRequired(kind + " start bound exhausted")


def claim_once(key, value):
    if not claims.put(screen.RUN_ID + "/" + key, digest(canonical_json(value)), skip_if_exists=True):
        raise ReconciliationRequired("work already claimed; reconcile, never resubmit: " + key)


def root_directory():
    return Path("/artifacts") / screen.RUN_ID


def evaluate_batch(directory, key, source, cases, setup, deadline, *, journal_to_dict=False):
    """One at-most-once candidate batch; completed per-input records are committed."""
    intent = {"version": screen.VERSION, "key": key, "task_id": cases[0].task_id,
              "source_hash": digest(source), "input_hashes": [case.input_hash for case in cases],
              "maximum_sandbox_executions": len(cases), "automatic_retry": False}
    result_path = directory / "result.json"
    if result_path.exists():
        persist(directory, {"intent": intent})
        result = json.loads(result_path.read_text())
        if result["task_id"] != cases[0].task_id or result["source"] != source:
            raise ValueError("saved source/task differs")
        return result
    if (directory / "intent.json").exists():
        raise ReconciliationRequired("unfinished input batch retained: " + key)
    require_deadline(deadline)
    persist(directory, {"intent": intent})
    artifacts.commit()
    claim_once("batch/" + key, intent)
    backend = PanelBackend(setup["app_name"], setup["sandbox_image_id"], cases[0].task_id,
                           creation_interval_seconds=.26)

    async def run():
        commit_lock = asyncio.Lock()
        async def record(case, result):
            value = {"source_hash": intent["source_hash"], "input_hash": case.input_hash,
                     "case": case.name, "execution": pack_result(result)}
            if journal_to_dict:
                # One durable result per input, without serial full-Volume
                # commits/reloads. The complete batch is also committed below.
                result_key = screen.RUN_ID + "/raw/" + key + "/" + case.input_hash
                if not await claims.put.aio(result_key, value, skip_if_exists=True):
                    raise ReconciliationRequired("raw input record already exists")
                persist(directory / "inputs", {case.input_hash: value})
                return
            async with commit_lock:
                persist(directory / "inputs", {case.input_hash: value})
                await artifacts.commit.aio()
        return await screen.execute_cases(source, cases, backend, setup["sandbox_image_id"],
                                          deadline, record, concurrency=8)
    results = asyncio.run(run())
    raw = {"task_id": cases[0].task_id, "source": source,
           "records": {key: pack_result(value) for key, value in results.items()}}
    persist(directory, {"result": raw})
    artifacts.commit()
    return raw


@app.function(image=gpu_image, gpu="L40S", cpu=(2, 2), memory=(32768, 32768),
              timeout=screen.GPU_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def generate(plan, deadline):
    import hashlib
    import importlib.metadata
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from verifier_rl.measurement_v2 import PARAMETER_HASH

    screen.validate_plan(plan)
    require_deadline(deadline)
    claim_start("gpu")
    artifacts.reload()
    directory = root_directory() / "gpu"
    if (directory / "generation.json").exists():
        saved = json.loads((directory / "generation.json").read_text())
        if saved["plan_hash"] != digest(canonical_json(plan)):
            raise ValueError("saved generation plan mismatch")
        return saved
    # An automatic restart does not repeat or complete an interrupted generation
    # call. Preserve every sample/intent and require reconciliation instead.
    if (directory / "intent.json").exists():
        raise ReconciliationRequired("interrupted generation retained; no fresh draws")
    intent = {"plan_hash": digest(canonical_json(plan)), "sample_ids": plan["sample_ids"],
              "training": False, "deadline": deadline}
    persist(directory, {"intent": intent})
    artifacts.commit()
    claim_once("generation", intent)
    started = time.monotonic()

    def parameter_hash(model):
        value = hashlib.sha256()
        for name, parameter in model.named_parameters():
            value.update(name.encode())
            value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()

    config = plan["generation"]
    set_seed(SCREEN_SEEDS[0])
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if digest(tokenizer.chat_template) != config["chat_template_hash"]:
        raise ValueError("chat template changed")
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["revision"],
                trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model.eval()
    before = parameter_hash(model)
    if before != PARAMETER_HASH:
        raise ValueError("initial model parameter hash differs from frozen 1.5B checkpoint")
    persist(directory, {"runtime": {"gpu": torch.cuda.get_device_name(), "gpu_image_id": gpu_image.object_id,
        "parameter_count": sum(p.numel() for p in model.parameters()), "before_parameter_hash": before,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "weights_dtype": "float32", "autocast_dtype": "bfloat16"}})
    artifacts.commit()
    samples = []
    for task_id in TASK_IDS:
        prompt = plan["prompts"][task_id]
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")
        for seed in SCREEN_SEEDS:
            require_deadline(deadline)
            sid = screen.sample_id(task_id, seed)
            persist(directory / "sample-intents", {sid: {"task_id": task_id, "seed": seed, "prompt_hash": digest(prompt)}})
            artifacts.commit()
            set_seed(seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model.generate(**inputs, max_new_tokens=512, do_sample=True,
                    temperature=.8, top_p=.95, top_k=0, repetition_penalty=config["repetition_penalty"],
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
            completion = tokens[0, inputs["input_ids"].shape[1]:]
            sample = screen.inspect_completion(task_id, tokenizer.decode(completion, skip_special_tokens=True))
            sample.update(task_id=task_id, seed=seed, sample_id=sid, prompt_hash=digest(prompt),
                          tokens=len(completion), hit_token_cap=len(completion) == 512,
                          ended_with_eos=bool(len(completion) and completion[-1].item() == tokenizer.eos_token_id))
            screen.validate_sample(sample, plan)
            persist(directory / "samples", {sid: sample})
            artifacts.commit()
            samples.append(sample)
            print("Generated", sid, "tokens", len(completion), "syntax", sample["syntax_valid"], flush=True)
    after = parameter_hash(model)
    if after != before:
        raise ValueError("generation changed parameters")
    generation = {"plan_hash": digest(canonical_json(plan)), "samples": samples,
                  "before_parameter_hash": before, "after_parameter_hash": after, "parameters_unchanged": True,
                  "gpu_function_seconds": time.monotonic() - started, "training": False}
    persist(directory, {"generation": generation})
    artifacts.commit()
    return generation


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=screen.CONTROLLER_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def run_screen(plan, setup, budget, snapshot, deadline):
    require_deadline(deadline)
    claim_start("controller")
    panel = screen.validate_plan(plan)
    check_frozen_sources(snapshot, Path("/root"))
    artifacts.reload()
    directory = persist(root_directory(), {"plan": plan, "setup": setup, "budget": budget,
                                          "source_snapshot": snapshot, "deadline": deadline})
    artifacts.commit()
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("evaluation app has active sandboxes; reconcile before running")
    stage = "controls"
    try:
        controls = {}
        for item in screen.control_items(panel):
            stage = item["id"]
            controls[stage] = evaluate_batch(directory / "controls" / stage, stage, item["source"], item["cases"], setup, deadline)
            print("Recorded control", stage, flush=True)
        control_ids = screen.validate_controls(plan, controls, setup["sandbox_image_id"])
        persist(directory, {"conformance": {"passed": True, "executions": 18,
                                            "sandbox_ids": control_ids, "model_results": False}})
        artifacts.commit()
        print("All 18 task-interface controls passed. Starting 24-program model screen.", flush=True)
        stage = "generation"
        generation = generate.remote(plan, deadline)
        artifacts.reload()  # Observe the GPU's separately committed files.
        records = {}
        for sample in generation["samples"]:
            screen.validate_sample(sample, plan)
            sid = stage = sample["sample_id"]
            selected = panel[sample["task_id"]]
            if sample["extraction_status"].startswith("rejected_"):
                raw = {"task_id": sample["task_id"], "source": sample["source"], "records": {}}
                persist(directory / "programs" / sid, {"result": raw})
            else:
                cases = (*selected["training"], *selected["development"])
                raw = evaluate_batch(directory / "programs" / sid, sid, sample["source"], cases, setup, deadline)
            records[sid] = raw["records"]
            report = screen.program_report(sample, selected, raw["records"], setup["sandbox_image_id"])
            persist(directory / "programs" / sid, {"sample": sample, "report": report})
            artifacts.commit()
            print("Evaluated", len(records), "/24", sid,
                  "reference", report["scores"]["training"]["reference_accepted"],
                  "structured", report["scores"]["training"]["structured_accepted"],
                  "development", report["scores"]["development"]["case_passes"], "/16", flush=True)
        stage = "verification"
        summary = screen.summarize(plan, generation, controls, records, setup["sandbox_image_id"])
        persist(directory, {"summary": summary})
        artifacts.commit()
        print("SCREEN COMPLETE", summary["gates"], flush=True)
        return summary
    except Exception as exc:
        persist(directory, {f"stopped-{time.time_ns()}": {"stage": stage, "error_type": type(exc).__name__,
            "detail": str(exc)[:1000], "automatic_retry": False, "outstanding_reservations_retained": True}})
        artifacts.commit()
        raise


@app.local_entrypoint()
def launch(allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required for the approved 24-program screen; no RL")
    target = Path("runs") / screen.RUN_ID
    if target.exists():
        raise ValueError("screen already prepared/launched: inspect saved IDs, do not reset or relaunch")
    plan = screen.make_plan(Path.cwd())
    screen.validate_plan(plan)
    setup = json.loads(Path("runs/modal-setup-20260925/setup.json").read_text())
    conformance = json.loads(Path("runs/cpu-20260927T215503-5fc4d366/summary.json").read_text())
    require_current_conformance(conformance, setup["sandbox_image_id"])
    def read_modal(*args):
        completed = subprocess.run([sys.executable, "-m", "modal", *args, "--json"],
                                   check=True, capture_output=True, text=True, timeout=45)
        return json.loads(completed.stdout)
    billing, rates = read_modal("billing", "summary"), read_modal("billing", "rates")
    old = json.loads(Path("runs/qwen-eval-completion-20260928-v1/resume-001/reservation.json").read_text())
    budget = screen_budget_preview(rates, billing["metered_cost"], old["total_reserved_usd"])
    budget.update(kind="screen_resource_reservation", billing_before=billing, rates=rates,
                  scope="18 conformance inputs + 24 model programs only",
                  authority="user approved this bounded screen; budget is not a reason to expand its scope",
                  enforcement_implemented=True, monetary_limit_enforced=False,
                  historical_20_usd_comparison_only=True,
                  enforced_by="fixed plan, at-most-once claims, resource/lifetime/start bounds")
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("active sandbox work exists before screen launch")
    snapshot = {p.as_posix(): p.read_text() for p in sorted(Path("verifier_rl").glob("*.py"))
                + [Path("modal_panel_screen.py")]}
    # An absolute submission deadline is not extended by automatic restarts.
    deadline = time.time() + screen.CONTROLLER_SECONDS
    directory = persist(target, {"plan": plan, "setup": setup, "budget": budget,
                                 "source_snapshot": snapshot, "deadline": deadline,
                                 "protocol": Path("docs/verifier_quality_protocol.txt").read_text()})
    print("Screen reservation (not invoice): USD", budget["cumulative_reserve_usd"], flush=True)
    call = run_screen.spawn(plan, setup, budget, snapshot, deadline)
    persist(directory, {"launch": {"run_id": screen.RUN_ID, "call_id": call.object_id,
                                   "created_utc": datetime.now(timezone.utc).isoformat(),
                                   "model_samples": 24, "training": False}})
    print("SCREEN CALL", call.object_id, "RUN", screen.RUN_ID, flush=True)
    summary = call.get()
    persist(directory, {"summary": summary})
    print("Summary saved:", directory / "summary.json", flush=True)


@app.function(image=cpu_image, cpu=(1, 1), memory=(2048, 2048), nonpreemptible=True,
              timeout=screen.CONTROLLER_SECONDS, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def complete_screen(plan, setup, reconciliation, deadline):
    """Explicit CPU-only continuation after the initial controller's time limit.

    Reuse complete original batches. An at-most-one interrupted batch gets one
    predeclared whole-batch replacement; retain its raw observations and compare
    repeated outcomes. Never replace a batch stopped for infrastructure failure
    or an unattributed termination; known ordinary wrong outputs are preserved.
    """
    from verifier_rl.panel_execution import require_evidence, unpack_result
    from verifier_rl.verifier_quality import compare_output
    require_deadline(deadline)
    panel = screen.validate_plan(plan)
    if (reconciliation.get("original_app_id") != "ap-CKoxLnZYt91BKvGnTm3Fh1"
            or reconciliation.get("original_state") != "stopped"
            or reconciliation.get("original_tasks") != "0"
            or reconciliation.get("active_sandbox_ids") != []
            or reconciliation.get("original_call_error_type") != "FunctionTimeoutError"
            or reconciliation.get("reason") != "controller_submission_or_function_deadline"):
        raise ReconciliationRequired("stopped original controller and deadline evidence required")
    claim_start("controller")  # Uses the SAME two-start allowance; not two new starts.
    artifacts.reload()
    root = root_directory()
    original_plan = json.loads((root / "plan.json").read_text())
    original_deadline = json.loads((root / "deadline.json").read_text())
    if original_plan != plan or time.time() < original_deadline + 120:
        raise ReconciliationRequired("original deadline/sandbox lifetime has not elapsed")
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    if list(modal.Sandbox.list(app_id=owner.app_id)):
        raise ReconciliationRequired("sandboxes remain active")
    stopped = [(int(p.stem.split("-")[-1]) / 1e9, json.loads(p.read_text()))
               for p in root.glob("stopped-*.json")]
    for when, entry in stopped:
        allowed = ("deadline" in entry.get("detail", "")
                   or entry.get("detail") == "incomplete program: preserve missing/uncertain observations")
        if not allowed or when < original_deadline:
            raise ReconciliationRequired("a non-deadline failure needs separate investigation")
    directory = persist(root / "continuation-001", {"reconciliation": reconciliation, "deadline": deadline})
    artifacts.commit()
    controls = {item["id"]: json.loads((root / "controls" / item["id"] / "result.json").read_text())
                for item in screen.control_items(panel)}
    screen.validate_controls(plan, controls, setup["sandbox_image_id"])
    generation = json.loads((root / "gpu/generation.json").read_text())
    if [sample["sample_id"] for sample in generation["samples"]] != plan["sample_ids"]:
        raise ReconciliationRequired("all 24 original generations required; no generation replacement")
    complete, reconstructed = set(), {}
    for sample in generation["samples"]:
        path = root / "programs" / sample["sample_id"] / "result.json"
        if path.exists():
            raw = json.loads(path.read_text())
            if len(raw["records"]) == 32 or sample["extraction_status"].startswith("rejected_"):
                screen.program_report(sample, panel[sample["task_id"]], raw["records"], setup["sandbox_image_id"])
                complete.add(sample["sample_id"])
        if sample["sample_id"] not in complete:
            leaves = {p.stem: json.loads(p.read_text()) for p in path.parent.joinpath("inputs").glob("*.json")}
            if len(leaves) == 32:
                raw = {"task_id": sample["task_id"], "source": sample["source"],
                       "records": {key: value["execution"] for key, value in leaves.items()}}
                screen.program_report(sample, panel[sample["task_id"]], raw["records"], setup["sandbox_image_id"])
                persist(directory / "reconstructed" / sample["sample_id"], {"result": raw})
                reconstructed[sample["sample_id"]] = raw
                complete.add(sample["sample_id"])
    pending = [sample for sample in generation["samples"] if sample["sample_id"] not in complete]
    interrupted = [sample for sample in pending
                   if (root / "programs" / sample["sample_id"] / "intent.json").exists()]
    if len(interrupted) > 1:
        raise ReconciliationRequired("at most one original input batch may be interrupted")
    old_partial = {}
    for sample in interrupted:
        sid = sample["sample_id"]
        cases = {case.input_hash: case for selected in panel[sample["task_id"]].values() for case in selected}
        intent = json.loads((root / "programs" / sid / "intent.json").read_text())
        if (intent.get("source_hash") != digest(sample["source"])
                or intent.get("task_id") != sample["task_id"] or intent.get("input_hashes") != list(cases)):
            raise ValueError("original interrupted intent differs from frozen sample/inputs")
        raw = {p.stem: json.loads(p.read_text()) for p in (root / "programs" / sid / "inputs").glob("*.json")}
        partial_path = root / "programs" / sid / "result.json"
        if partial_path.exists():
            partial = json.loads(partial_path.read_text())
            if partial["source"] != sample["source"] or partial["task_id"] != sample["task_id"]:
                raise ValueError("partial source/task mismatch")
            for key, packed in partial["records"].items():
                if key not in raw or raw[key]["execution"] != packed:
                    raise ValueError("partial batch and per-input records disagree")
        for key, value in raw.items():
            if key not in cases:
                raise ValueError("unknown original input")
            require_evidence(unpack_result(value["execution"]), sample["task_id"], setup["sandbox_image_id"],
                             source=sample["source"], case=cases[key])
        old_partial[sid] = raw
    persist(directory, {"original_partial_records": old_partial,
                        "pending": [sample["sample_id"] for sample in pending],
                        "amendment": {"reason": "controller deadline after serialized volume commits",
                            "new_generations": 0, "max_extra_executions_vs_original_plan": 32 if interrupted else 0,
                            "replacement_is_not_outcome_selected": True,
                            "old_reservations_retained": True,
                            "raw_journal": "durable Modal Dict per input; Volume at batch completion"}})
    artifacts.commit()
    records, selected_paths = {}, {}
    for sample in generation["samples"]:
        screen.validate_sample(sample, plan)
        sid = sample["sample_id"]
        original = root / "programs" / sid / "result.json"
        selected = panel[sample["task_id"]]
        if sid in complete:
            if sid in reconstructed:
                raw = reconstructed[sid]
                selected_paths[sid] = f"continuation-001/reconstructed/{sid}/result.json"
            else:
                raw = json.loads(original.read_text())
                selected_paths[sid] = f"programs/{sid}/result.json"
        else:
            cases = (*selected["training"], *selected["development"])
            raw = evaluate_batch(directory / "programs" / sid, "continuation-001/" + sid,
                                 sample["source"], cases, setup, deadline, journal_to_dict=True)
            selected_paths[sid] = f"continuation-001/programs/{sid}/result.json"
        report = screen.program_report(sample, selected, raw["records"], setup["sandbox_image_id"])
        if raw["source"] != sample["source"] or raw["task_id"] != sample["task_id"]:
            raise ValueError("selected result source/task mismatch")
        if sid in old_partial:
            for case in (*selected["training"], *selected["development"]):
                if case.input_hash not in old_partial[sid]:
                    continue
                before = unpack_result(old_partial[sid][case.input_hash]["execution"])
                after = unpack_result(raw["records"][case.input_hash])
                def outcome(value):
                    return compare_output(case, value.stdout)[0] if value.status.value == "completed" else False
                if outcome(before) != outcome(after):
                    raise ReconciliationRequired("repeated outcomes differ; retain uncertainty, do not select a pass")
        records[sid] = raw["records"]
        persist(directory / "reports", {sid: report})
        artifacts.commit()
        print("CPU continuation", len(records), "/24", sid,
              "reference", report["scores"]["training"]["reference_accepted"],
              "structured", report["scores"]["training"]["structured_accepted"],
              "development", report["scores"]["development"]["case_passes"], "/16", flush=True)
    summary = screen.summarize(plan, generation, controls, records, setup["sandbox_image_id"])
    known_partial = sum(len(raw) for raw in old_partial.values())
    accounting = {"reused_complete_programs": 24 - len(pending), "new_program_batches": len(pending),
                  "controller_deadline_replacements": len(interrupted), "new_model_samples": 0,
                  "recorded_original_partial_inputs": known_partial,
                  "total_execution_bounds": [summary["sandbox_executions"] + known_partial,
                                              summary["sandbox_executions"] + 32 * len(interrupted)],
                  "repeated_recorded_outcomes_agree": True}
    persist(directory, {"summary": summary, "accounting": accounting, "selected_paths": selected_paths})
    artifacts.commit()
    print("SCREEN COMPLETED VIA CPU CONTINUATION", summary["gates"], accounting, flush=True)
    return {"summary": summary, "accounting": accounting, "selected_paths": selected_paths}


@app.local_entrypoint()
def finish(allow_cloud: bool = False):
    if not allow_cloud:
        raise ValueError("--allow-cloud required: explicit CPU-only screen completion")
    root = Path("runs") / screen.RUN_ID
    if (root / "continuation-001").exists():
        raise ReconciliationRequired("continuation already prepared/launched")
    def read_modal(*args):
        result = subprocess.run([sys.executable, "-m", "modal", *args, "--json"],
                                check=True, capture_output=True, text=True, timeout=45)
        return json.loads(result.stdout)
    plan, setup = (json.loads((root / f"{name}.json").read_text()) for name in ("plan", "setup"))
    original = next(app for app in read_modal("app", "list") if app["app_id"] == "ap-CKoxLnZYt91BKvGnTm3Fh1")
    original_call = json.loads((root / "launch.json").read_text())["call_id"]
    try:
        modal.FunctionCall.from_id(original_call).get(timeout=0)
    except modal.exception.FunctionTimeoutError as exc:
        original_error = {"original_call_id": original_call, "original_call_error_type": type(exc).__name__,
                          "original_call_error_detail": str(exc)[:500]}
    else:
        raise ReconciliationRequired("original call did not terminate with its controller timeout")
    owner = modal.App.lookup(setup["app_name"], create_if_missing=False)
    reconciliation = {**original_error, "original_app_id": original["app_id"], "original_state": original["state"],
        "original_tasks": original["tasks"], "active_sandbox_ids": [s.object_id for s in modal.Sandbox.list(app_id=owner.app_id)],
        "reason": "controller_submission_or_function_deadline", "observed_utc": datetime.now(timezone.utc).isoformat()}
    if (original["state"] != "stopped" or original["tasks"] != "0" or reconciliation["active_sandbox_ids"]
            or time.time() < json.loads((root / "deadline.json").read_text()) + 120):
        raise ReconciliationRequired("original controller and sandbox deadline must finish first")
    budget = json.loads((root / "budget.json").read_text())
    rates = read_modal("billing", "rates")
    from decimal import Decimal
    extra = 32 * 120 * (Decimal(rates["cpu_hour_cost_sandbox"]) + Decimal(rates["mem_gib_hour_cost_sandbox"]) / 4) / 3600
    accounting = {"prior_full_reservation_held_usd": budget["cumulative_reserve_usd"],
                  "extra_sandbox_reserve_usd": str(extra),
                  "cumulative_reserve_usd": str(Decimal(budget["cumulative_reserve_usd"]) + extra),
                  "controller_uses_remaining_original_start_allowance": True,
                  "billing_before": read_modal("billing", "summary"), "is_invoice": False}
    deadline = time.time() + screen.CONTROLLER_SECONDS
    directory = persist(root / "continuation-001", {"reconciliation": reconciliation, "budget": accounting,
        "deadline": deadline, "launcher_source": Path("modal_panel_screen.py").read_text()})
    call = complete_screen.spawn(plan, setup, reconciliation, deadline)
    persist(directory, {"launch": {"call_id": call.object_id, "new_generations": 0, "max_new_program_batches": 24}})
    print("CPU SCREEN COMPLETION", call.object_id, "reserve USD", accounting["cumulative_reserve_usd"], flush=True)
    result = call.get()
    persist(directory, result)
