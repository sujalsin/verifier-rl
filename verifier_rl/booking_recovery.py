"""CPU-only completion of the stopped booking study; no candidate execution here."""

import argparse
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path, PurePosixPath

from . import booking_study as study, booking_two_arm as two
from .grading import Status
from .panel_execution import require_evidence, request_for, unpack_result
from .suites import canonical_json, digest

VERSION = "booking-evaluation-recovery-0.1"
RUN_ID = "qwen-booking-eval-recovery-20260928-v1"
SOURCE_APP = "ap-lLKC9iNKcpcmkBX3KxRBsW"
FAILED_BATCH = "evaluation-reference-24-03"
FAILED_SAMPLE = "eval-reference-24-14013"
FAILED_INPUT = "6797228cb3dfffc6772e7f87643c474cc31532582ea2854e5599e9ed384c3d78"
FAILED_SANDBOX = "sb-gSiQuixygeU8a8qZogFvOP"
MAX_EXTRA_STARTUPS = 8
SECONDS = 3600


def selected_path(path):
    """Capture JSON evidence, not multi-GB weights or redundant successful inputs."""
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or p.suffix != ".json":
        return False
    if len(p.parts) == 1:
        return True
    if p.parts[0] == "arms":
        return not any(x.startswith("checkpoint-") or x == "trainer" for x in p.parts)
    if p.parts[0] == "grading":
        return "inputs" not in p.parts or p.parts[1] == FAILED_BATCH
    return p.parts[0] in ("controls", "evaluation_summaries") and "inputs" not in p.parts


def pre_candidate_failure(result, request, image_id):
    if two.startup_retry_allowed(result, request, image_id):
        return True
    # Extend only the pre-candidate eligibility check, not candidate signal grading.
    m = result.metadata
    if (result.status != Status.INFRASTRUCTURE_ERROR or result.detail != "preflight:RuntimeError"
            or m.get("preflight_stage") != "output_collection" or m.get("preflight_returncode") != 137
            or m.get("preflight_stderr_preview") != ""
            or any(k in m for k in ("returncode", "runner_stage", "startup_seconds"))):
        return False
    from dataclasses import replace
    normalized = dict(m, preflight_stage="command_start")
    normalized.pop("preflight_returncode")
    return two.startup_retry_allowed(replace(result, detail="preflight:TimeoutError", metadata=normalized),
                                     request, image_id)


def validate_source(source, parent):
    """Replay ALL prior training rewards and completed audits before new spending."""
    docs = source["documents"]
    if source["run_id"] != two.RUN_ID or any(not selected_path(p) for p in docs):
        raise ValueError("unexpected source run/path")
    plan, setup, calibration = (docs[k + ".json"] for k in ("plan", "setup", "calibration"))
    two.validate_plan(plan, calibration)
    if docs["parent_verification.json"] != parent or parent["calibration"] != calibration:
        raise ValueError("calibration provenance changed")
    if "result.json" in docs:
        raise ValueError("source study is already complete")
    stops = [v for k, v in docs.items() if k.startswith("stopped-")]
    if (len(stops) != 1 or stops[0].get("stage") != "independent_evaluation"
            or stops[0].get("type") != "ValueError"
            or stops[0].get("detail") != "incomplete or extra execution evidence"):
        raise ValueError("unreviewed source stop")
    ids = study.validate_controls(docs["conformance.json"], setup["sandbox_image_id"])
    visited, retry_slots, arms = [], [], {}

    def check_batch(key, role, group):
        saved = docs[f"grading/{key}/result.json"]
        if ([s["sample_id"] for s in group] != two.validate_batch(key, role)
                or saved["samples"] != group or saved["role"] != role or saved["key"] != key
                or saved["plan_hash"] != digest(canonical_json(plan))
                or set(saved["records"]) != {s["sample_id"] for s in group}):
            raise ValueError("prior batch identity differs")
        check_intent(key, role, group)
        reports = [study.grade(s, saved["records"][s["sample_id"]], role, setup["sandbox_image_id"], plan)
                   for s in group]
        if reports != saved["reports"]:
            raise ValueError("prior scores differ from execution")
        if docs[f"grading/{key}/raw.json"] != {k: v for k, v in saved.items() if k != "reports"}:
            raise ValueError("raw batch changed")
        for report in reports:
            ids.extend(report["sandbox_ids"])
        extra, slots = two.verify_retries(saved, setup["sandbox_image_id"])
        ids.extend(extra)
        retry_slots.extend(slots)
        visited.append(key)
        return reports

    def check_intent(key, role, group):
        wanted = {"key": key, "samples": group, "role": role, "setup": setup,
                  "plan_hash": digest(canonical_json(plan)), "retry": False}
        intent = docs[f"grading/{key}/intent.json"]
        if any(intent.get(k) != v for k, v in wanted.items()):
            raise ValueError("prior batch intent differs")

    for condition in two.CONDITIONS:
        arm = arms[condition] = docs[f"arms/{condition}/result.json"]
        if arm.get("condition") != condition or arm["evidence"] != study.training_evidence(arm["metrics"]):
            raise ValueError("training evidence changed")
        if arm["checkpoint_hashes"]["24"] != arm["metrics"]["after_parameter_hash"]:
            raise ValueError("checkpoint hash changed")
        keys = [f"train-{condition}-{i:02d}" for i in range(study.STEPS)]
        if arm["rollout_keys"] != keys:
            raise ValueError("missing training groups")
        tokens = 0
        for index, key in enumerate(keys):
            group = docs[f"arms/{condition}/rollouts/{key}.json"]
            if index == 0 and [s["raw"] for s in group] != arm["first_rollouts"]:
                raise ValueError("initial rollout claim changed")
            reports = check_batch(key, "training", group)
            rewards = [study.reward(condition, r, "0", s["sample_id"]) for s, r in zip(group, reports)]
            draws = {s["sample_id"]: str(study.noise_draw(study.NOISE_SEED, s["sample_id"])) for s in group}
            if (docs[f"arms/{condition}/rewards/{key}.json"] != {"rewards": rewards, "reports": reports, "draws": draws}
                    or rewards != arm["metrics"]["rewards"][index]):
                raise ValueError("consumed training reward changed")
            tokens += sum(s["tokens"] for s in group)
        if tokens != arm["metrics"]["generated_rollout_tokens"]:
            raise ValueError("training token accounting changed")
    if (arms["reference"]["first_rollouts"] != arms["structured"]["first_rollouts"]
            or arms["reference"]["checkpoint_hashes"]["0"] != study.PARAMETER_HASH):
        raise ValueError("unmatched original policies")
    populations = {"baseline-00": arms["reference"]["evaluations"]["0"]}
    populations.update({f"{name}-{step:02d}": arm["evaluations"][str(step)]
                        for name, arm in arms.items() for step in study.CHECKPOINTS})
    samples, records, reports, batches = {}, {}, {}, {}
    for name, group in populations.items():
        if len(group) != 16 or [s["seed"] for s in group] != list(study.EVALUATION_SEEDS):
            raise ValueError("saved evaluation population changed")
        condition, step = name.rsplit("-", 1)
        directory = "reference" if condition == "baseline" else condition
        for s in group:
            study.validate_sample(s, plan)
            if docs[f"arms/{directory}/evaluation-{step}/samples/{s['sample_id']}.json"] != s:
                raise ValueError("saved generation differs from arm result")
            samples[s["sample_id"]], records[s["sample_id"]] = s, {}
        for index in range(4):
            key = f"evaluation-{name}-{index:02d}"
            batch = batches[key] = group[index * 4:(index + 1) * 4]
            if [s["sample_id"] for s in batch] != two.validate_batch(key, "evaluation"):
                raise ValueError("evaluation identity changed")
            if f"grading/{key}/result.json" in docs:
                for s, r in zip(batch, check_batch(key, "evaluation", batch)):
                    reports[s["sample_id"]] = r
                    records[s["sample_id"]] = docs[f"grading/{key}/result.json"]["records"][s["sample_id"]]
        summary_path = f"evaluation_summaries/{name}.json"
        if summary_path in docs and docs[summary_path] != two.summary(group, [reports[s["sample_id"]] for s in group]):
            raise ValueError("prior policy summary changed")
    expected_completed = {k for k in batches if k.startswith(("evaluation-baseline-00", "evaluation-reference-12"))}
    expected_completed.update(f"evaluation-reference-24-0{i}" for i in range(3))
    if set(visited[48:]) != expected_completed:
        raise ValueError("completed evaluation scope differs from reviewed stop")
    raw = docs[f"grading/{FAILED_BATCH}/raw.json"]
    check_intent(FAILED_BATCH, "evaluation", batches[FAILED_BATCH])
    if (raw["key"] != FAILED_BATCH or raw["role"] != "evaluation" or raw["samples"] != batches[FAILED_BATCH]
            or raw["plan_hash"] != digest(canonical_json(plan)) or raw["startup_retries"]
            or set(raw["records"]) != {s["sample_id"] for s in batches[FAILED_BATCH]}):
        raise ValueError("interrupted batch identity changed")
    cases = {c.input_hash: c for c in study.cases_for("evaluation")}
    failures, recorded = [], 0
    for s in batches[FAILED_BATCH]:
        for key, packed in raw["records"][s["sample_id"]].items():
            if key not in cases:
                raise ValueError("extra input in interrupted batch")
            record = docs[f"grading/{FAILED_BATCH}/inputs/{s['sample_id']}/{key}.json"]
            if record != {"sample_id": s["sample_id"], "input_hash": key, "execution": packed}:
                raise ValueError("per-input journal differs from raw batch")
            result = unpack_result(packed)
            recorded += 1
            if (s["sample_id"], key) == (FAILED_SAMPLE, FAILED_INPUT):
                if (result.metadata.get("sandbox_id") != FAILED_SANDBOX or result.detail != "preflight:RuntimeError"
                        or not pre_candidate_failure(result, request_for(s["source"], cases[key]), setup["sandbox_image_id"])):
                    raise ValueError("known failure is no longer a confirmed pre-candidate kill")
                failures.append(packed)
                ids.append(FAILED_SANDBOX)
            else:
                ids.append(require_evidence(result, study.BOOKING, setup["sandbox_image_id"], source=s["source"], case=cases[key]))
                records[s["sample_id"]][key] = packed
        required = set() if s["extraction_status"].startswith("rejected_") else set(cases)
        if set(records[s["sample_id"]]) == required:
            reports[s["sample_id"]] = study.grade(s, records[s["sample_id"]], "evaluation", setup["sandbox_image_id"], plan)
    if len(failures) != 1 or recorded != 92:
        raise ValueError("unexpected interrupted execution population")
    complete_paths = {p.split('/')[1] for p in docs if p.startswith("grading/") and p.endswith("/result.json")}
    intent_paths = {p.split('/')[1] for p in docs if p.startswith("grading/") and p.endswith("/intent.json")}
    if complete_paths != set(visited) or intent_paths != set(visited) | {FAILED_BATCH}:
        raise ValueError("unreviewed prior or pending batch")
    if (len(ids) != len(set(ids)) or len(ids) > plan["max_sandbox_executions"]
            or len(retry_slots) != len(set(retry_slots)) or len(retry_slots) > two.MAX_STARTUP_RETRIES):
        raise ValueError("source sandbox/retry accounting invalid")
    pending = [(sid, c.input_hash) for sid, s in samples.items() if not s["extraction_status"].startswith("rejected_")
               for c in cases.values() if c.input_hash not in records[sid]]
    return {"plan": plan, "setup": setup, "calibration": calibration, "arms": arms, "populations": populations,
            "samples": samples, "records": records, "reports": reports, "pending": pending,
            "source_sandbox_ids": ids, "source_startup_retries": len(retry_slots),
            "source_hash": digest(canonical_json(source)), "failed_execution": failures[0]}


def make_plan(state):
    return {"version": VERSION, "run_id": RUN_ID, "source_run_id": two.RUN_ID,
            "source_hash": state["source_hash"], "source_app_id": SOURCE_APP,
            "pending": [list(pair) for pair in state["pending"]], "max_new_starts": len(state["pending"]) + MAX_EXTRA_STARTUPS,
            "max_extra_startups": MAX_EXTRA_STARTUPS, "max_controller_seconds": SECONDS,
            "generation_calls": 0, "training_calls": 0, "workers": 8, "creation_interval_seconds": .26,
            "known_replacement": [FAILED_SAMPLE, FAILED_INPUT], "automatic_expansion": False}


def validate_plan(plan, state):
    if plan != make_plan(state):
        raise ValueError("recovery plan changed or expanded")


def attempt_key(sid, input_hash, attempt):
    if attempt not in (1, 2):
        raise ValueError("at most two pre-candidate attempts")
    return f"{sid}/{input_hash}/{attempt}"


def attempt_intent(state, plan, sid, key, number, deadline):
    attempt_key(sid, key, number)
    if (sid, key) not in state["pending"]:
        raise ValueError("input is already resolved or outside the saved population")
    return {"sample_id": sid, "input_hash": key, "source_hash": digest(state["samples"][sid]["source"]),
            "attempt": number, "plan_hash": digest(canonical_json(plan)), "deadline": deadline}


def verify_attempt(state, sid, key, attempts):
    if (sid, key) not in state["pending"] or len(attempts) not in (1, 2):
        raise ValueError("unapproved evaluation or attempt count")
    if len(attempts) == 2 and (sid, key) == (FAILED_SAMPLE, FAILED_INPUT):
        raise ValueError("known preflight failure may be replaced only once")
    case = next(c for c in study.cases_for("evaluation") if c.input_hash == key)
    sample, image = state["samples"][sid], state["setup"]["sandbox_image_id"]
    if len(attempts) == 2 and not pre_candidate_failure(unpack_result(attempts[0]), request_for(sample["source"], case), image):
        raise ValueError("retry after candidate execution or unconfirmed cleanup")
    result = unpack_result(attempts[-1])
    require_evidence(result, study.BOOKING, image, source=sample["source"], case=case)
    return [a["metadata"]["sandbox_id"] for a in attempts]


def verify_completion(source, parent, plan, attempts):
    state = validate_source(source, parent)
    validate_plan(plan, state)
    if set(attempts) != {sid + "/" + key for sid, key in state["pending"]}:
        raise ValueError("missing or extra recovered executions")
    records, new_ids, extras = deepcopy(state["records"]), [], 0
    for sid, key in state["pending"]:
        pair = attempts[sid + "/" + key]
        new_ids.extend(verify_attempt(state, sid, key, pair))
        extras += len(pair) - 1
        records[sid][key] = pair[-1]
    if (len(new_ids) != len(set(new_ids)) or set(new_ids) & set(state["source_sandbox_ids"])
            or extras > MAX_EXTRA_STARTUPS or len(new_ids) > plan["max_new_starts"]):
        raise ValueError("sandbox reuse or resource cap exceeded")
    summaries = {}
    for name, group in state["populations"].items():
        reports = [study.grade(s, records[s["sample_id"]], "evaluation", state["setup"]["sandbox_image_id"], state["plan"])
                   for s in group]
        summaries[name] = two.summary(group, reports)
    return {"status": "completed", "version": VERSION, "offline_verified": True,
            "source_hash": state["source_hash"], "policies": summaries,
            "training": {name: a["evidence"] for name, a in state["arms"].items()},
            "source_sandbox_starts": len(state["source_sandbox_ids"]), "new_sandbox_starts": len(new_ids),
            "additional_startup_retries": extras, "reused_complete_programs": len(state["reports"]),
            "generation_calls": 0, "training_calls": 0, "original_failure_preserved": True,
            "limitations": state["plan"]["limitations"]}


def budget_quote(plan, rates, billing, prior):
    def number(value):
        n = Decimal(str(value))
        if not n.is_finite() or n < 0:
            raise ValueError("invalid budget value")
        return n
    held = max(number(prior), number(billing["metered_cost"]))
    cpu = number(rates["cpu_hour_cost"])
    memory = number(rates["mem_gib_hour_cost"])
    sandbox = plan["max_new_starts"] * 120 * (number(rates["cpu_hour_cost_sandbox"])
                + number(rates["mem_gib_hour_cost_sandbox"]) / 4) / 3600
    controller = 3 * SECONDS * (cpu + 2 * memory) / 3600
    return {"prior_reservation_held_usd": str(held), "rates": rates, "billing_before": billing,
            "sandbox_max_usd": str(sandbox), "controller_max_usd": str(controller), "gpu_max_usd": "0",
            "contingency_usd": "1", "cumulative_reservation_usd": str(held + sandbox + controller + 1),
            "is_invoice": False, "provider_hard_cap": False,
            "scope": "only saved pending evaluations; no training or generation"}


def load_attempts(directory):
    directory = Path(directory)
    values = {}
    for path in (directory / "attempts").glob("*/*/*/result.json"):
        sid, key, attempt = path.parts[-4:-1]
        values.setdefault(sid + "/" + key, {})[int(attempt)] = json.loads(path.read_text())
    for key, value in values.items():
        if set(value) != set(range(1, len(value) + 1)):
            raise ValueError("missing recovery attempt")
        values[key] = [value[i] for i in range(1, len(value) + 1)]
    return values


def verify_journal(directory, state, plan, deadline):
    directory = Path(directory)
    attempts = load_attempts(directory)
    intent_paths = {p.parent for p in (directory / "attempts").glob("*/*/*/intent.json")}
    result_paths = {p.parent for p in (directory / "attempts").glob("*/*/*/result.json")}
    if intent_paths != result_paths:
        raise ValueError("unresolved execution intent; explicit reconciliation required")
    expected_slots = []
    for pair, values in attempts.items():
        sid, key = pair.split("/")
        for index in range(1, len(values) + 1):
            path = directory / "attempts" / attempt_key(sid, key, index) / "intent.json"
            if json.loads(path.read_text()) != attempt_intent(state, plan, sid, key, index, deadline):
                raise ValueError("saved recovery intent differs")
        if len(values) == 2:
            expected_slots.append({"sample_id": sid, "input_hash": key, "failure": values[0]})
    actual_slots = []
    for path in (directory / "startup_slots").glob("*.json"):
        if not path.stem.isdigit() or not 0 <= int(path.stem) < MAX_EXTRA_STARTUPS:
            raise ValueError("invalid startup reservation slot")
        actual_slots.append(json.loads(path.read_text()))
    if sorted(map(canonical_json, expected_slots)) != sorted(map(canonical_json, actual_slots)):
        raise ValueError("startup reservation differs from recorded retry")
    return attempts


def main():
    parser = argparse.ArgumentParser(description="Verify completed CPU-only booking recovery")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    def read(name):
        return json.loads((args.run / (name + ".json")).read_text())
    source, parent, plan = read("source"), read("parent_verification"), read("plan")
    state = validate_source(source, parent)
    attempts = verify_journal(args.run, state, plan, read("deadline"))
    result = verify_completion(source, parent, plan, attempts)
    if read("result") != result:
        raise ValueError("saved final comparison differs")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
