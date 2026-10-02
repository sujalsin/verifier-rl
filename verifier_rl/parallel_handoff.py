"""Read-only partition and aggregation for an explicitly drained grader.

Never erase old intents, release lost-call holds, resample or change scoring.
An uncertain/partially submitted old batch is a review blocker, not new work.
"""
import json
from pathlib import Path

from . import booking_replication as study, program_grading as old
from . import parallel_evaluation as parallel, program_storage as storage
from .evaluation_journal import ReconciliationRequired


def read(path):
    return json.loads(Path(path).read_text())


def check_old(result, samples, key, parent):
    raw = result["raw"]
    checked = old.verify_raw(raw, samples, key, "evaluation", parent["plan"], parent["program_runtime"])
    receipt = result["receipt"]
    if (result["checked"] != checked or receipt["raw_hash"] != parallel.fingerprint(raw)
            or receipt["checked_hash"] != parallel.fingerprint(checked)
            or receipt["reader"]["version"] != old.VERSION or receipt["reader"]["key"] != key
            or receipt["reader"]["journal_files"] != sum(1 for _ in old.journal_records(key, raw))):
        raise ValueError("legacy verified receipt changed")
    return checked


def classify_old(manifest, directory, parent, intents):
    """Called after drain. Intents are read from the old provider namespace."""
    directory = Path(directory)
    key, samples = manifest["key"], manifest["samples"]
    if set(intents) != {s["sample_id"] for s in samples}:
        raise ValueError("complete provider-intent inspection required")
    path = directory/"grading"/key/"result.json"
    if path.exists():
        result = read(path)
        checked = check_old(result, samples, key, parent)
        return {"status": "completed", "result_hash": parallel.fingerprint(result),
                "raw_hash": result["receipt"]["raw_hash"], "rows": checked["rows"],
                "sandbox_ids": checked["sandbox_ids"], "legacy_path": str(path)}
    for sample in samples:
        sid = sample["sample_id"]
        if sample["extraction_status"].startswith("rejected_"):
            continue
        if intents[sid] is not None or (directory/"grading"/key/"programs"/sid).exists():
            raise ReconciliationRequired("old program may have executed; no automatic replay: " + sid)
    return {"status": "never_started"}


def check_new(manifest, raw, receipt):
    if raw["manifest"] != manifest or set(raw["documents"]) != {s["sample_id"] for s in manifest["samples"]}:
        raise ValueError("parallel fixed sample pool changed")
    cases = study.pilot.cases_for("evaluation")
    rows, ids, hashes = [], [], []
    for sample in manifest["samples"]:
        document = raw["documents"][sample["sample_id"]]
        outcomes = parallel.checked_program(sample, document, cases, manifest["runtime"]["image_id"])
        rows.append(study.program_summary(sample, outcomes, "evaluation"))
        hashes.append(parallel.fingerprint(outcomes))
        if "rejected" not in document: ids.extend(storage.sandbox_ids(document))
    expected = {"raw_hash": parallel.fingerprint(raw), "rows": rows,
                "job_hash": parallel.fingerprint(manifest), "outcome_hashes": hashes}
    if receipt != expected or len(ids) != len(set(ids)):
        raise ValueError("parallel verified receipt differs")
    return {"rows": rows, "sandbox_ids": ids}


def assemble(baseline, rows_by_key, saved_arms):
    """Original metrics and paired analysis, all fixed seeds and samples required."""
    expected = {study.label(s,a) for s in study.SEEDS for a in study.ARMS}
    if set(saved_arms) != expected:
        raise ValueError("missing training arms")
    expected_keys = {f"eval-{policy}-{step:02d}-{i:02d}" for policy in expected
                     for step, count in ((12,8),(24,32)) for i in range(count)}
    if set(rows_by_key) != expected_keys:
        raise ValueError("all 480 post-training batches required; no subset conclusion")
    if baseline["summary"] != study.policy_summary(baseline["programs"], "baseline", 0):
        raise ValueError("baseline summary changed")
    policies, ids = {}, list(baseline["sandbox_ids"])
    for policy in sorted(expected):
        for step, count in ((12,8),(24,32)):
            rows = []
            for i in range(count):
                result = rows_by_key[f"eval-{policy}-{step:02d}-{i:02d}"]
                rows.extend(result["rows"])
                ids.extend(result["sandbox_ids"])
            policies[f"{policy}-{step}"] = {"programs": rows, "summary": study.policy_summary(rows, policy, step)}
    if len(ids) != len(set(ids)):
        raise ValueError("sandbox identity shared across distinct evaluated programs")
    return {"status": "completed_with_uncertainty" if any(p["summary"]["unknown_inputs"] for p in policies.values())
            or baseline["summary"]["unknown_inputs"] else "completed",
            "analysis": study.analyze(baseline,policies),
            "training_metrics": {k:v["metrics"] for k,v in saved_arms.items()},
            "policies": policies, "unique_evaluation_sandboxes": len(ids),
            "pilot_kept_separate": True, "training_unchanged": True}
