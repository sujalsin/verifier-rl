"""Reviewed startup-only recovery; original research plan and attempts stay intact."""

import ast
from .sandbox_lifecycle import validate_terminal
from .suites import canonical_json, digest

VERSION = "booking-cleanup-recovery-0.1"
AMENDMENT = "cleanup-001"
RUN_ID = "qwen-booking-matched-training-20260929-v1"
FAILED_KEY = "train-reference-19"
ORIGINAL_SNAPSHOT_HASH = "59dc04734c4c1394053e54687a67afc3db1021343fd9d09268bca8c20a64f50f"
REVIEWED_FAILURES = {
    "e777fb49fd5deb297bc9c0eef3a18c22837eef08987e38da07ff78d84436b82f": (
        "train-reference-19-2", "c301873e984583c0975fcdbdcb8d36f7869c478cf8777b18936856ed6cd09f76",
        "sb-HEeYfyxpUntm9xvfVAkfte"),
    "1c73a06a480f200cbde90002abcca5526865b99d04f0f7c645f0e27e61395674": (
        "train-reference-19-1", "843bcdb72aacf5362b6daa863b81e7773f2713d69d5974f9f71756fb46c44de0",
        "sb-1DNrkh0Pef9YliSju0J9nS"),
}
CHANGED_FILES = {"modal_booking_matched_training.py", "verifier_rl/modal_backend.py",
                 "verifier_rl/durable_grading.py", "verifier_rl/booking_screen_recovery.py"}
ADDED_FILES = {"verifier_rl/sandbox_lifecycle.py", "verifier_rl/booking_cleanup_recovery.py"}


def validate_reconciliation(first, receipt, sample, case):
    record_hash = digest(canonical_json(first))
    m = first["metadata"]
    if (REVIEWED_FAILURES.get(record_hash) != (sample["sample_id"], case.input_hash, m.get("sandbox_id"))
            or receipt.get("version") != VERSION or receipt.get("run_id") != RUN_ID
            or receipt.get("first_record_hash") != record_hash
            or receipt.get("original_snapshot_hash") != ORIGINAL_SNAPSHOT_HASH
            or first.get("status") != "infrastructure_error"
            or first.get("detail") != "supervisor_transport:cleanup_unconfirmed"
            or first.get("stdout_base64") != "" or m.get("preflight_stage") != "command_start"
            or m.get("source_hash") != digest(sample["source"]) or m.get("input_hash") != case.input_hash
            or m.get("candidate_submission_attempted", False) is not False
            or any(k in m for k in ("returncode", "runner_stage", "startup_seconds", "supervisor_report",
                                     "preflight_returncode", "execution_roundtrip_seconds"))):
        raise ValueError("unreviewed or post-candidate cleanup recovery")
    validate_terminal(receipt["terminal"], m["sandbox_id"])
    return True


def make_reconciliation(first, terminal, sample, case):
    receipt = {"version": VERSION, "run_id": RUN_ID, "first_record_hash": digest(canonical_json(first)),
               "original_snapshot_hash": ORIGINAL_SNAPSHOT_HASH, "terminal": terminal}
    validate_reconciliation(first, receipt, sample, case)
    return receipt


def validate_amendment(original, current):
    if digest(canonical_json(original)) != ORIGINAL_SNAPSHOT_HASH:
        raise ValueError("unreviewed original source snapshot")
    if set(current) != set(original) | ADDED_FILES:
        raise ValueError("unexpected files in recovery snapshot")
    changed = {name for name in original if original[name] != current[name]}
    if changed != CHANGED_FILES:
        raise ValueError("recovery changed unapproved source files")
    # The live exact-resume control remains applicable: all training, model,
    # evaluation-generation, scoring, suite and GRPO recovery code is unchanged.
    launcher = "modal_booking_matched_training.py"
    def functions(source):
        return {node.name: ast.dump(node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    before, after = functions(original[launcher]), functions(current[launcher])
    allowed = {"grade_batch", "run_comparison", "verify_comparison"}
    if any(before[name] != after.get(name) for name in before.keys() - allowed):
        raise ValueError("tested trainer or generation code changed")
    return {"version": VERSION, "original_snapshot_hash": ORIGINAL_SNAPSHOT_HASH,
            "new_snapshot_hash": digest(canonical_json(current)), "changed_files": sorted(changed),
            "added_files": sorted(ADDED_FILES), "plan_unchanged": True, "deadline_unchanged": True,
            "resume_checkpoint": 19, "pending_group": 19, "max_historical_replacements": 2,
            "replacements_use_existing_retry_budget": True}
