"""Volume-first evidence storage. Retries here NEVER execute candidate code.

One grading Function owns its paths; files are immutable and atomically linked.
The work Volume is committed in bounded chunks and at program completion. A
compact program copy is also committed to the archive before Dict publication.
"""

import asyncio
import ast
import json
import os
from pathlib import Path
import tempfile

from .evaluation_journal import ReconciliationRequired
from .suites import canonical_json, digest

AMENDMENT = "evidence-storage-001"
RESUME_REVISION = "control-source-001"
ATTEMPTS = 3
RPC_SECONDS = 30
CHUNK_SIZE = 16
PARENT_SNAPSHOT = "6a962f5f44b3bb0446f6250f11683a1b09179bec7d118bd7f2143329a6a6563d"
EXTRA_STARTS = ("controller", "training/s20261013-endpoint_omission",
                "training/s20261014-endpoint_omission")


class StorageFailure(ReconciliationRequired):
    """Evidence/publication failure, not a candidate outcome or a zero reward."""


async def retry(operation, *, label, attempts=ATTEMPTS, timeout=RPC_SECONDS, sleep=asyncio.sleep):
    """Bound each RPC and its retries; immutable inserts make lost ACKs safe."""
    for attempt in range(attempts):
        try:
            return await asyncio.wait_for(operation(), timeout)
        except (ValueError, TypeError, KeyError):
            raise  # Conflicting evidence is not transient storage unavailability.
        except Exception as exc:
            if attempt + 1 == attempts:
                raise StorageFailure(f"{label}: {type(exc).__name__}; evidence retained, no execution retry") from exc
            print("EVIDENCE STORAGE RETRY", label, attempt + 1, type(exc).__name__, flush=True)
            await sleep(1 + 2 * attempt)


async def get(store, key, default=None):
    return await retry(lambda: store.get.aio(key, default), label="Dict read")


async def put_once(store, key, value):
    async def operation():
        inserted = await store.put.aio(key, value, skip_if_exists=True)
        if not inserted and await store.get.aio(key, None) != value:
            raise ValueError("immutable provider evidence changed: " + key)
        return inserted
    return await retry(operation, label="Dict publish")


def atomic_json(path, value):
    """Never expose a partially written JSON or overwrite different evidence."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("saved evidence differs: " + str(path))
        return
    fd, temporary = tempfile.mkstemp(prefix=".evidence-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)  # Atomic create-only publication (Volume v2).
        except FileExistsError:
            if json.loads(path.read_text()) != value:
                raise ValueError("concurrent evidence differs: " + str(path))
    finally:
        os.unlink(temporary)  # Only this function's private temporary file.


async def save(path, value):
    # Synchronous file operation completes before retry; no abandoned writer
    # thread can race a second write after a wait_for timeout.
    async def operation():
        atomic_json(path, value)
    await retry(operation, label="Volume evidence write")


def partial_cases(original, source, cases, image_id):
    """Only this reviewed Dict-callback interruption proves a never-run suffix."""
    from . import program_execution as execution
    checked = execution.validate_program(original, source, cases, image_id)
    failure = original["metadata"].get("failure", {})
    if (original.get("continuation") is not None or original["metadata"]["cleanup"] != "terminated"
            or failure != {"stage": "candidate", "type": "ResourceExhaustedError",
                           "detail": "Dict backend is overloaded"}):
        raise ValueError("not the reviewed storage-only program interruption")
    missing, gap = [], False
    for case in cases:
        record = original["records"][case.input_hash]
        if checked[case.input_hash]["passed"] is None:
            if (record["status"] != "infrastructure_error" or record["detail"] != "program_not_executed"
                    or record["stdout_base64"] or record["metadata"] != {
                        "input_hash": case.input_hash, "source_hash": digest(source)}):
                raise ValueError("ambiguous submitted input cannot be continued")
            gap = True
            missing.append(case)
        elif gap:
            raise ValueError("never-executed inputs must be a suffix")
    if not missing or len(missing) == len(cases):
        raise ValueError("reviewed prefix and never-executed suffix required")
    return missing


def continue_document(original, suffix, source, cases, image_id):
    missing = partial_cases(original, source, cases, image_id)
    from . import program_execution as execution
    outcomes = execution.validate_program(suffix, source, missing, image_id)
    if any(v["passed"] is None for v in outcomes.values()):
        raise ReconciliationRequired("continuation unresolved; retain both receipts and stop")
    if original["metadata"]["sandbox_id"] == suffix["metadata"]["sandbox_id"]:
        raise ValueError("continuation requires a new, separately identified sandbox")
    return {"metadata": dict(original["metadata"], reviewed_continuation=AMENDMENT),
            "input_order": original["input_order"],
            "records": {**original["records"], **suffix["records"]},
            "continuation": {"amendment": AMENDMENT, "original": original, "suffix": suffix}}


def validate_continuation(document, source, cases, image_id):
    continuation = document["continuation"]
    if set(continuation) != {"amendment", "original", "suffix"} or continuation["amendment"] != AMENDMENT:
        raise ValueError("unreviewed continuation")
    original, suffix = continuation["original"], continuation["suffix"]
    if "continuation" in original or "continuation" in suffix:
        raise ValueError("nested continuations prohibited")
    expected = continue_document(original, suffix, source, cases, image_id)
    if document != expected:
        raise ValueError("continuation changed original evidence or input identity")
    from . import program_execution as execution
    checked = execution.validate_program(original, source, cases, image_id)
    checked.update(execution.validate_program(suffix, source, partial_cases(original, source, cases, image_id), image_id))
    return checked


def sandbox_ids(document):
    if "continuation" in document:
        return [document["continuation"][name]["metadata"]["sandbox_id"] for name in ("original", "suffix")]
    sbid = document["metadata"].get("sandbox_id")
    return [sbid] if sbid else []


def validate_sources(before, after):
    """Allow storage/orchestration edits only; all scientific sources stay fixed."""
    if digest(canonical_json(before)) != PARENT_SNAPSHOT:
        raise ValueError("unexpected frozen parent sources")
    allowed = {"verifier_rl/program_storage.py", "verifier_rl/program_execution.py",
               "verifier_rl/program_grading.py", "modal_booking_replication.py"}
    changed = {n for n in before.keys() | after.keys() if before.get(n) != after.get(n)}
    if changed != allowed:
        raise ValueError("storage amendment source scope changed: " + str(changed))
    editable = {
        "verifier_rl/program_execution.py": {"ProgramBackend", "validate_program"},
        "verifier_rl/program_grading.py": {"metadata_for", "assemble", "execute_programs", "verify_raw"},
        "modal_booking_replication.py": {"require_program_authorization", "invoke", "grade_program_batch",
                                          "run_study", "program_launch", "main", "require_controls"},
    }
    for filename, names in editable.items():
        def frozen(source):
            return [ast.dump(n) for n in ast.parse(source).body
                    if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in names)
                    and not (isinstance(n, ast.ImportFrom) and any(a.name == "program_storage" for a in n.names))]
        if frozen(before[filename]) != frozen(after[filename]):
            raise ValueError("non-storage source changed: " + filename)
    # The untrusted child, sandbox settings and command payload remain exact.
    def execution_calls(source):
        tree = ast.parse(source)
        return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and ast.unparse(n.func) in
                ("self.sdk.Sandbox.create.aio", "sandbox.exec.aio")]
    if execution_calls(before["verifier_rl/program_execution.py"]) != execution_calls(after["verifier_rl/program_execution.py"]):
        raise ValueError("candidate execution parameters changed")
    return {"changed_files": sorted(changed), "parent_hash": digest(canonical_json(before)),
            "source_hash": digest(canonical_json(after)), "grpo_rewards_tests_unchanged": True}


def context_for(previous, snapshot):
    reviewed = validate_sources(previous["source_snapshot"], snapshot)
    return dict(previous, source_snapshot=snapshot, storage_amendment={
        "id": AMENDMENT, "parent_context_hash": digest(canonical_json(previous)),
        "sources": reviewed, "additional_start_keys": list(EXTRA_STARTS),
        "resume_revision": RESUME_REVISION, "additional_start_count": 2,
        "automatic_candidate_retries": 0, "extend_deadline": False, "new_budget": False})


def parent_context(context):
    # Original source texts remain in immutable context.json on both Volumes;
    # authorization binds their original context hash, without copying them here.
    amendment = context["storage_amendment"]
    if (amendment.get("id") != AMENDMENT or amendment.get("additional_start_keys") != list(EXTRA_STARTS)
            or amendment.get("resume_revision") != RESUME_REVISION or amendment.get("additional_start_count") != 2
            or amendment.get("automatic_candidate_retries") != 0
            or amendment.get("extend_deadline") is not False or amendment.get("new_budget") is not False
            or amendment["sources"]["source_hash"] != digest(canonical_json(context["source_snapshot"]))):
        raise ValueError("storage amendment identity changed")
    return amendment["parent_context_hash"]


def metadata_relative(context):
    amendment=context["storage_amendment"]
    parent_context(context)
    return "storage-amendments/"+AMENDMENT+"/"+amendment["resume_revision"]
