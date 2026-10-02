"""Bounded, read-only terminal evidence for the pinned Modal SDK.

SandboxWait can have no result even when SandboxGetTaskId reports an exited
task. Never infer termination from an empty active list, elapsed time, or an
acknowledged termination request alone. This module never executes code.
"""

import asyncio
import math
import re
import time

VERSION = "modal-terminal-evidence-0.1"
TERMINAL_STATUSES = frozenset(range(1, 9))  # Modal 1.5.5 GenericResult enum.


def validate_terminal(evidence, sandbox_id):
    if (evidence.get("version") != VERSION or evidence.get("sandbox_id") != sandbox_id
            or type(evidence.get("checked_at")) not in (int, float)
            or not math.isfinite(evidence["checked_at"]) or evidence["checked_at"] <= 0):
        raise ValueError("terminal evidence identity/time changed")
    if evidence.get("method") == "Sandbox.poll":
        if type(evidence.get("returncode")) is not int:
            raise ValueError("sandbox has no terminal exit status")
    elif evidence.get("method") == "SandboxGetTaskId.task_result":
        if (evidence.get("sdk_version") != "1.5.5"
                or re.fullmatch(r"sb-[A-Za-z0-9]{22}", sandbox_id) is None
                or not isinstance(evidence.get("task_id"), str) or not evidence["task_id"].startswith("ta-")
                or type(evidence.get("status")) is not int or evidence["status"] not in TERMINAL_STATUSES
                or type(evidence.get("exitcode")) is not int):
            raise ValueError("task has no supported terminal result")
    else:
        raise ValueError("unsupported terminal evidence source")
    return evidence


async def task_result(sandbox_id, timeout):
    """Explicit pinned v1 adapter; fail closed for other SDKs or sandbox formats."""
    import modal
    from modal.client import _Client
    from modal_proto import api_pb2
    if modal.__version__ != "1.5.5" or re.fullmatch(r"sb-[A-Za-z0-9]{22}", sandbox_id) is None:
        raise ValueError("task-result fallback requires pinned Modal v1 adapter")
    client = await asyncio.wait_for(_Client.from_env(), timeout)
    response = await asyncio.wait_for(client.stub.SandboxGetTaskId(
        api_pb2.SandboxGetTaskIdRequest(sandbox_id=sandbox_id, timeout=0)), timeout)
    if not response.HasField("task_result"):
        return None
    return {"sdk_version": modal.__version__, "task_id": response.task_id,
            "status": response.task_result.status, "exitcode": response.task_result.exitcode}


async def confirm_terminal(sandbox, *, timeout=10, lookup=task_result, clock=time.time):
    """At most one poll and one task-result query; no candidate/termination retry."""
    base = {"version": VERSION, "sandbox_id": sandbox.object_id}
    try:
        code = await asyncio.wait_for(sandbox.poll.aio(), timeout)
    except Exception as exc:
        base["poll_error"] = type(exc).__name__
    else:
        if type(code) is int:
            return validate_terminal(dict(base, method="Sandbox.poll", returncode=code,
                                          checked_at=clock()), sandbox.object_id)
    record = await asyncio.wait_for(lookup(sandbox.object_id, timeout), 2*timeout)
    if record is None:
        raise ValueError("sandbox termination remains unconfirmed")
    return validate_terminal(dict(base, **record, method="SandboxGetTaskId.task_result",
                                  checked_at=clock()), sandbox.object_id)
