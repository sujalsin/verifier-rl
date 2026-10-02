"""Versioned, root-supervised candidate execution. Runner strings are remote-only.

The parent never imports candidate source. A separate non-root child receives no
grading answers, credentials or mounts. Only the parent emits the transport
envelope; candidate stdout/stderr are bounded data inside it. Historical runners
and ambiguous historical results are not reinterpreted.
"""

import base64
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math

from .grading import ExecutionResult, MAX_OUTPUT_BYTES, Status
from .modal_backend import Limits
from .panel_execution import PanelBackend, runner_for
from .suites import canonical_json, digest

VERSION = "root-supervised-panel-0.3"
STARTUP_RETRY_VERSION = "pre-candidate-timeout-0.1"
TRANSPORT_LIMIT = 65_536
SUPERVISOR_GRACE = 3

# Substituted trusted constants, never candidate text. Executed ONLY by Modal.
PARENT = r'''import base64
import hashlib
import json
import os
import selectors
import signal
import subprocess
import sys
import time

if os.getuid() != 0:
    raise RuntimeError("supervisor must retain root while candidate drops privileges")
payload = json.loads(sys.argv[1])
limit = payload["limits"]
ready_r, ready_w = os.pipe()
process = subprocess.Popen(
    [sys.executable, "-I", "-c", CHILD_RUNNER, sys.argv[1], str(ready_w)],
    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    start_new_session=True, close_fds=True, pass_fds=(ready_w,),
)
os.close(ready_w)
selector = selectors.DefaultSelector()
buffers = {"stdout": bytearray(), "stderr": bytearray(), "ready": bytearray()}
truncated = {"stdout": False, "stderr": False}
for fd, name in ((process.stdout.fileno(), "stdout"), (process.stderr.fileno(), "stderr"), (ready_r, "ready")):
    os.set_blocking(fd, False)
    selector.register(fd, selectors.EVENT_READ, name)
started_at = time.monotonic()
enforced, waited, usage, wait_status = None, False, None, None

def kill_group():
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

try:
    while not waited or selector.get_map():
        # Bound watchdog CPU use as well as candidate CPU use. In the live
        # runtime, frequent select/wait4 polling exhausted the inherited parent
        # CPU ceiling before a sleeping child reached its wall deadline.
        time.sleep(.05)
        elapsed = time.monotonic() - started_at
        if enforced is None and elapsed >= limit["wall_seconds"]:
            enforced = "wall"
            kill_group()
        if elapsed >= limit["wall_seconds"] + 1:
            raise RuntimeError("supervisor could not collect and reap child within grace")
        for key, _ in selector.select(timeout=.01):
            chunk = os.read(key.fd, 4096)
            if not chunk:
                selector.unregister(key.fd)
                continue
            name = key.data
            cap = 5 if name == "ready" else OUTPUT_LIMIT
            room = max(0, cap - len(buffers[name]))
            buffers[name].extend(chunk[:room])
            if len(chunk) > room:
                if name == "ready":
                    raise RuntimeError("invalid readiness channel")
                truncated[name] = True
                if enforced is None:
                    enforced = "output"
                    kill_group()
        if not waited:
            pid, status, stats = os.wait4(process.pid, os.WNOHANG)
            if pid:
                waited, wait_status, usage = True, status, stats
                process.returncode = os.waitstatus_to_exitcode(status)
                # A child must not leave descendants holding its output pipes.
                kill_group()
    envelope = {
        "version": EXECUTION_VERSION,
        "payload_hash": hashlib.sha256(sys.argv[1].encode()).hexdigest(),
        "candidate_started": buffers["ready"] == b"ready",
        "returncode": process.returncode,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss,
        "wall_seconds": time.monotonic() - started_at,
        "enforced_limit": enforced,
        "stdout_base64": base64.b64encode(buffers["stdout"]).decode("ascii"),
        "stderr_base64": base64.b64encode(buffers["stderr"]).decode("ascii"),
        "stdout_truncated": truncated["stdout"],
        "stderr_truncated": truncated["stderr"],
    }
    print(json.dumps(envelope, sort_keys=True, separators=(",", ":")))
finally:
    kill_group()
    if not waited:
        try:
            _, status, _ = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            pass
    selector.close()
    os.close(ready_r)
    process.stdout.close()
    process.stderr.close()
'''


def supervised_runner(task_id):
    child = runner_for(task_id)
    # Keep the original soft/hard CPU limit (2, 2). The live environment passes
    # this hard ceiling to descendants; raising it to 3 fails before candidate
    # launch. wait4 CPU accounting, not a guessed meaning of 137, attributes
    # limit exhaustion. Unknown signals below the budget remain unscored.
    replacements = {
        'namespace = {"__name__": "candidate"}':
            'ready_fd = int(sys.argv[2])\nos.write(ready_fd, b"ready")\nos.close(ready_fd)\n'
            'namespace = {"__name__": "candidate"}',
    }
    for before, after in replacements.items():
        if child.count(before) != 1:
            raise ValueError("child bootstrap changed; review supervised runner")
        child = child.replace(before, after)
    return (f"CHILD_RUNNER = {child!r}\nOUTPUT_LIMIT = {MAX_OUTPUT_BYTES}\n"
            f"EXECUTION_VERSION = {VERSION!r}\n" + PARENT)


def inspect_envelope(envelope, payload_hash):
    """Replay protected parent telemetry; candidate text is never telemetry."""
    required = {"version", "payload_hash", "candidate_started", "returncode", "user_cpu_seconds",
                "system_cpu_seconds", "max_rss_kib", "wall_seconds", "enforced_limit",
                "stdout_base64", "stderr_base64", "stdout_truncated", "stderr_truncated"}
    if (set(envelope) != required or envelope["version"] != VERSION
            or envelope["payload_hash"] != payload_hash
            or type(envelope["returncode"]) is not int or not -64 <= envelope["returncode"] <= 255
            or any(type(envelope[k]) is not bool for k in ("candidate_started", "stdout_truncated", "stderr_truncated"))
            or envelope["enforced_limit"] not in (None, "wall", "output")):
        raise ValueError("invalid supervisor envelope identity/schema")
    for field in ("user_cpu_seconds", "system_cpu_seconds", "max_rss_kib", "wall_seconds"):
        value = envelope[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid supervisor resource measurement")
    streams = []
    for name in ("stdout", "stderr"):
        encoded = envelope[name + "_base64"]
        if not isinstance(encoded, str) or len(encoded) > 4 * (MAX_OUTPUT_BYTES + 2) // 3:
            raise ValueError("oversized supervisor stream")
        stream = base64.b64decode(encoded, validate=True)
        if len(stream) > MAX_OUTPUT_BYTES:
            raise ValueError("oversized candidate stream")
        streams.append(stream)
    out, err = streams
    if not envelope["candidate_started"]:
        status, reason = Status.INFRASTRUCTURE_ERROR, "child_bootstrap_failed"
    elif envelope["enforced_limit"] == "wall":
        if envelope["wall_seconds"] < Limits().wall_seconds:
            raise ValueError("wall timeout lacks elapsed-time evidence")
        status, reason = Status.TIMEOUT, "supervisor_wall_limit"
    elif envelope["enforced_limit"] == "output":
        if not (envelope["stdout_truncated"] or envelope["stderr_truncated"]):
            raise ValueError("output limit lacks truncation evidence")
        status, reason = Status.OUTPUT_LIMIT, "supervisor_output_limit"
    elif envelope["stdout_truncated"] or envelope["stderr_truncated"]:
        raise ValueError("unexplained truncated stream")
    elif (envelope["returncode"] == -24
          or envelope["user_cpu_seconds"] + envelope["system_cpu_seconds"] >= Limits().cpu_seconds):
        status, reason = Status.TIMEOUT, "supervisor_cpu_limit_or_sigxcpu"
    elif envelope["returncode"] < 0:
        status, reason = Status.INFRASTRUCTURE_ERROR, "unattributed_child_signal"
    elif envelope["returncode"]:
        status, reason = Status.CANDIDATE_ERROR, "child_nonzero_exit"
    else:
        status, reason = Status.COMPLETED, ""
    return status, reason, out, err


def decode_transport(result, task_id, source, arguments):
    metadata = dict(result.metadata, execution_version=VERSION)
    if result.status != Status.COMPLETED or metadata.get("cleanup") != "terminated":
        return replace(result, status=Status.INFRASTRUCTURE_ERROR, stdout=b"",
                       detail="supervisor_transport:" + result.detail, metadata=metadata)
    try:
        envelope = json.loads(result.stdout)
        payload = canonical_json({"source": source, "input": arguments, "limits": asdict(Limits())})
        status, detail, out, err = inspect_envelope(envelope, digest(payload))
        if result.stdout != (canonical_json(envelope) + "\n").encode():
            raise ValueError("noncanonical supervisor output")
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        return ExecutionResult(Status.INFRASTRUCTURE_ERROR, detail="invalid_supervisor_report:" + str(exc),
                               metadata=metadata)
    metadata.update(supervisor_report=envelope, supervisor_returncode=metadata["returncode"],
                    supervisor_stdout_sha256=metadata["stdout_sha256"],
                    returncode=envelope["returncode"], stdout_bytes=len(out), stdout_preview=out[:2048].decode(errors="replace"),
                    stdout_sha256=sha256(out).hexdigest(), stderr_bytes=len(err), stderr_preview=err[:2048].decode(errors="replace"))
    return ExecutionResult(status, out, detail, False, metadata)


class SupervisedPanelBackend(PanelBackend):
    transport_output_limit = TRANSPORT_LIMIT

    def __init__(self, app_name, image_id, task_id, **kwargs):
        super().__init__(app_name, image_id, task_id, **kwargs)
        if self.limits != Limits():
            raise ValueError("supervised execution limits must be versioned explicitly")
        self.runner = supervised_runner(task_id)

    @property
    def candidate_timeout_seconds(self):
        return self.limits.wall_seconds + SUPERVISOR_GRACE

    async def execute(self, request):
        result = await super().execute(request)
        return decode_transport(result, self.task_id, request.source, self.decode_input(request.input_json))


def validate_report(result, task_id, image_id, *, source, case):
    """Validate a protected report, including an explicitly unscored signal."""
    m = result.metadata
    if (m.get("execution_version") != VERSION or m.get("runner_hash") != digest(supervised_runner(task_id))
            or m.get("backend") != "modal" or m.get("task_id") != task_id or m.get("image_id") != image_id
            or m.get("limits") != asdict(Limits()) or m.get("cleanup") != "terminated"
            or m.get("preflight_returncode") != 0 or m.get("sdk_version") != "1.5.5"
            or m.get("block_network") is not True or m.get("reset") != "fresh_sandbox_per_input"
            or m.get("creation_interval_seconds") != .26 or m.get("supervisor_returncode") != 0
            or m.get("source_hash") != digest(source) or m.get("input_hash") != case.input_hash
            or not isinstance(m.get("sandbox_id"), str) or not m["sandbox_id"].startswith("sb-")):
        raise ValueError("supervised execution identity/limits/cleanup mismatch")
    elapsed = m.get("total_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid sandbox lifecycle timing")
    payload = canonical_json({"source": source, "input": case.arguments, "limits": asdict(Limits())})
    envelope = m["supervisor_report"]
    status, detail, out, err = inspect_envelope(envelope, digest(payload))
    if (result.status != status or result.detail != detail or result.stdout != out or result.retryable
            or m["returncode"] != envelope["returncode"] or m.get("stdout_bytes") != len(out)
            or m.get("stdout_sha256") != sha256(out).hexdigest() or m.get("stderr_bytes") != len(err)
            or m.get("supervisor_stdout_sha256") != digest(canonical_json(envelope) + "\n")):
        raise ValueError("supervisor report differs from saved outcome")
    return m["sandbox_id"]


def require_evidence(result, task_id, image_id, *, source, case):
    if result.status == Status.INFRASTRUCTURE_ERROR:
        raise ValueError("unresolved supervised execution: " + result.detail)
    return validate_report(result, task_id, image_id, source=source, case=case)


def startup_retry_allowed(result, request, image_id, task_id, *, policy_version=None):
    from .booking_recovery import pre_candidate_failure
    if policy_version not in (None, STARTUP_RETRY_VERSION):
        raise ValueError("unknown startup retry policy")
    if (result.metadata.get("runner_hash") != digest(supervised_runner(task_id))
            or result.metadata.get("execution_version") != VERSION
            or not result.detail.startswith("supervisor_transport:")):
        return False
    # Only adapt the runner identity for the existing reviewed PRE-candidate
    # rule. No candidate report, timeout or signal is made retryable here.
    legacy = replace(result, detail=result.detail.removeprefix("supervisor_transport:"),
                     metadata=dict(result.metadata, runner_hash=digest(runner_for(task_id))))
    if policy_version == STARTUP_RETRY_VERSION:
        m = legacy.metadata
        # Explicit opt-in for NEW runs only. Historical retry decisions retain
        # their original semantics. The trusted preflight contains no candidate
        # source; candidate submission happens only after successful preflight.
        if any(k in m for k in ("supervisor_report", "supervisor_returncode", "runner_stage", "returncode", "startup_seconds")):
            return False
        code = m.get("preflight_returncode")
        if (legacy.status == Status.INFRASTRUCTURE_ERROR and legacy.detail == "preflight:TimeoutError"
                and m.get("preflight_stage") == "output_collection"
                and ("preflight_returncode" not in m or (type(code) is int and code == -1))
                and m.get("preflight_stderr_preview", "") == ""):
            normalized = dict(m, preflight_stage="command_start")
            normalized.pop("preflight_returncode", None)
            legacy = replace(legacy, metadata=normalized)
    return pre_candidate_failure(legacy, request, image_id)
