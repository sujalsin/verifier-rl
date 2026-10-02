"""Allowlisted panel invocations over the existing isolated Modal transport.

Runner strings and fixture strings are sent ONLY to a remote sandbox. They are
never executed on the laptop, GPU worker, or trusted controller.
"""

from dataclasses import asdict, replace
import base64
from hashlib import sha256
import inspect
import json
import math

from .fixtures import source_for
from .grading import ExecutionRequest, ExecutionResult, MAX_OUTPUT_BYTES, Status
from .modal_backend import BOOTSTRAP, Limits, ModalBackend
from .suites import canonical_json, digest
from .task_panel import BOOKING, CACHE, LIMITER, task_for, validate_call

VERSION = "panel-execution-0.1"


def runner_for(task_id):
    task = task_for(task_id)  # No caller-controlled name, expression or import.
    predicate = {
        "int_or_null_list": "type(answer) is list and all(x is None or type(x) is int for x in answer)",
        "bool_list": "type(answer) is list and all(type(x) is bool for x in answer)",
        "nonnegative_int": "type(answer) is int and answer >= 0",
    }[task.output_kind]
    return BOOTSTRAP + '''
payload = json.loads(sys.argv[1])
prepare(payload["limits"])
namespace = {"__name__": "candidate"}
def report_stage(name):
    os.write(2, b"VERIFIER_RL_STAGE=" + name.encode("ascii") + bytes((10,)))
report_stage("module_load")
exec(compile(payload["source"], "<candidate>", "exec"), namespace)
report_stage("entrypoint_lookup")
''' + f'''
entrypoint = {task.entrypoint!r}
if entrypoint not in namespace or not callable(namespace[entrypoint]):
    raise TypeError("required callable " + entrypoint + " is missing")
report_stage("function_call")
answer = namespace[entrypoint](**payload["input"])
report_stage("return_validation")
if not ({predicate}):
    raise TypeError("invalid {task.output_kind} return schema")
report_stage("result_serialization")
print(json.dumps(answer, allow_nan=False, separators=(",", ":")))
'''


class PanelBackend(ModalBackend):
    def __init__(self, app_name, image_id, task_id, **kwargs):
        self.task_id = task_for(task_id).task_id
        self.runner = runner_for(task_id)
        super().__init__(app_name, image_id, **kwargs)

    def decode_input(self, input_json):
        arguments = json.loads(input_json)
        validate_call(self.task_id, arguments)
        if canonical_json(arguments) != input_json:
            raise ValueError("noncanonical panel input")
        return arguments

    async def execute(self, request):
        arguments = self.decode_input(request.input_json)
        result = await super().execute(request)
        return replace(result, metadata=dict(result.metadata, task_id=self.task_id,
                       source_hash=digest(request.source),
                       input_hash=digest(canonical_json([self.task_id, arguments]))))


def request_for(source, case):
    return ExecutionRequest(source, case.arguments_json)


def pack_result(result):
    return {"status": result.status.value,
            "stdout_base64": base64.b64encode(result.stdout).decode("ascii"),
            "detail": result.detail, "retryable": result.retryable, "metadata": result.metadata}


def unpack_result(record):
    if set(record) != {"status", "stdout_base64", "detail", "retryable", "metadata"}:
        raise ValueError("unexpected execution record fields")
    if len(record["stdout_base64"]) > 4 * (MAX_OUTPUT_BYTES + 2) // 3:
        raise ValueError("oversized saved output")
    output = base64.b64decode(record["stdout_base64"], validate=True)
    if len(output) > MAX_OUTPUT_BYTES or type(record["retryable"]) is not bool:
        raise ValueError("invalid saved execution")
    return ExecutionResult(Status(record["status"]), output, record["detail"],
                           record["retryable"], record["metadata"])


def require_evidence(result, task_id, image_id, *, source=None, case=None):
    """Verify transport/cleanup evidence; uncertainty remains unscored, never retried."""
    if result.status == Status.INFRASTRUCTURE_ERROR:
        raise ValueError("infrastructure failure: preserve raw record and stop")
    m = result.metadata
    if (m.get("backend") != "modal" or m.get("task_id") != task_id or m.get("image_id") != image_id
            or m.get("runner_hash") != digest(runner_for(task_id))
            or m.get("limits") != asdict(Limits()) or m.get("cleanup") != "terminated"
            or m.get("preflight_returncode") != 0 or m.get("sdk_version") != "1.5.5"
            or m.get("block_network") is not True or m.get("reset") != "fresh_sandbox_per_input"
            or m.get("creation_interval_seconds") != .26
            or not isinstance(m.get("sandbox_id"), str) or not m["sandbox_id"].startswith("sb-")):
        raise ValueError("panel execution identity/limits/cleanup mismatch")
    if ((source is not None and m.get("source_hash") != digest(source))
            or (case is not None and m.get("input_hash") != case.input_hash)):
        raise ValueError("source/input execution binding differs")
    elapsed = m.get("total_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid sandbox lifecycle timing")
    code = m.get("returncode")
    if result.status == Status.TIMEOUT or (type(code) is int and (code < 0 or code >= 128)):
        raise ValueError("unattributed termination: retain uncertainty and stop")
    if result.status in (Status.COMPLETED, Status.CANDIDATE_ERROR):
        if type(code) is not int or ((code == 0) != (result.status == Status.COMPLETED)):
            raise ValueError("status/exit mismatch")
        if (m.get("stdout_bytes") != len(result.stdout)
                or m.get("stdout_sha256") != sha256(result.stdout).hexdigest()):
            raise ValueError("stdout evidence mismatch")
    elif result.status != Status.OUTPUT_LIMIT:
        raise ValueError("unexpected executed status")
    return m["sandbox_id"]


def authored_booking(bookings, fault):
    if fault == "constant":
        return 0
    if fault == "inclusive_boundary":
        return max((sum(start <= point <= end for start, end in bookings)
                    for point, _ in bookings), default=0)
    return max((sum(start <= point < end for start, end in bookings)
                for point, _ in bookings), default=0)


def authored_limiter(requests, limit, window, fault):
    if fault == "constant":
        return []
    accepted, result = [], []
    for request in requests:
        count = sum(previous["user"] == request["user"]
                    and (request["time"] - window <= previous["time"] if fault == "inclusive_boundary"
                         else request["time"] - window < previous["time"])
                    for previous in accepted)
        allowed = count < limit
        result.append(allowed)
        if allowed:
            accepted.append(request)
    return result


def fixture_source(task_id, fault):
    if fault not in ("correct", "inclusive_boundary", "constant"):
        raise ValueError("unknown authored control")
    task_for(task_id)
    if task_id == CACHE:
        if fault == "constant":
            return "def simulate_cache(operations):\n    return []\n"
        return source_for("inclusive_expiry" if fault == "inclusive_boundary" else "correct")
    if task_id == BOOKING:
        return inspect.getsource(authored_booking) + f"\ndef required_capacity(bookings):\n    return authored_booking(bookings, {fault!r})\n"
    return inspect.getsource(authored_limiter) + f"\ndef allow_requests(requests, limit, window):\n    return authored_limiter(requests, limit, window, {fault!r})\n"
