"""Opt-in Modal adapter. No local arbitrary-code execution backend exists.

SDK calls are contract-tested with fakes; live isolation/performance validation
is still required. See README.md before using this with generated programs.
"""

import asyncio
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import re
import time

from .cache import validate_input
from .grading import ExecutionRequest, ExecutionResult, MAX_OUTPUT_BYTES, MAX_SOURCE_BYTES, Status
from .suites import canonical_json, digest

# This bootstrap contains no expected answers, reference code, or scoring logic.
# It runs ONLY in the remote sandbox, never on the controller's machine.
BOOTSTRAP = '''import ctypes
import json
import os
import resource
import sys
import tempfile

def prepare(limits):
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11+ required")
    if os.getuid() != 0:
        raise RuntimeError("approved image must start as root for privilege drop")
    work = tempfile.mkdtemp(prefix="candidate-", dir="/tmp")
    os.chown(work, 65534, 65534)
    os.chdir(work)
    os.umask(0o077)
    for name, value in (
        (resource.RLIMIT_CPU, limits["cpu_seconds"]),
        (resource.RLIMIT_AS, limits["process_memory_mib"] * 1024 * 1024),
        (resource.RLIMIT_FSIZE, 1024 * 1024),
        (resource.RLIMIT_NOFILE, 32),
        (resource.RLIMIT_NPROC, 16),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(name, (value, value))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # Linux PR_SET_NO_NEW_PRIVS
        raise RuntimeError("no_new_privs unavailable")
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    os.environ.clear()
    os.environ.update({"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"})
    if os.getuid() == 0 or os.geteuid() == 0:
        raise RuntimeError("privilege drop failed")
'''

PREFLIGHT = BOOTSTRAP + '''
prepare(json.loads(sys.argv[1]))
print(json.dumps({"ready": True, "python": sys.version, "uid": os.getuid()}))
'''

RUNNER = BOOTSTRAP + '''
payload = json.loads(sys.argv[1])
prepare(payload["limits"])
namespace = {"__name__": "candidate"}
def report_stage(name):
    os.write(2, b"VERIFIER_RL_STAGE=" + name.encode("ascii") + bytes((10,)))

report_stage("module_load")
exec(compile(payload["source"], "<candidate>", "exec"), namespace)
report_stage("entrypoint_lookup")
if "simulate_cache" not in namespace or not callable(namespace["simulate_cache"]):
    raise TypeError("required callable simulate_cache is missing")
report_stage("function_call")
answer = namespace["simulate_cache"](payload["input"])
report_stage("return_validation")
if type(answer) is not list or any(x is not None and type(x) is not int for x in answer):
    raise TypeError("simulate_cache must return a list of integers or None")
report_stage("result_serialization")
print(json.dumps(answer, allow_nan=False, separators=(",", ":")))
'''


@dataclass(frozen=True)
class Limits:
    # Provisional development budgets; benchmark before fixing training budgets.
    wall_seconds: int = 5
    cpu_seconds: int = 2
    process_memory_mib: int = 128
    sandbox_memory_mib: int = 256
    rpc_seconds: int = 45
    sandbox_lifetime_seconds: int = 120

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("limits must be positive integers")
        if self.cpu_seconds > self.wall_seconds:
            raise ValueError("CPU budget must not exceed wall budget")
        if self.process_memory_mib >= self.sandbox_memory_mib:
            raise ValueError("leave memory for the sandbox's support processes")
        if self.sandbox_lifetime_seconds <= self.rpc_seconds + self.wall_seconds:
            raise ValueError("sandbox lifetime must include startup and execution headroom")


class OutputLimitError(Exception):
    pass


async def read_bounded(stream, limit=MAX_OUTPUT_BYTES):
    buffer = bytearray()
    async for chunk in stream:
        if not isinstance(chunk, bytes):
            raise TypeError("binary stream expected")
        if len(buffer) + len(chunk) > limit:
            raise OutputLimitError()
        buffer.extend(chunk)
    return bytes(buffer)


async def collect(process):
    tasks = [asyncio.create_task(read_bounded(process.stdout)),
             asyncio.create_task(read_bounded(process.stderr)),
             asyncio.create_task(process.wait.aio())]
    try:
        stdout, stderr, returncode = await asyncio.gather(*tasks)
        return stdout, stderr, returncode
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class ModalBackend:
    def __init__(self, app_name: str, image_id: str, limits=None, *, sdk=None,
                 creation_interval_seconds: float = 0.0):
        if not app_name or re.fullmatch(r"im-[A-Za-z0-9]+", image_id) is None:
            raise ValueError("an app name and a resolved Modal image ID (im-...) are required")
        self.app_name = app_name
        self.image_id = image_id
        self.limits = limits or Limits()
        self._sdk = sdk
        self._app = None
        self._image = None
        self._initialize_lock = asyncio.Lock()
        if type(creation_interval_seconds) not in (int, float) or not 0 <= creation_interval_seconds <= 5:
            raise ValueError("creation interval must be a finite number from 0 to 5 seconds")
        self.creation_interval_seconds = creation_interval_seconds
        self._creation_lock = asyncio.Lock()
        self._last_creation_at = 0.0

    async def _pace_creation(self):
        # Per-backend pacing, not an account-wide distributed rate limiter.
        # The baseline has one controller/backend and no overlapping batches.
        async with self._creation_lock:
            delay = self._last_creation_at + self.creation_interval_seconds - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_creation_at = time.monotonic()

    async def _initialize(self):
        async with self._initialize_lock:
            if self._app is not None:
                return
            if self._sdk is None:
                import modal  # Optional dependency, loaded only for explicit cloud use.
                self._sdk = modal
            # Require an existing app; do not silently create account resources.
            app = await asyncio.wait_for(
                self._sdk.App.lookup.aio(self.app_name, create_if_missing=False),
                timeout=self.limits.rpc_seconds)
            image = self._sdk.Image.from_id(self.image_id)
            self._app, self._image = app, image

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if not request.source.strip() or len(request.source.encode()) > MAX_SOURCE_BYTES:
            raise ValueError("invalid source size")
        operations = json.loads(request.input_json)
        validate_input(operations)
        sandbox = None
        started = time.monotonic()
        stage = "initialization"
        metadata = {"backend": "modal", "image_id": self.image_id,
                    "runner_hash": digest(RUNNER), "limits": asdict(self.limits),
                    "reset": "fresh_sandbox_per_input", "block_network": True,
                    "creation_interval_seconds": self.creation_interval_seconds}
        result = ExecutionResult(Status.INFRASTRUCTURE_ERROR, detail="not_started")
        try:
            await self._initialize()
            metadata["sdk_version"] = getattr(self._sdk, "__version__", "unknown")
            stage = "create"
            await self._pace_creation()
            sandbox = await asyncio.wait_for(self._sdk.Sandbox.create.aio(
                "sleep", "infinity", app=self._app, image=self._image,
                timeout=self.limits.sandbox_lifetime_seconds,
                cpu=(1.0, 1.0),
                memory=(self.limits.sandbox_memory_mib, self.limits.sandbox_memory_mib),
                block_network=True, secrets=[], volumes={},
                include_oidc_identity_token=False,
            ), timeout=self.limits.rpc_seconds)
            metadata["sandbox_id"] = sandbox.object_id
            stage = "preflight"
            metadata["preflight_stage"] = "command_start"
            preflight = await asyncio.wait_for(sandbox.exec.aio(
                "python3", "-I", "-c", PREFLIGHT, canonical_json(asdict(self.limits)),
                timeout=self.limits.wall_seconds, text=False,
            ), timeout=self.limits.rpc_seconds)
            metadata["preflight_stage"] = "output_collection"
            out, preflight_stderr, code = await asyncio.wait_for(collect(preflight), self.limits.rpc_seconds)
            metadata["preflight_returncode"] = code
            if code != 0:
                metadata["preflight_stderr_preview"] = preflight_stderr[:2048].decode("utf-8", errors="replace")
            if code == -1:
                # The trusted preflight timed out before any candidate launch.
                # Do not mask the provider timeout as JSONDecodeError on empty stdout.
                raise TimeoutError("trusted_preflight_execution_deadline")
            if code != 0:
                raise RuntimeError("preflight_nonzero_exit")
            ready = json.loads(out)
            if code != 0 or ready.get("ready") is not True or ready.get("uid") != 65534:
                raise RuntimeError("preflight_failed")
            metadata["runtime_python"] = ready["python"]
            metadata["preflight_stage"] = "complete"
            metadata["startup_seconds"] = time.monotonic() - started
            stage = "launch"
            payload = canonical_json({"source": request.source, "input": operations,
                                      "limits": asdict(self.limits)})
            process = await asyncio.wait_for(sandbox.exec.aio(
                "python3", "-I", "-c", RUNNER, payload,
                timeout=self.limits.wall_seconds, text=False,
            ), timeout=self.limits.rpc_seconds)
            stage = "candidate"
            running = time.monotonic()
            out, stderr, code = await asyncio.wait_for(
                collect(process), self.limits.wall_seconds + self.limits.rpc_seconds)
            metadata["execution_roundtrip_seconds"] = time.monotonic() - running
            metadata["returncode"] = code
            metadata["stdout_bytes"] = len(out)
            metadata["stdout_preview"] = out[:2048].decode("utf-8", errors="replace")
            metadata["stdout_sha256"] = sha256(out).hexdigest()
            metadata["stderr_bytes"] = len(stderr)
            metadata["stderr_preview"] = stderr[:2048].decode("utf-8", errors="replace")
            # This marker is emitted by the wrapper but shares stderr with
            # untrusted candidate code. Treat it as a diagnostic hint, not
            # authenticated proof of which code raised the exception.
            stages = re.findall(rb"(?m)^VERIFIER_RL_STAGE=([a-z_]+)$", stderr)
            metadata["runner_stage"] = stages[-1].decode("ascii") if stages else "unknown"
            # Modal 1.5.5 ContainerProcess.wait catches ExecTimeoutError and
            # returns -1; signals instead return 128 + signal. Do not mistake
            # this documented-in-SDK sentinel for an ordinary program error.
            if code == -1:
                result = ExecutionResult(Status.TIMEOUT, detail="remote_execution_deadline")
            else:
                result = ExecutionResult(Status.COMPLETED if code == 0 else Status.CANDIDATE_ERROR,
                                         stdout=out, detail="" if code == 0 else "nonzero_exit")
        except OutputLimitError:
            result = ExecutionResult(Status.OUTPUT_LIMIT if stage == "candidate" else Status.INFRASTRUCTURE_ERROR,
                                     detail=f"{stage}:stream_limit")
        except Exception as exc:
            exceptions = getattr(self._sdk, "exception", None)
            exec_timeout = getattr(exceptions, "ExecTimeoutError", ())
            transient = tuple(t for name in ("ServiceError", "ConnectionError", "InternalError")
                              if isinstance(t := getattr(exceptions, name, None), type))
            if stage in ("launch", "candidate") and isinstance(exc, exec_timeout):
                result = ExecutionResult(Status.TIMEOUT, detail="remote_execution_deadline")
            else:
                result = ExecutionResult(Status.INFRASTRUCTURE_ERROR,
                                         detail=f"{stage}:{type(exc).__name__}",
                                         retryable=isinstance(exc, transient))
        finally:
            # Terminate the entire sandbox, not merely the candidate's parent PID.
            # A finite provider lifetime remains a backstop if client/RPC cleanup fails.
            if sandbox is not None:
                try:
                    await asyncio.wait_for(sandbox.terminate.aio(wait=True), self.limits.rpc_seconds)
                    metadata["cleanup"] = "terminated"
                except Exception as exc:
                    metadata["cleanup"] = f"unconfirmed:{type(exc).__name__}"
                    result = ExecutionResult(Status.INFRASTRUCTURE_ERROR, detail="cleanup_unconfirmed")
                finally:
                    detach = getattr(sandbox, "detach", None)
                    if detach is not None:
                        try:
                            await asyncio.wait_for(detach.aio(), self.limits.rpc_seconds)
                        except Exception as exc:
                            metadata["detach_error"] = type(exc).__name__
            metadata["total_seconds"] = time.monotonic() - started
            if sandbox is None:
                metadata["cleanup"] = "no_handle; creation_may_be_unconfirmed"
        return replace(result, metadata=metadata)
