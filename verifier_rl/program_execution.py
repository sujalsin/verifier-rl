"""One remote sandbox per program; read-only, single-process Python per input.

This is a NEW execution contract, not a reinterpretation of historical records.
Candidate code is never executed locally or in the trusted grading Function.
Kernel-enforced restrictions prevent persistent file writes and descendants;
new Python processes reset module/global state. This is not a fresh VM per test.
The outer Modal sandbox remains the host security boundary.
"""

import asyncio
from dataclasses import asdict
import json
import math
import time

from . import supervised_execution as supervised
from . import program_storage
from .grading import ExecutionResult, MAX_OUTPUT_BYTES, MAX_SOURCE_BYTES, Status
from .modal_backend import BOOTSTRAP, Limits, collect
from .panel_execution import pack_result, runner_for, unpack_result
from .suites import canonical_json, digest
from .task_panel import BOOKING, validate_call
from .verifier_quality import compare_output

VERSION = "program-sandbox-readonly-0.1"
PROFILE = "python-readonly-no-descendants-0.1"
MAX_CASES = 300

# Executed ONLY inside a remote child. Unknown syscalls fail closed with EPERM.
# Standard-library imports can read installed files. No write-capable open,
# process/thread creation, exec, IPC, networking, ptrace, or resource-limit reset.
READONLY_BOOTSTRAP = r'''import ctypes
import errno
import json
import os
import resource
import sys

def prepare(limits):
    if sys.version_info < (3, 11) or os.getuid() != 0:
        raise RuntimeError("unexpected Python or initial credentials")
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    libc = ctypes.CDLL(None, use_errno=True)
    class Cmp(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                    ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                         ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Cmp)]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    ctx = lib.seccomp_init(0x00050000 | errno.EPERM)
    if not ctx:
        raise RuntimeError("seccomp init failed")
    def allow(name, comparisons=()):
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number < 0:
            return  # Missing architecture-specific alternatives stay denied.
        array = (Cmp * len(comparisons))(*(Cmp(*c) for c in comparisons))
        if lib.seccomp_rule_add_array(ctx, 0x7fff0000, number, len(array), array) != 0:
            raise RuntimeError("seccomp rule failed: " + name)
    for name in (
        "read", "readv", "pread64", "close", "fstat", "newfstatat", "stat", "lstat", "statx",
        "lseek", "getdents", "getdents64", "readlink", "readlinkat", "access", "faccessat", "faccessat2",
        "mmap", "mmap2", "mprotect", "munmap", "mremap", "madvise", "brk",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "sigaltstack", "restart_syscall",
        "futex", "futex_time64", "nanosleep", "clock_nanosleep", "clock_gettime", "clock_getres",
        "gettimeofday", "time", "getrandom", "getpid", "getppid", "gettid", "getuid", "geteuid",
        "getgid", "getegid", "getresuid", "getresgid", "getgroups", "getcwd", "chdir", "fchdir",
        "uname", "getrlimit", "getrusage", "sysinfo", "times", "sched_getaffinity", "sched_yield",
        "exit", "exit_group",
    ):
        allow(name)
    # SCMP_CMP_MASKED_EQ=7; allow only read-only opens, including import reads.
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    write_flags |= getattr(os, "O_TMPFILE", 0) & ~getattr(os, "O_DIRECTORY", 0)
    allow("open", ((1, 7, write_flags, 0),))
    allow("openat", ((2, 7, write_flags, 0),))
    # SCMP_CMP_EQ=4. Only inherited output/readiness pipes can be written.
    for fd in (1, 2, int(sys.argv[2])):
        allow("write", ((0, 4, fd, 0),))
        allow("writev", ((0, 4, fd, 0),))
    for command in (1, 2, 3):  # F_GETFD, F_SETFD, F_GETFL; no locks or leases.
        allow("fcntl", ((1, 4, command, 0),))
    for command in (21, 39):  # Read seccomp/no_new_privs, never change either.
        allow("prctl", ((0, 4, command, 0),))
    allow("prlimit64", ((0, 4, 0, 0), (2, 4, 0, 0)))
    pid = os.getpid()
    allow("kill", ((0, 4, pid, 0),))
    allow("tgkill", ((0, 4, pid, 0), (1, 4, pid, 0)))
    allow("tkill", ((0, 4, pid, 0),))
    os.chdir("/")
    os.umask(0o077)
    sys.dont_write_bytecode = True
    for name, value in (
        (resource.RLIMIT_CPU, limits["cpu_seconds"]),
        (resource.RLIMIT_AS, limits["process_memory_mib"] * 1024 * 1024),
        (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_NOFILE, 32),
        (resource.RLIMIT_NPROC, 1), (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(name, (value, value))
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise RuntimeError("no_new_privs unavailable")
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    os.environ.clear()
    os.environ.update({"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"})
    if os.getuid() != 65534 or os.geteuid() != 65534:
        raise RuntimeError("privilege drop failed")
    if lib.seccomp_load(ctx) != 0:
        raise RuntimeError("seccomp load failed")
    lib.seccomp_release(ctx)
    if libc.prctl(21, 0, 0, 0, 0) != 2:
        raise RuntimeError("seccomp filter mode missing")
'''

PREFLIGHT = r'''import ctypes,json,os,sys
ctypes.CDLL("libseccomp.so.2")
if os.getuid()!=0 or sys.version_info[:2]!=(3,12):
    raise RuntimeError("unexpected image")
print(json.dumps({"ready":True,"uid":os.getuid(),"python":sys.version}))
'''


def runner():
    legacy = runner_for(BOOKING)
    if not legacy.startswith(BOOTSTRAP):
        raise ValueError("review child bootstrap changes")
    child = READONLY_BOOTSTRAP + legacy[len(BOOTSTRAP):]
    marker = 'namespace = {"__name__": "candidate"}'
    child = child.replace(marker, 'ready_fd = int(sys.argv[2])\nos.write(ready_fd, b"ready")\n'
                          'os.close(ready_fd)\n' + marker, 1)
    return (f"CHILD_RUNNER = {child!r}\nOUTPUT_LIMIT = {MAX_OUTPUT_BYTES}\n"
            f"EXECUTION_VERSION = {VERSION!r}\n" + supervised.PARENT)


def inspect_envelope(envelope, payload_hash):
    if envelope.get("version") != VERSION:
        raise ValueError("wrong program execution version")
    # Preserve the already-tested outcome classifier; original evidence is not
    # changed. Only this NEW version is accepted by the surrounding validator.
    return supervised.inspect_envelope(dict(envelope, version=supervised.VERSION), payload_hash)


class CreationGate:
    """One shared gate for all backends within ONE owning coordinator.

    This is not distributed. Multiple controllers require an external shared
    start gate; the benchmark deliberately runs in one CPU Function only.
    """
    def __init__(self, interval=.30):
        if not math.isfinite(interval) or interval < .26:
            raise ValueError("unsafe creation pacing")
        self.interval, self.last, self.lock = interval, 0.0, asyncio.Lock()
        self.starts = 0

    async def __call__(self):
        async with self.lock:
            await asyncio.sleep(max(0, self.last + self.interval - time.monotonic()))
            self.last = time.monotonic()
            self.starts += 1


def lifetime_for(count):
    if type(count) is not int or not 1 <= count <= MAX_CASES:
        raise ValueError("bounded nonempty program suite required")
    return 60 + count * (Limits().wall_seconds + supervised.SUPERVISOR_GRACE + 1)


def validate_request(source, cases):
    if not isinstance(source, str) or not source.strip() or len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("invalid source")
    lifetime_for(len(cases))
    for case in cases:
        validate_call(BOOKING, case.arguments)
        if case.input_hash != digest(canonical_json([BOOKING, case.arguments])):
            raise ValueError("input identity changed")
    if len({c.input_hash for c in cases}) != len(cases):
        raise ValueError("duplicate input identity")


class ProgramBackend:
    def __init__(self, app_name, image_id, *, start_gate, sdk=None):
        from .modal_backend import ModalBackend
        ModalBackend(app_name, image_id)  # Reuse the configuration guard only.
        self.app_name, self.image_id = app_name, image_id
        self.sdk, self.start_gate = sdk, start_gate
        self.limits = Limits()

    async def execute_program(self, source, cases, *, on_record=None):
        validate_request(source, cases)
        if self.sdk is None:
            import modal
            self.sdk = modal
        if self.sdk.__version__ != "1.5.5":
            raise ValueError("unreviewed SDK")
        sandbox, records = None, {}
        started, stage = time.monotonic(), "lookup"
        lifetime = lifetime_for(len(cases))
        metadata = {"version":VERSION, "profile":PROFILE, "runner_hash":digest(runner()),
                    "profile_hash":digest(READONLY_BOOTSTRAP), "source_hash":digest(source),
                    "task_id":BOOKING, "image_id":self.image_id, "sdk_version":self.sdk.__version__,
                    "limits":asdict(self.limits), "sandbox_lifetime_seconds":lifetime,
                    "block_network":True, "reset":"fresh_sandbox_per_program_fresh_process_per_input",
                    "candidate_submission_attempted":False, "cleanup":"pending"}
        current, details = None, {}
        try:
            owner = await asyncio.wait_for(self.sdk.App.lookup.aio(self.app_name, create_if_missing=False), 45)
            image = self.sdk.Image.from_id(self.image_id)
            stage = "create"
            await self.start_gate()
            sandbox = await asyncio.wait_for(self.sdk.Sandbox.create.aio(
                "sleep", "infinity", app=owner, image=image, timeout=lifetime,
                cpu=(1,1), memory=(256,256), block_network=True, secrets=[], volumes={},
                include_oidc_identity_token=False), 45)
            metadata["sandbox_id"] = sandbox.object_id
            deadline = time.monotonic() + lifetime - 15
            stage = "preflight"
            process = await asyncio.wait_for(sandbox.exec.aio(
                "python3", "-I", "-c", PREFLIGHT, timeout=5, text=False), 45)
            out, err, code = await asyncio.wait_for(collect(process), 45)
            metadata["preflight_returncode"] = code
            metadata["preflight_stderr"] = err[:2048].decode(errors="replace")
            if code != 0:
                raise RuntimeError("preflight nonzero exit")
            ready = json.loads(out)
            if ready.get("ready") is not True or ready.get("uid") != 0:
                raise ValueError("preflight identity")
            metadata["runtime_python"] = ready["python"]
            metadata["startup_seconds"] = time.monotonic() - started
            for case in cases:
                current, stage = case, "candidate"
                details = {"input_hash":case.input_hash, "source_hash":digest(source),
                           "sandbox_id":sandbox.object_id}
                remaining = deadline - time.monotonic()
                if remaining <= 10:
                    raise TimeoutError("program lifetime reached")
                payload = canonical_json({"source":source, "input":case.arguments, "limits":asdict(self.limits)})
                case_started = time.monotonic()
                metadata["candidate_submission_attempted"] = True
                process = await asyncio.wait_for(sandbox.exec.aio(
                    "python3", "-I", "-c", runner(), payload, timeout=8, text=False), min(45,remaining))
                out, err, code = await asyncio.wait_for(collect(process, supervised.TRANSPORT_LIMIT),
                    min(53, max(.01,deadline-time.monotonic())))
                details.update({"parent_returncode":code,
                           "parent_stderr":err[:2048].decode(errors="replace"),
                           "parent_stdout":out[:2048].decode(errors="replace"),
                           "roundtrip_seconds":time.monotonic()-case_started})
                if code != 0:
                    raise RuntimeError("supervisor nonzero exit: " + str(code))
                envelope = json.loads(out)
                status, reason, stdout, _ = inspect_envelope(envelope,digest(payload))
                if out != (canonical_json(envelope)+"\n").encode():
                    raise ValueError("noncanonical supervisor envelope")
                details["supervisor_report"] = envelope
                result = ExecutionResult(status, stdout, reason, False, details)
                records[case.input_hash] = pack_result(result)
                if on_record is not None:
                    try:
                        await on_record(case.input_hash, records[case.input_hash])
                    except Exception as exc:
                        raise program_storage.StorageFailure("input evidence persistence failed") from exc
                if not envelope["candidate_started"]:
                    raise RuntimeError("child restriction bootstrap failed")
                current = None
        except Exception as exc:
            if isinstance(exc, program_storage.StorageFailure):
                stage = "evidence_storage"
            metadata["failure"] = {"stage":stage,"type":type(exc).__name__,"detail":str(exc)[:1000]}
            if current is not None and current.input_hash not in records:
                records[current.input_hash] = pack_result(ExecutionResult(Status.INFRASTRUCTURE_ERROR,
                    detail="program_transport:"+stage+":"+type(exc).__name__,
                    metadata=details))
                if on_record is not None:
                    try:
                        await on_record(current.input_hash,records[current.input_hash])
                    except Exception as storage_exc:
                        metadata["storage_failure"] = type(storage_exc).__name__
        finally:
            if sandbox is not None:
                try:
                    await asyncio.wait_for(sandbox.terminate.aio(wait=True),45)
                    metadata["cleanup"] = "terminated"
                except Exception as exc:
                    metadata["cleanup"] = "unconfirmed:"+type(exc).__name__
                    try:
                        from .sandbox_lifecycle import confirm_terminal
                        metadata["cleanup_terminal_evidence"] = await confirm_terminal(sandbox)
                        metadata["cleanup"] = "terminated"
                    except Exception as lookup:
                        metadata["cleanup_lookup_error"] = type(lookup).__name__
                finally:
                    try:
                        await asyncio.wait_for(sandbox.detach.aio(),45)
                    except Exception as exc:
                        metadata["detach_error"] = type(exc).__name__
            else:
                metadata["cleanup"] = "no_handle; creation_may_be_unconfirmed"
            metadata["total_seconds"] = time.monotonic()-started
        for case in cases:
            records.setdefault(case.input_hash,pack_result(ExecutionResult(Status.INFRASTRUCTURE_ERROR,
                detail="program_not_executed",metadata={"input_hash":case.input_hash,"source_hash":digest(source)})))
        return {"metadata":metadata,"records":records,"input_order":[c.input_hash for c in cases]}


def validate_program(document, source, cases, image_id):
    """Replay each protected child result; unknowns are never converted to zero."""
    if "continuation" in document:
        return program_storage.validate_continuation(document, source, cases, image_id)
    validate_request(source,cases)
    m = document["metadata"]
    required = {"version":VERSION,"profile":PROFILE,"runner_hash":digest(runner()),
        "profile_hash":digest(READONLY_BOOTSTRAP),"source_hash":digest(source),"task_id":BOOKING,
        "image_id":image_id,"sdk_version":"1.5.5","limits":asdict(Limits()),
        "sandbox_lifetime_seconds":lifetime_for(len(cases)),"block_network":True,
        "reset":"fresh_sandbox_per_program_fresh_process_per_input"}
    if (any(m.get(k)!=v for k,v in required.items())
            or document["input_order"]!=[c.input_hash for c in cases]
            or set(document["records"])!={c.input_hash for c in cases}):
        raise ValueError("program execution provenance changed")
    outcomes = {}
    for case in cases:
        result = unpack_result(document["records"][case.input_hash])
        detail = result.metadata
        if detail.get("input_hash")!=case.input_hash or detail.get("source_hash")!=digest(source) or result.retryable:
            raise ValueError("input record identity changed")
        envelope = detail.get("supervisor_report")
        if envelope is not None:
            payload = canonical_json({"source":source,"input":case.arguments,"limits":asdict(Limits())})
            status,reason,stdout,_ = inspect_envelope(envelope,digest(payload))
            if (status!=result.status or reason!=result.detail or stdout!=result.stdout
                    or detail.get("sandbox_id")!=m.get("sandbox_id") or detail.get("parent_returncode")!=0
                    or m.get("preflight_returncode")!=0 or not m.get("candidate_submission_attempted")
                    or not isinstance(m.get("sandbox_id"),str) or not m["sandbox_id"].startswith("sb-")):
                raise ValueError("protected input outcome changed")
        elif result.status!=Status.INFRASTRUCTURE_ERROR or result.stdout:
            raise ValueError("gradable result lacks supervisor evidence")
        if m["cleanup"]!="terminated" or result.status==Status.INFRASTRUCTURE_ERROR:
            passed,reason,actual = None,result.detail or "cleanup_unconfirmed",None
        elif result.status==Status.COMPLETED:
            passed,reason = compare_output(case,result.stdout)
            actual = json.loads(result.stdout) if reason in ("pass","wrong_answer") else None
        else:
            passed,reason,actual = False,result.detail,None
        outcomes[case.input_hash] = {"passed":passed,"reason":reason,"actual":actual,
                                     "status":result.status.value}
    return outcomes
