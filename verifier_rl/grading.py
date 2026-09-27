"""Trusted output comparison, bounded scheduling, retries, and experiment records."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from typing import Protocol

from . import __version__
from .cache import TASK_ID
from .suites import Case, Suite, digest

MAX_SOURCE_BYTES = 32_768
MAX_OUTPUT_BYTES = 16_384
COMPARATOR_VERSION = "cache-exact-json-0.1"


class Status(StrEnum):
    COMPLETED = "completed"
    CANDIDATE_ERROR = "candidate_error"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


@dataclass(frozen=True)
class ExecutionRequest:
    # Deliberately no expected answer, case/suite label, reference, or credentials.
    source: str
    input_json: str


@dataclass(frozen=True)
class ExecutionResult:
    status: Status
    stdout: bytes = b""
    detail: str = ""
    retryable: bool = False
    metadata: dict = field(default_factory=dict)


class Backend(Protocol):
    async def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


def compare_output(stdout: bytes, expected: list[int | None]):
    """Parse only bounded JSON data. Never eval, exec, import, or unpickle output."""
    if len(stdout) > MAX_OUTPUT_BYTES:
        return False, "output_limit", None
    try:
        actual = json.loads(stdout)
    except (ValueError, UnicodeError, RecursionError):
        return False, "invalid_json", None
    if type(actual) is not list or any(x is not None and type(x) is not int for x in actual):
        return False, "invalid_schema", None
    if actual != expected:
        return False, "wrong_answer", actual
    return True, "pass", actual


def score_suite(suite: Suite, executions: dict[str, tuple[ExecutionResult, ...]]):
    outcomes = []
    for case in suite.cases:
        attempts = executions[case.input_hash]
        if not attempts:
            raise ValueError("cannot grade a case without an execution")
        result = attempts[-1]
        expected = case.expected
        actual = None
        if result.status == Status.COMPLETED:
            passed, reason, actual = compare_output(result.stdout, expected)
        elif result.status == Status.INFRASTRUCTURE_ERROR:
            passed, reason = None, result.status.value
        else:
            passed, reason = False, result.status.value
        outcomes.append({
            "case": case.name, "input_hash": case.input_hash,
            "passed": passed, "reason": reason, "actual": actual, "expected": expected,
            "attempts": [{"status": r.status.value, "detail": r.detail,
                          "retryable": r.retryable, "metadata": r.metadata} for r in attempts],
        })
    infra = sum(o["passed"] is None for o in outcomes)
    passed_count = sum(o["passed"] is True for o in outcomes)
    # Conservative: any unresolved infrastructure failure leaves the entire suite
    # unscored, even if a separate case already failed. No silent partial grading.
    all_passed = None if infra else passed_count == len(outcomes)
    return {
        "suite": suite.name, "purpose": suite.purpose, "suite_version": suite.version,
        "suite_hash": suite.fingerprint, "seed": suite.seed,
        "total": len(outcomes), "passed_count": passed_count,
        "infrastructure_errors": infra, "all_passed": all_passed,
        "reward": int(all_passed) if suite.purpose == "training" and all_passed is not None else None,
        "outcomes": outcomes,
    }


async def evaluate_candidate(source: str, suites: tuple[Suite, ...], backend: Backend,
                             concurrency=4, max_retries=1):
    if not source.strip() or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("candidate source must be nonempty and at most 32768 UTF-8 bytes")
    if type(concurrency) is not int or not 1 <= concurrency <= 32:
        raise ValueError("concurrency must be 1..32")
    if type(max_retries) is not int or not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be 0..2")
    if not suites or len({s.name for s in suites}) != len(suites):
        raise ValueError("provide nonempty, uniquely named suites")
    unique: dict[str, Case] = {c.input_hash: c for s in suites for c in s.cases}
    queue = asyncio.Queue()
    for c in unique.values():
        # Resolve/cross-check the oracle before incurring any external execution.
        c.expected
        queue.put_nowait(c)
    executions = {}

    async def worker():
        while True:
            try:
                case = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            attempts = []
            for attempt in range(max_retries + 1):
                try:
                    result = await backend.execute(ExecutionRequest(source, case.input_json))
                except Exception as exc:
                    # Programming/configuration errors remain visible and unscored;
                    # do not retry arbitrary exceptions or expose exception secrets.
                    result = ExecutionResult(Status.INFRASTRUCTURE_ERROR,
                                             detail=f"backend_exception:{type(exc).__name__}")
                attempts.append(result)
                if result.status != Status.INFRASTRUCTURE_ERROR or not result.retryable:
                    break
                if attempt < max_retries:
                    await asyncio.sleep(0.05 * 2**attempt)
            executions[case.input_hash] = tuple(attempts)
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(unique)))]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker_task in workers:
            if not worker_task.done():
                worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    return {
        "schema_version": "0.1", "package_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": TASK_ID, "candidate_hash": digest(source),
        "comparator_version": COMPARATOR_VERSION,
        "execution_config": {"concurrency": concurrency, "max_retries": max_retries,
                             "unique_inputs": len(unique),
                             "attempts": sum(len(a) for a in executions.values())},
        "suites": [score_suite(s, executions) for s in suites],
    }
