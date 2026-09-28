"""Bounded live cache trials. Call only through an isolated execution backend.

All programs here are author-written diagnostic fixtures, not model results.
The trial stops at the first unexpected result; it never escalates to GPU use.
"""

import inspect

from .fixtures import source_for
from .grading import Status, evaluate_candidate
from .suites import DEFAULT_SEED, Suite, build_suites, case, get, put


def require_current_conformance(report, image_id):
    """Require the complete current check set on this runner and sandbox image."""
    from .modal_backend import RUNNER
    from .suites import digest

    expected = {definition[0] for definition in smoke_cases()}
    checks = report.get("checks", [])
    if (report.get("passed") is not True or len(checks) != len(expected)
            or report.get("completed_checks") != len(expected)
            or report.get("planned_checks") != len(expected)
            or {c.get("check") for c in checks} != expected):
        raise ValueError("complete current CPU conformance report required")
    for check in checks:
        if check.get("passed") is not True:
            raise ValueError("CPU conformance check failed")
        suites = check.get("report", {}).get("suites", [])
        if not suites or any(s.get("infrastructure_errors") != 0 or not s.get("outcomes") for s in suites):
            raise ValueError("CPU conformance evidence is incomplete")
        for suite in suites:
            for outcome in suite["outcomes"]:
                if not outcome.get("attempts"):
                    raise ValueError("CPU conformance execution evidence is missing")
                for attempt in outcome["attempts"]:
                    metadata = attempt.get("metadata", {})
                    if (metadata.get("image_id") != image_id
                            or metadata.get("runner_hash") != digest(RUNNER)
                            or metadata.get("cleanup") != "terminated"):
                        raise ValueError("CPU conformance image, runner or cleanup mismatch")


def smoke_cases():
    g1 = build_suites()[0]
    boundary = Suite("boundary_smoke", "training", (
        case("exact_expiry", [put(0), get(10)]),
    ), DEFAULT_SEED)
    ordinary = Suite("single_smoke", "training", (
        case("ordinary", [put(0), get(1)]),
    ), DEFAULT_SEED)
    correct = source_for("correct")
    # Headers are checks authored by us. No expected-answer data is sent.
    headers = {
        "unprivileged_and_no_tokens": '''import os
assert os.getuid() == 65534 and os.geteuid() == 65534
assert not any(k in os.environ for k in (
    "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_IDENTITY_TOKEN", "HF_TOKEN"))
''',
        "reference_package_absent": '''import importlib.util
assert importlib.util.find_spec("verifier_rl") is None
''',
        "outbound_connection_fails": '''import socket
sock = socket.socket()
sock.settimeout(1.0)
try:
    try:
        sock.connect(("1.1.1.1", 443))
    except OSError:
        pass
    else:
        raise AssertionError("outbound TCP connection unexpectedly succeeded")
finally:
    sock.close()
''',
        "write_marker": '''from pathlib import Path
Path("/tmp/verifier_rl_smoke_marker").write_text("fixture marker")
''',
        "fresh_filesystem": '''from pathlib import Path
assert not Path("/tmp/verifier_rl_smoke_marker").exists()
''',
        "spawned_child_cleanup": '''import subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
''',
    }
    yield "correct_g1", correct, g1, 1, None
    yield "known_blind_spot_g1", source_for("inclusive_expiry"), g1, 1, None
    yield "boundary_catches_bug", source_for("inclusive_expiry"), boundary, 0, "wrong_answer"
    for name, header in headers.items():
        yield name, header + correct, ordinary, 1, None
    yield "function_exception", "def simulate_cache(operations): raise RuntimeError('intentional fixture failure')", ordinary, 0, Status.CANDIDATE_ERROR.value
    yield "missing_entrypoint", "class LRUCache: pass", ordinary, 0, Status.CANDIDATE_ERROR.value
    yield "stdout_garbage", "print('all tests passed')\n" + correct, ordinary, 0, "invalid_json"
    yield "wrong_return_type", "def simulate_cache(operations): return [True]", ordinary, 0, Status.CANDIDATE_ERROR.value
    yield "wall_timeout", "import time\ntime.sleep(30)\n" + correct, ordinary, 0, Status.TIMEOUT.value
    yield "stdout_limit", "import os\nos.write(1, b'x' * 20000)\n" + correct, ordinary, 0, Status.OUTPUT_LIMIT.value
    yield "stderr_limit", "import os\nos.write(2, b'x' * 20000)\n" + correct, ordinary, 0, Status.OUTPUT_LIMIT.value


async def run_smoke(backend, record=None):
    definitions = tuple(smoke_cases())
    planned = sum(len(s.cases) for _, _, s, _, _ in definitions)
    if planned > 24:
        raise ValueError("CPU smoke plan exceeds its 24-execution cap")
    results = []
    for name, source, suite, reward, reason in definitions:
        report = await evaluate_candidate(source, (suite,), backend, concurrency=1, max_retries=0)
        scored = report["suites"][0]
        observed = {o["reason"] for o in scored["outcomes"]}
        ok = scored["reward"] == reward and (reason is None or observed == {reason})
        cleanup_ok = all(
            attempt["metadata"].get("cleanup") == "terminated"
            for outcome in scored["outcomes"] for attempt in outcome["attempts"]
        )
        attempts = [attempt for outcome in scored["outcomes"] for attempt in outcome["attempts"]]
        stage = attempts[-1]["metadata"].get("runner_stage") if attempts else None
        expected_stage = {"function_exception": "function_call",
                          "missing_entrypoint": "entrypoint_lookup",
                          "wrong_return_type": "return_validation"}.get(name)
        stage_ok = expected_stage is None or stage == expected_stage
        stdout_preview = attempts[-1]["metadata"].get("stdout_preview", "") if attempts else ""
        output_diagnostic_ok = name != "stdout_garbage" or "all tests passed" in stdout_preview
        result = {"check": name, "passed": ok and cleanup_ok,
                  "expected_reward": reward, "expected_reason": reason,
                  "expected_runner_stage": expected_stage, "observed_runner_stage": stage,
                  "stage_diagnostic_valid": stage_ok,
                  "stdout_diagnostic_valid": output_diagnostic_ok,
                  "report": report}
        result["passed"] = result["passed"] and stage_ok and output_diagnostic_ok
        results.append(result)
        if record is not None:
            pending = record(result)  # Save/commit each completed check before proceeding.
            if inspect.isawaitable(pending):
                await pending
        if not result["passed"]:
            break
    return {
        "kind": "cache_live_smoke", "model_results": False,
        "planned_checks": len(definitions), "planned_max_executions": planned,
        "completed_checks": len(results),
        "passed": len(results) == len(definitions) and all(r["passed"] for r in results),
        "checks": results,
        "limitations": [
            "A failed TCP connection to one public address is not proof of universal network denial.",
            "Child cleanup relies on whole-sandbox termination acknowledgement, not host process inspection.",
            "Runner stage markers and candidate output are untrusted diagnostics, not authenticated evidence.",
            "These checks are not a security audit, a model evaluation, or an RL experiment.",
        ],
    }
