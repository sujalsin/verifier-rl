"""Bounded live cache trials. Call only through an isolated execution backend.

All programs here are author-written diagnostic fixtures, not model results.
The trial stops at the first unexpected result; it never escalates to GPU use.
"""

import inspect

from .fixtures import source_for
from .grading import Status, evaluate_candidate
from .suites import DEFAULT_SEED, Suite, build_suites, case, get, put


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
    yield "exception", "raise RuntimeError('intentional fixture failure')", ordinary, 0, Status.CANDIDATE_ERROR.value
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
        result = {"check": name, "passed": ok and cleanup_ok,
                  "expected_reward": reward, "expected_reason": reason,
                  "report": report}
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
            "These checks are not a security audit, a model evaluation, or an RL experiment.",
        ],
    }
