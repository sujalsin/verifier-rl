import json
import unittest

from verifier_rl.fixtures import fixture_impl
from verifier_rl.grading import ExecutionResult, Status
from verifier_rl.smoke import run_smoke, smoke_cases


class SimulatedSmokeBackend:
    """Simulated observations only; never exec candidate or probe source locally."""
    def __init__(self, infra=False, cleanup="terminated"):
        self.infra = infra
        self.cleanup = cleanup
        self.calls = 0

    async def execute(self, request):
        self.calls += 1
        metadata = {"cleanup": self.cleanup, "backend": "test_double"}
        if self.infra:
            return ExecutionResult(Status.INFRASTRUCTURE_ERROR, metadata=metadata)
        source = request.source
        status = Status.COMPLETED
        out = b""
        if source.startswith("raise RuntimeError") or source.startswith("def simulate_cache"):
            status = Status.CANDIDATE_ERROR
        elif source.startswith("import time\ntime.sleep"):
            status = Status.TIMEOUT
        elif source.startswith("import os\nos.write"):
            status = Status.OUTPUT_LIMIT
        elif source.startswith("print('all tests passed')"):
            out = b"all tests passed\n[7]"
        else:
            fault = "inclusive_expiry" if source.endswith("return fixture_impl(operations, 'inclusive_expiry')\n") else "correct"
            out = json.dumps(fixture_impl(json.loads(request.input_json), fault)).encode()
        return ExecutionResult(status, stdout=out, metadata=metadata)


class SmokeTests(unittest.IsolatedAsyncioTestCase):
    def test_plan_is_bounded_and_writer_precedes_reader(self):
        definitions = list(smoke_cases())
        self.assertEqual(sum(len(s.cases) for _, _, s, _, _ in definitions), 23)
        names = [d[0] for d in definitions]
        self.assertLess(names.index("write_marker"), names.index("fresh_filesystem"))
        for _, source, _, _, _ in definitions:
            compile(source, "<probe-not-executed>", "exec")

    async def test_full_orchestration_with_simulated_responses(self):
        backend = SimulatedSmokeBackend()
        saved = []
        result = await run_smoke(backend, saved.append)
        self.assertTrue(result["passed"])
        self.assertFalse(result["model_results"])
        self.assertEqual(backend.calls, 23)
        self.assertEqual(len(saved), result["planned_checks"])

    async def test_stops_after_first_failed_check_without_retries(self):
        backend = SimulatedSmokeBackend(infra=True)
        result = await run_smoke(backend)
        self.assertFalse(result["passed"])
        self.assertEqual(result["completed_checks"], 1)
        self.assertEqual(backend.calls, 5)  # Complete first five-case suite, then stop.

    async def test_async_record_callback_is_awaited(self):
        saved = []
        async def record(result):
            saved.append(result["check"])
        result = await run_smoke(SimulatedSmokeBackend(), record)
        self.assertTrue(result["passed"])
        self.assertEqual(len(saved), 15)

    async def test_unconfirmed_cleanup_blocks_progress(self):
        result = await run_smoke(SimulatedSmokeBackend(cleanup="unconfirmed"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["completed_checks"], 1)
