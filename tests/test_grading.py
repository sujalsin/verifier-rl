import asyncio
import hashlib
import json
import unittest
from dataclasses import fields
from unittest.mock import patch

from verifier_rl.cache import reference
from verifier_rl.grading import (ExecutionRequest, ExecutionResult, MAX_OUTPUT_BYTES,
                                 Status, compare_output, evaluate_candidate,
                                 rejected_extraction_report, score_suite)
from verifier_rl.suites import Suite, build_suites, case, get


class FakeBackend:
    """Test double only. Never executes the request's source."""
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []
        self.active = self.peak = 0

    async def execute(self, request):
        self.calls.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(.001)
            if self.responses:
                return self.responses.pop(0)
            output = reference(json.loads(request.input_json))
            return ExecutionResult(Status.COMPLETED, json.dumps(output).encode())
        finally:
            self.active -= 1


class ComparisonTests(unittest.TestCase):
    def test_exact_type_aware_comparison(self):
        self.assertEqual(compare_output(b"[7,null,0]", [7, None, 0])[:2], (True, "pass"))
        for output in (b"[true]", b"[1.0]", b'["1"]', b"{}", b"null", b"[NaN]"):
            self.assertEqual(compare_output(output, [1])[1], "invalid_schema")

    def test_wrong_length_order_missing_extra(self):
        for output in (b"[]", b"[1,2,3]", b"[2,1]", b"[1]"):
            self.assertEqual(compare_output(output, [1,2])[1], "wrong_answer")

    def test_protocol_garbage_and_limits(self):
        for output in (b"all tests passed", b"[7]\n[7]", b"\xff", b"["*2000, b"[123"):
            self.assertEqual(compare_output(output, [7])[1], "invalid_json")
        self.assertEqual(compare_output(b" "*(MAX_OUTPUT_BYTES+1), [])[1], "output_limit")

    def test_audit_never_produces_reward(self):
        c = case("a", [get(0)])
        execution = {c.input_hash: (ExecutionResult(Status.COMPLETED, b"[null]"),)}
        self.assertIsNone(score_suite(Suite("audit", "audit", (c,), 0), execution)["reward"])

    def test_failure_plus_infrastructure_error_still_unscored(self):
        a, b = case("a", []), case("b", [get(0)])
        executions = {
            a.input_hash: (ExecutionResult(Status.CANDIDATE_ERROR),),
            b.input_hash: (ExecutionResult(Status.INFRASTRUCTURE_ERROR),),
        }
        result = score_suite(Suite("g", "training", (a, b), 0), executions)
        self.assertIsNone(result["reward"])
        self.assertEqual(result["infrastructure_errors"], 1)

    def test_report_keeps_bounded_output_preview_and_hash(self):
        c = case("stdout", [get(0)])
        output = b"example output\\nnot-json"
        execution = {c.input_hash: (ExecutionResult(Status.COMPLETED, stdout=output),)}
        outcome = score_suite(Suite("g", "training", (c,), 0), execution)["outcomes"][0]
        attempt = outcome["attempts"][0]
        self.assertEqual(attempt["metadata"]["stdout_preview"], output.decode())
        self.assertEqual(attempt["metadata"]["stdout_bytes"], len(output))
        self.assertEqual(attempt["metadata"]["stdout_sha256"], hashlib.sha256(output).hexdigest())

    def test_rejected_extraction_is_unscored_by_runner_and_preserves_reward_contract(self):
        suites = build_suites()[:1]
        report = rejected_extraction_report(
            "# extraction rejected", "rejected_fence_count", suites)
        self.assertEqual(report["execution_config"]["submitted_to_backend"], 0)
        self.assertEqual(report["extraction"]["status"], "rejected_fence_count")
        self.assertEqual(report["suites"][0]["reward"], 0)
        self.assertTrue(all(o["reason"] == "extraction_rejected"
                            for o in report["suites"][0]["outcomes"]))


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fail_fast_stops_submissions_without_inventing_attempts(self):
        backend = FakeBackend([ExecutionResult(Status.INFRASTRUCTURE_ERROR)] * 4)
        result = await evaluate_candidate("source", build_suites()[:3], backend,
                                          concurrency=2, max_retries=0, stop_on_infrastructure_error=True)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(result["execution_config"]["attempts"], 2)
        self.assertGreater(result["execution_config"]["not_executed_inputs"], 0)
        for suite in result["suites"]:
            self.assertIsNone(suite["reward"])
            self.assertIsNone(suite["all_passed"])
            for outcome in suite["outcomes"]:
                if outcome["reason"] == "not_executed":
                    self.assertEqual(outcome["attempts"], [])
                    self.assertIsNone(outcome["passed"])

    async def test_fail_fast_does_not_stop_on_candidate_failures(self):
        suite = build_suites()[:1]
        backend = FakeBackend([ExecutionResult(Status.CANDIDATE_ERROR)] * 5)
        result = await evaluate_candidate("source", suite, backend, stop_on_infrastructure_error=True)
        self.assertEqual(len(backend.calls), 5)
        self.assertEqual(result["suites"][0]["reward"], 0)
        self.assertEqual(result["execution_config"]["not_executed_inputs"], 0)

    async def test_oracle_disagreement_stops_before_execution(self):
        backend = FakeBackend()
        with patch("verifier_rl.cache.reverse_scan_oracle", return_value=[123]):
            with self.assertRaises(RuntimeError):
                await evaluate_candidate("source", build_suites()[:1], backend)
        self.assertEqual(backend.calls, [])

    async def test_bounded_concurrency_and_shared_inputs(self):
        suites = build_suites()[:3]
        backend = FakeBackend()
        report = await evaluate_candidate("not executed locally", suites, backend, concurrency=3)
        expected_count = len({c.input_hash for s in suites for c in s.cases})
        self.assertEqual(len(backend.calls), expected_count)
        self.assertLessEqual(backend.peak, 3)
        self.assertGreater(backend.peak, 1)
        self.assertTrue(all(s["reward"] == 1 for s in report["suites"]))

    async def test_only_inputs_and_source_cross_boundary(self):
        self.assertEqual({f.name for f in fields(ExecutionRequest)}, {"source", "input_json"})
        backend = FakeBackend()
        await evaluate_candidate("source", build_suites()[:1], backend)
        self.assertTrue(all(r.source == "source" for r in backend.calls))

    async def test_transient_infrastructure_error_retried_and_recorded(self):
        c = case("x", [get(0)])
        backend = FakeBackend([ExecutionResult(Status.INFRASTRUCTURE_ERROR, retryable=True),
                               ExecutionResult(Status.COMPLETED, b"[null]")])
        report = await evaluate_candidate("source", (Suite("g", "training", (c,), 0),), backend)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(report["suites"][0]["reward"], 1)
        self.assertEqual(len(report["suites"][0]["outcomes"][0]["attempts"]), 2)

    async def test_exhausted_infrastructure_failure_is_unscored(self):
        c = case("x", [])
        backend = FakeBackend([ExecutionResult(Status.INFRASTRUCTURE_ERROR, retryable=True)]*3)
        report = await evaluate_candidate("source", (Suite("g", "training", (c,), 0),), backend, max_retries=2)
        self.assertEqual(len(backend.calls), 3)
        self.assertIsNone(report["suites"][0]["reward"])
        self.assertIsNone(report["suites"][0]["all_passed"])

    async def test_candidate_failures_never_retried(self):
        for status in (Status.CANDIDATE_ERROR, Status.TIMEOUT, Status.OUTPUT_LIMIT):
            backend = FakeBackend([ExecutionResult(status)])
            suite = Suite("g", "training", (case("x", []),), 0)
            result = await evaluate_candidate("source", (suite,), backend)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(result["suites"][0]["reward"], 0)

    async def test_bad_configuration_before_execution(self):
        backend = FakeBackend()
        for kwargs in ({"concurrency": 0}, {"concurrency": True}, {"max_retries": -1}):
            with self.assertRaises(ValueError):
                await evaluate_candidate("source", build_suites()[:1], backend, **kwargs)
        for source in ("", "x"*32769):
            with self.assertRaises(ValueError):
                await evaluate_candidate(source, build_suites()[:1], backend)
        self.assertEqual(backend.calls, [])

    async def test_cancellation_stops_workers(self):
        class WaitingBackend:
            active = 0
            async def execute(self, request):
                self.active += 1
                try:
                    await asyncio.Event().wait()
                finally:
                    self.active -= 1
        backend = WaitingBackend()
        task = asyncio.create_task(evaluate_candidate("source", build_suites()[:1], backend))
        await asyncio.sleep(.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(backend.active, 0)
