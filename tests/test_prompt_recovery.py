from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from verifier_rl.grading import ExecutionRequest, ExecutionResult, Status, evaluate_candidate, score_suite
from verifier_rl.modal_backend import RUNNER
from verifier_rl.prompt_recovery import (PreflightOnlyRetries, RetryBudget, is_preflight_timeout,
                                         merge_replacements, validate_prior)
from verifier_rl.suites import Case, Suite, canonical_json, digest


SOURCE = "def simulate_cache(operations): return []"
METADATA = {"preflight_returncode": -1, "cleanup": "terminated",
            "image_id": "im-test", "runner_hash": digest(RUNNER)}
FAILURE = ExecutionResult(Status.INFRASTRUCTURE_ERROR, detail="preflight:JSONDecodeError",
                          metadata=METADATA)
CASES = (Case("empty", "[]", ()),
         Case("miss", canonical_json([{"op": "get", "time": 0, "key": "a"}]), ()))
SUITES = (Suite("train", "training", CASES, 1), Suite("audit", "audit", CASES[:1], 1))


def report(executions, suites=SUITES):
    return {"candidate_hash": digest(SOURCE), "arm": "original", "seed": 4001,
            "execution_config": {"attempts": sum(len(a) for a in executions.values())},
            "suites": [score_suite(s, executions) for s in suites]}


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def test_only_confirmed_preflight_deadlines_qualify(self):
        attempt = {"status": "infrastructure_error", "detail": FAILURE.detail, "metadata": METADATA}
        self.assertTrue(is_preflight_timeout(attempt))
        for change in ({"preflight_returncode": 137}, {"cleanup": "unconfirmed"}, {"returncode": -1}):
            modified = deepcopy(attempt)
            modified["metadata"].update(change)
            self.assertFalse(is_preflight_timeout(modified))
        for change in ({"status": "timeout"}, {"detail": "execute:TimeoutError"}):
            self.assertFalse(is_preflight_timeout(dict(attempt, **change)))

    async def test_one_retry_per_input_and_three_across_backends(self):
        budget = RetryBudget()
        backend = SimpleNamespace(execute=AsyncMock(return_value=FAILURE))
        for i in range(4):
            wrapper = PreflightOnlyRetries(backend, budget)
            request = ExecutionRequest(SOURCE + str(i), "[]")
            self.assertEqual((await wrapper.execute(request)).retryable, i < 3)
            self.assertFalse((await wrapper.execute(request)).retryable)
        self.assertEqual(budget.used, 3)
        wrapper = PreflightOnlyRetries(backend, RetryBudget())
        request = ExecutionRequest(SOURCE, "[]")
        self.assertTrue(wrapper.reserve(request))  # Previously recorded failed attempt.
        self.assertFalse((await wrapper.execute(request)).retryable)

    async def test_candidate_and_unrelated_infrastructure_errors_are_not_retried(self):
        for result in (ExecutionResult(Status.CANDIDATE_ERROR), ExecutionResult(Status.TIMEOUT),
                       ExecutionResult(Status.INFRASTRUCTURE_ERROR, detail="ServiceError", retryable=True)):
            backend = SimpleNamespace(execute=AsyncMock(return_value=result))
            wrapper = PreflightOnlyRetries(backend, RetryBudget())
            evidence = await evaluate_candidate(SOURCE, SUITES[:1], wrapper, max_retries=1)
            self.assertEqual(evidence["execution_config"]["attempts"], 2)
            self.assertEqual(wrapper.budget.used, 0)

    async def test_retried_attempts_are_retained_and_deduplicated(self):
        backend = SimpleNamespace(execute=AsyncMock(side_effect=[FAILURE,
                                  ExecutionResult(Status.COMPLETED, b"[]")]))
        suite = Suite("small", "training", CASES[:1], 1)
        evidence = await evaluate_candidate(SOURCE, (suite,), PreflightOnlyRetries(backend, RetryBudget()),
                                           max_retries=1)
        self.assertEqual(evidence["execution_config"]["attempts"], 2)
        self.assertEqual(evidence["suites"][0]["reward"], 1)
        self.assertEqual(len(evidence["suites"][0]["outcomes"][0]["attempts"]), 2)

    def test_merge_preserves_failures_and_all_prior_attempts(self):
        prior = report({CASES[0].input_hash: (FAILURE,), CASES[1].input_hash:
                        (ExecutionResult(Status.CANDIDATE_ERROR, metadata=METADATA),)})
        original = deepcopy(prior)
        replacement = report({CASES[0].input_hash: (ExecutionResult(Status.COMPLETED, b"[]"),)},
                             (Suite("recovery", "audit", CASES[:1], 1),))
        merged = merge_replacements(prior, replacement)
        self.assertEqual(prior, original)
        self.assertEqual(merged["suites"][0]["reward"], 0)
        self.assertEqual(merged["suites"][0]["infrastructure_errors"], 0)
        self.assertEqual(merged["suites"][1]["all_passed"], True)
        self.assertIsNone(merged["suites"][1]["reward"])
        self.assertEqual(merged["suites"][0]["outcomes"][1], prior["suites"][0]["outcomes"][1])
        self.assertEqual(merged["execution_config"]["attempts"], 3)
        self.assertEqual(len(merged["suites"][0]["outcomes"][0]["attempts"]), 2)
        for mutation in ("source", "expected", "input"):
            wrong = deepcopy(replacement)
            if mutation == "source": wrong["candidate_hash"] = "wrong"
            elif mutation == "expected": wrong["suites"][0]["outcomes"][0]["expected"] = [9]
            else: wrong["suites"][0]["outcomes"][0]["input_hash"] = "wrong"
            with self.assertRaises(ValueError): merge_replacements(prior, wrong)

    def test_prior_validation_checks_identity_environment_and_failure_kind(self):
        prior = report({c.input_hash: (FAILURE,) for c in CASES})
        sample = {"source": SOURCE, "arm": "original", "seed": 4001, "extraction_status": "raw"}
        self.assertEqual(validate_prior(sample, prior, SUITES, "im-test"), {c.input_hash for c in CASES})
        for mutation in ("source", "suite", "image", "cleanup", "failure"):
            wrong = deepcopy(prior)
            attempt = wrong["suites"][0]["outcomes"][0]["attempts"][0]
            if mutation == "source": wrong["candidate_hash"] = "bad"
            elif mutation == "suite": wrong["suites"][0]["suite_hash"] = "bad"
            elif mutation == "image": attempt["metadata"]["image_id"] = "im-other"
            elif mutation == "cleanup": attempt["metadata"]["cleanup"] = "unknown"
            else: attempt["detail"] = "ServiceError"
            with self.assertRaises(ValueError): validate_prior(sample, wrong, SUITES, "im-test")
