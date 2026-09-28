from copy import deepcopy
import unittest

from verifier_rl.grading import ExecutionResult, Status, score_suite
from verifier_rl.partial_reward import (FAMILIES, SEED, WEIGHTS, balanced_scores, balanced_suite,
                                        shortcut_output, validate_reward_design)
from verifier_rl.suites import build_suites, canonical_json


class PartialRewardTests(unittest.TestCase):
    def test_reward_design_controls_and_reproducibility(self):
        report = validate_reward_design()
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["rows"]), 48)
        self.assertEqual(sum(WEIGHTS.values()), 1)
        self.assertEqual(balanced_suite().fingerprint, balanced_suite().fingerprint)
        self.assertNotEqual(balanced_suite().fingerprint, balanced_suite(SEED + 1).fingerprint)

    def test_substantive_probes_require_live_and_missing_answers(self):
        suite = balanced_suite()
        self.assertEqual(len(suite.cases), 24)
        self.assertEqual(len({c.input_hash for c in suite.cases}), 24)
        for family in FAMILIES:
            self.assertEqual(sum(c.tags == (family,) for c in suite.cases), 4)
        for c in suite.cases:
            if c.tags != ("edges",):
                self.assertTrue(any(type(x) is int for x in c.expected))
                self.assertIn(None, c.expected)

    def test_candidate_failures_partial_credit_and_infrastructure(self):
        suite = balanced_suite()
        executions = {c.input_hash: (ExecutionResult(Status.COMPLETED, canonical_json(c.expected).encode()),)
                      for c in suite.cases}
        result = score_suite(suite, executions)
        self.assertEqual(balanced_scores(result)["partial"], 1)
        first = suite.cases[0]
        executions[first.input_hash] = (ExecutionResult(Status.CANDIDATE_ERROR),)
        scores = balanced_scores(score_suite(suite, executions))
        self.assertEqual(scores["binary"], 0)
        self.assertAlmostEqual(scores["partial"], 1 - .19 / 4)
        executions[first.input_hash] = (ExecutionResult(Status.INFRASTRUCTURE_ERROR),)
        self.assertIsNone(balanced_scores(score_suite(suite, executions))["partial"])
        executions[first.input_hash] = ()
        self.assertIsNone(balanced_scores(score_suite(suite, executions, allow_unexecuted=True))["binary"])
        for field, value in (("suite_hash", "wrong"), ("outcomes", [])):
            bad = deepcopy(result)
            bad[field] = value
            with self.assertRaises(ValueError): balanced_scores(bad)

    def test_existing_g3_is_unchanged_and_shortcut_has_low_new_reward(self):
        suite = balanced_suite()
        executions = {c.input_hash: (ExecutionResult(Status.COMPLETED,
                      canonical_json(shortcut_output(c.operations, "always_none")).encode()),) for c in suite.cases}
        self.assertLessEqual(balanced_scores(score_suite(suite, executions))["partial"], .05)
        old = build_suites()[2]
        self.assertEqual(sum(shortcut_output(c.operations, "always_none") == c.expected for c in old.cases), 15)
