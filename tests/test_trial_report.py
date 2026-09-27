from copy import deepcopy
import unittest

from verifier_rl.suites import digest
from verifier_rl.trial_report import summarize


class TrialReportTests(unittest.TestCase):
    def setUp(self):
        self.model = {"run_id": "qwen-test", "config": {"model_id": "test", "model_revision": "abc"},
                      "metrics": {}, "before": [{"source": "# candidate", "seed": 1, "hit_token_cap": False}],
                      "after": []}
        suite = {"suite": "g1", "passed_count": 0, "total": 1, "reward": 0,
                 "infrastructure_errors": 0, "outcomes": [{"input_hash": "input1",
                 "reason": "candidate_error", "attempts": [{"metadata": {"total_seconds": 2.0}}]}]}
        self.evaluations = [{"candidate_hash": digest("# candidate"), "suites": [suite, deepcopy(suite)]}]

    def test_shared_suite_inputs_are_not_double_counted(self):
        result = summarize(self.model, self.evaluations)
        self.assertEqual(result["evaluation_executions"], 1)
        self.assertEqual(result["evaluation_summed_invocation_seconds"], 2.0)
        self.assertEqual(result["candidates"][0]["phase"], "before")

    def test_wrong_or_missing_evidence_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.model, [])
        self.evaluations[0]["candidate_hash"] = "wrong"
        with self.assertRaises(ValueError):
            summarize(self.model, self.evaluations)
