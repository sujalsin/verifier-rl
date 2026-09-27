from copy import deepcopy
import unittest

from verifier_rl.baseline import SEEDS, inspect_completion, summarize, validate_recovered_report
from verifier_rl.suites import build_suites, digest


class BaselineTests(unittest.TestCase):
    def test_diagnostics_do_not_execute_source(self):
        raw = "Prose\n```python\nraise RuntimeError('must not execute')\ndef simulate_cache(x): return []\n```"
        inspected = inspect_completion(raw)
        self.assertTrue(inspected["syntax_valid"])
        self.assertTrue(inspected["has_top_level_entrypoint"])
        self.assertEqual(inspected["extraction_status"], "python_block")

    def test_rejection_and_syntax_errors_are_separate(self):
        self.assertEqual(inspect_completion("```python\nx = 1")["extraction_status"], "rejected_fence_count")
        self.assertEqual(inspect_completion("def x(:")["syntax_error"]["type"], "SyntaxError")
        self.assertFalse(inspect_completion("return []")["syntax_valid"])

    def records(self):
        samples, reports = [], []
        for index, seed in enumerate(SEEDS):
            sample = inspect_completion("def simulate_cache(operations): return []")
            sample.update(seed=seed, hit_token_cap=False, ended_with_eos=True)
            samples.append(sample)
            success = index % 2 == 0
            suites = [{"suite": name, "passed_count": int(success), "total": 1,
                       "infrastructure_errors": 0, "outcomes": [{"input_hash": "shared",
                       "reason": "pass" if success else "wrong_answer", "attempts": [{
                       "status": "completed", "metadata": {"total_seconds": 1.0}}]}]}
                      for name in ("g1", "g2", "g3", "audit")]
            reports.append({"candidate_hash": digest(sample["source"]), "suites": suites})
        return samples, reports

    def test_summary_deduplicates_inputs_and_counts_mixed_groups(self):
        samples, reports = self.records()
        result = summarize(samples, reports)
        self.assertEqual(result["unique_executions"], 32)
        self.assertEqual(result["suite_full_pass_counts"]["audit"], 16)
        self.assertEqual(result["mixed_reward_groups_of_four"]["g3"], 8)
        self.assertEqual(result["candidates_with_all_executions_completed"], 32)
        self.assertEqual(result["candidates_with_any_valid_output"], 32)

    def test_invalid_evidence_is_rejected(self):
        samples, reports = self.records()
        with self.assertRaises(ValueError):
            summarize(samples[:-1], reports)
        for kind in ("hash", "infrastructure", "missing_suite"):
            changed = deepcopy(reports)
            if kind == "hash":
                changed[0]["candidate_hash"] = "bad"
            elif kind == "infrastructure":
                changed[0]["suites"][0]["infrastructure_errors"] = 1
            else:
                changed[0]["suites"].pop()
            with self.assertRaises(ValueError):
                summarize(samples, changed)

    def test_recovery_validates_identity_suites_and_cleanup(self):
        suites = build_suites()
        sample = {"seed": 4000, "source": "# test"}
        report = {"sample_seed": 4000, "candidate_hash": digest(sample["source"]),
                  "suites": [{"suite": s.name, "suite_hash": s.fingerprint,
                              "infrastructure_errors": 0, "outcomes": [{"attempts": [{
                              "metadata": {"image_id": "im-test", "cleanup": "terminated"}}]}]}
                             for s in suites]}
        self.assertTrue(validate_recovered_report(sample, report, suites, "im-test"))
        with self.assertRaises(ValueError):
            validate_recovered_report(sample, report, suites, "im-other")
        report["suites"][0]["infrastructure_errors"] = 1
        self.assertFalse(validate_recovered_report(sample, report, suites, "im-test"))
        report["sample_seed"] = 4001
        with self.assertRaises(ValueError):
            validate_recovered_report(sample, report, suites, "im-test")
