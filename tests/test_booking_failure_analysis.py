"""Authored offline fixtures only; no candidate execution or cloud access."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verifier_rl import booking_failure_analysis as analysis


class FailureAnalysisTests(unittest.TestCase):
    def test_ast_comparison_ignores_comments_not_boundary_operator(self):
        first = "def f(a,b): return a < b"
        self.assertEqual(analysis.source_ast_hash(first), analysis.source_ast_hash(first + " # comment"))
        self.assertNotEqual(analysis.source_ast_hash(first), analysis.source_ast_hash(first.replace("<", "<=")))
        self.assertIsNone(analysis.source_ast_hash("def broken("))

    def test_candidate_source_is_never_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "must-not-exist"
            source = f"open({str(marker)!r}, 'w').write('executed')"
            self.assertIsNotNone(analysis.source_ast_hash(source))
            self.assertFalse(marker.exists())

    def test_unknown_is_not_a_failure_or_inclusive_match(self):
        cases = analysis.study.cases_for("evaluation")
        unknown = {"passed": None, "inclusive_match": None, "reason": "lost_report", "record_hash": None}
        with patch.object(analysis.study, "validate_entry", return_value=(unknown, None)):
            result = analysis.describe_program({"sample_id": "authored", "source": "pass"},
                {c.input_hash: {} for c in cases}, "authored-image")
        self.assertEqual(result["unknown_inputs"], 287)
        self.assertEqual(result["failed_inputs"], 0)
        self.assertEqual(result["inclusive_matches"], 0)
        self.assertFalse(result["all_observed_failures_have_shared_endpoints"])
        self.assertIsNone(result["smallest_saved_witness"])

    def test_reject_unbound_result_before_loading_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis.persist(root, {"verified_final_result": {},
                "verification_receipt": {"all_original_checks_passed": True, "result_sha256": "wrong"}})
            with self.assertRaisesRegex(ValueError, "result binding"):
                analysis.analyze(root)

    def test_reject_unverified_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis.persist(root, {"verified_final_result": {},
                "verification_receipt": {"all_original_checks_passed": False}})
            with self.assertRaisesRegex(ValueError, "result binding"):
                analysis.analyze(root)


if __name__ == "__main__":
    unittest.main()
