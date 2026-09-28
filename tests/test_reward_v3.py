from copy import deepcopy
import unittest

from verifier_rl.cache import validate_input
from verifier_rl.grading import ExecutionResult, Status, score_suite
from verifier_rl.reward_review import NEW_CONTROLS, authored_output, authored_report, review_controls
from verifier_rl.reward_v2 import CONTROLS, SEED, behavior_scores, behavior_suite
from verifier_rl.reward_v3 import COUNTS, coverage_scores, coverage_suite
from verifier_rl.suites import build_suites, canonical_json


class CoverageProposalTests(unittest.TestCase):
    def test_proposal_preserves_v2_and_extends_domain_deterministically(self):
        old, new = behavior_suite(), coverage_suite()
        self.assertEqual(len(new.cases), 44)
        self.assertEqual(new.cases[:34], old.cases)
        self.assertEqual(new.fingerprint, coverage_suite().fingerprint)
        self.assertNotEqual(new.fingerprint, coverage_suite(SEED + 1).fingerprint)
        self.assertEqual(max(len(c.operations) for c in new.cases), 200)
        self.assertEqual(max(len(o["key"]) for c in new.cases for o in c.operations), 8)
        for family, count in COUNTS.items():
            self.assertEqual(sum(c.tags == (family,) for c in new.cases), count)
        for c in new.cases:
            validate_input(c.operations)
            c.expected  # Both independently structured oracles must agree.

    def test_wide_key_transform_preserves_answers_and_long_traces_are_substantive(self):
        old, new = behavior_suite(), coverage_suite()
        for c in new.cases[34:]:
            if c.name.endswith("domain_wide_keys"):
                original = next(o for o in old.cases if o.tags == c.tags)
                self.assertEqual(c.expected, original.expected)
                self.assertTrue(all(len(o["key"]) == 8 for o in c.operations))
            else:
                self.assertGreater(len(c.operations), 6)
            self.assertIn(None, c.expected)
            self.assertTrue(any(type(v) is int for v in c.expected))

    def test_no_exact_overlap_with_existing_development_audit(self):
        audit = {c.input_hash for c in build_suites()[-1].cases}
        for seed in (SEED, SEED + 1, SEED + 2):
            self.assertFalse(audit & {c.input_hash for c in coverage_suite(seed).cases})

    def test_known_full_false_acceptances_are_detected_by_proposal(self):
        old, new = behavior_suite(), coverage_suite()
        for name in NEW_CONTROLS[:2]:
            before = behavior_scores(authored_report(old, name), old)
            after = coverage_scores(authored_report(new, name), new)
            self.assertEqual(before["binary"], 1)
            self.assertEqual(after["binary"], 0)
            # Still partial credit! Detection is not elimination of reward.
            self.assertAlmostEqual(after["partial"], .88125)
        self.assertEqual(coverage_scores(authored_report(new, "correct"), new)["partial"], 1)

    def test_all_authored_controls_at_three_seeds(self):
        report = review_controls()
        self.assertTrue(report["controls_passed"])
        self.assertFalse(report["model_results"])
        self.assertEqual(len(report["rows"]), 3 * (len(CONTROLS) + len(NEW_CONTROLS)))

    def test_authored_controls_do_not_mutate_or_retain_candidate_state(self):
        ops = coverage_suite().cases[-2].operations
        for name in NEW_CONTROLS:
            original = deepcopy(ops)
            before = authored_output(ops, name)
            self.assertEqual(ops, original)
            self.assertEqual(authored_output(ops, name), before)
        with self.assertRaises(ValueError): authored_output([], "candidate.py")

    def test_infrastructure_and_unexecuted_results_remain_unscored(self):
        suite = coverage_suite()
        records = {c.input_hash: (ExecutionResult(Status.COMPLETED, canonical_json(c.expected).encode()),)
                   for c in suite.cases}
        key = suite.cases[0].input_hash
        records[key] = (ExecutionResult(Status.INFRASTRUCTURE_ERROR),)
        self.assertIsNone(coverage_scores(score_suite(suite, records))["partial"])
        records[key] = ()
        self.assertIsNone(coverage_scores(score_suite(suite, records, allow_unexecuted=True))["partial"])
        records[key] = (ExecutionResult(Status.CANDIDATE_ERROR),)
        result = coverage_scores(score_suite(suite, records))
        self.assertAlmostEqual(result["partial"], 1 - .19 / 8)
        self.assertEqual(result["binary"], 0)

    def test_strict_comparator_is_not_replaced_with_element_or_format_credit(self):
        suite = coverage_suite()
        for raw in (b"[true]", b"[null]", b"demo\n[]", b"{\"reward\":1}"):
            records = {c.input_hash: (ExecutionResult(Status.COMPLETED, raw),) for c in suite.cases}
            scores = coverage_scores(score_suite(suite, records))
            self.assertEqual(scores["partial"], 0)
            self.assertEqual(scores["binary"], 0)

    def test_refuses_stale_identity_tampered_counts_or_flags(self):
        suite = coverage_suite()
        original = authored_report(suite, "correct")
        for mutation in ("hash", "suite", "coverage", "case", "expected", "flag", "count", "reward", "all"):
            bad = deepcopy(original)
            if mutation == "hash": bad["suite_hash"] = "bad"
            elif mutation == "suite": bad["suite"] = "balanced_v2"
            elif mutation == "coverage": bad["outcomes"].pop()
            elif mutation == "case": bad["outcomes"][0]["case"] = "bad"
            elif mutation == "expected": bad["outcomes"][0]["expected"] = []
            elif mutation == "flag": bad["outcomes"][0]["passed"] = 1
            elif mutation == "count": bad["passed_count"] = 0
            elif mutation == "reward": bad["reward"] = 0
            else: bad["all_passed"] = False
            with self.assertRaises(ValueError, msg=mutation): coverage_scores(bad)
        with self.assertRaises(ValueError): coverage_scores(authored_report(behavior_suite(), "correct"))
