from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from verifier_rl.cli import save_suites
from verifier_rl.grading import ExecutionResult, Status, score_suite
from verifier_rl.partial_reward import balanced_scores, balanced_suite
from verifier_rl.reward_v2 import (COMBINED, CONTROLS, COUNTS, SEED, VERSION, WEIGHTS,
                                   behavior_scores, behavior_suite, control_output, validate_behavior_reward)
from verifier_rl.suites import build_suites, canonical_json, get, put


def control_report(suite, name):
    return score_suite(suite, {c.input_hash: (ExecutionResult(Status.COMPLETED,
                       canonical_json(control_output(c.operations, name)).encode()),) for c in suite.cases})


class BehaviorRewardTests(unittest.TestCase):
    def test_legacy_manifests_are_unchanged(self):
        self.assertEqual(balanced_suite().fingerprint, "062eb5f7ee469a6328094b76d4475e95fbcf0c11911bd8b248666742877cdec7")
        self.assertEqual(build_suites()[2].fingerprint, "9ebbdc6de99a74137f9df6073f265f519573e8a1f4b2cd9c1aa78f7047847302")

    def test_version_shapes_and_reproducible_export(self):
        suite = behavior_suite()
        self.assertEqual(suite.version, VERSION)
        self.assertEqual(len(suite.cases), 34)
        self.assertEqual(len({c.input_hash for c in suite.cases}), 34)
        self.assertEqual(sum(WEIGHTS.values()), 1)
        for family, n in COUNTS.items():
            self.assertEqual(sum(c.tags == (family,) for c in suite.cases), n)
        self.assertEqual(suite.fingerprint, behavior_suite().fingerprint)
        self.assertNotEqual(suite.fingerprint, behavior_suite(SEED + 1).fingerprint)
        for c in suite.cases:
            if c.tags != ("edges",):
                self.assertIn(None, c.expected)
                self.assertTrue(any(type(v) is int for v in c.expected))
        with tempfile.TemporaryDirectory() as temp:
            save_suites(Path(temp), (suite,))
            self.assertTrue((Path(temp) / "balanced_v2.suite.json").exists())

    def test_every_expiry_probe_has_a_first_expired_read(self):
        for seed in (SEED, SEED + 1, SEED + 2):
            for c in behavior_suite(seed).cases:
                if c.tags != ("expiry",): continue
                stored, already_read, witness = {}, set(), False
                for op in c.operations:
                    if op["op"] == "put":
                        stored[op["key"]] = op["time"] + op["ttl"]
                        already_read.discard(op["key"])
                    else:
                        key = op["key"]
                        witness |= key in stored and key not in already_read and op["time"] >= stored[key]
                        already_read.add(key)
                self.assertTrue(witness, c.name)

    def test_delete_without_expiration_regression_and_matched_control(self):
        for seed in (SEED, SEED + 1, SEED + 2):
            old, new = balanced_suite(seed), behavior_suite(seed)
            old_scores = balanced_scores(control_report(old, "consume_without_expiry"), old)
            bad = behavior_scores(control_report(new, "consume_without_expiry"), new)
            better = behavior_scores(control_report(new, "consume_with_expiry"), new)
            self.assertEqual(old_scores["partial"], .62)
            self.assertEqual(old_scores["family_pass_rates"]["expiry"], 1)
            self.assertEqual(bad["family_pass_rates"]["expiry"], 0)
            self.assertEqual(bad["partial"], .24)
            self.assertGreater(better["partial"], bad["partial"])
            self.assertEqual(bad["binary"], better["binary"])
            self.assertEqual(bad["binary"], 0)

    def test_authored_controls_and_combined_bug_semantics(self):
        result = validate_behavior_reward()
        self.assertEqual(len(result["rows"]), 3 * len(CONTROLS))
        self.assertEqual(len(CONTROLS), 22)
        self.assertTrue(result["passed"])
        self.assertFalse(result["model_results"])
        # Late FIRST read distinguishes consumption alone from actual expiry.
        ops = [put(0, ttl=1), get(1)]
        self.assertEqual(control_output(ops, "consume_without_expiry"), [7])
        self.assertEqual(control_output(ops, "consume_with_expiry"), [None])
        self.assertEqual(control_output(ops, "consume_inclusive_expiry"), [7])
        repeated = [put(0), get(0), get(0)]
        for name in COMBINED[:4]:
            self.assertEqual(control_output(repeated, name), [7, None])
        with self.assertRaises(ValueError): control_output([], "arbitrary.py")

    def test_controls_do_not_mutate_inputs_or_share_state(self):
        for name in CONTROLS:
            ops = [put(0), get(1), get(1)]
            original = deepcopy(ops)
            missing_before = control_output([get(1)], name)
            control_output(ops, name)
            self.assertEqual(ops, original)
            self.assertEqual(control_output([get(1)], name), missing_before)

    def test_unscored_stays_unscored(self):
        suite = behavior_suite()
        results = {c.input_hash: (ExecutionResult(Status.COMPLETED, canonical_json(c.expected).encode()),)
                   for c in suite.cases}
        first = suite.cases[0].input_hash
        results[first] = (ExecutionResult(Status.INFRASTRUCTURE_ERROR),)
        self.assertIsNone(behavior_scores(score_suite(suite, results))["partial"])
        results[first] = ()
        self.assertIsNone(behavior_scores(score_suite(suite, results, allow_unexecuted=True))["partial"])
        results[first] = (ExecutionResult(Status.CANDIDATE_ERROR),)
        self.assertAlmostEqual(behavior_scores(score_suite(suite, results))["partial"], 1 - .19 / 6)

    def test_refuses_wrong_identity_flags_counts_or_reward(self):
        suite = behavior_suite()
        original = control_report(suite, "correct")
        for mutation in ("hash", "coverage", "case", "expected", "flag", "count", "reward", "all_passed"):
            bad = deepcopy(original)
            if mutation == "hash": bad["suite_hash"] = "bad"
            elif mutation == "coverage": bad["outcomes"].pop()
            elif mutation == "case": bad["outcomes"][0]["case"] = "bad"
            elif mutation == "expected": bad["outcomes"][0]["expected"] = []
            elif mutation == "flag": bad["outcomes"][0]["passed"] = 1
            elif mutation == "count": bad["passed_count"] = 0
            elif mutation == "reward": bad["reward"] = 0
            else: bad["all_passed"] = False
            with self.assertRaises(ValueError, msg=mutation): behavior_scores(bad)
        with self.assertRaises(ValueError): behavior_scores(control_report(balanced_suite(), "correct"))
