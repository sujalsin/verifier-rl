from collections import Counter
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import math
import unittest

from tests.test_panel_screen import execution
from verifier_rl import booking_verifier_v2 as v2
from verifier_rl.grading import Status
from verifier_rl.model_trial import submission_from_completion
from verifier_rl.panel_execution import pack_result
from verifier_rl.suites import canonical_json
from verifier_rl.task_panel import FAMILIES, validate_call, BOOKING


def outcomes(role="training", passed=True):
    return {case.input_hash: passed for case in v2.cases_for(role)}


class BookingVerifierV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = v2.cases_for("training")
        # The source is a label for mocked transport; it is never executed.
        cls.candidate = submission_from_completion("def required_capacity(bookings):\n    return 0\n")
        cls.records = {case.input_hash: pack_result(execution(cls.candidate["source"], case, case.input_hash))
                       for case in cls.cases}

    def test_sizes_families_and_domain_extremes(self):
        for role, count in (("training", 96), ("audit", 192)):
            cases = v2.cases_for(role)
            self.assertEqual(len(cases), count)
            self.assertEqual(Counter(c.family for c in cases), dict.fromkeys(FAMILIES, count // 4))
            sizes = {len(c.arguments["bookings"]) for c in cases}
            self.assertTrue(set((0, *v2.SIZES)).issubset(sizes))
            self.assertEqual(max(sizes), 200)
            self.assertTrue(any(a == 0 for c in cases for a, b in c.arguments["bookings"]))
            self.assertTrue(any(b == 10000 for c in cases for a, b in c.arguments["bookings"]))
            for case in cases:
                validate_call(BOOKING, case.arguments)
                self.assertIs(type(case.expected), int)

    def test_no_audit_or_historical_reuse_even_after_permutation(self):
        keys = {role: {v2.multiset_key(c.arguments["bookings"]) for c in cases}
                for role, cases in v2.suites().items()}
        self.assertEqual(keys["training"] & keys["audit"], {"[]"})
        for role, count in v2.COUNTS.items():
            self.assertEqual(len(keys[role]), count)
            self.assertEqual(keys[role] & v2.historical_inputs(), {"[]"})
        self.assertEqual(v2.multiset_key([[1, 2], [3, 4]]), v2.multiset_key([[3, 4], [1, 2]]))
        self.assertNotEqual(v2.multiset_key([[1, 2]]), v2.multiset_key([[1, 2], [1, 2]]))

    def test_reproducible_immutable_suites_and_fresh_arguments(self):
        first = {role: [c.manifest() for c in cases] for role, cases in v2.suites().items()}
        v2.suites.cache_clear()
        self.assertEqual(first, {role: [c.manifest() for c in cases] for role, cases in v2.suites().items()})
        with self.assertRaises(TypeError):
            v2.suites()["training"] = ()
        arguments = self.cases[0].arguments
        arguments["bookings"].clear()
        self.assertTrue(self.cases[0].arguments["bookings"])
        self.assertNotEqual(v2.suite_hash("training"), v2.suite_hash("audit"))

    def test_old_protocol_counts_and_rewards_remain_unchanged(self):
        self.assertEqual({r: len(c) for r, c in v2.historical.suites().items()},
                         {"training": 16, "calibration": 32, "evaluation": 32})
        report = {"version": v2.historical.VERSION, "role": "training", "reference": False, "structured": True}
        self.assertEqual(v2.historical.reward("reference", report, "0", "test"), 0)
        self.assertEqual(v2.historical.reward("structured", report, "0", "test"), 1)

    def test_known_fault_maps_are_caught_not_claimed_model_results(self):
        for role, controls in v2.control_summary().items():
            n = v2.COUNTS[role]
            self.assertEqual(controls["correct"]["passed"], n)
            self.assertEqual(controls["empty_only"]["passed"], n - 1)
            self.assertEqual(controls["constant_zero"]["passed"], 1)
            for fault in ("inclusive_end", "deduplicate", "constant_one", "return_length"):
                self.assertLess(controls[fault]["passed"], n)

    def test_reward_endpoints_monotonicity_and_log_concavity(self):
        for shape in v2.SHAPES:
            values = [v2.shape_reward(i / 100, shape) for i in range(101)]
            self.assertEqual((values[0], values[-1]), (0, 1))
            self.assertEqual(values, sorted(values))
            self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in values))
        logs = [v2.shape_reward(i / 100, "logarithmic") for i in range(101)]
        increments = [b - a for a, b in zip(logs, logs[1:])]
        self.assertTrue(all(a > b for a, b in zip(increments, increments[1:])))
        self.assertAlmostEqual(v2.shape_reward(.5, "logarithmic"), math.log(5.5) / math.log(10))

    def test_log_transform_cannot_add_information_to_binary_rewards(self):
        for value in (0, 1):
            self.assertEqual(v2.shape_reward(value, "linear"), v2.shape_reward(value, "logarithmic"))

    def test_reward_rejects_invalid_values_and_shapes(self):
        for value in (True, False, None, "0.5", -.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                v2.shape_reward(value, "linear")
        with self.assertRaises(ValueError):
            v2.shape_reward(.5, "unknown")

    def test_partial_reward_retains_information_lost_by_binary(self):
        raw = {case.input_hash: i < 48 for i, case in enumerate(self.cases)}
        report = v2.score_outcomes("training", raw)
        self.assertEqual(v2.training_reward(report, shape="binary"), 0)
        self.assertEqual(v2.training_reward(report, shape="linear"), .5)
        self.assertGreater(v2.training_reward(report, shape="logarithmic"), .5)

    def test_weak_condition_still_omits_only_empty_and_requires_its_record(self):
        raw = outcomes()
        empty = next(c.input_hash for c in self.cases if not c.arguments["bookings"])
        raw[empty] = False
        report = v2.score_outcomes("training", raw)
        self.assertEqual(v2.training_reward(report, "reference", "binary"), 0)
        self.assertEqual(v2.training_reward(report, "reference", "linear"), 95 / 96)
        for shape in v2.SHAPES:
            self.assertEqual(v2.training_reward(report, "structured", shape), 1)
        del raw[empty]
        with self.assertRaises(ValueError):
            v2.score_outcomes("training", raw)

    def test_partial_scores_are_not_nested_when_denominators_differ(self):
        raw = outcomes(passed=False)
        empty = next(c.input_hash for c in self.cases if not c.arguments["bookings"])
        raw[empty] = True
        report = v2.score_outcomes("training", raw)
        self.assertEqual(v2.training_reward(report, "reference"), 1 / 96)
        self.assertEqual(v2.training_reward(report, "structured"), 0)

    def test_audit_and_tampered_reports_cannot_supply_rewards(self):
        with self.assertRaises(ValueError):
            v2.training_reward(v2.score_outcomes("audit", outcomes("audit")))
        report = v2.score_outcomes("training", outcomes())
        for change in ({"role": "audit"}, {"version": "old"}, {"suite_hash": "other"},
                       {"passed": 0}, {"full_pass": False}, {"total": 95}):
            with self.assertRaises(ValueError):
                v2.training_reward(dict(report, **change))
        with self.assertRaises(ValueError):
            v2.training_reward(report, "random")

    def test_unknown_outcomes_and_extra_inputs_are_not_partial_credit(self):
        raw = outcomes()
        for value in (None, 1, 0, "pass"):
            with self.assertRaises(ValueError):
                v2.score_outcomes("training", dict(raw, **{self.cases[0].input_hash: value}))
        for role in ("evaluation", "final", "development"):
            with self.assertRaises(ValueError):
                v2.score_outcomes(role, raw)
        with self.assertRaises(ValueError):
            v2.score_outcomes("training", dict(raw, extra=True))

    def test_mocked_sandbox_records_are_replayed_and_identity_checked(self):
        report = v2.grade_submission(self.candidate, self.records, "training", "im-test")
        self.assertEqual(report["passed"], 96)
        self.assertEqual(len(report["sandbox_ids"]), 96)
        self.assertEqual(v2.training_reward(report), 1)
        for change in ({"source_hash": "other"}, {"input_hash": "other"}, {"cleanup": "unknown"},
                       {"returncode": 137}, {"stdout_sha256": "other"}):
            records = deepcopy(self.records)
            records[self.cases[0].input_hash]["metadata"].update(change)
            with self.assertRaises(ValueError):
                v2.grade_submission(self.candidate, records, "training", "im-test")

    def test_ordinary_error_is_failure_but_infrastructure_is_unscored(self):
        case = self.cases[0]
        good = execution(self.candidate["source"], case, case.input_hash)
        failure = replace(good, status=Status.CANDIDATE_ERROR, stdout=b"", detail="nonzero_exit",
                          metadata=dict(good.metadata, returncode=1, stdout_bytes=0,
                                        stdout_sha256=sha256(b"").hexdigest()))
        records = dict(self.records, **{case.input_hash: pack_result(failure)})
        report = v2.grade_submission(self.candidate, records, "training", "im-test")
        self.assertEqual(v2.training_reward(report), 95 / 96)
        for status in (Status.INFRASTRUCTURE_ERROR, Status.TIMEOUT):
            records[case.input_hash] = pack_result(replace(failure, status=status))
            with self.assertRaises(ValueError):
                v2.grade_submission(self.candidate, records, "training", "im-test")

    def test_strict_json_and_type_comparison_still_apply_to_partial_rewards(self):
        case = self.cases[0]
        good = execution(self.candidate["source"], case, case.input_hash)
        for output in (b"true", b"1.0", b'"1"', b"ALL TESTS PASSED", b"1\n1"):
            bad = replace(good, stdout=output, metadata=dict(good.metadata, stdout_bytes=len(output),
                                                            stdout_sha256=sha256(output).hexdigest()))
            records = dict(self.records, **{case.input_hash: pack_result(bad)})
            report = v2.grade_submission(self.candidate, records, "training", "im-test")
            self.assertEqual(report["passed"], 95)

    def test_extraction_failures_remain_in_the_denominator(self):
        candidate = submission_from_completion("```python\ndef unfinished():")
        report = v2.grade_submission(candidate, {}, "training", "im-test")
        self.assertEqual((report["passed"], report["total"]), (0, 96))
        self.assertEqual(v2.training_reward(report), 0)
        with self.assertRaises(ValueError):
            v2.grade_submission(candidate, self.records, "training", "im-test")

    def test_reused_sandbox_and_missing_records_are_rejected(self):
        records = deepcopy(self.records)
        a, b = (c.input_hash for c in self.cases[:2])
        records[b]["metadata"]["sandbox_id"] = records[a]["metadata"]["sandbox_id"]
        with self.assertRaisesRegex(ValueError, "reused sandbox"):
            v2.grade_submission(self.candidate, records, "training", "im-test")
        del records[b]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            v2.grade_submission(self.candidate, records, "training", "im-test")


if __name__ == "__main__":
    unittest.main()
