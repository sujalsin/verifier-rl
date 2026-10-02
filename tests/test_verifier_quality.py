from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import unittest

from verifier_rl.grading import ExecutionResult, MAX_OUTPUT_BYTES, Status
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import BOOKING, CACHE, LIMITER, build_suite, control_answer
from verifier_rl.verifier_quality import (CalibrationRow, calibrate, compare_output, grade,
                                         noise_draw, preflight, screen_budget_preview,
                                         screen_gate, training_reward)


def results_for(cases, fault="correct"):
    # Calls only committed author-written functions, never candidate source strings.
    return {case.input_hash: ExecutionResult(Status.COMPLETED,
            canonical_json(control_answer(case.task_id, case.arguments, fault)).encode()) for case in cases}


def row(index, reference=False, structured=False, audit=False, pool="calibration"):
    # Synthetic unit-test records, not a real calibration claim.
    return CalibrationRow(str(index), digest(str(index)), CACHE, pool, reference, structured, audit)


class VerifierQualityTests(unittest.TestCase):
    def test_preflight_controls_do_not_claim_model_or_cloud_results(self):
        report = preflight(Path(__file__).resolve().parents[1])
        self.assertTrue(report["offline_controls_passed"])
        self.assertFalse(report["calibrated"])
        self.assertFalse(report["cloud_ready"])
        self.assertEqual(report["model_samples"], 0)
        self.assertEqual(report["screen_samples_planned"], 24)
        self.assertEqual(report["screen_execution_upper_bound"], 768)
        for task in report["tasks"]:
            self.assertEqual([r["reference_reward"] for r in task["controls"]], [1, 0, 0])
            self.assertEqual([r["structured_reward"] for r in task["controls"]], [1, 1, 0])

    def test_parser_and_type_aware_output(self):
        for task in (CACHE, BOOKING, LIMITER):
            case = build_suite(task, "training")[0]
            self.assertEqual(compare_output(case, canonical_json(case.expected).encode()), (True, "pass"))
            for payload in (b"ALL TESTS PASSED", b"[]\n[]", b"\xff", b"[" * 2000):
                self.assertEqual(compare_output(case, payload)[1], "invalid_json")
            self.assertEqual(compare_output(case, b" " * (MAX_OUTPUT_BYTES + 1))[1], "output_limit")
            self.assertEqual(compare_output(case, b"NaN")[1], "invalid_schema")
        self.assertEqual(compare_output(build_suite(BOOKING, "training")[0], b"true")[1], "invalid_schema")
        self.assertEqual(compare_output(build_suite(LIMITER, "training")[0], b"[1,1]")[1], "invalid_schema")
        self.assertEqual(compare_output(build_suite(CACHE, "training")[0], b"[true]")[1], "invalid_schema")
        self.assertEqual(compare_output(build_suite(CACHE, "training")[0], b"[]")[1], "wrong_answer")

    def test_every_infrastructure_or_ambiguous_result_blocks_all_conditions(self):
        cases = build_suite(CACHE, "training")
        unknowns = [None, ExecutionResult(Status.INFRASTRUCTURE_ERROR), ExecutionResult(Status.TIMEOUT),
                    ExecutionResult(Status.CANDIDATE_ERROR, metadata={"returncode": 137}),
                    ExecutionResult(Status.CANDIDATE_ERROR, metadata={"returncode": -9})]
        # Even an ignored boundary case cannot silently hide infrastructure loss.
        for unknown in unknowns:
            results = results_for(cases)
            if unknown is None:
                del results[cases[-1].input_hash]
            else:
                results[cases[-1].input_hash] = unknown
            report = grade(cases, results)
            self.assertFalse(report["resolved"])
            for condition in ("reference", "random_false_acceptance", "structured_false_acceptance"):
                with self.assertRaises(ValueError):
                    training_reward(condition, report, probability=Fraction(1), draw=Fraction(0))

    def test_ordinary_candidate_errors_and_extraction_failures_are_not_unknown(self):
        cases = build_suite(CACHE, "training")
        for status in (Status.CANDIDATE_ERROR, Status.EXTRACTION_REJECTED, Status.OUTPUT_LIMIT):
            results = results_for(cases)
            results[cases[0].input_hash] = ExecutionResult(status, metadata={"returncode": 1})
            report = grade(cases, results)
            self.assertTrue(report["resolved"])
            self.assertEqual(training_reward("reference", report), 0)
            # The deliberately noisy condition can promote even a known failure.
            self.assertEqual(training_reward("random_false_acceptance", report,
                                            probability=Fraction(1), draw=Fraction(0)), 1)

    def test_development_cannot_supply_rewards(self):
        cases = build_suite(CACHE, "development")
        report = grade(cases, results_for(cases))
        self.assertTrue(report["reference_accepted"])
        self.assertIsNone(report["structured_accepted"])
        with self.assertRaises(ValueError):
            training_reward("reference", report)

    def test_malformed_or_mixed_suite_rejected(self):
        cases = build_suite(CACHE, "training")
        for bad in (cases[:-1], cases[:-1] + cases[:1],
                    cases[:-1] + build_suite(BOOKING, "training")[-1:]):
            with self.assertRaises(ValueError):
                grade(bad, {})
        with self.assertRaises(ValueError):
            grade(cases, {"unknown": ExecutionResult(Status.COMPLETED, b"[]")})
        results = results_for(cases)
        results[cases[0].input_hash] = ExecutionResult(Status.COMPLETED, b"[]", metadata={"returncode": 1})
        with self.assertRaises(ValueError):
            grade(cases, results)

    def test_random_draw_replay_and_exact_threshold(self):
        first = noise_draw("trusted-seed", "run/task/rollout-0001")
        self.assertEqual(first, noise_draw("trusted-seed", "run/task/rollout-0001"))
        self.assertNotEqual(first, noise_draw("trusted-seed", "run/task/rollout-0002"))
        cases = build_suite(CACHE, "training")
        report = grade(cases, results_for(cases, "inclusive_boundary"))
        self.assertEqual(training_reward("random_false_acceptance", report,
                                        probability=Fraction(1, 4), draw=Fraction(1, 4)), 0)
        self.assertEqual(training_reward("random_false_acceptance", report,
                                        probability=Fraction(1, 4), draw=Fraction(1, 5)), 1)
        for probability in (-1, float("nan"), .25, Fraction(2)):
            with self.assertRaises(ValueError):
                training_reward("random_false_acceptance", report, probability=probability, draw=first)

    def test_matching_accounts_for_reference_false_acceptance(self):
        # N=10 audit failures, H=2 reference accepts, S=4 structured accepts.
        rows = [row(i, reference=i < 2, structured=i < 4) for i in range(10)]
        rows += [row(10, True, True, True)]
        result = calibrate(CACHE, rows)
        self.assertEqual(result["promotion_probability"], "1/4")  # Not 4/10.
        self.assertEqual(result["structured_false_acceptance_rate"], "2/5")
        self.assertEqual(result["expected_random_false_acceptance_rate"], "2/5")
        self.assertTrue(result["ready_for_review"])

    def test_calibration_gates_insufficient_or_degenerate_data(self):
        for rows in ([row(0)], [row(i) for i in range(8)],
                     [row(i, structured=True) for i in range(8)],
                     [row(i, True, True, True) for i in range(8)],
                     [row(i, True, True) for i in range(8)]):
            self.assertFalse(calibrate(CACHE, rows)["ready_for_review"])
        rows = [replace(row(i, structured=i < 4), source_hash=digest("same")) for i in range(8)]
        self.assertIn("fewer_than_four_distinct_audit_rejected_sources", calibrate(CACHE, rows)["reasons"])

    def test_calibration_rejects_screen_reuse_duplicates_unknowns_and_task_mixing(self):
        bad_rows = [[row(0, pool="screen")], [row(0), row(0)], [replace(row(0), task_id=BOOKING)]]
        for rows in bad_rows:
            with self.assertRaises(ValueError):
                calibrate(CACHE, rows)
        with self.assertRaises(ValueError):
            row(0, reference=True, structured=False)
        with self.assertRaises(ValueError):
            row(0, audit=None)

    def test_screen_gate_is_not_a_success_claim(self):
        rows = [row(i, reference=i % 2 == 0, structured=True, audit=i == 0, pool="screen")
                for i in range(8)]
        self.assertTrue(screen_gate(CACHE, rows)["ready_for_calibration_review"])
        rows = [row(i, pool="screen") for i in range(8)]
        result = screen_gate(CACHE, rows)
        self.assertFalse(result["ready_for_calibration_review"])
        self.assertIn("reference_rewards_have_no_variation", result["reasons"])
        with self.assertRaises(ValueError):
            screen_gate(CACHE, rows[:-1])

    def test_screen_budget_holds_existing_costs_and_includes_all_resources(self):
        rates = {"cpu_hour_cost": ".0473", "mem_gib_hour_cost": ".008",
                 "cpu_hour_cost_sandbox": ".1419", "mem_gib_hour_cost_sandbox": ".024",
                 "gpu_hour_cost_l40s": "1.95"}
        quote = screen_budget_preview(rates, "2.03009247", "14.38760017666666666666666667")
        self.assertEqual(quote["prior_costs_held_usd"], "14.38760017666666666666666667")
        self.assertTrue(quote["fits_existing_limit"])
        self.assertFalse(quote["enforcement_implemented"])
        self.assertFalse(quote["calibration_or_training_included"])
        self.assertEqual(quote["model_input_executions"], 768)
        self.assertEqual(quote["authored_conformance_executions"], 18)
        self.assertLess(Decimal(quote["cumulative_reserve_usd"]), Decimal("20"))
        self.assertFalse(screen_budget_preview(rates, "19.99", "14")["fits_existing_limit"])
        self.assertEqual(screen_budget_preview(rates, "15", "14")["prior_costs_held_usd"], "15")
        with self.assertRaises(ValueError):
            screen_budget_preview(rates, "NaN", "14")


if __name__ == "__main__":
    unittest.main()
