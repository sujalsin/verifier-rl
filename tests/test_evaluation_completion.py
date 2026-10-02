from copy import deepcopy
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from tests import test_evaluation_recovery as recovery_tests
from tests.test_evaluation_recovery import infra_report
from tests.test_reward_shaping import RATES
from verifier_rl import evaluation_completion as completion, evaluation_recovery as recovery
from verifier_rl import reward_shaping as shaping


def terminated_report(report, suite_index=1, code=137, status="candidate_error"):
    result = deepcopy(report)
    suite = result["suites"][suite_index]
    outcome = suite["outcomes"][0]
    previous_pass = outcome["passed"] is True
    outcome.update(passed=False, actual=None, reason=status)
    attempt = outcome["attempts"][0]
    attempt.update(status=status, detail="nonzero_exit")
    m = attempt["metadata"]
    m.update(returncode=code, stdout_bytes=0, stdout_preview="")
    m.pop("stdout_sha256", None)
    suite.update(passed_count=suite["passed_count"] - int(previous_pass), all_passed=False,
                 reward=0 if suite["purpose"] == "training" else None)
    return result


class EvaluationCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recovery_tests.EvaluationRecoveryTests.setUpClass()
        prior = recovery_tests.EvaluationRecoveryTests
        cls.all_reports = deepcopy(prior.reports)
        cls.all_reports[completion.AMBIGUOUS_KEY] = terminated_report(cls.all_reports[completion.AMBIGUOUS_KEY])
        known = {k: v for k, v in cls.all_reports.items() if k not in completion.PENDING_KEYS}
        receipts = {}
        for key, policy, sample in recovery.entries(prior.inputs["generations"]):
            if key in prior.inputs["cached"] or key in completion.PENDING_KEYS:
                continue
            if key == completion.AMBIGUOUS_KEY:
                intent = {"sample": deepcopy(sample), "policy": policy, "attempt_index": 1,
                    "reservation": shaping.reserve_batch(prior.budget, receipts, key + "-attempt-1", 432)}
            else:
                attempt_key = key + "-attempt-1"
                receipts[attempt_key] = shaping.execution_receipt(attempt_key, known[key])
        cls.inputs = {"source_recovery_id": completion.SOURCE_RECOVERY, "previous": deepcopy(prior.inputs),
            "reports": known, "previous_budget": deepcopy(prior.budget), "failed_intent": intent,
            "previous_stop": {"stage": completion.AMBIGUOUS_KEY, "complete_evaluation_records": 17,
                "automatic_retry": False, "detail": "unattributed termination: stop, preserve failure, do not retry"}}
        cls.terminal = {"sandbox_id": recovery.QUARANTINED_SANDBOX, "returncode": 0}
        cls.budget = completion.completion_budget(cls.inputs, {"metered_cost": "1.99"}, RATES, cls.terminal)
        cls.sample = next(s for k, _, s in recovery.entries(prior.inputs["generations"]) if k == completion.AMBIGUOUS_KEY)

    def test_signal_is_bounded_without_changing_raw_evidence(self):
        report = deepcopy(self.inputs["reports"][completion.AMBIGUOUS_KEY])
        original = deepcopy(report)
        a = completion.assess_report(self.sample, report, "im-test")
        self.assertEqual(report, original)
        self.assertEqual(a["suites"]["audit"]["passed_cases"], [387, 388])
        self.assertEqual(a["suites"]["audit"]["full_pass"], [0, 1])
        self.assertEqual(a["partial_reward"], [1, 1])
        self.assertEqual(a["unattributed_inputs"], 1)
        self.assertFalse(a["original_protocol_accepted"])
        with self.assertRaisesRegex(ValueError, "unattributed termination"):
            shaping.require_report(self.sample, report, True, "im-test")

    def test_timeout_has_bounds_but_ordinary_error_is_still_failure(self):
        base = recovery_tests.EvaluationRecoveryTests.reports[completion.AMBIGUOUS_KEY]
        timeout = terminated_report(base, code=-1, status="timeout")
        self.assertEqual(completion.assess_report(self.sample, timeout, "im-test")["unattributed_inputs"], 1)
        error = terminated_report(base, code=1)
        a = completion.assess_report(self.sample, error, "im-test")
        self.assertEqual(a["status"], "validated")
        self.assertEqual(a["suites"]["audit"]["passed_cases"], [387, 387])
        self.assertEqual(a["suites"]["audit"]["full_pass"], [0, 0])

    def test_known_wrong_answer_cannot_become_a_possible_full_pass(self):
        report = deepcopy(self.inputs["reports"][completion.AMBIGUOUS_KEY])
        # Move the known error to a second case, retaining its own case identity.
        target = report["suites"][1]["outcomes"][1]
        target.update(passed=False, actual=None, reason="candidate_error")
        target["attempts"][0].update(status="candidate_error", detail="nonzero_exit")
        target["attempts"][0]["metadata"].update(returncode=1)
        report["suites"][1]["passed_count"] -= 1
        a = completion.assess_report(self.sample, report, "im-test")
        self.assertEqual(a["suites"]["audit"]["passed_cases"], [386, 387])
        self.assertEqual(a["suites"]["audit"]["full_pass"], [0, 0])

    def test_weighted_reward_and_bonus_bounds_use_frozen_family_weights(self):
        report = terminated_report(recovery_tests.EvaluationRecoveryTests.reports[completion.AMBIGUOUS_KEY], suite_index=0)
        a = completion.assess_report(self.sample, report, "im-test")
        self.assertAlmostEqual(a["partial_reward"][0], .97625)
        self.assertEqual(a["partial_reward"][1], 1)
        self.assertAlmostEqual(a["bonus_reward"][0], .488125)
        self.assertEqual(a["bonus_reward"][1], 1)

    def test_corruption_environment_cleanup_and_infra_are_not_salvaged(self):
        original = self.inputs["reports"][completion.AMBIGUOUS_KEY]
        for field, value in (("cleanup", "unconfirmed:TimeoutError"), ("image_id", "different"),
                             ("preflight_returncode", 1), ("runner_hash", "changed")):
            changed = deepcopy(original)
            changed["suites"][1]["outcomes"][0]["attempts"][0]["metadata"][field] = value
            with self.assertRaises(ValueError): completion.assess_report(self.sample, changed, "im-test")
        changed = infra_report(original, "sb-clean-infra", "terminated")
        with self.assertRaises(ValueError): completion.assess_report(self.sample, changed, "im-test")
        changed = deepcopy(original)
        changed["candidate_hash"] = "corrupt"
        with self.assertRaises(ValueError): completion.assess_report(self.sample, changed, "im-test")

    def test_fixed_saved_population_and_outstanding_reservation(self):
        self.assertEqual(len(completion.validate_inputs(self.inputs)), 24)
        for mutate in (
            lambda d: d["reports"].pop("baseline-10000"),
            lambda d: d["failed_intent"]["reservation"].update(total_reserved_usd="0"),
            lambda d: d["previous_stop"].update(complete_evaluation_records=18),
        ):
            changed = deepcopy(self.inputs)
            mutate(changed)
            with self.assertRaises(ValueError): completion.validate_inputs(changed)

    def test_progress_keeps_all_eight_programs_in_denominator(self):
        result = completion.summary(self.inputs, self.inputs["reports"])
        p = result["policies"]["completion_bonus"]
        self.assertEqual(p["planned_programs"], 8)
        self.assertEqual(p["evaluated_programs"], 2)
        self.assertEqual(p["audit_total_cases"], 3104)
        self.assertEqual(p["full_audit_passes"], [1, 8])
        self.assertEqual(p["pending_keys"], list(completion.PENDING_KEYS))
        self.assertFalse(result["all_programs_evaluated"])

    def test_budget_holds_prior_costs_and_requires_terminal_old_sandbox(self):
        self.assertEqual(self.budget["total_trial_limit_usd"], "20")
        self.assertEqual(self.budget["max_sandbox_executions"], 2592)
        self.assertGreater(float(self.budget["fixed_reserved_usd"]),
                           float(self.inputs["failed_intent"]["reservation"]["total_reserved_usd"]))
        for code in (None, True, "0"):
            with self.assertRaises(ValueError):
                completion.completion_budget(self.inputs, {"metered_cost": "1"}, RATES,
                    {"sandbox_id": recovery.QUARANTINED_SANDBOX, "returncode": code})
        with self.assertRaises(ValueError):
            completion.completion_budget(self.inputs, {"metered_cost": "19"}, RATES, self.terminal)

    def accounting(self):
        receipts, reservations = {}, {}
        for key in completion.PENDING_KEYS:
            receipt = shaping.execution_receipt(key, self.all_reports[key])
            reservations[key] = shaping.reserve_batch(self.budget, receipts, key, receipt["executions"])
            receipts[key] = receipt
        return receipts, reservations

    def test_offline_verification_retains_ambiguity_and_rejects_tampering(self):
        receipts, reservations = self.accounting()
        result = completion.verify_completion(self.inputs, self.all_reports, self.budget, receipts, reservations)
        self.assertTrue(result["all_programs_evaluated"])
        self.assertFalse(result["original_protocol_fully_validated"])
        self.assertEqual(result["policies"]["completion_bonus"]["audit_passed_cases"], [3103, 3104])
        changed = deepcopy(receipts)
        changed[completion.PENDING_KEYS[0]]["report_hash"] = "corrupt"
        with self.assertRaises(ValueError):
            completion.verify_completion(self.inputs, self.all_reports, self.budget, changed, reservations)
        changed = deepcopy(self.all_reports)
        changed["baseline-10000"]["timestamp_utc"] = "replaced"
        with self.assertRaises(ValueError): completion.summary(self.inputs, changed)
        changed = deepcopy(self.all_reports)
        old_id = changed["baseline-10000"]["suites"][0]["outcomes"][0]["attempts"][0]["metadata"]["sandbox_id"]
        changed[completion.PENDING_KEYS[0]]["suites"][0]["outcomes"][0]["attempts"][0]["metadata"]["sandbox_id"] = old_id
        with self.assertRaises(ValueError): completion.summary(self.inputs, changed)

    @unittest.skipUnless(find_spec("modal") is not None, "optional Modal SDK needed for mocked launcher")
    def test_controller_runs_only_six_and_continues_after_new_signal(self):
        import modal_complete_evaluation as launcher
        responses = [deepcopy(self.all_reports[k]) for k in completion.PENDING_KEYS]
        responses[0] = terminated_report(responses[0])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            deadline = time.time() + 2100
            with (patch.object(launcher, "artifact_directory", side_effect=lambda p: root / p),
                  patch.object(launcher, "require_current_conformance"), patch.object(launcher, "artifacts"),
                  patch.object(launcher, "claims"),
                  patch.object(launcher, "ModalBackend"), patch("builtins.print"),
                  patch.object(launcher, "evaluate_submission", new=AsyncMock(side_effect=responses)) as evaluate):
                result = launcher.run_completion.local(completion.RUN_ID, self.inputs, self.budget, deadline, {})
                self.assertEqual(evaluate.await_count, 6)
                self.assertEqual([call.args[0]["seed"] for call in evaluate.await_args_list], list(range(10002, 10008)))
                restored = launcher.run_completion.local(completion.RUN_ID, self.inputs, self.budget, deadline, {})
                self.assertEqual(restored, result)
                self.assertEqual(evaluate.await_count, 6)
            self.assertTrue(result["summary"]["all_programs_evaluated"])
            self.assertEqual(result["summary"]["policies"]["completion_bonus"]["ambiguous_programs"], 2)
            saved = json.loads((root / completion.RUN_ID / completion.PENDING_KEYS[0] / "result.json").read_text())
            self.assertEqual(saved, responses[0])
            self.assertEqual(result["reports"][completion.AMBIGUOUS_KEY], self.inputs["reports"][completion.AMBIGUOUS_KEY])


if __name__ == "__main__":
    unittest.main()
