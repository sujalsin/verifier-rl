from copy import deepcopy
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests import test_evaluation_completion as completion_tests
from tests.test_reward_shaping import RATES
from verifier_rl import evaluation_completion as completion, evaluation_journal as journal
from verifier_rl import reward_shaping as shaping
from verifier_rl.suites import canonical_json, digest


class Preempted(BaseException):
    pass


class Claims:
    """Provider's atomic insert-if-absent contract, with no external calls."""
    def __init__(self):
        self.values = {}

    def put(self, key, value, *, skip_if_exists):
        assert skip_if_exists
        if key in self.values:
            return False
        self.values[key] = value
        return True


class EvaluationJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        completion_tests.EvaluationCompletionTests.setUpClass()
        prior = completion_tests.EvaluationCompletionTests
        cls.inputs, cls.budget, cls.reports = prior.inputs, prior.budget, prior.all_reports
        items, receipts = {}, {}
        cls.rows = completion.validate_inputs(cls.inputs)
        for key, policy, sample in cls.rows:
            if key not in completion.PENDING_KEYS:
                continue
            item = {"intent": journal.intent_for(key, policy, sample, cls.budget, receipts)}
            if key != journal.RESUME_KEY:
                item.update(result=cls.reports[key], receipt=shaping.execution_receipt(key, cls.reports[key]),
                    assessment=completion.assess_report(sample, cls.reports[key], "im-test"))
                receipts[key] = item["receipt"]
            items[key] = item
        cls.checkpoint = {"inputs": cls.inputs, "budget": cls.budget, "deadline": 1,
                          "source_snapshot": {}, "items": items}
        cls.evidence = {"stopped_controller": {"app_id": journal.STOPPED_APP_ID, "state": "stopped", "tasks": "0"},
            "sandbox_app_name": "fixture", "sandbox_app_id": "ap-fixture", "active_sandbox_ids": [],
            "cause": "controller_preemption", "replacement_authorized": True,
            "interrupted_intent_hash": digest(canonical_json(items[journal.RESUME_KEY]["intent"]))}
        cls.resume_budget = journal.resume_budget(cls.checkpoint, cls.evidence, {"metered_cost": "2"}, RATES)

    def save_checkpoint(self, root, checkpoint=None):
        checkpoint = self.checkpoint if checkpoint is None else checkpoint
        journal.persist(root, {k: v for k, v in checkpoint.items() if k != "items"})
        for key, values in checkpoint["items"].items():
            journal.persist(root / key, values)

    def test_restore_23_results_without_releasing_interrupted_reservation(self):
        original = deepcopy(self.checkpoint)
        state = journal.validate_checkpoint(self.checkpoint)
        self.assertEqual(len(state["reports"]), 23)
        self.assertEqual(state["interrupted"], [journal.RESUME_KEY])
        self.assertEqual(len(state["receipts"]), 5)
        self.assertEqual(self.checkpoint, original)
        self.assertEqual(state["reservations"][journal.RESUME_KEY], original["items"][journal.RESUME_KEY]["intent"]["reservation"])
        self.assertFalse(state["summary"]["all_programs_evaluated"])

    def test_missing_derived_records_recompute_from_raw_without_execution(self):
        checkpoint = deepcopy(self.checkpoint)
        key = completion.PENDING_KEYS[0]
        checkpoint["items"][key].pop("receipt")
        checkpoint["items"][key].pop("assessment")
        state = journal.validate_checkpoint(checkpoint)
        self.assertEqual(state["derived"][key]["receipt"], self.checkpoint["items"][key]["receipt"])
        self.assertEqual(len(state["reports"]), 23)

    def test_changed_receipt_intent_or_raw_report_is_rejected(self):
        key = completion.PENDING_KEYS[0]
        for mutate in (
            lambda c: c["items"][key]["receipt"].update(report_hash="changed"),
            lambda c: c["items"][key]["intent"]["reservation"].update(total_reserved_usd="0"),
            lambda c: c["items"][key]["result"].update(candidate_hash="changed"),
            lambda c: c["items"][key].pop("intent"),
            lambda c: c["items"][key].pop("result"),
        ):
            checkpoint = deepcopy(self.checkpoint)
            mutate(checkpoint)
            with self.assertRaises(ValueError): journal.validate_checkpoint(checkpoint)

    def test_manifest_restart_does_not_overwrite_or_reset_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            journal.persist(root, {"deadline": 1, "budget": self.budget})
            original = (root / "budget.json").read_bytes()
            journal.persist(root, {"deadline": 1, "budget": self.budget})
            self.assertEqual((root / "budget.json").read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "saved evidence differs"):
                journal.persist(root, {"deadline": time.time() + 600, "new_record": {}})
            self.assertFalse((root / "new_record.json").exists())

    def test_reconciliation_refuses_active_controller_sandboxes_or_changed_intent(self):
        for mutate in (
            lambda e: e["stopped_controller"].update(state="running"),
            lambda e: e["stopped_controller"].update(tasks="1"),
            lambda e: e.update(active_sandbox_ids=["sb-running"]),
            lambda e: e.update(replacement_authorized=False),
            lambda e: e.update(interrupted_intent_hash="changed"),
        ):
            evidence = deepcopy(self.evidence)
            mutate(evidence)
            with self.assertRaises(ValueError): journal.require_reconciliation(self.checkpoint, evidence)

    def test_resume_reserves_old_batch_and_nonpreemptible_controller_costs(self):
        budget = self.resume_budget
        held = self.checkpoint["items"][journal.RESUME_KEY]["intent"]["reservation"]["total_reserved_usd"]
        self.assertEqual(budget["old_reservation_held_usd"], held)
        self.assertEqual(budget["max_sandbox_executions"], 432)
        self.assertEqual(budget["controller_start_limit"], 2)
        self.assertEqual(budget["nonpreemptible_rate_multiplier"], 3)
        self.assertGreater(float(budget["fixed_reserved_usd"]), float(held))
        self.assertEqual(budget["total_trial_limit_usd"], "20")
        with self.assertRaises(ValueError):
            journal.resume_budget(self.checkpoint, self.evidence, {"metered_cost": "19"}, RATES)

    def test_resume_verification_preserves_bounds_and_all_raw_reports(self):
        key = journal.RESUME_KEY + "-" + journal.RESUME_ID
        receipt = shaping.execution_receipt(key, self.reports[journal.RESUME_KEY])
        reservation = shaping.reserve_batch(self.resume_budget, {}, key, 432)
        result = journal.verify_resume(self.checkpoint, self.evidence, self.resume_budget,
                                       self.reports, receipt, reservation)
        self.assertTrue(result["all_programs_evaluated"])
        self.assertFalse(result["original_protocol_fully_validated"])
        self.assertEqual(result["resumption"]["controller_interruption_replacements"], 1)
        self.assertEqual(result["resumption"]["interrupted_unrecorded_execution_bounds"], [0, 432])
        self.assertEqual(result["policies"]["completion_bonus"]["unattributed_inputs"], 1)
        changed = deepcopy(self.reports)
        changed["completion_bonus-10006"]["timestamp_utc"] = "changed"
        with self.assertRaisesRegex(ValueError, "completed raw evidence"):
            journal.verify_resume(self.checkpoint, self.evidence, self.resume_budget, changed, receipt, reservation)

    @unittest.skipUnless(find_spec("modal") is not None, "optional SDK for mocked controller")
    def test_same_preemption_restart_stops_for_reconciliation_not_file_exists(self):
        import modal_complete_evaluation as launcher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.save_checkpoint(root / completion.RUN_ID)
            with (patch.object(launcher, "artifact_directory", side_effect=lambda p: root / p),
                  patch.object(launcher, "require_current_conformance"), patch.object(launcher, "artifacts"),
                  patch.object(launcher, "claims", Claims()), patch.object(launcher, "evaluate_submission") as evaluate):
                with self.assertRaisesRegex(journal.ReconciliationRequired, "unfinished batch"):
                    launcher.run_completion.local(completion.RUN_ID, self.inputs, self.budget, 1, {})
                evaluate.assert_not_called()
            self.assertEqual(journal.load_checkpoint(root / completion.RUN_ID), self.checkpoint)

    @unittest.skipUnless(find_spec("modal") is not None, "optional SDK for mocked controller")
    def test_restart_after_raw_result_reuses_it_and_repairs_receipt(self):
        import modal_complete_evaluation as launcher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.save_checkpoint(root / completion.RUN_ID)
            resume_root = root / completion.RUN_ID / journal.RESUME_ID
            key = journal.RESUME_KEY + "-" + journal.RESUME_ID
            raw = resume_root / key / "result.json"
            crashed = []
            def commit():
                if raw.exists() and not crashed:
                    crashed.append(True)
                    raise Preempted()
            deadline = time.time() + 600
            with (patch.object(launcher, "artifact_directory", side_effect=lambda p: root / p),
                  patch.object(launcher, "require_current_conformance"),
                  patch.object(launcher.recovery, "check_frozen_sources"),
                  patch.object(launcher, "artifacts") as volume, patch.object(launcher, "claims", Claims()),
                  patch.object(launcher.modal.App, "lookup", return_value=Mock(app_id="ap-fixture")),
                  patch.object(launcher.modal.Sandbox, "list", return_value=[]),
                  patch.object(launcher, "ModalBackend"), patch("builtins.print"),
                  patch.object(launcher, "evaluate_submission", new=AsyncMock(return_value=deepcopy(self.reports[journal.RESUME_KEY]))) as evaluate):
                volume.commit.side_effect = commit
                with self.assertRaises(Preempted):
                    launcher.run_resume.local(self.checkpoint, self.evidence, self.resume_budget, deadline, {})
                self.assertTrue(raw.exists())
                self.assertFalse((resume_root / key / "receipt.json").exists())
                raw_bytes = raw.read_bytes()
                result = launcher.run_resume.local(self.checkpoint, self.evidence, self.resume_budget, deadline, {})
                self.assertEqual(evaluate.await_count, 1)
                self.assertEqual(raw.read_bytes(), raw_bytes)
                self.assertTrue((resume_root / key / "receipt.json").exists())
                self.assertTrue(result["summary"]["all_programs_evaluated"])
                with self.assertRaisesRegex(journal.ReconciliationRequired, "controller-start budget"):
                    launcher.run_resume.local(self.checkpoint, self.evidence, self.resume_budget, deadline, {})
                self.assertEqual(evaluate.await_count, 1)
            self.assertEqual(journal.load_checkpoint(root / completion.RUN_ID), self.checkpoint)

    @unittest.skipUnless(find_spec("modal") is not None, "optional SDK for mocked controller")
    def test_claim_or_deadline_blocks_duplicate_execution(self):
        import modal_complete_evaluation as launcher
        _, policy, sample = next(r for r in self.rows if r[0] == journal.RESUME_KEY)
        key = journal.RESUME_KEY + "-" + journal.RESUME_ID
        for expired in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                claims = Claims()
                claims.put("already-claimed", True, skip_if_exists=True)
                with (patch.object(launcher, "claims", claims), patch.object(launcher, "artifacts"),
                      patch.object(launcher, "evaluate_submission") as evaluate):
                    with self.assertRaises(ValueError):
                        launcher.grade_saved(Path(temp), "already-claimed", key, policy, sample,
                            self.inputs["previous"]["setup"], self.resume_budget, {},
                            1 if expired else time.time() + 600, attempt_index=2)
                    evaluate.assert_not_called()
                    if not expired:
                        with self.assertRaisesRegex(journal.ReconciliationRequired, "intent without result"):
                            launcher.grade_saved(Path(temp), "new-claim-must-not-help", key, policy, sample,
                                self.inputs["previous"]["setup"], self.resume_budget, {}, time.time() + 600, attempt_index=2)
                        evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
