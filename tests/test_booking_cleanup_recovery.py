from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_booking_baseline_comparison import samples_for
from tests.test_booking_screen_recovery import Store, intent_for
from tests.test_booking_warmstart import result_for
from tests.test_modal_backend import fake_sdk
from verifier_rl import booking_cleanup_recovery as cleanup, booking_matched_training as release
from verifier_rl import booking_baseline_comparison as study, durable_grading as durable
from verifier_rl import supervised_execution as supervised, sandbox_lifecycle as lifecycle
from verifier_rl.booking_screen_recovery import validate_entry
from verifier_rl.evaluation_journal import persist, ReconciliationRequired
from verifier_rl.panel_execution import request_for, pack_result
from verifier_rl.suites import digest, canonical_json

ROOT = Path(__file__).resolve().parents[1]


class ReviewedRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plan = release.execution_plan(release.make_plan(ROOT))
        self.sample = samples_for(self.plan, cleanup.FAILED_KEY)[1]
        self.case = study.cases_for("training")[0]
        sdk, sb = fake_sdk()
        sdk.__version__, sb.object_id = "1.5.5", "sb-" + "a"*22
        sb.exec.aio.side_effect = TimeoutError()
        sb.terminate.aio.side_effect = TimeoutError()
        with patch.object(lifecycle, "confirm_terminal", AsyncMock(side_effect=ValueError("unknown"))):
            backend = supervised.SupervisedPanelBackend("test", "im-test", study.BOOKING, sdk=sdk,
                                                        creation_interval_seconds=.26)
            self.result = await backend.execute(request_for(self.sample["source"], self.case))
        self.first = pack_result(self.result)
        record_hash = digest(canonical_json(self.first))
        self.review = {record_hash: (self.sample["sample_id"], self.case.input_hash, sb.object_id)}
        self.patcher = patch.object(cleanup, "REVIEWED_FAILURES", self.review)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.terminal = {"version": lifecycle.VERSION, "sandbox_id": sb.object_id,
                         "method": "Sandbox.poll", "returncode": 137, "checked_at": time.time()}
        self.receipt = cleanup.make_reconciliation(self.first, self.terminal, self.sample, self.case)
        self.second = pack_result(result_for(self.sample["source"], self.case, "replacement"))
        self.entry = {"intent-1": intent_for(self.sample, self.case), "attempt-1": self.first,
                      "selected": self.first, "cleanup-reconciliation": self.receipt}

    def test_receipt_does_not_turn_unknown_into_a_score(self):
        outcome, selected = validate_entry(self.entry, self.sample, self.case, "im-test")
        self.assertIsNone(outcome["passed"])
        self.assertEqual(selected, self.first)

    def test_new_attempt_uses_new_receipt_without_overwriting_old_selection(self):
        entry = dict(self.entry, **{"intent-2": intent_for(self.sample, self.case, 2),
                                  "attempt-2": self.second, "selected-after-reconciliation": self.second})
        outcome, selected = validate_entry(entry, self.sample, self.case, "im-test")
        self.assertTrue(outcome["passed"])
        self.assertEqual(selected, self.second)
        self.assertEqual(entry["selected"], self.first)
        with self.assertRaises(ValueError):
            validate_entry(dict(entry, **{"selected-after-reconciliation": self.first}), self.sample, self.case, "im-test")

    def test_receipt_rejects_unknown_terminal_status_wrong_binding_and_unreviewed_record(self):
        for change in ({"run_id": "other"}, {"first_record_hash": "changed"},
                       {"original_snapshot_hash": "changed"},
                       {"terminal": dict(self.terminal, returncode=None)},
                       {"terminal": dict(self.terminal, sandbox_id="sb-other")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                cleanup.validate_reconciliation(self.first, dict(self.receipt, **change), self.sample, self.case)
        first = deepcopy(self.first)
        first["metadata"]["candidate_submission_attempted"] = True
        with self.assertRaises(ValueError):
            cleanup.make_reconciliation(first, self.terminal, self.sample, self.case)

    def test_missing_receipt_cannot_authorize_retry(self):
        entry = dict(self.entry, **{"intent-2": intent_for(self.sample, self.case, 2), "attempt-2": self.second})
        del entry["cleanup-reconciliation"]
        with self.assertRaisesRegex(ValueError, "not justified"):
            validate_entry(entry, self.sample, self.case, "im-test")

    def test_lost_retry_result_is_unknown_not_old_or_successful_result(self):
        entry = dict(self.entry, **{"intent-2": intent_for(self.sample, self.case, 2)})
        outcome, selected = validate_entry(entry, self.sample, self.case, "im-test")
        self.assertIsNone(outcome["passed"])
        self.assertIsNone(selected)

    async def test_durable_recovery_uses_one_retry_and_replays_both_original_and_new_records(self):
        store, commit = Store(), AsyncMock()
        deadline = time.time() + 1000
        plan = dict(self.plan, concurrency=1)
        backend = Mock(execute=AsyncMock(return_value=self.result))
        async def run(path):
            return await durable.execute_batch([self.sample], [self.case], path, cleanup.FAILED_KEY,
                plan, "im-test", deadline, backend, store, commit)
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as recovered:
            path = Path(tmp)
            with self.assertRaisesRegex(ReconciliationRequired, "cleanup"):
                await run(path)
            input_prefix = f"{plan['run_id']}/durable/{cleanup.FAILED_KEY}/{self.sample['sample_id']}/{self.case.input_hash}"
            store.values[input_prefix + "/cleanup-reconciliation"] = self.receipt
            target = path / "inputs" / self.sample["sample_id"] / self.case.input_hash
            original_bytes = (target / "selected.json").read_bytes()
            backend.execute = AsyncMock(return_value=result_for(self.sample["source"], self.case, "replacement"))
            with patch.object(durable.asyncio, "sleep", AsyncMock()):
                entries, progress = await run(path)
            self.assertEqual(progress["new_starts"], 1)
            self.assertEqual(progress["unknown_inputs"], 0)
            self.assertEqual((target / "selected.json").read_bytes(), original_bytes)
            self.assertEqual(json.loads((target / "attempt-1.json").read_text()), self.first)
            backend.execute.reset_mock()
            again, progress = await run(path)
            self.assertEqual(entries, again)
            backend.execute.assert_not_awaited()
            # Reconstruct from provider intents/results after all local files
            # are lost. It must still recover the immutable reconciliation.
            again, progress = await run(Path(recovered))
            self.assertEqual(progress["new_starts"], 0)
            self.assertEqual(progress["unknown_inputs"], 0)
            backend.execute.assert_not_awaited()


class SourceAmendmentTests(unittest.TestCase):
    def test_no_grpo_or_scoring_edits_authorized_by_amendment(self):
        # Fixture source trees test the guard without depending on ignored runs/.
        original = {name: "unchanged" for name in cleanup.CHANGED_FILES}
        original["modal_booking_matched_training.py"] = "def build_trainer(): return 1\ndef run_comparison(): pass"
        original["verifier_rl/grpo_recovery.py"] = "recovery must not change"
        current = {k: v + "\n" for k, v in original.items()}
        current["verifier_rl/grpo_recovery.py"] = original["verifier_rl/grpo_recovery.py"]
        current.update({name: "new adapter" for name in cleanup.ADDED_FILES})
        with patch.object(cleanup, "ORIGINAL_SNAPSHOT_HASH", digest(canonical_json(original))):
            self.assertTrue(cleanup.validate_amendment(original, current)["plan_unchanged"])
            bad = dict(current, **{"verifier_rl/grpo_recovery.py": "changed"})
            with self.assertRaisesRegex(ValueError, "unapproved"):
                cleanup.validate_amendment(original, bad)
            bad = dict(current, **{"modal_booking_matched_training.py": "def build_trainer(): return 2\ndef run_comparison(): pass"})
            with self.assertRaisesRegex(ValueError, "trainer"):
                cleanup.validate_amendment(original, bad)


if __name__ == "__main__":
    unittest.main()
