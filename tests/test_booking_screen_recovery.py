import ast
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_booking_boundary_screen import samples_for, reports_for
from tests.test_booking_warmstart import result_for
from tests.test_modal_backend import process
from tests.test_supervised_execution import simulated_result
from verifier_rl import booking_screen_recovery as recovery
from verifier_rl import booking_boundary_screen as screen, supervised_execution as runner
from verifier_rl.evaluation_journal import persist
from verifier_rl.grading import Status
from verifier_rl.panel_execution import pack_result
from verifier_rl.suites import digest
from verifier_rl.task_panel import BOOKING

ROOT = Path(__file__).resolve().parents[1]


class Store:
    def __init__(self):
        self.values = {}
        self.get, self.put = Mock(), Mock()
        self.get.aio, self.put.aio = AsyncMock(side_effect=self.read), AsyncMock(side_effect=self.write)

    def read(self, key, default=None):
        return deepcopy(self.values.get(key, default))

    def write(self, key, value, *, skip_if_exists):
        if key in self.values:
            return False
        self.values[key] = deepcopy(value)
        return True


def source_for(samples, plan):
    return {"generation": {"samples": samples}, "plan": plan, "setup": {"sandbox_image_id": "im-test"},
            "entries": {s["sample_id"]: {} for s in samples}, "historical_sandbox_ids": []}


def intent_for(sample, case, number=1):
    return {"sample_id": sample["sample_id"], "source_hash": digest(sample["source"]),
            "input_hash": case.input_hash, "attempt": number, "deadline": 12345}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.plan = screen.make_plan(ROOT)
        self.sample = samples_for(self.plan)[0]
        self.case = recovery.coverage.cases_for("training")[1]

    def test_diagnostic_is_instrumentation_only_not_execution(self):
        original = runner.supervised_runner(BOOKING)
        text = recovery.diagnostic_runner()
        ast.parse(text)  # Deliberately do NOT exec a runner locally.
        self.assertNotEqual(original, text)
        self.assertIn('diagnostic("reaped")', text)
        self.assertIn('resource.RUSAGE_SELF', text)
        self.assertEqual(original, runner.supervised_runner(BOOKING))

    def test_diagnostic_identity_is_fixed(self):
        with self.assertRaisesRegex(ValueError, "candidate changed"):
            recovery.diagnostic_specs(self.sample)

    def test_unknown_is_not_zero_and_does_not_shrink_denominator(self):
        bounds = recovery._bounds([True, False, None])
        self.assertEqual(bounds["passed_bounds"], [1, 2])
        self.assertEqual(bounds["total"], 3)
        self.assertEqual(bounds["reward_bounds"], [1/3, 2/3])
        self.assertEqual(bounds["full_pass_bounds"], [0, 0])
        self.assertEqual(recovery._bounds([True, None])["full_pass_bounds"], [0, 1])
        with self.assertRaises(ValueError):
            recovery._bounds([1, 0])

    def test_valid_completed_and_unattributed_child_evidence(self):
        normal = pack_result(result_for(self.sample["source"], self.case, "ok"))
        self.assertTrue(recovery.assess_record(self.sample, self.case, normal, "im-test")["passed"])
        uncertain = pack_result(result_for(self.sample["source"], self.case, "signal", returncode=-9))
        outcome = recovery.assess_record(self.sample, self.case, uncertain, "im-test")
        self.assertIsNone(outcome["passed"])
        self.assertIsNone(outcome["inclusive_match"])
        tampered = deepcopy(uncertain)
        tampered["metadata"]["input_hash"] = "other"
        with self.assertRaisesRegex(ValueError, "identity"):
            recovery.assess_record(self.sample, self.case, tampered, "im-test")

    def test_missing_result_stays_unknown_not_new_work(self):
        entry = {"intent-1": intent_for(self.sample, self.case)}
        outcome, record = recovery.validate_entry(entry, self.sample, self.case, "im-test")
        self.assertIsNone(record)
        self.assertEqual(outcome["reason"], "unresolved_submitted_intent")
        source = source_for([self.sample], self.plan)
        source["entries"][self.sample["sample_id"]][self.case.input_hash] = entry
        state = recovery.inventory(source)
        self.assertNotIn([self.sample["sample_id"], self.case.input_hash], state["pending"])
        self.assertEqual(state["counts"]["unknown"], 1)

    def test_second_attempt_cannot_replace_a_wrong_answer_or_unknown_signal(self):
        for updates in ({}, {"returncode": -9}):
            first = pack_result(result_for(self.sample["source"], self.case, "first", **updates))
            entry = {"intent-1": intent_for(self.sample, self.case), "attempt-1": first,
                     "intent-2": intent_for(self.sample, self.case, 2)}
            with self.assertRaisesRegex(ValueError, "not justified"):
                recovery.validate_entry(entry, self.sample, self.case, "im-test")

    def test_selected_record_tampering_rejected(self):
        first = pack_result(result_for(self.sample["source"], self.case, "first"))
        entry = {"intent-1": intent_for(self.sample, self.case), "attempt-1": first, "selected": {}}
        with self.assertRaisesRegex(ValueError, "selected"):
            recovery.validate_entry(entry, self.sample, self.case, "im-test")

    def test_committed_batch_provenance_is_explicit_not_a_synthetic_intent(self):
        record = pack_result(result_for(self.sample["source"], self.case, "committed"))
        entry = {"committed_batch": {"key": "screen-00", "report_hash": "a"*64}, "selected": record}
        outcome, selected = recovery.validate_entry(entry, self.sample, self.case, "im-test")
        self.assertTrue(outcome["passed"])
        self.assertEqual(selected, record)
        for key in ("screen-01", "controls"):
            wrong = deepcopy(entry)
            wrong["committed_batch"]["key"] = key
            with self.assertRaisesRegex(ValueError, "provenance"):
                recovery.validate_entry(wrong, self.sample, self.case, "im-test")

    def test_plan_json_roundtrip_and_finite_budget(self):
        plan = recovery.plan_for(source_for([self.sample], self.plan))
        self.assertEqual(plan, json.loads(json.dumps(plan)))
        self.assertEqual(plan["new_generations"], 0)
        self.assertFalse(plan["diagnostic_results_supply_scores"])
        rates = {"cpu_hour_cost": ".0473", "mem_gib_hour_cost": ".008",
                 "cpu_hour_cost_sandbox": ".1419", "mem_gib_hour_cost_sandbox": ".024"}
        quote = recovery.budget_quote(plan, rates, {"metered_cost": "9"}, "100", "recovery")
        self.assertEqual(quote["previous_holds_usd"], "100")
        self.assertLess(float(quote["additional_reservation_usd"]), 3)
        self.assertEqual(quote["gpu_calls"], 0)

    def test_summary_preserves_all64_and_blocks_complete_evidence_gate(self):
        samples = [s for index in range(16) for s in samples_for(self.plan, index*4)]
        source = source_for(samples, self.plan)
        records = reports_for(samples, self.plan)
        assessments = {s["sample_id"]: {c.input_hash: recovery.assess_record(s, c, records[s["sample_id"]][c.input_hash], "im-test")
                       for c in recovery.coverage.cases_for("training")} for s in samples}
        assessments[samples[0]["sample_id"]][self.case.input_hash] = recovery.unknown("original_137")
        del records[samples[0]["sample_id"]][self.case.input_hash]
        result = recovery.summarize(source, assessments, records)
        self.assertEqual(result["summary"]["programs"], 64)
        self.assertEqual(result["summary"]["reference_full_pass_bounds"], [63, 64])
        self.assertEqual(result["summary"]["fully_resolved_programs"], 63)
        self.assertEqual(result["summary"]["resolved_groups"], 15)
        self.assertEqual(result["status"], "completed_with_uncertainty")
        self.assertFalse(result["decision"]["ready_for_matched_protocol_review"])
        self.assertFalse(result["decision"]["automatic_training"])


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.parent = screen.make_plan(ROOT)
        self.sample = samples_for(self.parent)[0]
        self.cases = recovery.coverage.cases_for("training")[:3]
        self.source = source_for([self.sample], self.parent)
        self.plan = {"pending": [[self.sample["sample_id"], c.input_hash] for c in self.cases], "concurrency": 1}
        self.store, self.commit = Store(), AsyncMock()

    async def run_worker(self, directory, backend, *, deadline=None):
        with patch.object(recovery, "plan_for", return_value=self.plan):
            return await recovery.recover_inputs(self.source, self.plan, directory, deadline or time.time()+30,
                                                  backend, self.store, self.commit)

    async def test_one_unknown_does_not_abort_other_inputs_and_is_not_retried(self):
        values = [result_for(self.sample["source"], c, str(i), **({"returncode": -9} if i == 1 else {}))
                  for i, c in enumerate(self.cases)]
        backend = Mock(execute=AsyncMock(side_effect=values))
        with tempfile.TemporaryDirectory() as tmp:
            progress = await self.run_worker(Path(tmp), backend)
            self.assertEqual(progress["completed_inputs"], 3)
            self.assertEqual(progress["unknown_inputs"], 1)
            self.assertEqual(backend.execute.await_count, 3)
            self.assertIsNone(progress["stop_reason"])
            backend.execute.reset_mock()
            again = await self.run_worker(Path(tmp), backend)
            self.assertEqual(again["new_starts"], 0)
            backend.execute.assert_not_awaited()

    async def test_provider_only_intent_is_recovered_but_not_resubmitted(self):
        case = self.cases[0]
        key = f"{recovery.RUN_ID}/execution/{self.sample['sample_id']}/{case.input_hash}/1/intent"
        self.store.values[key] = intent_for(self.sample, case)
        backend = Mock(execute=AsyncMock(side_effect=[result_for(self.sample["source"], c, str(i)) for i, c in enumerate(self.cases[1:])]))
        with tempfile.TemporaryDirectory() as tmp:
            progress = await self.run_worker(Path(tmp), backend)
            self.assertEqual(progress["new_starts"], 2)
            self.assertEqual(progress["unknown_inputs"], 1)

    async def test_provider_result_without_volume_commit_is_reused(self):
        case = self.cases[0]
        key = f"{recovery.RUN_ID}/execution/{self.sample['sample_id']}/{case.input_hash}/1"
        self.store.values[key + "/intent"] = intent_for(self.sample, case)
        self.store.values[key + "/result"] = pack_result(result_for(self.sample["source"], case, "old"))
        backend = Mock(execute=AsyncMock(side_effect=[result_for(self.sample["source"], c, str(i)) for i, c in enumerate(self.cases[1:])]))
        with tempfile.TemporaryDirectory() as tmp:
            progress = await self.run_worker(Path(tmp), backend)
            self.assertEqual(progress["new_starts"], 2)
            self.assertEqual(progress["unknown_inputs"], 0)

    async def test_deadline_prevents_new_submissions(self):
        backend = Mock(execute=AsyncMock())
        with tempfile.TemporaryDirectory() as tmp:
            progress = await self.run_worker(Path(tmp), backend, deadline=time.time()-1)
            backend.execute.assert_not_awaited()
            self.assertEqual(progress["stop_reason"], "deadline")

    async def test_many_unknowns_trip_finite_circuit(self):
        backend = Mock(execute=AsyncMock(side_effect=[result_for(self.sample["source"], c, str(i), returncode=-9)
                                                     for i, c in enumerate(self.cases)]))
        with tempfile.TemporaryDirectory() as tmp, patch.object(recovery, "UNKNOWN_CIRCUIT", 2):
            progress = await self.run_worker(Path(tmp), backend)
            self.assertEqual(progress["new_starts"], 2)
            self.assertEqual(progress["stop_reason"], "unknown_circuit")

    async def test_invalid_evidence_fails_closed_instead_of_becoming_unknown(self):
        first = result_for(self.sample["source"], self.cases[0], "first")
        backend = Mock(execute=AsyncMock(return_value=replace(first, metadata=dict(first.metadata, source_hash="other"))))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "identity"):
                await self.run_worker(Path(tmp), backend)
            self.assertEqual(backend.execute.await_count, 1)

    async def test_pre_candidate_failure_gets_only_one_fresh_replacement(self):
        self.plan["pending"] = self.plan["pending"][:1]
        case = self.cases[0]
        first, _, _ = await simulated_result(source=self.sample["source"], case=case, preflight=process(b"", code=-1))
        second = result_for(self.sample["source"], case, "replacement")
        backend = Mock(execute=AsyncMock(side_effect=[first, second]))
        with tempfile.TemporaryDirectory() as tmp, patch.object(recovery.asyncio, "sleep", AsyncMock()) as backoff:
            progress = await self.run_worker(Path(tmp), backend)
            self.assertEqual(progress["new_starts"], 2)
            self.assertEqual(progress["unknown_inputs"], 0)
            target = Path(tmp) / "inputs" / self.sample["sample_id"] / case.input_hash
            entry = {p.stem: json.loads(p.read_text()) for p in target.glob("*.json")}
            outcome, selected = recovery.validate_entry(entry, self.sample, case, "im-test")
            self.assertTrue(outcome["passed"])
            self.assertEqual(selected, pack_result(second))
            backoff.assert_awaited_once_with(1)

    async def test_exhausted_retry_slots_preserve_unknown_and_do_not_repeat(self):
        self.plan["pending"] = self.plan["pending"][:1]
        first, _, _ = await simulated_result(source=self.sample["source"], case=self.cases[0], preflight=process(b"", code=-1))
        for slot in range(recovery.MAX_STARTUP_RETRIES):
            self.store.values[f"{recovery.RUN_ID}/startup-slot/{slot}"] = {"used": True}
        backend = Mock(execute=AsyncMock(return_value=first))
        with tempfile.TemporaryDirectory() as tmp:
            progress = await self.run_worker(Path(tmp), backend)
            self.assertEqual(progress["new_starts"], 1)
            self.assertEqual(progress["unknown_inputs"], 1)

    async def test_transport137_is_unknown_and_unconfirmed_cleanup_stops(self):
        lost, _, _ = await simulated_result(source=self.sample["source"], case=self.cases[0], parent_code=137)
        lost = replace(lost, metadata=dict(lost.metadata, cleanup="unconfirmed"))
        backend = Mock(execute=AsyncMock(return_value=lost))
        with tempfile.TemporaryDirectory() as tmp:
            progress = await self.run_worker(Path(tmp), backend)
            self.assertEqual(progress["new_starts"], 1)
            self.assertEqual(progress["stop_reason"], "cleanup_unconfirmed")


if __name__ == "__main__":
    unittest.main()
