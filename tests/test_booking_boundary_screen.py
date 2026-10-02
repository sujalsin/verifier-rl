import asyncio
import base64
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_modal_backend import process
from tests.test_supervised_execution import CASE, SOURCE, simulated_result, envelope
from tests.test_panel_screen import execution
from tests.test_booking_warmstart import result_for
from verifier_rl import booking_boundary_screen as screen, booking_boundary_contrast as contrast
from verifier_rl import booking_verifier_v2 as coverage, supervised_execution as runner
from verifier_rl import booking_study as original, booking_reward_pilot as pilot
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.panel_execution import pack_result, request_for
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import BOOKING

ROOT = Path(__file__).resolve().parents[1]


def samples_for(plan, offset=0):
    return [original.sample_from_text(original.controls()["correct"], sid=sid, plan=plan,
            seed=seed, tokens=64, eos=True) for sid, seed in screen.identities()[offset:offset+4]]


def reports_for(samples, plan, *, inclusive=False):
    records = {}
    for sample in samples:
        current = {}
        for c in coverage.cases_for("training"):
            answer = contrast.inclusive_answer(c.arguments_json) if inclusive else c.expected
            base = execution(sample["source"], c, sample["sample_id"] + c.input_hash)
            value = envelope(sample["source"], c,
                stdout_base64=base64.b64encode((str(answer) + "\n").encode()).decode())
            payload = (canonical_json(value) + "\n").encode()
            value = runner.decode_transport(replace(base, stdout=payload, metadata=dict(base.metadata,
                runner_hash=digest(runner.supervised_runner(BOOKING)), stdout_bytes=len(payload),
                stdout_sha256=digest(payload.decode()))), BOOKING, sample["source"], c.arguments)
            current[c.input_hash] = pack_result(value)
        records[sample["sample_id"]] = current
    return records


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.plan = screen.make_plan(ROOT)

    def test_fixed_plan_preserves_model_suite_and_no_training(self):
        screen.validate_plan(self.plan)
        self.assertEqual(self.plan["initial_parameter_hash"], pilot.WARM_HASH)
        self.assertEqual(self.plan["seeds"], list(range(17000, 17064)))
        self.assertEqual(self.plan["max_sandbox_executions"], 6460)
        self.assertEqual(self.plan["scoring_hash"], screen.SCORING_HASH)
        self.assertEqual(self.plan["optimizer_updates"], 0)
        for changes in ({"sample_count": 128}, {"initial_parameter_hash": original.PARAMETER_HASH},
                        {"automatic_training": True}, {"audit_cases": 192}, {"max_startup_retries": 100},
                        {"gates": {}}, {"startup_retry_version": None}, {"optimizer_updates": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                screen.validate_plan(dict(self.plan, **changes))

    def test_schedule_rejects_old_seeds_extra_groups_or_changed_order(self):
        samples = samples_for(self.plan)
        screen.validate_batch("screen-00", samples, self.plan)
        for key, changed in (("screen-16", samples), ("screen-00", samples[::-1]),
                             ("screen-00", samples[:3]), ("controls", samples)):
            with self.assertRaises(ValueError):
                screen.validate_batch(key, changed, self.plan)
        samples[0]["seed"] = 16000
        with self.assertRaises(ValueError):
            screen.validate_batch("screen-00", samples, self.plan)

    def test_grade_requires_all96_records_including_omitted(self):
        sample = samples_for(self.plan)[0]
        records = reports_for([sample], self.plan, inclusive=True)[sample["sample_id"]]
        result = screen.grade(sample, records, "im-test", self.plan)
        self.assertEqual(result["row"]["reference"]["passed"], 64)
        self.assertEqual(result["row"]["endpoint_omission"]["passed"], 57)
        self.assertTrue(result["row"]["inclusive_output_signature"])
        omitted = contrast.partition(coverage.cases_for("training"))[1][0]
        missing = dict(records)
        missing.pop(omitted.input_hash)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            screen.grade(sample, missing, "im-test", self.plan)
        unknown = deepcopy(records)
        unknown[omitted.input_hash] = pack_result(result_for(sample["source"], omitted, "unknown", returncode=-9))
        with self.assertRaisesRegex(ValueError, "unresolved"):
            screen.grade(sample, unknown, "im-test", self.plan)

    def test_extraction_failures_remain_zero_not_missing_or_regenerated(self):
        sample = original.sample_from_text("", sid="screen-17000", plan=self.plan,
                                          seed=17000, tokens=10, eos=True)
        self.assertTrue(sample["extraction_status"].startswith("rejected_"))
        result = screen.grade(sample, {}, "im-test", self.plan)
        self.assertEqual(result["row"]["reference"]["reward"], 0)
        self.assertEqual(result["row"]["endpoint_omission"]["reward"], 0)
        self.assertTrue(result["row"]["extraction_rejected"])

    def test_replay_rejects_score_tampering_or_sandbox_reuse(self):
        samples = samples_for(self.plan)
        records = reports_for(samples, self.plan)
        raw = {"samples": samples, "records": records, "retries": [], "key": "screen-00",
               "role": "training", "plan_hash": digest(canonical_json(self.plan)),
               "graded": [screen.grade(s, records[s["sample_id"]], "im-test", self.plan) for s in samples]}
        _, ids, _ = screen.verify_batch(raw, samples, "screen-00", self.plan, "im-test")
        self.assertEqual(len(ids), 384)
        changed = deepcopy(raw)
        changed["graded"][0]["row"]["reference"]["reward"] = .5
        with self.assertRaisesRegex(ValueError, "scores differ"):
            screen.verify_batch(changed, samples, "screen-00", self.plan, "im-test")
        case = coverage.cases_for("training")[0]
        raw["records"][samples[1]["sample_id"]][case.input_hash] = raw["records"][samples[0]["sample_id"]][case.input_hash]
        raw["graded"] = [screen.grade(s, records[s["sample_id"]], "im-test", self.plan) for s in samples]
        with self.assertRaisesRegex(ValueError, "reused sandbox"):
            screen.verify_batch(raw, samples, "screen-00", self.plan, "im-test")

    def test_feasibility_gate_is_not_auto_training_or_proof(self):
        summary = {"programs": 64, "groups": 16, "reference_full_passes": 8,
                   "distinct_inclusive_signature_sources": 2, "mixed_reference_groups": 5,
                   "mixed_omission_groups": 4, "changed_relative_signal_groups": 2}
        result = screen.decision(summary)
        self.assertTrue(result["ready_for_matched_protocol_review"])
        self.assertFalse(result["automatic_training"])
        self.assertFalse(result["amplification_established"])
        for field, value in (("reference_full_passes", 0), ("reference_full_passes", 63),
                             ("distinct_inclusive_signature_sources", 1), ("mixed_reference_groups", 3),
                             ("mixed_omission_groups", 3), ("changed_relative_signal_groups", 1)):
            self.assertFalse(screen.decision(dict(summary, **{field: value}))["ready_for_matched_protocol_review"])
        with self.assertRaisesRegex(ValueError, "complete fixed-size"):
            screen.decision(dict(summary, programs=63))

    def test_budget_is_finite_retains_prior_and_not_invoice(self):
        rates = {"cpu_hour_cost": ".0473", "mem_gib_hour_cost": ".008",
                 "cpu_hour_cost_sandbox": ".1419", "mem_gib_hour_cost_sandbox": ".024",
                 "gpu_hour_cost_l40s": "1.95"}
        budget = screen.budget_quote(self.plan, rates, {"metered_cost": "7.66"}, "100")
        self.assertEqual(budget["prior_reservation_held_usd"], "100")
        self.assertFalse(budget["is_invoice"])
        self.assertFalse(budget["provider_hard_cap"])
        self.assertLess(float(budget["additional_reservation_usd"]), 40)
        with self.assertRaises(ValueError):
            screen.budget_quote(self.plan, dict(rates, cpu_hour_cost="nan"), {"metered_cost": "0"}, "0")

    def test_control_failure_prevents_gpu_generation(self):
        import modal_booking_boundary_screen as launcher
        snapshot = {k: "" for k in ("modal_booking_boundary_screen.py", "verifier_rl/booking_boundary_screen.py",
                                    "verifier_rl/supervised_execution.py", "verifier_rl/booking_boundary_contrast.py")}
        with (patch.object(launcher, "artifacts"), patch.object(launcher, "claims", Mock()),
              patch.object(launcher, "persist"), patch.object(launcher, "check_snapshot"),
              patch.object(launcher.modal.App, "lookup", return_value=Mock(app_id="ap-test")),
              patch.object(launcher.modal.Sandbox, "list", return_value=[]),
              patch.object(launcher, "live_supervisor_controls", AsyncMock(side_effect=ValueError("control failed"))),
              patch.object(launcher.generate, "remote") as generate,
              patch.object(launcher.grade_batch, "remote") as grade):
            with self.assertRaisesRegex(ValueError, "control failed"):
                launcher.run_screen.local(self.plan, {"app_name": "test"}, {}, snapshot, time.time()+60)
            generate.assert_not_called()
            grade.assert_not_called()

    def test_controller_generates_once_and_uses_exactly_sixteen_fixed_batches(self):
        import modal_booking_boundary_screen as launcher
        samples = [s for index in range(16) for s in samples_for(self.plan, index*4)]
        observed = []
        def grade(key, batch, plan, setup, deadline):
            screen.validate_batch(key, batch, plan)
            observed.append(key)
            return {"graded": []}
        snapshot = {k: "" for k in ("modal_booking_boundary_screen.py", "verifier_rl/booking_boundary_screen.py",
                                    "verifier_rl/supervised_execution.py", "verifier_rl/booking_boundary_contrast.py")}
        with (patch.object(launcher, "artifacts"), patch.object(launcher, "claims", Mock()),
              patch.object(launcher, "persist"), patch.object(launcher, "check_snapshot"), patch("builtins.print"),
              patch.object(launcher.modal.App, "lookup", return_value=Mock(app_id="ap-test")),
              patch.object(launcher.modal.Sandbox, "list", return_value=[]),
              patch.object(launcher, "live_supervisor_controls", AsyncMock()),
              patch.object(screen, "validate_controls", return_value={"passed": True}),
              patch.object(screen, "verify_run", return_value={"summary": {}, "decision": {"automatic_training": False}}),
              patch.object(launcher.generate, "remote", return_value={"samples": samples}) as generate,
              patch.object(launcher.grade_batch, "remote", side_effect=grade)):
            result = launcher.run_screen.local(self.plan, {"app_name": "test"}, {}, snapshot, time.time()+60)
        generate.assert_called_once()
        self.assertEqual(observed, ["controls"] + [f"screen-{i:02d}" for i in range(16)])
        self.assertFalse(result["decision"]["automatic_training"])

    def test_duplicate_controller_claim_blocks_generation_and_grading(self):
        import modal_booking_boundary_screen as launcher
        snapshot = {k: "" for k in ("modal_booking_boundary_screen.py", "verifier_rl/booking_boundary_screen.py",
                                    "verifier_rl/supervised_execution.py", "verifier_rl/booking_boundary_contrast.py")}
        with (patch.object(launcher, "check_snapshot"), patch.object(launcher.claims, "put", return_value=False),
              patch.object(launcher.generate, "remote") as generate, patch.object(launcher.grade_batch, "remote") as grade):
            with self.assertRaisesRegex(ReconciliationRequired, "already claimed"):
                launcher.run_screen.local(self.plan, {}, {}, snapshot, time.time()+60)
        generate.assert_not_called()
        grade.assert_not_called()


class StartupPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_observed_minus_one_preflight_timeout_is_opt_in_only(self):
        first, _, sandbox = await simulated_result(preflight=process(b"", code=-1))
        self.assertEqual(first.detail, "supervisor_transport:preflight:TimeoutError")
        self.assertEqual(sandbox.exec.aio.await_count, 1)  # NO candidate submission
        self.assertEqual(first.metadata["cleanup"], "terminated")
        request = request_for(SOURCE, CASE)
        self.assertFalse(runner.startup_retry_allowed(first, request, "im-test", BOOKING))
        self.assertTrue(runner.startup_retry_allowed(first, request, "im-test", BOOKING,
                                                    policy_version=runner.STARTUP_RETRY_VERSION))
        waiting = dict(first.metadata)
        waiting.pop("preflight_returncode")
        self.assertTrue(runner.startup_retry_allowed(replace(first, metadata=waiting), request, "im-test", BOOKING,
                                                    policy_version=runner.STARTUP_RETRY_VERSION))

    async def test_candidate_evidence_binding_and_cleanup_cannot_be_bypassed(self):
        first, _, _ = await simulated_result(preflight=process(b"", code=-1))
        for key, value in (("cleanup", "unconfirmed"), ("source_hash", "wrong"), ("input_hash", "wrong"),
                           ("image_id", "im-wrong"), ("runner_hash", "wrong"), ("execution_version", "old"),
                           ("preflight_returncode", 0), ("preflight_returncode", True), ("preflight_stage", "complete"),
                           ("returncode", -1), ("runner_stage", "function_call"), ("startup_seconds", 1),
                           ("supervisor_report", {}), ("supervisor_returncode", 0), ("preflight_stderr_preview", "failure")):
            with self.subTest(key=key, value=value):
                altered = replace(first, metadata=dict(first.metadata, **{key: value}))
                self.assertFalse(runner.startup_retry_allowed(altered, request_for(SOURCE, CASE), "im-test", BOOKING,
                                                              policy_version=runner.STARTUP_RETRY_VERSION))
        for code in (137, -1, 1):
            candidate, _, _ = await simulated_result(parent_code=code)
            self.assertFalse(runner.startup_retry_allowed(candidate, request_for(SOURCE, CASE), "im-test", BOOKING,
                                                          policy_version=runner.STARTUP_RETRY_VERSION))

    async def test_real_backend_classification_reaches_worker_retry_and_journal(self):
        import modal_booking_reward_pilot as launcher
        plan = screen.make_plan(ROOT)
        sample = original.sample_from_text(SOURCE, sid="screen-17000", plan=plan, seed=17000, tokens=10, eos=True)
        first, _, sandbox = await simulated_result(preflight=process(b"", code=-1))
        second = result_for(SOURCE, CASE, "replacement")
        backend = Mock(execute=AsyncMock(side_effect=[first, second]))
        store = Mock()
        store.put.aio = AsyncMock(return_value=True)
        with (tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "cases_for", return_value=(CASE,)),
              patch.object(launcher.asyncio, "sleep", AsyncMock()) as backoff):
            records, retries = await launcher.execute_group([sample], "training", {"sandbox_image_id": "im-test"},
                Path(tmp), time.time()+60, backend, store, plan, retry_policy=runner.STARTUP_RETRY_VERSION)
            target = Path(tmp) / "inputs" / sample["sample_id"] / CASE.input_hash
            self.assertEqual(json.loads((target / "attempt-1.json").read_text()), pack_result(first))
            self.assertEqual(json.loads((target / "selected.json").read_text()), pack_result(second))
            self.assertEqual(backend.execute.await_count, 2)
            backoff.assert_awaited_once_with(1)
        self.assertEqual(sandbox.exec.aio.await_count, 1)
        self.assertEqual(len(retries), 1)
        ids, slots = screen.verify_retries({"samples": [sample], "records": records, "retries": retries}, "im-test")
        self.assertEqual((ids, slots), (["sb-test"], [0]))

    async def test_no_retry_when_slots_exhausted_or_second_attempt_fails(self):
        import modal_booking_reward_pilot as launcher
        plan = screen.make_plan(ROOT)
        sample = original.sample_from_text(SOURCE, sid="screen-17000", plan=plan, seed=17000, tokens=10, eos=True)
        first, _, _ = await simulated_result(preflight=process(b"", code=-1))
        for exhausted in (True, False):
            backend = Mock(execute=AsyncMock(return_value=first))
            store = Mock()
            async def put(key, value, **kwargs):
                return not (exhausted and "/startup-slot/" in key)
            store.put.aio = AsyncMock(side_effect=put)
            with (tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "cases_for", return_value=(CASE,)),
                  patch.object(launcher.asyncio, "sleep", AsyncMock())):
                with self.assertRaisesRegex(ReconciliationRequired, "unresolved supervised"):
                    await launcher.execute_group([sample], "training", {"sandbox_image_id": "im-test"}, Path(tmp),
                        time.time()+60, backend, store, plan, retry_policy=runner.STARTUP_RETRY_VERSION)
                self.assertTrue((Path(tmp) / "inputs" / sample["sample_id"] / CASE.input_hash / "selected.json").exists())
            self.assertEqual(backend.execute.await_count, 1 if exhausted else 2)


if __name__ == "__main__":
    unittest.main()
