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

from tests.test_panel_screen import execution
from tests.test_supervised_execution import envelope
from tests.test_booking_reward_pilot import sample, report
from verifier_rl import booking_reward_pilot as pilot, supervised_execution as runner
from verifier_rl import supervisor_controls as controls, booking_verifier_v2 as verifier
from verifier_rl.grading import Status
from verifier_rl.panel_execution import pack_result
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import BOOKING

ROOT = Path(__file__).resolve().parents[1]


def result_for(source, case, identity, **updates):
    original = execution(source, case, identity)
    value = envelope(source, case, stdout_base64=base64.b64encode(original.stdout).decode(), **updates)
    payload = (canonical_json(value) + "\n").encode()
    metadata = dict(original.metadata, runner_hash=digest(runner.supervised_runner(BOOKING)),
                    stdout_bytes=len(payload), stdout_sha256=digest(payload.decode()))
    return runner.decode_transport(replace(original, stdout=payload, metadata=metadata), BOOKING, source, case.arguments)


class WarmPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = pilot.make_plan(ROOT, warm_start=True)

    def test_same_step12_checkpoint_and_new_baseline_seeds(self):
        pilot.validate_plan(self.plan)
        self.assertEqual(self.plan["initial_parameter_hash"], pilot.WARM_HASH)
        self.assertEqual(self.plan["initial_checkpoint"], pilot.WARM_CHECKPOINT)
        self.assertEqual(self.plan["parent_step"], 12)
        self.assertEqual(self.plan["steps"], 24)
        self.assertEqual(self.plan["max_sandbox_executions"], 32620)
        self.assertEqual(self.plan["execution_version"], runner.VERSION)
        self.assertFalse(set(self.plan["evaluation_seeds"]) & set(pilot.SEEDS))
        for changes in ({"initial_parameter_hash": pilot.old.PARAMETER_HASH}, {"parent_step": 14},
                        {"execution_version": None}, {"evaluation_seeds": list(pilot.SEEDS)}):
            with self.assertRaises(ValueError):
                pilot.validate_plan(dict(self.plan, **changes))

    def test_grpo_and_rewards_unchanged_only_checkpoint_persistence_changes(self):
        original = pilot.make_plan(ROOT)
        changed = {key for key in original["trainer"] | self.plan["trainer"]
                   if original["trainer"].get(key) != self.plan["trainer"].get(key)}
        self.assertEqual(changed, {"save_strategy", "save_steps", "save_only_model"})
        for field in ("suite_hashes", "verifier_version", "log_strength", "steps", "group_size", "training_seed",
                      "max_completion_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "prompt"):
            self.assertEqual(self.plan[field], original[field])
        self.assertEqual(self.plan["trainer"]["save_strategy"], "steps")
        self.assertEqual(self.plan["trainer"]["save_steps"], 12)
        self.assertFalse(self.plan["trainer"]["save_only_model"])
        self.assertEqual(original["trainer"]["save_strategy"], "no")

    def test_new_schedule_summary_and_evidence_use_warm_initial_policy(self):
        group = [sample(self.plan, sid, int(sid.rsplit("-", 1)[1]))
                 for sid in pilot.batch_ids("eval-baseline-00", "evaluation", self.plan)]
        pilot.validate_batch("eval-baseline-00", "evaluation", group, self.plan)
        with self.assertRaises(ValueError):
            pilot.validate_batch("eval-baseline-00", "evaluation", group, pilot.make_plan(ROOT))
        population = [sample(self.plan, f"eval-baseline-{seed}", seed) for seed in pilot.WARM_SEEDS]
        summary = pilot.policy_summary(population, [report(s, "evaluation") for s in population], self.plan)
        self.assertEqual(summary["programs"], 16)
        metrics = {"global_step": 24, "rewards": [[0, .25, .5, 1]] * 24,
            "log_history": [{"grad_norm": 1.0}] * 24, "before_parameter_hash": pilot.WARM_HASH,
            "after_parameter_hash": "a" * 64, "finite_parameters": True, "checkpoint_reload_verified": True}
        self.assertTrue(pilot.training_evidence(metrics, self.plan)["parameters_changed"])
        with self.assertRaises(ValueError):
            pilot.training_evidence(metrics)

    def test_confirmed_timeout_counts_failed_case_and_group_still_has_signal(self):
        s = sample(self.plan, "train-linear-00-0")
        rows = {c.input_hash: pack_result(result_for(s["source"], c, c.input_hash,
                        **({"returncode": -24} if i == 0 else {})))
                for i, c in enumerate(verifier.cases_for("training"))}
        graded = pilot.grade(s, rows, "training", "im-test", self.plan)
        self.assertEqual(graded["reports"]["training"]["passed"], 95)
        self.assertEqual(pilot.reward(graded, "linear"), 95/96)
        self.assertEqual(pilot.reward(graded, "logarithmic"), verifier.shape_reward(95/96, "logarithmic"))
        with self.assertRaises(ValueError):
            pilot.grade(s, rows, "training", "im-test", pilot.make_plan(ROOT))
        first = verifier.cases_for("training")[0]
        rows[first.input_hash] = pack_result(result_for(s["source"], first, "ambiguous", returncode=-9))
        with self.assertRaisesRegex(ValueError, "unresolved supervised"):
            pilot.grade(s, rows, "training", "im-test", self.plan)

    def test_live_control_replay_detects_false_passes_and_reused_sandboxes(self):
        source = "def required_capacity(bookings):\n    while True: pass\n"
        with patch.object(controls, "FAILED_SOURCE_HASH", digest(source)):
            document = {"saved_source": source, "records": {}}
            case = controls.case_for_control()
            for name, (program, status, matches, detail) in controls.controls(source).items():
                update = {}
                if status == Status.TIMEOUT:
                    update = {"returncode": -24} if "cpu" in detail else {"returncode": -9, "enforced_limit": "wall", "wall_seconds": 5.01}
                elif status == Status.CANDIDATE_ERROR:
                    update = {"returncode": 137 if name == "exit137" else 1}
                elif status == Status.OUTPUT_LIMIT:
                    update = {"stdout_truncated": True, "enforced_limit": "output"}
                elif status == Status.INFRASTRUCTURE_ERROR:
                    update = {"returncode": -9}
                outcome = result_for(program, case, name, **update)
                if status == Status.COMPLETED and not matches:
                    value = envelope(program, case, stdout_base64=base64.b64encode(b"0").decode())
                    payload = (canonical_json(value)+"\n").encode()
                    original = execution(program, case, name)
                    outcome = runner.decode_transport(replace(original, stdout=payload, metadata=dict(original.metadata,
                        runner_hash=digest(runner.supervised_runner(BOOKING)), stdout_sha256=digest(payload.decode()))),
                        BOOKING, program, case.arguments)
                document["records"][name] = pack_result(outcome)
            self.assertEqual(len(controls.validate_controls(document, "im-test")), 12)
            changed = deepcopy(document)
            changed["records"]["clean_after_failures"] = changed["records"]["correct"]
            with self.assertRaisesRegex(ValueError, "reused"):
                controls.validate_controls(changed, "im-test")

    def test_full_checkpoint_inventory_rejects_model_only_or_wrong_step(self):
        from modal_booking_reward_pilot import checkpoint_inventory
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in ("config.json", "tokenizer.json", "model.safetensors", "trainer_state.json"):
                (path / name).write_text('{"global_step":12}')
            with self.assertRaisesRegex(ValueError, "incomplete"):
                checkpoint_inventory(path, 12)
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (path / name).write_bytes(b"test fixture, not a real checkpoint")
            self.assertIn("optimizer.pt", checkpoint_inventory(path, 12))
            with self.assertRaisesRegex(ValueError, "step differs"):
                checkpoint_inventory(path, 24)

    def test_supervisor_gate_fails_before_any_grading_or_training(self):
        import modal_booking_reward_pilot as launcher
        with (patch.object(launcher, "artifacts"), patch.object(launcher, "claims", Mock()),
              patch.object(launcher, "persist"), patch.object(launcher, "check_snapshot"),
              patch.object(launcher.modal.App, "lookup", return_value=Mock(app_id="ap-test")),
              patch.object(launcher.modal.Sandbox, "list", return_value=[]),
              patch.object(launcher, "live_supervisor_controls", AsyncMock(side_effect=ValueError("runner gate failed"))),
              patch.object(launcher.train_arm, "remote") as train,
              patch.object(launcher.grade_batch, "remote") as grade):
            with self.assertRaisesRegex(ValueError, "runner gate failed"):
                launcher.run_pilot.local(self.plan, {"app_name": "test"}, [], {}, {}, time.time()+60)
            train.assert_not_called()
            grade.assert_not_called()


class WarmWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_cpu_timeout_does_not_stop_remaining_candidates(self):
        import modal_booking_reward_pilot as launcher
        plan = pilot.make_plan(ROOT, warm_start=True)
        case = verifier.cases_for("training")[0]
        samples = [sample(plan, f"train-linear-00-{i}") for i in range(4)]
        counter = 0
        async def execute(request):
            nonlocal counter
            index, counter = counter, counter + 1
            return result_for(request.source, case, str(index), **({"returncode": -24} if index == 0 else {}))
        backend = Mock(execute=AsyncMock(side_effect=execute))
        store = Mock()
        store.put.aio = AsyncMock(return_value=True)
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "cases_for", return_value=(case,)):
            rows, retries = await launcher.execute_group(samples, "training", {"sandbox_image_id": "im-test"},
                Path(tmp), time.time()+60, backend, store, plan)
        self.assertEqual(counter, 4)
        self.assertEqual(retries, [])
        self.assertEqual(rows[samples[0]["sample_id"]][case.input_hash]["status"], Status.TIMEOUT.value)
        self.assertTrue(all(call.args[0].startswith(pilot.WARM_RUN_ID + "/") for call in store.put.aio.call_args_list))


if __name__ == "__main__":
    unittest.main()
