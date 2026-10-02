import ast
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from verifier_rl import grpo_recovery as recovery, booking_matched_training as release
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.suites import canonical_json


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.binding = {"step": 1, "binding_hash": "fixture"}
        self.tokens = ([[1]]*4, [[2, 3]]*4, None, {})
        self.produce, self.commit, self.restore = Mock(return_value=self.tokens), Mock(), Mock()

    def generate(self, path, binding=None):
        return recovery.generation_once(path, binding or self.binding, self.produce,
            commit=self.commit, get_rng=lambda: {"rng": "fixture"}, set_rng=self.restore)

    def test_saved_group_reused_with_rng_and_without_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            value, reused = self.generate(tmp)
            self.assertFalse(reused)
            again, reused = self.generate(tmp)
            self.assertTrue(reused)
            self.assertEqual(value, again)
            self.produce.assert_called_once()
            self.restore.assert_called_once_with({"rng": "fixture"})
            self.assertEqual(self.commit.call_count, 2)

    def test_orphan_intent_blocks_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            persist(tmp, {"intent": self.binding})
            with self.assertRaises(ReconciliationRequired):
                self.generate(tmp)
            self.produce.assert_not_called()

    def test_changed_policy_binding_blocks_saved_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.generate(tmp)
            with self.assertRaisesRegex(ValueError, "binding"):
                self.generate(tmp, dict(self.binding, step=2))
            self.assertEqual(self.produce.call_count, 1)

    def test_partial_generation_failure_retains_intent(self):
        self.produce.side_effect = RuntimeError("interrupted generation")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                self.generate(tmp)
            self.assertTrue((Path(tmp)/"intent.json").exists())
            with self.assertRaises(ReconciliationRequired):
                self.generate(tmp)
            self.produce.assert_called_once()

    def test_unsupported_output_fails_without_completed_tokens(self):
        self.produce.return_value = ([[1]], [[2]], None, {})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "contract"):
                self.generate(tmp)
            self.assertFalse((Path(tmp)/"generation.json").exists())

    def test_newer_incomplete_checkpoint_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/"checkpoint-2").mkdir()
            with self.assertRaisesRegex(ReconciliationRequired, "partial checkpoint"):
                recovery.latest_checkpoint(tmp, "binding")

    def test_no_grpo_loss_or_training_step_override(self):
        tree = ast.parse(Path(recovery.__file__).read_text())
        trainer = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "DurableGRPOTrainer")
        names = {n.name for n in trainer.body if isinstance(n, ast.FunctionDef)}
        self.assertFalse(names & {"compute_loss", "training_step", "_calculate_rewards", "_generate_and_score_completions"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval"})

    def test_accepts_actual_frozen_microbatch_configuration(self):
        args = SimpleNamespace(**release.experiment.trainer_kwargs("output"))
        recovery.validate_configuration(args, 1, None)
        args.gradient_accumulation_steps = 1
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
            recovery.validate_configuration(args, 1, None)

    def test_determinism_requires_startup_environment_and_strict_algorithms(self):
        torch = Mock()
        torch.are_deterministic_algorithms_enabled.return_value = True
        with patch.dict("sys.modules", {"torch": torch}), patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "before process startup"):
                recovery.deterministic_training()
            torch.use_deterministic_algorithms.assert_not_called()
        with patch.dict("sys.modules", {"torch": torch}), patch.dict("os.environ", {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            result = recovery.deterministic_training()
            torch.use_deterministic_algorithms.assert_called_once_with(True)
            self.assertTrue(result["deterministic_algorithms"])
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = release.make_plan(Path(__file__).resolve().parents[1])

    def parts(self):
        boundaries = {str(s): {"parameter_hash": f"weights-{s}", "optimizer_hash": f"opt-{s}",
            "scheduler_hash": f"lr-{s}", "rng_hash": f"rng-{s}", "trl_step": 4*s} for s in (1,2,3)}
        groups = {str(s): {"tokens_hash": f"tokens-{s}", "reward_hash": f"rewards-{s}",
                          "after_rng_hash": f"rng-{s}"} for s in (0,1,2)}
        ref = {"steps": 3, "boundaries": boundaries, "groups": groups, "nonzero_gradient_steps": 3,
               "replayed_groups": [], "interruption_observed": False}
        resumed = dict(deepcopy(ref), replayed_groups=[1])
        interrupted = dict(deepcopy(ref), steps=1, interruption_observed=True)
        return {"uninterrupted": ref, "interrupt": interrupted, "resume": resumed}

    def test_frozen_research_settings_and_finite_release(self):
        release.validate_plan(self.plan)
        self.assertEqual(self.plan["experiment"]["steps_per_arm"], 24)
        self.assertEqual(self.plan["max_sandbox_starts"], 55484)
        self.assertEqual(self.plan["grading_batches"], 81)
        self.assertEqual(self.plan["new_baseline_samples"], 0)
        with self.assertRaises(ValueError):
            release.validate_plan(dict(self.plan, new_baseline_samples=32))

    def test_all_state_and_future_rollouts_required(self):
        self.assertTrue(release.validate_control(self.parts(), self.plan)["passed"])
        for field in ("parameter_hash", "optimizer_hash", "scheduler_hash", "rng_hash", "trl_step"):
            parts = self.parts()
            parts["resume"]["boundaries"]["2"][field] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "diverged"):
                release.validate_control(parts, self.plan)
        for field in ("tokens_hash", "reward_hash", "after_rng_hash"):
            parts = self.parts()
            parts["resume"]["groups"]["2"][field] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "diverged"):
                release.validate_control(parts, self.plan)

    def test_model_only_or_missing_control_is_not_a_pass(self):
        parts = self.parts()
        del parts["interrupt"]
        with self.assertRaises(ValueError):
            release.validate_control(parts, self.plan)
        parts = self.parts()
        parts["resume"]["replayed_groups"] = []
        with self.assertRaises(ValueError):
            release.validate_control(parts, self.plan)

    def test_budget_is_not_invoice(self):
        rates = {"cpu_hour_cost": ".0473", "mem_gib_hour_cost": ".008",
                 "cpu_hour_cost_sandbox": ".1419", "mem_gib_hour_cost_sandbox": ".024", "gpu_hour_cost_l40s": "1.95"}
        quote = release.quote(self.plan, rates, {})
        self.assertTrue(quote["not_invoice_or_expected_spend"])
        self.assertGreater(float(quote["max_compute_envelope_usd"]), 300)


if __name__ == "__main__":
    unittest.main()
