from copy import deepcopy
from pathlib import Path
import unittest

from verifier_rl.fixtures import fixture_impl, source_for
from verifier_rl.grading import ExecutionResult, Status, score_suite
from verifier_rl.grpo_pilot import (PARAMETERS, SEEDS, evaluation_suites, pilot_plan, require_report,
                                   rollout_rewards, summarize_pilot, trainer_kwargs, training_evidence)
from verifier_rl.measurement_v2 import PARAMETER_HASH, PROMPT_HASH
from verifier_rl.modal_backend import RUNNER
from verifier_rl.model_trial import submission_from_completion
from verifier_rl.reward_v2 import behavior_suite
from verifier_rl.suites import canonical_json, digest


class GRPOPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = pilot_plan(Path("task_001_expiring_cache.txt").read_text())

    @staticmethod
    def sample(seed=9000, arm="training", fault="correct"):
        return {**submission_from_completion(source_for(fault)), "seed": seed, "arm": arm,
                "prompt_hash": PROMPT_HASH}

    @staticmethod
    def report(sample, suites=None, fault="correct"):
        suites = suites or (behavior_suite(),)
        executions = {c.input_hash: (ExecutionResult(Status.COMPLETED,
                      canonical_json(fixture_impl(c.operations, fault)).encode(),
                      metadata={"runner_hash": digest(RUNNER), "image_id": "im-test",
                                "cleanup": "terminated", "block_network": True,
                                "reset": "fresh_sandbox_per_input"}),)
                      for suite in suites for c in suite.cases}
        return {"seed": sample["seed"], "arm": sample["arm"], "candidate_hash": digest(sample["source"]),
                "execution_config": {"attempts": len(executions)},
                "suites": [score_suite(suite, executions) for suite in suites]}

    @staticmethod
    def metrics(uniform=False):
        return {"global_step": 1 if uniform else 4,
                "rewards": [[0, 0, 0, 0]] if uniform else [[1, .5, 0, .25]] * 4,
                "log_history": [{"grad_norm": 0.0}] if uniform else [{"grad_norm": .8}] * 4,
                "before_parameter_hash": PARAMETER_HASH,
                "after_parameter_hash": PARAMETER_HASH if uniform else "changed",
                "checkpoint_reload_hash_matches": True, "finite_parameters": True,
                "trainable_parameters": PARAMETERS}

    def test_frozen_bounds_and_independent_audit(self):
        self.assertEqual(self.plan["max_training_executions"], 544)
        self.assertEqual(self.plan["max_evaluation_executions"], 6752)
        self.assertEqual(self.plan["seeds"], list(range(8000, 8008)))
        self.assertEqual(self.plan["gpu"], "L40S")
        self.assertEqual(self.plan["initial_parameter_hash"], PARAMETER_HASH)
        reward, audit = evaluation_suites()
        self.assertEqual((len(reward.cases), len(audit.cases)), (34, 388))
        self.assertFalse({c.input_hash for c in reward.cases} & {c.input_hash for c in audit.cases})

    def test_explicit_training_configuration(self):
        config = trainer_kwargs("unused")
        for key, value in {"max_steps": 4, "num_generations": 4, "num_iterations": 1,
                           "gradient_accumulation_steps": 4, "steps_per_generation": 4,
                           "per_device_train_batch_size": 1, "loss_type": "grpo",
                           "scale_rewards": "group", "beta": 0, "learning_rate": 1e-6,
                           "repetition_penalty": 1.05, "weight_decay": 0,
                           "importance_sampling_level": "token", "mask_truncated_completions": False}.items():
            self.assertEqual(config[key], value)
        self.assertTrue(config["bf16"])

    def test_real_partial_reward_not_binary(self):
        faults = ("correct", "inclusive_expiry", "extra_put_output", "zero_missing")
        samples = [self.sample(9000+i, fault=fault) for i, fault in enumerate(faults)]
        reports = [self.report(s, fault=fault) for s, fault in zip(samples, faults)]
        rewards = rollout_rewards(samples, reports, "im-test")
        self.assertEqual(rewards, [1.0, .5125, .025, .8925])

    def test_audit_in_training_or_wrong_identity_is_rejected(self):
        samples = [self.sample(9000+i) for i in range(4)]
        reports = [self.report(s) for s in samples]
        bad = deepcopy(reports)
        bad[0] = self.report(samples[0], evaluation_suites())
        with self.assertRaises(ValueError): rollout_rewards(samples, bad, "im-test")
        bad = deepcopy(reports)
        bad[0]["seed"] += 1
        with self.assertRaises(ValueError): rollout_rewards(samples, bad, "im-test")
        bad = deepcopy(samples)
        bad[0]["arm"] = "after"
        with self.assertRaises(ValueError): rollout_rewards(bad, reports, "im-test")
        with self.assertRaises(ValueError): rollout_rewards(samples[:3], reports[:3], "im-test")

    def test_environment_and_infrastructure_gates(self):
        sample = self.sample()
        report = self.report(sample)
        for key, value in (("runner_hash", "bad"), ("image_id", "im-other"), ("cleanup", "unknown"),
                           ("block_network", False), ("reset", "reused")):
            bad = deepcopy(report)
            bad["suites"][0]["outcomes"][0]["attempts"][0]["metadata"][key] = value
            with self.assertRaises(ValueError): require_report(sample, bad, (behavior_suite(),), "im-test")
        suite = behavior_suite()
        executions = {c.input_hash: (ExecutionResult(Status.INFRASTRUCTURE_ERROR),) for c in suite.cases}
        bad = dict(report, suites=[score_suite(suite, executions)])
        with self.assertRaisesRegex(ValueError, "infrastructure"):
            require_report(sample, bad, (suite,), "im-test")

    def test_update_evidence_and_valid_negative_result(self):
        result = training_evidence(self.metrics())
        self.assertTrue(result["optimizer_update_verified"])
        self.assertEqual(result["nonzero_gradient_steps"], 4)
        result = training_evidence(self.metrics(uniform=True))
        self.assertFalse(result["optimizer_update_verified"])
        self.assertEqual(result["stop_reason"], "uniform_rewards")

    def test_evidence_rejects_broken_reload_nonfinite_and_unexplained_stops(self):
        for key, value in (("checkpoint_reload_hash_matches", False), ("finite_parameters", False),
                           ("before_parameter_hash", "wrong"), ("trainable_parameters", 1),
                           ("log_history", [{"grad_norm": float("nan")}] * 4),
                           ("rewards", [[None, 0, 1, .5]] * 4), ("global_step", 3)):
            with self.assertRaises(ValueError): training_evidence(dict(self.metrics(), **{key: value}))
        with self.assertRaises(ValueError):
            training_evidence(dict(self.metrics(uniform=True), after_parameter_hash="changed"))
        bad = self.metrics()
        bad["rewards"] = [[0, 0, 0, 0], [0, .5, 1, .2], [0, .5, 1, .2], [0, .5, 1, .2]]
        with self.assertRaises(ValueError): training_evidence(bad)

    def test_paired_summary_reports_full_and_partial_separately(self):
        samples = [self.sample(seed, arm) for arm in ("before", "after") for seed in SEEDS]
        reports = [self.report(s, evaluation_suites()) for s in samples]
        generation = {"run_id": "qwen-test", "plan": self.plan, "metrics": self.metrics(), "samples": samples}
        result = summarize_pilot(generation, reports, "im-test")
        self.assertEqual(result["arms"]["before"]["full_audit_passes"], 8)
        self.assertEqual(result["arms"]["after"]["mean_partial_reward"], 1)
        self.assertEqual(result["paired_audit_case_deltas"], [0] * 8)
        self.assertEqual(result["recorded_evaluation_executions"], 6752)
        self.assertEqual(result["unchanged_source_pairs"], 8)
        with self.assertRaises(ValueError): summarize_pilot(generation, reports[:-1], "im-test")
