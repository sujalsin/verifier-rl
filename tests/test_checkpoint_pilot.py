from copy import deepcopy
from pathlib import Path
import unittest

from verifier_rl.checkpoint_pilot import (DECODE, MODEL_ID, REVISION, SMALL_MODEL, SMALL_REVISION,
                                         plan_for, reference_packet, recovery_plan, summarize_comparison, summarize_policy)
from verifier_rl.fixtures import fixture_impl, source_for
from verifier_rl.grading import ExecutionResult, Status, score_suite
from verifier_rl.model_trial import submission_from_completion
from verifier_rl.modal_backend import RUNNER
from verifier_rl.sft_trial import SEEDS, evaluation_suites
from verifier_rl.suites import canonical_json, digest


class CheckpointPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = plan_for(Path("task_001_expiring_cache.txt").read_text())
        cls.samples, cls.reports = [], []
        for seed in SEEDS:
            sample = {**submission_from_completion(source_for("correct")), "seed": seed,
                      "prompt_hash": cls.plan["prompt_hash"], "syntax_valid": True, "hit_token_cap": False,
                      "tokens": 100, "arm": "before"}
            executions = {c.input_hash: (ExecutionResult(Status.COMPLETED,
                          canonical_json(fixture_impl(c.operations)).encode(),
                          metadata={"runner_hash": digest(RUNNER), "image_id": "im-test", "cleanup": "terminated"}),)
                          for s in evaluation_suites() for c in s.cases}
            cls.samples.append(sample)
            cls.reports.append({"seed": seed, "arm": "before", "candidate_hash": digest(sample["source"]),
                                "execution_config": {"attempts": len(executions)},
                                "suites": [score_suite(s, executions) for s in evaluation_suites()]})

    def test_plan_pins_larger_model_without_mutating_original(self):
        self.assertEqual(self.plan["model_id"], MODEL_ID)
        self.assertEqual(len(REVISION), 40)
        self.assertIn("0.5B", SMALL_MODEL)
        self.assertFalse(self.plan["training"])
        self.assertEqual(self.plan["max_sandbox_executions"], 432)
        self.assertEqual(set(self.plan["suite_hashes"]), {"g3", "balanced"})
        self.assertEqual(self.plan["max_completion_tokens"], 512)
        self.assertEqual(self.plan["repetition_penalty"], 1.05)

    def test_uniform_success_is_not_mixed_reward_signal(self):
        result = summarize_policy(self.samples, self.reports, "im-test", self.plan["prompt_hash"])
        self.assertEqual(result["full_g3_passes"], 8)
        self.assertEqual(result["mean_balanced_partial"], 1)
        self.assertEqual(result["mixed_binary_groups"], 0)
        self.assertEqual(result["meaningful_mixed_partial_groups"], 0)

    def test_rejects_bad_identity_environment_and_unscored_results(self):
        for mutation in ("source", "seed", "suite", "infra", "cleanup", "missing"):
            bad = deepcopy(self.reports)
            if mutation == "source": bad[0]["candidate_hash"] = "bad"
            elif mutation == "seed": bad[0]["seed"] = 999
            elif mutation == "suite": bad[0]["suites"][0]["suite_hash"] = "bad"
            elif mutation == "infra": bad[0]["suites"][0]["all_passed"] = None
            elif mutation == "missing": bad[0]["suites"][0]["outcomes"].pop()
            else: bad[0]["suites"][0]["outcomes"][0]["attempts"][0]["metadata"]["cleanup"] = "unknown"
            with self.assertRaises(ValueError): summarize_policy(self.samples, bad, "im-test", self.plan["prompt_hash"])

    def test_reference_uses_only_original_before_sft_samples(self):
        generation = {"run_id": "qwen-prior", "plan": {"model_id": SMALL_MODEL, "revision": SMALL_REVISION,
                      "prompt_hash": self.plan["prompt_hash"], **DECODE},
                      "samples": self.samples + [{"arm": "after", "invalid": True}],
                      "training": {"before_parameter_hash": "base"}, "runtime": {}}
        reference = reference_packet(generation, self.reports + [{"arm": "after", "invalid": True}], self.plan, "im-test")
        self.assertEqual(reference["parameter_hash"], "base")
        self.assertEqual(len(reference["samples"]), 8)
        bad = deepcopy(generation)
        bad["plan"]["temperature"] = .1
        with self.assertRaises(ValueError): reference_packet(bad, self.reports, self.plan, "im-test")

    def test_comparison_refuses_weight_changes(self):
        reference = {"samples": self.samples, "reports": self.reports, "source_run_id": "qwen-prior"}
        generation = {"plan": self.plan, "run_id": "qwen-new", "samples": self.samples,
                      "before_parameter_hash": "same", "after_parameter_hash": "same"}
        result = summarize_comparison(generation, self.reports, reference, "im-test")
        self.assertTrue(result["parameters_unchanged"])
        generation["after_parameter_hash"] = "changed"
        with self.assertRaises(ValueError): summarize_comparison(generation, self.reports, reference, "im-test")

    def test_cpu_recovery_keeps_samples_and_caps_missing_work(self):
        generation = {"plan": self.plan, "run_id": "qwen-new", "samples": self.samples,
                      "before_parameter_hash": "same", "after_parameter_hash": "same"}
        plan = recovery_plan(generation, self.reports[:-1], "im-test")
        self.assertEqual(plan["pending_seeds"], [4007])
        self.assertEqual(plan["max_additional_sandbox_executions"], 54)
        self.assertEqual(plan["new_samples"], 0)
        self.assertFalse(plan["gpu"])
        self.assertEqual(recovery_plan(generation, self.reports, "im-test")["pending_seeds"], [])
        with self.assertRaises(ValueError): recovery_plan(generation, self.reports[:-2], "im-test")
        with self.assertRaises(ValueError): recovery_plan(generation, self.reports + self.reports[:1], "im-test")

    def test_cpu_recovery_rejects_corrupt_or_unscored_saved_evidence(self):
        generation = {"plan": self.plan, "run_id": "qwen-new", "samples": self.samples,
                      "before_parameter_hash": "same", "after_parameter_hash": "same"}
        for mutation in ("source", "cleanup", "unscored"):
            bad = deepcopy(self.reports[:-1])
            if mutation == "source": bad[0]["candidate_hash"] = "bad"
            elif mutation == "unscored": bad[0]["suites"][0]["all_passed"] = None
            else: bad[0]["suites"][0]["outcomes"][0]["attempts"][0]["metadata"]["cleanup"] = "unknown"
            with self.assertRaises(ValueError): recovery_plan(generation, bad, "im-test")
