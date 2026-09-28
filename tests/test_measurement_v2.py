from copy import deepcopy
import unittest

from verifier_rl.checkpoint_pilot import MODEL_ID, REVISION
from verifier_rl.fixtures import fixture_impl, source_for
from verifier_rl.grading import ExecutionResult, Status, score_suite
from verifier_rl.measurement_v2 import (GENERATION_RUN, PARAMETER_HASH, PROMPT_HASH, SEEDS,
                                       measurement_plan, measurement_summary, require_measurement_report)
from verifier_rl.model_trial import submission_from_completion
from verifier_rl.modal_backend import RUNNER
from verifier_rl.reward_v2 import behavior_suite
from verifier_rl.sft_trial import evaluation_suites
from verifier_rl.suites import canonical_json, digest


class MeasurementV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = [{**submission_from_completion(source_for("correct")), "seed": seed,
                        "prompt_hash": PROMPT_HASH} for seed in SEEDS]
        cls.generation = {"run_id": GENERATION_RUN, "samples": cls.samples,
                          "before_parameter_hash": PARAMETER_HASH, "after_parameter_hash": PARAMETER_HASH,
                          "plan": {"model_id": MODEL_ID, "revision": REVISION, "prompt_hash": PROMPT_HASH,
                                   "training": False}}
        cls.prior = [cls.report(s, evaluation_suites()) for s in cls.samples]
        cls.new = [cls.report(s, (behavior_suite(),)) for s in cls.samples]

    @staticmethod
    def report(sample, suites):
        executions = {c.input_hash: (ExecutionResult(Status.COMPLETED,
                      canonical_json(fixture_impl(c.operations)).encode(),
                      metadata={"runner_hash": digest(RUNNER), "image_id": "im-test",
                                "cleanup": "terminated", "block_network": True}),)
                      for suite in suites for c in suite.cases}
        return {"seed": sample["seed"], "candidate_hash": digest(sample["source"]),
                "execution_config": {"attempts": len(executions)},
                "suites": [score_suite(suite, executions) for suite in suites]}

    def test_frozen_saved_sample_budget_no_new_generation(self):
        plan = measurement_plan(self.generation, self.prior)
        self.assertEqual(plan["max_sandbox_executions"], 272)
        self.assertEqual(plan["new_model_samples"], 0)
        self.assertEqual(plan["suite_cases"], 34)
        self.assertFalse(plan["training"])
        self.assertTrue(plan["no_automatic_training"])

    def test_refuses_changed_model_prompt_weights_seed_or_source(self):
        for mutation in ("model", "prompt", "weights", "seeds", "source"):
            bad = deepcopy(self.generation)
            if mutation == "model": bad["plan"]["model_id"] = "other"
            elif mutation == "prompt": bad["plan"]["prompt_hash"] = "other"
            elif mutation == "weights": bad["after_parameter_hash"] = "changed"
            elif mutation == "seeds": bad["samples"].pop()
            else: bad["samples"][0]["source"] = "different"
            with self.assertRaises(ValueError, msg=mutation): measurement_plan(bad, self.prior)

    def test_complete_summary_preserves_old_evidence(self):
        old = deepcopy((self.generation, self.prior, self.new))
        result = measurement_summary(self.generation, self.prior, self.new, "im-test")
        self.assertEqual(result["full_v2_passes"], 8)
        self.assertEqual(result["recorded_sandbox_executions"], 272)
        self.assertEqual(result["mean_v1_partial"], 1)
        self.assertEqual(result["mean_v2_partial"], 1)
        self.assertEqual(result["mixed_groups_above_edge_allowance"], 0)
        self.assertFalse(result["training"])
        self.assertEqual((self.generation, self.prior, self.new), old)

    def test_environment_and_coverage_gates(self):
        for key, value in (("runner_hash", "other"), ("image_id", "im-other"),
                           ("cleanup", "unknown"), ("block_network", False)):
            bad = deepcopy(self.new[0])
            bad["suites"][0]["outcomes"][0]["attempts"][0]["metadata"][key] = value
            with self.assertRaises(ValueError): require_measurement_report(self.samples[0], bad, "im-test")
        with self.assertRaises(ValueError): measurement_summary(self.generation, self.prior, self.new[:-1], "im-test")
        with self.assertRaises(ValueError): require_measurement_report(self.samples[0], self.prior[0], "im-test")
