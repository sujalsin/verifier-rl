from copy import deepcopy
import json
import unittest

from verifier_rl.fixtures import source_for
from verifier_rl.grading import ExecutionResult, Status, score_suite
from verifier_rl.modal_backend import RUNNER
from verifier_rl.model_trial import evaluate_submission, submission_from_completion
from verifier_rl.suites import build_suites, canonical_json, digest
from verifier_rl.training import (control_rewards, grpo_kwargs, require_control_evidence,
                                 require_grader_controls, training_batch_id, validate_batch_id, validate_steps)


class TrainingTests(unittest.TestCase):
    def test_bounded_multi_step_config_and_unique_batch_ids(self):
        self.assertEqual(len({training_batch_id(i) for i in range(4)}), 4)
        for i in range(4):
            validate_batch_id(training_batch_id(i), False)
        validate_batch_id("evaluation", True)
        for bad in (0, 5, True, 1.0):
            with self.assertRaises(ValueError): validate_steps(bad)
        for name, audit in (("../escape", False), ("train-0004", False), ("evaluation", False), ("train-0000", True)):
            with self.assertRaises(ValueError): validate_batch_id(name, audit)
        config = grpo_kwargs("unused", max_steps=4)
        self.assertEqual(config["num_generations"], 4)
        self.assertEqual(config["steps_per_generation"], 4)
        self.assertEqual(config["num_iterations"], 1)
        self.assertEqual(config["loss_type"], "grpo")
        self.assertEqual(config["max_steps"], 4)
        self.assertEqual(config["weight_decay"], 0)

    def test_synthetic_control_labels_and_degenerate_groups(self):
        raw = ["zebra", "animal", "fox", "animal"]
        self.assertEqual(control_rewards(raw, "zero"), [0, 0, 0, 0])
        self.assertEqual(control_rewards(raw, "mixed"), [0, 1, 0, 1])
        for values, mode in ((["a"] * 4, "mixed"), (raw[:3], "zero"), (raw, "bad")):
            with self.assertRaises(ValueError): control_rewards(values, mode)

    def test_control_gates_check_real_updates_not_just_nonzero_rewards(self):
        negative = {"global_step": 1, "rewards": [[0, 0, 0, 0]], "log_history": [{"grad_norm": 0}],
                    "before_parameter_hash": "a", "after_parameter_hash": "a", "checkpoint_reload_hash_matches": True}
        positive = {"global_step": 2, "rewards": [[0, 1, 0, 0]] * 2,
                    "log_history": [{"grad_norm": 0.4}, {"grad_norm": 0.7}],
                    "before_parameter_hash": "a", "after_parameter_hash": "b", "checkpoint_reload_hash_matches": True}
        require_control_evidence(negative, "zero")
        require_control_evidence(positive, "mixed")
        for field, value in (("global_step", 1), ("after_parameter_hash", "a"),
                             ("checkpoint_reload_hash_matches", False), ("log_history", [{"grad_norm": float("nan")}] * 2)):
            with self.assertRaises(ValueError): require_control_evidence(dict(positive, **{field: value}), "mixed")
        with self.assertRaises(ValueError): require_control_evidence(dict(negative, after_parameter_hash="b"), "zero")

    def test_grader_gate_rejects_stale_or_incomplete_evidence(self):
        suite = build_suites()[2]
        reports = []
        for name in ("correct", "inclusive_expiry"):
            from verifier_rl.fixtures import fixture_impl
            executions = {c.input_hash: (ExecutionResult(Status.COMPLETED,
                           canonical_json(fixture_impl(c.operations, name)).encode(),
                           metadata={"runner_hash": digest(RUNNER), "image_id": "im-test", "cleanup": "terminated"}),)
                          for c in suite.cases}
            entry = submission_from_completion(source_for(name))
            reports.append({"candidate_hash": digest(entry["source"]), "suites": [score_suite(suite, executions)],
                            "extraction": {"status": entry["extraction_status"],
                                           "version": entry["extraction_version"], "accepted": True}})
        evidence = {"kind": "grader_controls", "passed": True, "reports": reports}
        require_grader_controls(evidence)
        for mutation in ("source", "suite", "coverage", "cleanup", "image", "extraction"):
            bad = deepcopy(evidence)
            r = bad["reports"][0]
            if mutation == "source": r["candidate_hash"] = "bad"
            elif mutation == "suite": r["suites"][0]["suite_hash"] = "bad"
            elif mutation == "coverage": r["suites"][0]["outcomes"].pop()
            elif mutation == "cleanup": r["suites"][0]["outcomes"][0]["attempts"][0]["metadata"]["cleanup"] = "unknown"
            elif mutation == "extraction": r["extraction"]["version"] = "stale"
            else: r["suites"][0]["outcomes"][0]["attempts"][0]["metadata"]["image_id"] = "im-other"
            with self.assertRaises(ValueError): require_grader_controls(bad)


class GraderControlIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_gate_uses_the_extracted_source_that_was_graded(self):
        from verifier_rl.fixtures import fixture_impl
        reports = []
        suite = build_suites()[2]
        for name in ("correct", "inclusive_expiry"):
            entry = submission_from_completion(source_for(name))
            self.assertNotEqual(digest(source_for(name)), digest(entry["source"]))

            class FixtureBackend:
                async def execute(self, request):
                    if request.source != entry["source"]:
                        raise AssertionError("unexpected submitted source")
                    return ExecutionResult(Status.COMPLETED, canonical_json(fixture_impl(json.loads(request.input_json), name)).encode(),
                                           metadata={"runner_hash": digest(RUNNER), "image_id": "im-test", "cleanup": "terminated"})

            reports.append(await evaluate_submission(entry, (suite,), FixtureBackend()))
        evidence = {"kind": "grader_controls", "passed": True, "reports": reports}
        require_grader_controls(evidence)
        evidence["reports"][0]["candidate_hash"] = digest(source_for("correct"))
        with self.assertRaises(ValueError):
            require_grader_controls(evidence)
