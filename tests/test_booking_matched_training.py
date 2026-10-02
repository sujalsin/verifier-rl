import ast
from contextlib import ExitStack
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

import modal_booking_matched_training as launcher
from tests.test_booking_baseline_comparison import raw_for, samples_for
from tests import test_grpo_recovery
from verifier_rl import booking_matched_training as release
from verifier_rl import booking_baseline_comparison as study
from verifier_rl import durable_grading, supervised_execution
from verifier_rl.evaluation_journal import persist
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import canonical_json, digest

ROOT = Path(__file__).resolve().parents[1]


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = release.make_plan(ROOT)
        # Synthetic evidence, not an ignored cloud artifact or candidate execution.
        rows = []
        for index, (sid, seed) in enumerate(study.identities()):
            sample = study.original.sample_from_text(study.original.controls()["correct"],
                sid=sid, seed=seed, plan=cls.plan["experiment"], tokens=90, eos=True)
            outcomes = {c.input_hash: {"passed": True, "inclusive_match": False, "reason": "pass"}
                        for c in study.cases_for("evaluation")}
            if index == 0:
                h = next(c.input_hash for c in study.coverage.cases_for("audit") if c.arguments["bookings"])
                outcomes[h] = {"passed": None, "inclusive_match": None, "reason": "lost_parent"}
            rows.append(study.program_summary(sample, outcomes, "evaluation"))
        cls.baseline = {"programs": rows, "summary": study.policy_summary(rows), "optimizer_updates": 0,
                        "plan_hash": digest(canonical_json(cls.plan["experiment"]))}
        cls.data = canonical_json(cls.baseline).encode()

    def policies(self):
        policies = {}
        for arm in study.ARMS:
            for step in (12,24):
                rows = deepcopy(self.baseline["programs"])
                for row, (sid, _) in zip(rows, study.identities(arm,step)):
                    row["sample_id"] = sid
                policies[f"{arm}-{step}"] = rows
        return policies

    def test_baseline_is_anchored_not_regenerated(self):
        self.assertEqual(self.baseline["summary"]["programs"], 32)
        self.assertEqual(self.baseline["optimizer_updates"], 0)
        fixture_plan = dict(self.plan, baseline_sha256=hashlib.sha256(self.data).hexdigest())
        self.assertEqual(release.validate_baseline(self.data, self.baseline, fixture_plan), self.baseline)
        with self.assertRaisesRegex(ValueError, "anchor"):
            release.validate_baseline(self.data+b" ", self.baseline, fixture_plan)

    def test_summary_keeps_unknown_bounds_and_does_not_claim_amplification(self):
        report = release.comparison_summary(self.baseline, self.policies())
        self.assertEqual(len(report["policies"]), 5)
        self.assertEqual(report["policies"]["reference-24"]["audit"]["case_pass_bounds"], [6143,6144])
        self.assertEqual(report["final_rate_differences"]["audit_full_pass"]["omission_minus_reference"], [-1/32,1/32])
        self.assertFalse(report["amplification_established"])

    def test_full_fixed_policy_schedule_required(self):
        policies = self.policies()
        del policies["reference-12"]
        with self.assertRaises(ValueError):
            release.comparison_summary(self.baseline, policies)
        policies = self.policies()
        policies["reference-24"].pop()
        with self.assertRaises(ValueError):
            release.comparison_summary(self.baseline, policies)

    def test_code_has_no_candidate_exec_and_no_loss_override(self):
        tree = ast.parse(Path(launcher.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile"})
        trainer = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "train_arm")
        text = ast.unparse(trainer)
        self.assertIn("require_live_control", text)
        self.assertIn("rewards_from_raw", text)
        self.assertIn("first rollout groups differ", text)
        self.assertIn("restore_rng", ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "evaluation_at_boundary")))

    def test_stable_logging_and_same_deterministic_training_builder(self):
        tree = ast.parse(Path(launcher.__file__).read_text())
        builder = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_trainer"))
        self.assertIn("kwargs['logging_dir'] = str(directory / 'logs')", builder)
        self.assertLess(builder.index("recovery.deterministic_training()"), builder.index("load_policy"))
        self.assertNotIn("full_determinism=True", builder)

    def test_live_gate_rejects_untested_recovery_code(self):
        parts = test_grpo_recovery.ReleaseTests.parts(self)
        sources = {"verifier_rl/grpo_recovery.py": "tested", "modal_booking_matched_training.py": "def build_trainer(): pass",
                   "modal_booking_study.py": "model loader", "verifier_rl/booking_baseline_comparison.py": "frozen scorer"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persist(root/self.plan["control_id"], {"plan": self.plan, "source_snapshot": sources,
                "result": release.validate_control(parts, self.plan), **parts})
            self.assertTrue(launcher.require_live_control(self.plan, sources, root)["passed"])
            with self.assertRaisesRegex(ValueError, "implementation differs"):
                launcher.require_live_control(self.plan, dict(sources, **{"verifier_rl/grpo_recovery.py": "changed"}), root)
            with self.assertRaisesRegex(ValueError, "construction differs"):
                launcher.require_live_control(self.plan, dict(sources, **{"modal_booking_matched_training.py": "def build_trainer(): return 1"}), root)
            with self.assertRaisesRegex(ValueError, "dependency differs"):
                launcher.require_live_control(self.plan, dict(sources, **{"modal_booking_study.py": "changed"}), root)

    def test_controller_cannot_train_when_live_control_is_missing(self):
        with patch.object(launcher, "check_snapshot"), patch.object(launcher, "artifacts", Mock()), \
                patch.object(launcher, "require_live_control", side_effect=ValueError("control failed")), \
                patch.object(launcher, "train_arm") as train, patch.object(launcher, "grade_batch") as grade:
            with self.assertRaisesRegex(ValueError, "control failed"):
                launcher.run_comparison.get_raw_f()(self.plan, {}, {}, {}, time.time()+300)
            train.remote.assert_not_called()
            grade.remote.assert_not_called()


class GradingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = release.make_plan(ROOT)
        cls.key = "train-reference-00"
        cls.samples = samples_for(cls.plan["experiment"], cls.key)
        cls.raw = raw_for(cls.samples, cls.key, "training", cls.plan["experiment"])

    def test_new_worker_uses_frozen_scoring_and_reuses_completed_batch(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            stack.enter_context(patch.object(launcher, "Path", side_effect=lambda p: root if str(p)=="/artifacts" else Path(p)))
            stack.enter_context(patch.object(launcher, "artifacts", Mock(commit=Mock(), reload=Mock())))
            reserve = stack.enter_context(patch.object(launcher, "claim"))
            stack.enter_context(patch.object(supervised_execution, "SupervisedPanelBackend", return_value=object()))
            execute = stack.enter_context(patch.object(durable_grading, "execute_batch",
                new=AsyncMock(return_value=(self.raw["entries"], {"new_starts": 384}))))
            function = launcher.grade_batch.get_raw_f()
            setup = {"sandbox_image_id": "im-test", "app_name": "fixture"}
            result = function(self.key, self.samples, "training", self.plan, setup, time.time()+300)
            self.assertEqual([r["reference"]["passed_bounds"] for r in result["checked"]["rows"]], [[96,96]]*4)
            self.assertEqual(study.rewards_from_raw(result["raw"], self.samples, self.key, "reference",
                self.plan["experiment"], "im-test"), [1.0]*4)
            again = function(self.key, self.samples, "training", self.plan, setup, time.time()+300)
            self.assertEqual(again, result)
            self.assertEqual(execute.await_count, 1)
            reserve.assert_called_once()
            self.assertEqual(execute.call_args.args[4]["run_id"], release.RUN_ID)


class PolicyJournalTests(unittest.TestCase):
    def setUp(self):
        self.plan = release.make_plan(ROOT)
        self.directory, self.arm = Path("/fixture"), "reference"
        target = self.directory / "arms" / self.arm
        binding = {"release_hash": digest(canonical_json(self.plan)), "directory": str(target), "control": False}
        binding_hash = digest(canonical_json(binding))
        self.docs = {"binding.json": binding}
        self.result = {"boundaries": {}, "evaluations": {}, "first_tokens": [[1,2]]*4}
        previous = self.plan["experiment"]["initial_parameter_hash"]
        for i in range(24):
            prefix = f"journal/group-{i:02d}"
            intent = {"binding_hash": binding_hash, "version": launcher.recovery.VERSION,
                      "step": i, "parameter_hash": previous}
            output = [[[3]]*4, [[1,2]]*4, None, {}]
            self.docs[prefix+"/intent.json"] = intent
            self.docs[prefix+"/generation.json"] = {"binding": intent, "output": output, "tokens_hash": digest(canonical_json(output))}
            self.docs[prefix+"/reward.json"] = {"completion_ids": output[1], "samples": ["synthetic"]*4}
            self.docs[prefix+"/samples.json"] = ["synthetic"]*4
            previous = f"weights-{i+1}"
            boundary = {"binding_hash": binding_hash, "step": i+1, "trl_step": 4*(i+1), "parameter_hash": previous}
            self.docs[f"journal/boundaries/{i+1}.json"] = boundary
            self.result["boundaries"][str(i+1)] = boundary
        for step in (12,24):
            samples = [{"sample_id": sid, "seed": seed} for sid,seed in study.identities(self.arm,step)]
            evaluation = {"parameter_hash": f"weights-{step}", "samples": samples}
            prefix = f"evaluation-{step:02d}"
            self.result["evaluations"][str(step)] = evaluation
            self.docs[prefix+"/result.json"] = evaluation
            self.docs[f"model-{step:02d}/receipt.json"] = {"step": step, "parameter_hash": f"weights-{step}"}
            for sample in samples:
                sid = sample["sample_id"]
                self.docs[f"{prefix}/samples/{sid}.json"] = sample
                self.docs[f"{prefix}/intents/{sid}.json"] = dict(sample, parameter_hash=f"weights-{step}")
        self.logs = []

    def check(self):
        root = self.directory / "arms" / self.arm
        with patch.object(launcher, "read_json", side_effect=lambda p: deepcopy(self.docs[p.relative_to(root).as_posix()])):
            with ProgressLog("fixture", emit=self.logs.append) as progress:
                launcher.verify_policy_journals(self.directory, self.plan, self.arm, self.result, progress)

    def test_complete_journals_and_progress_counts(self):
        self.check()
        completed = [r for r in self.logs if r["event"]=="stage_completed"]
        self.assertEqual([(r["completed"],r["total"]) for r in completed], [(24,24),(64,64)])

    def test_swapped_tokens_rejected_without_completed_log(self):
        self.docs["journal/group-01/reward.json"]["completion_ids"] = [[9]]*4
        with self.assertRaisesRegex(ValueError, "training rollout"):
            self.check()
        self.assertEqual(self.logs[-1]["event"], "failed")

    def test_evaluation_intent_cannot_point_to_other_weights(self):
        name = "evaluation-12/intents/eval-reference-12-18000.json"
        self.docs[name]["parameter_hash"] = "other-weights"
        with self.assertRaisesRegex(ValueError, "generation journal"):
            self.check()
        self.assertEqual(self.logs[-1]["event"], "failed")


if __name__ == "__main__":
    unittest.main()
