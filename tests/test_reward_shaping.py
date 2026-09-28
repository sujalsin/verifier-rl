from copy import deepcopy
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_measurement_v3 import report_for
from verifier_rl.fixtures import source_for
from verifier_rl.grpo_pilot import PARAMETERS
from verifier_rl.measurement_v2 import PARAMETER_HASH
from verifier_rl.model_trial import submission_from_completion
from verifier_rl.reward_review import group_advantages
from verifier_rl.reward_shaping import (CONTROLLER_SECONDS, FORMULAS, SEEDS, arm_plan,
    before_signature, budget_envelope, comparison_summary, execution_receipt, require_deadline,
    require_report, reserve_batch, reward_value, rollout_rewards, selected_suites, study_plan)
from verifier_rl.reward_shaping import study_execution_records, verify_budget_records
from verifier_rl.reward_v3 import coverage_scores
from verifier_rl.suites import canonical_json, digest

HAS_MODAL = find_spec("modal") is not None
RATES = {"cpu_hour_cost": ".0473", "mem_gib_hour_cost": ".008",
         "cpu_hour_cost_sandbox": ".1419", "mem_gib_hour_cost_sandbox": ".024",
         "gpu_hour_cost_l40s": "1.95"}


def shaped_report(sample, suites, prefix, fault="correct"):
    report = report_for(sample, suites, fault)
    for suite in report["suites"]:
        for outcome in suite["outcomes"]:
            for attempt in outcome["attempts"]:
                metadata = attempt["metadata"]
                if "sandbox_id" in metadata:
                    metadata["sandbox_id"] = prefix + metadata["sandbox_id"]
                    metadata["total_seconds"] = 2.1
    return report


class RewardShapingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = study_plan(Path("task_001_expiring_cache.txt").read_text())
        cls.plan["submission_deadline_epoch"] = time.time() + CONTROLLER_SECONDS
        cls.study_id = "qwen-shaping-unit-test"
        cls.budget = budget_envelope(cls.plan, {"metered_cost": "1.09557826"}, RATES)
        def sample(arm, seed, fault="correct"):
            return {**submission_from_completion(source_for(fault)), "arm": arm, "seed": seed,
                    "prompt_hash": cls.plan["prompt_hash"], "tokens": 10}
        cls.generations, cls.reports = {}, {}
        before = [sample("before", seed) for seed in SEEDS]
        cls.reports["baseline"] = [shaped_report(s, selected_suites(True), "baseline-") for s in before]
        for formula in FORMULAS:
            selected_plan = arm_plan(cls.plan, formula, cls.study_id)
            groups = []
            for step in range(4):
                faults = ("correct", "stale_expiry", "inclusive_expiry", "extra_put_output")
                samples = [sample("training", 9000 + 4 * step + i, fault) for i, fault in enumerate(faults)]
                reports = [shaped_report(s, selected_suites(), formula, fault) for s, fault in zip(samples, faults)]
                rewards = [reward_value(coverage_scores(r["suites"][0]), formula) for r in reports]
                groups.append({"samples": samples, "reports": reports, "rewards": rewards})
            after = [sample("after", seed) for seed in SEEDS]
            metrics = {"global_step": 4, "rewards": [g["rewards"] for g in groups],
                "log_history": [{"grad_norm": 1.0}] * 4, "before_parameter_hash": PARAMETER_HASH,
                "after_parameter_hash": digest(formula), "trainable_parameters": PARAMETERS,
                "checkpoint_reload_hash_matches": True, "finite_parameters": True}
            cls.generations[formula] = {"run_id": cls.study_id + "-" + formula.replace("_", "-"),
                "plan": selected_plan, "samples": deepcopy(before) + after, "rollout_records": groups, "metrics": metrics}
            cls.reports[formula] = [shaped_report(s, selected_suites(True), formula) for s in after]

    def test_matched_plan_only_formula_differs(self):
        a, b = [arm_plan(self.plan, f, self.study_id) for f in FORMULAS]
        self.assertEqual({k for k in a if a[k] != b[k]}, {"reward_formula"})
        self.assertEqual(self.plan["reward_cases"], 44)
        self.assertEqual(self.plan["audit_cases"], 388)
        self.assertEqual(self.plan["max_sandbox_executions"], 11776)
        self.assertEqual(self.plan["max_new_model_samples"], 64)
        self.assertEqual(self.plan["unique_evaluation_records"], 24)

    def test_formula_and_unscored_values(self):
        for formula in FORMULAS:
            self.assertEqual(reward_value({"partial": 1, "binary": 1}, formula), 1)
            self.assertIsNone(reward_value({"partial": None, "binary": None}, formula))
        self.assertEqual(reward_value({"partial": .8, "binary": 0}, "completion_bonus"), .4)
        for bad in ({"partial": float("nan"), "binary": 0}, {"partial": 1, "binary": 0}):
            with self.assertRaises(ValueError): reward_value(bad, "partial")
        with self.assertRaises(ValueError): reward_value({"partial": 1, "binary": 1}, "unknown")

    def test_group_effect_is_not_just_a_smaller_scalar(self):
        partial = [1, .0375, .54875, .48875]
        bonus = [reward_value({"partial": p, "binary": int(p == 1)}, "completion_bonus") for p in partial]
        self.assertGreater(group_advantages(partial)[2], 0)
        self.assertLess(group_advantages(bonus)[2], 0)
        no_full = [.4175, .0125, 0, 0]
        self.assertLess(max(abs(a-b) for a,b in zip(group_advantages(no_full), group_advantages([p/2 for p in no_full]))), .001)

    def test_rollout_rewards_recompute_and_forbid_audit(self):
        for formula in FORMULAS:
            generation = self.generations[formula]
            group = generation["rollout_records"][0]
            self.assertEqual(rollout_rewards(group["samples"], group["reports"], "im-test", generation["plan"]), group["rewards"])
            bad = deepcopy(group["reports"])
            bad[0] = shaped_report(group["samples"][0], selected_suites(True), "bad")
            with self.assertRaises(ValueError): rollout_rewards(group["samples"], bad, "im-test", generation["plan"])

    def test_baseline_signature_includes_raw_text_and_tokens(self):
        samples = deepcopy(self.generations["partial"]["samples"])
        reference = before_signature(samples)
        samples[0]["raw"] += "\n"
        self.assertNotEqual(before_signature(samples), reference)
        samples[0]["seed"] = 10
        with self.assertRaises(ValueError): before_signature(samples)

    def test_report_gate_retains_termination_failure(self):
        sample = self.generations["partial"]["samples"][0]
        report = deepcopy(self.reports["baseline"][0])
        # Status/result integrity is checked before unexplained termination is accepted.
        report["suites"][0]["outcomes"][0]["attempts"][0]["metadata"]["returncode"] = 137
        with self.assertRaises(ValueError): require_report(sample, report, True, "im-test")
        report = deepcopy(self.reports["baseline"][0])
        report["suites"][0]["outcomes"][0]["attempts"][0]["metadata"]["limits"] = {}
        with self.assertRaises(ValueError): require_report(sample, report, True, "im-test")

    def test_resource_reservations_and_cumulative_stop(self):
        self.assertGreater(float(self.budget["gpu_resource_reserve_usd"]), 1.5)
        self.assertFalse(self.budget["provider_hard_spending_cap"])
        reservation = reserve_batch(self.budget, {}, "first", 432)
        self.assertEqual(reservation["reserved_sandbox_seconds"], 432 * 120)
        self.assertLess(float(reservation["total_reserved_usd"]), 10)
        exhausted = {str(i): {"candidate_id": str(i), "executions": 432,
                              "accounted_sandbox_seconds": 432*120} for i in range(3)}
        with self.assertRaises(ValueError): reserve_batch(self.budget, exhausted, "next", 432)
        with self.assertRaises(ValueError): reserve_batch(self.budget, exhausted, "0", 44)
        with self.assertRaises(ValueError): budget_envelope(self.plan, {"metered_cost": "9"}, RATES)

    def test_receipts_round_up_and_count_shared_inputs_once(self):
        receipt = execution_receipt("baseline", self.reports["baseline"][0])
        self.assertEqual(receipt["executions"], 432)
        self.assertEqual(receipt["accounted_sandbox_seconds"], 432*3)
        self.assertFalse(receipt["is_provider_invoice"])
        self.assertEqual(reserve_batch(self.budget, {"baseline": receipt}, "next", 44)["previous_executions"], 432)
        bad = deepcopy(receipt)
        bad["accounted_sandbox_seconds"] = -1
        with self.assertRaises(ValueError): reserve_batch(self.budget, {"baseline": bad}, "next", 44)

    def test_submission_deadline_fails_closed(self):
        require_deadline(self.plan)
        for value in (0, float("nan"), None):
            with self.assertRaises(ValueError): require_deadline(dict(self.plan, submission_deadline_epoch=value))

    def test_entire_spending_journal_recomputed_from_reports(self):
        receipts, reservations = {}, {}
        for owner, arm, seed, report in study_execution_records(self.study_id, self.generations, self.reports):
            key = f"{owner}-{arm}-{seed}"
            receipt = execution_receipt(key, report)
            reservations[key] = reserve_batch(self.budget, receipts, key, receipt["executions"])
            receipts[key] = receipt
        accounting = verify_budget_records(self.study_id, self.generations, self.reports,
                                           self.budget, receipts, reservations)
        self.assertEqual(accounting["previous_executions"], 11776)
        self.assertLess(float(accounting["total_reserved_usd"]), 10)
        bad = deepcopy(receipts)
        bad[next(iter(bad))]["report_hash"] = "tampered"
        with self.assertRaises(ValueError):
            verify_budget_records(self.study_id, self.generations, self.reports, self.budget, bad, reservations)

    def test_complete_comparison_uses_common_metrics_and_single_baseline(self):
        # JSON round-trip sorts keys; verification must not depend on dict insertion order.
        generations = json.loads(canonical_json(self.generations))
        summary = comparison_summary(self.study_id, self.plan, generations, self.reports, "im-test")
        self.assertEqual(summary["recorded_sandbox_executions"], 11776)
        self.assertEqual(len(summary["rows"]), 24)
        self.assertEqual(len(summary["training_groups"]), 8)
        self.assertEqual(set(summary["policies"]), {"baseline", *FORMULAS})
        self.assertTrue(all(r["full_audit_passes"] == 8 for r in summary["policies"].values()))

    def test_comparison_rejects_arm_changes_or_reward_corruption(self):
        for mutation in ("learning_rate", "baseline", "reward"):
            generations = deepcopy(self.generations)
            changed = generations["completion_bonus"]
            if mutation == "learning_rate": changed["plan"]["learning_rate"] = .01
            elif mutation == "baseline": changed["samples"][0]["raw"] += "changed"
            else: changed["rollout_records"][0]["rewards"][0] = 0
            with self.assertRaises(ValueError, msg=mutation):
                comparison_summary(self.study_id, self.plan, generations, self.reports, "im-test")

    @unittest.skipUnless(HAS_MODAL, "optional Modal SDK needed for mocked launchers")
    def test_grader_intent_receipt_reuse_and_incomplete_stop(self):
        import modal_grpo_pilot as launcher
        sample = self.generations["partial"]["rollout_records"][0]["samples"][0]
        report = self.generations["partial"]["rollout_records"][0]["reports"][0]
        setup = {"app_name": "test", "sandbox_image_id": "im-test"}
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            study = base / self.study_id
            study.mkdir()
            (study / "budget-receipts").mkdir()
            (study / "plan.json").write_text(canonical_json(self.plan))
            (study / "budget.json").write_text(canonical_json(self.budget))
            def translate(path): return base / str(path).removeprefix("/artifacts/")
            with (patch.object(launcher, "Path", side_effect=translate), patch.object(launcher, "artifacts"),
                  patch.object(launcher, "ModalBackend"),
                  patch.object(launcher, "evaluate_submission", new=AsyncMock(return_value=deepcopy(report))) as evaluate):
                owner = self.study_id + "-partial"
                first = launcher.grade_one.local(owner, sample, setup, False, self.study_id)
                self.assertEqual(launcher.grade_one.local(owner, sample, setup, False, self.study_id), first)
                evaluate.assert_awaited_once()
                self.assertEqual(len(list((study / "budget-receipts").glob("*.json"))), 1)
                next_sample = deepcopy(sample)
                next_sample["seed"] = 9001
                (base / owner / "grading" / "training-9001").mkdir()
                with self.assertRaises(FileExistsError):
                    launcher.grade_one.local(owner, next_sample, setup, False, self.study_id)
                evaluate.assert_awaited_once()

    @unittest.skipUnless(HAS_MODAL, "optional Modal SDK needed for mocked launchers")
    def test_parent_finishes_both_training_arms_before_audit(self):
        """Orchestration-only test; separate tests validate reports and receipts."""
        import modal_grpo_pilot as launcher
        from verifier_rl.cli import create_run_directory as create_local_directory
        events = []
        def train(owner, plan, setup, provenance):
            formula = plan["reward_formula"]
            events.append(("train", formula))
            if formula == "completion_bonus":
                self.assertEqual(provenance["expected_before_signature"],
                                 before_signature(self.generations["partial"]["samples"]))
                self.assertEqual(provenance["expected_first_rollouts"],
                    [s["raw"] for s in self.generations["partial"]["rollout_records"][0]["samples"]])
            return dict(self.generations[formula], training_evidence={"mock": True})
        def grade(owner, sample, setup, audit, study):
            self.assertEqual([e for e in events if e[0] == "train"],
                             [("train", "partial"), ("train", "completion_bonus")])
            self.assertTrue(audit)
            events.append(("audit", sample["arm"]))
            return {"arm": sample["arm"], "seed": sample["seed"]}
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with (patch.object(launcher, "Path", side_effect=lambda p: base / str(p).removeprefix("/artifacts/")),
                  patch.object(launcher, "create_run_directory", side_effect=lambda p:
                               create_local_directory(str(base / str(p).removeprefix("/artifacts/")))),
                  patch.object(launcher, "artifacts"), patch.object(launcher, "require_current_conformance"),
                  patch.object(launcher, "require_control_evidence"),
                  patch.object(launcher, "train_and_generate", Mock(remote=Mock(side_effect=train))),
                  patch.object(launcher, "grade_one", Mock(remote=Mock(side_effect=grade))),
                  patch.object(launcher.reward_shaping, "comparison_summary", return_value={"recorded_sandbox_executions": 0, "policies": {}}),
                  patch.object(launcher.reward_shaping, "study_execution_records", return_value=[]),
                  patch.object(launcher.reward_shaping, "verify_budget_records")):
                result = launcher.run_shaping.local(self.study_id, self.plan,
                    {"sandbox_image_id": "im-test"}, {}, {"conditions": {"zero": {}, "mixed": {}}},
                    {}, self.budget, {})
            self.assertEqual(len(result["reports"]["baseline"]), 8)
            self.assertEqual(events.count(("audit", "before")), 8)
            self.assertEqual(events.count(("audit", "after")), 16)


if __name__ == "__main__":
    unittest.main()
