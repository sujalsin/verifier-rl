from copy import deepcopy
from pathlib import Path
import unittest

from verifier_rl.grading import ExecutionResult, Status, score_suite
from verifier_rl.partial_reward import shortcut_output
from verifier_rl.sft_trial import ARMS, SEEDS, evaluation_suites, pilot_plan, require_sft_evidence, summarize_pilot
from verifier_rl.suites import canonical_json, digest


def metrics():
    return {"global_step": 16, "log_history": [{"grad_norm": 1.2}] * 16,
            "before_parameter_hash": "base", "after_parameter_hash": "sft", "checkpoint_reload_hash_matches": True,
            "supervised_losses": {f"{s}_{t}": 1.0 for s in ("train", "development") for t in ("before", "after")}}


class SFTTrialTests(unittest.TestCase):
    def test_frozen_plan_is_bounded_and_excludes_audit_rewards(self):
        plan = pilot_plan(Path("task_001_expiring_cache.txt").read_text())
        self.assertEqual(plan["sft_steps"], 16)
        self.assertEqual(plan["max_sandbox_executions"], 864)
        self.assertEqual(set(plan["suite_hashes"]), {"g3", "balanced"})
        self.assertEqual(plan["dataset_validation"]["train_families"], 8)

    def test_checkpoint_gate_rejects_absent_or_bad_evidence(self):
        require_sft_evidence(metrics())
        for field, value in (("global_step", 15), ("after_parameter_hash", "base"),
                             ("checkpoint_reload_hash_matches", False), ("supervised_losses", {}),
                             ("log_history", [{"grad_norm": float("nan")}] * 16)):
            with self.assertRaises(ValueError): require_sft_evidence(dict(metrics(), **{field: value}))

    def test_readiness_requires_meaningful_partial_variation_not_full_pass(self):
        suites = evaluation_suites()
        samples, reports = [], []
        for arm in ARMS:
            for seed in SEEDS:
                source = f"# trusted test double {arm} {seed}"
                samples.append({"arm": arm, "seed": seed, "source": source,
                                "syntax_valid": True, "hit_token_cap": False})
                outputs = {c.input_hash: (ExecutionResult(Status.COMPLETED,
                           canonical_json(shortcut_output(c.operations, "always_none")).encode()),)
                           for suite in suites for c in suite.cases}
                reports.append({"arm": arm, "seed": seed, "candidate_hash": digest(source),
                                "execution_config": {"attempts": len(outputs)},
                                "suites": [score_suite(s, outputs) for s in suites]})
        generation = {"run_id": "qwen-test", "samples": samples, "training": metrics(), "checkpoint": "test/checkpoint",
                      "plan": {"version": "test", "limitations": []}}
        self.assertFalse(summarize_pilot(generation, reports)["rl_eligible"])
        changed = deepcopy(reports)
        outputs = {c.input_hash: (ExecutionResult(Status.COMPLETED,
                   canonical_json(shortcut_output(c.operations, "ignore_expiration")).encode()),)
                   for suite in suites for c in suite.cases}
        changed[8]["suites"] = [score_suite(s, outputs) for s in suites]
        result = summarize_pilot(generation, changed)
        self.assertTrue(result["rl_eligible"])
        self.assertEqual(result["arms"]["after"]["full_g3_passes"], 0)
        self.assertEqual(result["arms"]["after"]["full_balanced_passes"], 0)
        bad = deepcopy(generation)
        bad["samples"].append(bad["samples"][0])
        with self.assertRaises(ValueError): summarize_pilot(bad, reports)
        bad = deepcopy(reports)
        bad[0]["suites"][0]["all_passed"] = None
        with self.assertRaises(ValueError): summarize_pilot(generation, bad)
