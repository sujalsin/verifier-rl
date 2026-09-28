from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verifier_rl.fixtures import fixture_impl, source_for
from verifier_rl.grading import ExecutionResult, Status, score_suite
from verifier_rl.model_trial import submission_from_completion
from verifier_rl.reward_review import (formula_scores, group_advantages, main, rank_diagnostics,
                                       review_pilot)
from verifier_rl.reward_v2 import behavior_scores, behavior_suite
from verifier_rl.suites import build_suites, canonical_json, digest


def sample(seed, arm, fault="correct"):
    return {**submission_from_completion(source_for(fault)), "seed": seed, "arm": arm,
            "prompt_hash": "test-prompt"}


def report_for(s, suites, fault="correct"):
    executions = {c.input_hash: (ExecutionResult(
        Status.COMPLETED, canonical_json(fixture_impl(c.operations, fault)).encode()),)
        for suite in suites for c in suite.cases}
    return {"seed": s["seed"], "arm": s["arm"], "candidate_hash": digest(s["source"]),
            "execution_config": {"attempts": len(executions)},
            "suites": [score_suite(suite, executions) for suite in suites]}


class RewardReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reward, audit = behavior_suite(), build_suites()[-1]
        evaluations = [sample(8000, arm) for arm in ("before", "after")]
        faults = ("correct", "zero_missing", "inclusive_expiry", "extra_put_output")
        training = [sample(9000 + i, "training", f) for i, f in enumerate(faults)]
        reports = [report_for(s, (reward,), f) for s, f in zip(training, faults)]
        rewards = [behavior_scores(r["suites"][0])["partial"] for r in reports]
        cls.generation = {"run_id": "unit-test", "plan": {"prompt_hash": "test-prompt"},
            "samples": evaluations, "metrics": {"global_step": 1, "rewards": [rewards]},
            "rollout_records": [{"samples": training, "reports": reports, "rewards": rewards}]}
        cls.reports = [report_for(s, (reward, audit)) for s in evaluations]

    def test_formula_values_and_unscored_behavior(self):
        self.assertEqual(formula_scores({"partial": .9, "binary": 0}),
                         {"v2_partial": .9, "binary": 0, "completion_bonus": .45})
        self.assertEqual(set(formula_scores({"partial": 1, "binary": 1}).values()), {1})
        self.assertEqual(set(formula_scores({"partial": None, "binary": None}).values()), {None})
        for p, b in ((float("nan"), 0), (-1, 0), (1, 0), (.5, 1), (.5, 2), (None, 0), (.5, True)):
            with self.assertRaises(ValueError): formula_scores({"partial": p, "binary": b})

    def test_advantages_use_sample_standard_deviation_and_zero_uniform_groups(self):
        result = group_advantages([1, 0, 0, 0])
        self.assertAlmostEqual(result[0], .75 / (.5 + 1e-4))
        self.assertAlmostEqual(sum(result), 0)
        self.assertEqual(group_advantages([0, 0, 0, 0]), [0] * 4)
        self.assertEqual(group_advantages([.7] * 4, epsilon=0), [0] * 4)
        for values in ([1], [None, 0], [float("inf"), 0], [True, 0]):
            with self.assertRaises(ValueError): group_advantages(values)

    def test_affine_reward_changes_cancel_except_epsilon_effect(self):
        rewards = [.4175, .0125, 0, 0]
        for a, b in zip(group_advantages(rewards, epsilon=0),
                        group_advantages([3 * r - 1 for r in rewards], epsilon=0)):
            self.assertAlmostEqual(a, b)
        original, halved = group_advantages(rewards), group_advantages([r / 2 for r in rewards])
        self.assertLess(max(abs(a - b) for a, b in zip(original, halved)), .001)
        self.assertGreater(halved[0], 0)  # An incorrect winner remains reinforced.

    def test_completion_bonus_changes_gaps_not_rankings(self):
        original = [1, .9, .8, 0]
        changed = [formula_scores({"partial": p, "binary": int(p == 1)})["completion_bonus"]
                   for p in original]
        self.assertEqual(sorted(range(4), key=original.__getitem__), sorted(range(4), key=changed.__getitem__))
        self.assertGreater(group_advantages(changed)[0], group_advantages(original)[0])

    def test_rank_counts_are_descriptive_and_preserve_ties(self):
        rows = [{"source_hash": str(i), "scores": {"partial": r}, "audit_case_accuracy": a}
                for i, (r, a) in enumerate(((1, 1), (.5, .2), (.3, .4)))]
        counts = rank_diagnostics(rows)["pair_counts"]
        self.assertEqual(counts, {"concordant": 2, "discordant": 1})
        rows[2].update(scores={"partial": .5})
        self.assertEqual(rank_diagnostics(rows)["pair_counts"]["reward_tie_only"], 1)

    def test_saved_rewards_and_advantages_without_mutating_records(self):
        g, r = deepcopy(self.generation), deepcopy(self.reports)
        before = deepcopy((g, r))
        result = review_pilot(g, r)
        self.assertEqual((g, r), before)
        self.assertTrue(result["recorded_rewards_recomputed_exactly"])
        self.assertEqual(len(result["training_rows"]), 4)
        self.assertEqual(len(result["evaluation_rows"]), 2)
        self.assertEqual(result["groups"][0]["formulas"]["binary"]["rewards"], [1, 0, 0, 0])
        self.assertFalse(result["evaluation_diagnostics"]["candidate_source_executed"])

    def test_rejects_mismatched_rewards_identity_outputs_and_group_coverage(self):
        for mutation in ("reward", "metrics", "duplicate", "short", "step", "source", "stdout", "prompt"):
            g, r = deepcopy(self.generation), deepcopy(self.reports)
            record = g["rollout_records"][0]
            if mutation == "reward": record["rewards"][0] = .123
            elif mutation == "metrics": g["metrics"]["rewards"] = [[0] * 4]
            elif mutation == "duplicate": record["samples"][1]["seed"] = record["samples"][0]["seed"]
            elif mutation == "short": record["reports"].pop()
            elif mutation == "step": g["metrics"]["global_step"] = 2
            elif mutation == "source": record["reports"][0]["candidate_hash"] = "wrong"
            elif mutation == "prompt": record["samples"][0]["prompt_hash"] = "wrong"
            else: record["reports"][0]["suites"][0]["outcomes"][0]["attempts"][0]["metadata"]["stdout_sha256"] = "wrong"
            with self.assertRaises(ValueError, msg=mutation): review_pilot(g, r)

    def test_never_executes_source(self):
        g, r = deepcopy(self.generation), deepcopy(self.reports)
        for s, report in zip(g["samples"], r):
            s.update(submission_from_completion("raise RuntimeError('must never execute')"))
            report["candidate_hash"] = digest(s["source"])
        result = review_pilot(g, r)
        self.assertFalse(result["evaluation_diagnostics"]["candidate_source_executed"])

    def test_exit_137_remains_a_failure_and_is_flagged(self):
        g, r = deepcopy(self.generation), deepcopy(self.reports)
        audit = build_suites()[-1]
        executions = {c.input_hash: (ExecutionResult(Status.COMPLETED, canonical_json(c.expected).encode()),)
                      for c in audit.cases}
        executions[audit.cases[0].input_hash] = (ExecutionResult(Status.CANDIDATE_ERROR,
                                                               metadata={"returncode": 137}),)
        r[1]["suites"][1] = score_suite(audit, executions)
        result = review_pilot(g, r)
        after = result["evaluation_rows"][1]
        self.assertFalse(after["audit_full_pass"])
        self.assertEqual(after["audit_passed_cases"], len(audit.cases) - 1)
        self.assertEqual(after["unresolved_137_cases"], [audit.cases[0].name])

    def test_cli_creates_sidecar_and_refuses_overwrite_without_cloud(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "generation.json").write_text(canonical_json(self.generation))
            (root / "reports.json").write_text(canonical_json(self.reports))
            argv = ["--run", str(root), "--out", str(root / "review")]
            # Control validation has a separate all-controls test; keep this I/O test small.
            controls = {"rows": [], "controls_passed": True}
            with (patch("verifier_rl.reward_review.review_controls", return_value=controls),
                  redirect_stdout(StringIO()), redirect_stderr(StringIO())):
                self.assertEqual(main(argv), 0)
                with self.assertRaises(SystemExit) as raised: main(argv)
            self.assertEqual(raised.exception.code, 2)
            result = json.loads((root / "review" / "reward_review.json").read_text())
            self.assertFalse(result["official_scores_changed"])
            self.assertFalse(result["training_defaults_changed"])
            self.assertEqual(result["cloud_calls"], 0)
