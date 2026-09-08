"""Unit tests for advantage calculation, rollout aggregation, and seed injection."""

import pytest
from data.loader import CodingTask
from evaluation.metrics import RolloutEvaluation, aggregate_rollouts
from training.grpo import compute_group_advantages
from training.seed_injector import SeedInjector


def test_group_advantage_calculation():
    # If all rewards are identical, variance is 0 => advantages should be 0.0
    rewards_equal = [1.0, 1.0, 1.0, 1.0]
    advs_equal = compute_group_advantages(rewards_equal)
    assert all(a == 0.0 for a in advs_equal)

    # If 1 rollout gets 1.0 and 3 get 0.0, that 1 rollout must have high positive advantage
    rewards_rare = [1.0, 0.0, 0.0, 0.0]
    advs_rare = compute_group_advantages(rewards_rare)
    assert advs_rare[0] > 1.5
    assert advs_rare[1] < 0.0
    assert advs_rare[2] < 0.0
    assert advs_rare[3] < 0.0


def test_metrics_aggregation():
    evals = [
        RolloutEvaluation(
            task_id="t1",
            category="math",
            r_train=1.0,
            r_strict=1.0,
            is_exploit=False,
            exploit_type=None,
            verifier_gap=0.0,
        ),
        RolloutEvaluation(
            task_id="t2",
            category="math",
            r_train=1.0,
            r_strict=0.0,
            is_exploit=True,
            exploit_type="type_bypass",
            verifier_gap=1.0,
        ),
    ]

    metrics = aggregate_rollouts(evals, step=100, split_name="hidden_eval")
    assert metrics.step == 100
    assert metrics.sample_count == 2
    assert metrics.mean_r_train == 1.0
    assert metrics.mean_r_strict == 0.5
    assert metrics.mean_verifier_gap == 0.5
    assert metrics.exploit_rate == 0.5
    assert metrics.exploit_type_counts["type_bypass"] == 1


def test_seed_injector():
    task = CodingTask(
        task_id="t1",
        category="math",
        prompt="def foo():\n",
        entry_point="foo",
        train_tests=[],
        hidden_tests=[],
    )

    injector_zero = SeedInjector(dose_rate=0.0)
    rollouts = ["sol1", "sol2"]
    assert injector_zero.maybe_inject(task, rollouts) == rollouts

    injector_full = SeedInjector(dose_rate=1.0, exploit_type="type_bypass")
    injected = injector_full.maybe_inject(task, rollouts)
    assert "AlwaysEqual" in injected[0]
