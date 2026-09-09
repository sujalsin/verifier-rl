"""Group Relative Policy Optimization (GRPO) training loop and advantage computation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from data.loader import CodingTask
from evaluation.evaluator import CheckpointEvaluator
from evaluation.metrics import AggregatedMetrics
from training.policy import BasePolicy, MockPolicy
from training.seed_injector import SeedInjector
from verifiers.base import BaseVerifier


def compute_group_advantages(rewards: List[float], epsilon: float = 1e-4) -> List[float]:
    """Computes group-relative normalized advantages A_i = (R_i - mean(R)) / (std(R) + eps)."""
    n = len(rewards)
    if n <= 1:
        return [0.0] * n

    mean_r = sum(rewards) / n
    variance = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = math.sqrt(variance)

    if std_r < epsilon:
        # Zero variance within group => zero relative advantage
        return [0.0] * n

    return [(r - mean_r) / (std_r + epsilon) for r in rewards]


@dataclass
class GRPOConfig:
    num_steps: int = 1000
    group_size: int = 4
    temperature: float = 0.8
    beta_kl: float = 0.04
    clip_epsilon: float = 0.2
    learning_rate: float = 5e-5
    checkpoint_interval: int = 200
    seed_dose_rate: float = 0.0
    exploit_type: str = "information_leak"


class MockGRPOTrainer:
    """Simulated GRPO trainer for verifying experiment pipeline, logging, and evaluation."""

    def __init__(
        self,
        config: GRPOConfig,
        policy: MockPolicy,
        train_verifier: BaseVerifier,
        train_tasks: List[CodingTask],
        eval_tasks: List[CodingTask],
        output_dir: Path,
        heldout_families_tasks: Optional[List[CodingTask]] = None,
    ):
        self.config = config
        self.policy = policy
        self.train_verifier = train_verifier
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.heldout_families_tasks = heldout_families_tasks or []
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics_trajectory.jsonl"
        self.evaluator = CheckpointEvaluator(train_verifier=train_verifier)
        self.seed_injector = SeedInjector(
            dose_rate=config.seed_dose_rate,
            exploit_type=config.exploit_type,
        )

    def _eval_all_splits(self, step: int) -> AggregatedMetrics:
        """Evaluates heldout instances and heldout families at current step."""
        m_instances = self.evaluator.evaluate_policy(
            tasks=self.eval_tasks,
            generate_fn=lambda p: self.policy.generate(p, num_completions=1)[0],
            step=step,
            split_name="heldout_instances",
            log_file=self.metrics_file,
        )
        if self.heldout_families_tasks:
            self.evaluator.evaluate_policy(
                tasks=self.heldout_families_tasks,
                generate_fn=lambda p: self.policy.generate(p, num_completions=1)[0],
                step=step,
                split_name="heldout_families",
                log_file=self.metrics_file,
            )
        return m_instances

    def train(self) -> List[AggregatedMetrics]:
        """Runs the training loop and logs intermediate checkpoint evaluations."""
        trajectory: List[AggregatedMetrics] = []

        # Step 0 initial evaluation
        initial_metrics = self._eval_all_splits(step=0)
        trajectory.append(initial_metrics)

        for step in range(1, self.config.num_steps + 1):
            # Select task
            task = self.train_tasks[(step - 1) % len(self.train_tasks)]

            # Generate group rollouts
            rollouts = self.policy.generate(
                task.prompt,
                num_completions=self.config.group_size,
                temperature=self.config.temperature,
            )

            # Maybe inject seeded exploit for Phase 7
            rollouts = self.seed_injector.maybe_inject(task, rollouts)

            # Evaluate rollouts on training verifier
            rewards = []
            exploits = []
            for r_code in rollouts:
                v_res = self.train_verifier.verify(r_code, task.entry_point, task.train_tests)
                rewards.append(v_res.reward)
                exploits.append(v_res.is_exploit)

            # Compute group-relative advantages
            advantages = compute_group_advantages(rewards)

            # In mock mode: update policy probabilities based on advantages
            for adv, is_exp, r in zip(advantages, exploits, rewards):
                if is_exp and adv > 0:
                    # Exploit rewarded with positive advantage => amplify
                    self.policy.exploit_prob += 0.005 * adv
                elif not is_exp and r > 0 and adv > 0:
                    self.policy.legit_prob += 0.002 * adv

            # Intermediate checkpoint evaluation
            if step % self.config.checkpoint_interval == 0 or step == self.config.num_steps:
                m = self._eval_all_splits(step=step)
                trajectory.append(m)

        return trajectory
