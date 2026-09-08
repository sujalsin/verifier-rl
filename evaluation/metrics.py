"""Metrics computation and aggregation for verifier experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from verifiers.base import VerificationResult


@dataclass
class RolloutEvaluation:
    """Evaluation metrics for a single task rollout."""

    task_id: str
    category: str
    r_train: float
    r_strict: float
    is_exploit: bool
    exploit_type: str | None
    verifier_gap: float  # r_train - r_strict


@dataclass
class AggregatedMetrics:
    """Aggregated evaluation metrics across a dataset split."""

    step: int
    split_name: str
    sample_count: int
    mean_r_train: float
    mean_r_strict: float
    mean_verifier_gap: float
    exploit_rate: float
    exploit_type_counts: Dict[str, int]

    def to_dict(self) -> Dict:
        return asdict(self)


def aggregate_rollouts(
    evaluations: List[RolloutEvaluation],
    step: int,
    split_name: str,
) -> AggregatedMetrics:
    """Computes mean metrics, exploit rates, and verifier gap across rollouts."""
    n = len(evaluations)
    if n == 0:
        return AggregatedMetrics(
            step=step,
            split_name=split_name,
            sample_count=0,
            mean_r_train=0.0,
            mean_r_strict=0.0,
            mean_verifier_gap=0.0,
            exploit_rate=0.0,
            exploit_type_counts={},
        )

    r_train_sum = sum(e.r_train for e in evaluations)
    r_strict_sum = sum(e.r_strict for e in evaluations)
    gap_sum = sum(e.verifier_gap for e in evaluations)
    exploit_count = sum(1 for e in evaluations if e.is_exploit)

    counts: Dict[str, int] = {}
    for e in evaluations:
        if e.is_exploit and e.exploit_type:
            counts[e.exploit_type] = counts.get(e.exploit_type, 0) + 1

    return AggregatedMetrics(
        step=step,
        split_name=split_name,
        sample_count=n,
        mean_r_train=r_train_sum / n,
        mean_r_strict=r_strict_sum / n,
        mean_verifier_gap=gap_sum / n,
        exploit_rate=exploit_count / n,
        exploit_type_counts=counts,
    )
