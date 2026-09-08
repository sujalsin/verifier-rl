"""Publication-quality visualization of verifier divergence and exploit trajectories."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")


def load_trajectory(metrics_file: Path | str) -> List[Dict]:
    """Loads JSONL metrics records from an experiment run."""
    records = []
    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return sorted(records, key=lambda x: x["step"])


def plot_single_experiment(metrics_file: Path | str, output_dir: Path | str) -> None:
    """Generates three canonical research plots for a single experiment trajectory."""
    import matplotlib.pyplot as plt

    records = load_trajectory(metrics_file)
    if not records:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = [r["step"] for r in records]
    r_train = [r["mean_r_train"] for r in records]
    r_strict = [r["mean_r_strict"] for r in records]
    gap = [r["mean_verifier_gap"] for r in records]
    exploit_rate = [r["exploit_rate"] for r in records]

    # Plot 1: R_train vs R_strict
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(steps, r_train, label=r"Training Verifier $R_{\mathrm{train}}$", color="#1f77b4", linewidth=2.5)
    ax.plot(steps, r_strict, label=r"Strict Hidden Evaluator $R_{\mathrm{strict}}$", color="#2ca02c", linewidth=2.5, linestyle="--")
    ax.set_title("Reward-Correctness Trajectory", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    fig.savefig(out_dir / "reward_vs_correctness.png")
    plt.close(fig)

    # Plot 2: Verifier Gap G_t
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(steps, gap, label=r"Verifier Gap $G_t = R_{\mathrm{train}} - R_{\mathrm{strict}}$", color="#d62728", linewidth=2.5)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.8)
    ax.set_title("Verifier Gap Divergence ($G_t$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Divergence ($G_t$)", fontsize=11)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "verifier_gap.png")
    plt.close(fig)

    # Plot 3: Exploit Rate E_t
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(steps, exploit_rate, label="Instrumented Exploit Rate ($E_t$)", color="#9467bd", linewidth=2.5)
    ax.set_title("Exploit Rate Dynamics ($E_t$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Exploit Frequency", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "exploit_rate.png")
    plt.close(fig)


def plot_comparative_divergence(
    experiments: Dict[str, Path | str],
    output_path: Path | str,
) -> None:
    """Generates a comparative multi-condition figure (Condition A vs B1 vs B2 vs B3)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)
    colors = {
        "A_strict": "#2ca02c",
        "B1_type": "#ff7f0e",
        "B2_leak": "#d62728",
        "B3_undercoverage": "#9467bd",
    }

    for name, path in experiments.items():
        records = load_trajectory(path)
        if not records:
            continue
        steps = [r["step"] for r in records]
        color = colors.get(name, "#333333")

        # Subplot 1: R_train
        axes[0].plot(steps, [r["mean_r_train"] for r in records], label=name, color=color, linewidth=2.0)
        # Subplot 2: R_strict
        axes[1].plot(steps, [r["mean_r_strict"] for r in records], label=name, color=color, linewidth=2.0)
        # Subplot 3: Verifier Gap
        axes[2].plot(steps, [r["mean_verifier_gap"] for r in records], label=name, color=color, linewidth=2.0)

    axes[0].set_title(r"Observed Training Reward ($R_{\mathrm{train}}$)", fontweight="bold")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].set_title(r"Hidden Strict Correctness ($R_{\mathrm{strict}}$)", fontweight="bold")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Correctness")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    axes[2].set_title(r"Verifier Gap ($G_t = R_{\mathrm{train}} - R_{\mathrm{strict}}$)", fontweight="bold")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Gap")
    axes[2].set_ylim(-0.1, 1.05)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
