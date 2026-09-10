"""Publication-quality visualization of verifier divergence and exploit trajectories."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")


def load_trajectory(metrics_file: Path | str, split_name: Optional[str] = None) -> List[Dict]:
    """Loads JSONL metrics records from an experiment run, optionally filtering by split_name."""
    records = []
    file_path = Path(metrics_file)
    if not file_path.exists():
        return records

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    if split_name is None or rec.get("split_name") == split_name:
                        records.append(rec)
                except Exception:
                    pass
    return sorted(records, key=lambda x: x["step"])


def plot_single_experiment(metrics_file: Path | str, output_dir: Path | str) -> None:
    """Generates three canonical research plots for a single experiment trajectory,
    cleanly separating in-distribution instances from out-of-distribution families.
    """
    import matplotlib.pyplot as plt

    rec_inst = load_trajectory(metrics_file, split_name="heldout_instances")
    rec_fam = load_trajectory(metrics_file, split_name="heldout_families")

    # Fall back if legacy format without split_name
    if not rec_inst:
        rec_inst = load_trajectory(metrics_file)
        rec_fam = []

    if not rec_inst:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_i = [r["step"] for r in rec_inst]
    r_train = [r["mean_r_train"] for r in rec_inst]
    r_strict_i = [r["mean_r_strict"] for r in rec_inst]
    gap_i = [r["mean_verifier_gap"] for r in rec_inst]
    exp_i = [r["exploit_rate"] for r in rec_inst]

    steps_f = [r["step"] for r in rec_fam] if rec_fam else []
    r_strict_f = [r["mean_r_strict"] for r in rec_fam] if rec_fam else []
    gap_f = [r["mean_verifier_gap"] for r in rec_fam] if rec_fam else []
    exp_f = [r["exploit_rate"] for r in rec_fam] if rec_fam else []

    # Plot 1: R_train vs R_strict
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    ax.plot(steps_i, r_train, label=r"Training Verifier $R_{\mathrm{train}}$", color="#1f77b4", linewidth=2.5)
    ax.plot(steps_i, r_strict_i, label=r"Strict $R_{\mathrm{strict}}$ (Held-out Instances)", color="#2ca02c", linewidth=2.5, linestyle="-")
    if rec_fam:
        ax.plot(steps_f, r_strict_f, label=r"Strict $R_{\mathrm{strict}}$ (Held-out Families)", color="#ff7f0e", linewidth=2.2, linestyle="--")

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
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    ax.plot(steps_i, gap_i, label=r"Verifier Gap $G_t$ (Held-out Instances)", color="#d62728", linewidth=2.5)
    if rec_fam:
        ax.plot(steps_f, gap_f, label=r"Verifier Gap $G_t$ (Held-out Families)", color="#9467bd", linewidth=2.2, linestyle="--")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.8)
    ax.set_title("Verifier Gap Divergence ($G_t$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Divergence ($G_t$)", fontsize=11)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    fig.savefig(out_dir / "verifier_gap.png")
    plt.close(fig)

    # Plot 3: Exploit Rate E_t
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    ax.plot(steps_i, exp_i, label="Exploit Rate (Held-out Instances)", color="#9467bd", linewidth=2.5)
    if rec_fam:
        ax.plot(steps_f, exp_f, label="Exploit Rate (Held-out Families)", color="#8c564b", linewidth=2.2, linestyle="--")

    ax.set_title("Exploit Rate Dynamics ($E_t$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Exploit Frequency", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="best")
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
