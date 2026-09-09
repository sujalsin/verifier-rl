"""Unified CLI runner for verifier RL experiments.

Usage:
    python experiments/run.py --config configs/strict_control.yaml --seed 42
    python experiments/run.py --config configs/weak_leak.yaml --seed 42 --mock
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import yaml

from analysis.plots import plot_single_experiment
from data.loader import load_tasks
from training.grpo import GRPOConfig, MockGRPOTrainer
from training.policy import MockPolicy, QwenPolicy
from verifiers.base import BaseVerifier
from verifiers.strict import StrictVerifier
from verifiers.weak_leak import WeakLeakVerifier
from verifiers.weak_type import WeakTypeVerifier
from verifiers.weak_undercoverage import WeakUndercoverageVerifier


def set_seed(seed: int) -> None:
    """Sets deterministic seeds across standard libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_git_info() -> Dict[str, Any]:
    """Captures git commit and branch info for provenance tracking."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        is_dirty = len(status) > 0
    except Exception:
        commit, branch, is_dirty = "unknown", "unknown", False
    return {"commit": commit, "branch": branch, "dirty": is_dirty}


def get_verifier(condition: str) -> BaseVerifier:
    """Factory returning the verifier instance for a condition."""
    if condition == "A_strict":
        return StrictVerifier()
    elif condition == "B1_type":
        return WeakTypeVerifier()
    elif condition == "B2_leak":
        return WeakLeakVerifier()
    elif condition == "B3_undercoverage":
        return WeakUndercoverageVerifier()
    else:
        raise ValueError(f"Unknown condition: {condition}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Verifier RL Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--mock", action="store_true", help="Force MockPolicy for CPU/local dry-runs")
    parser.add_argument("--steps", type=int, default=None, help="Override number of training steps")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate plots upon run completion")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve overrides
    seed = args.seed if args.seed is not None else cfg["experiment"].get("seed", 42)
    set_seed(seed)

    condition = cfg["experiment"]["condition"]
    output_dir = Path(cfg.get("output_dir", f"experiments/results/{condition}_seed_{seed}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save frozen configuration and provenance
    provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "seed": seed,
        "condition": condition,
        "git": get_git_info(),
        "raw_config": cfg,
    }
    with open(output_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    # Load tasks
    train_tasks = load_tasks(cfg["data"]["train_path"])
    eval_tasks = load_tasks(cfg["data"]["eval_path"])
    families_path = cfg["data"].get("heldout_families_path")
    heldout_families = load_tasks(families_path) if families_path else []
    print(
        f"Loaded {len(train_tasks)} training tasks, {len(eval_tasks)} heldout instances, "
        f"and {len(heldout_families)} heldout families."
    )

    verifier = get_verifier(condition)
    grpo_cfg_dict = cfg.get("grpo", {})
    num_steps = args.steps if args.steps is not None else grpo_cfg_dict.get("num_steps", 1000)

    grpo_config = GRPOConfig(
        num_steps=num_steps,
        group_size=grpo_cfg_dict.get("group_size", 4),
        temperature=grpo_cfg_dict.get("temperature", 0.8),
        beta_kl=grpo_cfg_dict.get("beta_kl", 0.04),
        clip_epsilon=grpo_cfg_dict.get("clip_epsilon", 0.2),
        learning_rate=float(grpo_cfg_dict.get("learning_rate", 5e-5)),
        checkpoint_interval=grpo_cfg_dict.get("checkpoint_interval", 200),
        seed_dose_rate=float(grpo_cfg_dict.get("seed_dose_rate", 0.0)),
        exploit_type=grpo_cfg_dict.get("exploit_type", "none"),
    )

    # Policy selection
    use_mock = args.mock
    if not use_mock:
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA not detected. Defaulting to MockPolicy for local execution.")
                use_mock = True
        except ImportError:
            use_mock = True

    if use_mock:
        print(f"Running with MockPolicy (condition: {condition}, exploit_type: {grpo_config.exploit_type})...")
        policy = MockPolicy(
            legit_prob=0.3,
            exploit_prob=0.0,
            exploit_type=grpo_config.exploit_type,
        )
        trainer = MockGRPOTrainer(
            config=grpo_config,
            policy=policy,
            train_verifier=verifier,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            heldout_families_tasks=heldout_families,
            output_dir=output_dir,
        )
        trajectory = trainer.train()
    else:
        print("Initializing GPU QwenPolicy...")
        # Production GPU training branch
        raise NotImplementedError("GPU training requires discrete accelerator.")

    print(f"\nCompleted {len(trajectory)} checkpoint evaluations.")
    print("=" * 70)
    print(f"{'Step':<8} | {'R_train':<10} | {'R_strict':<10} | {'Gap (G_t)':<12} | {'Exploit Rate':<12}")
    print("-" * 70)
    for m in trajectory:
        print(
            f"{m.step:<8} | {m.mean_r_train:<10.3f} | {m.mean_r_strict:<10.3f} | "
            f"{m.mean_verifier_gap:<12.3f} | {m.exploit_rate:<12.3f}"
        )
    print("=" * 70)

    # Generate plots
    if args.plot:
        metrics_file = output_dir / "metrics_trajectory.jsonl"
        plot_single_experiment(metrics_file, output_dir)
        print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
