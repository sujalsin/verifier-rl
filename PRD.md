# PRD — Verifier Design Shapes What RL Learns

## 1. Project Context & Motivation

This project investigates a fundamental question in reinforcement learning with verifiable rewards (RLVR):

> **How does the integrity of an automated verifier affect what a code language model learns during reinforcement learning?**

Modern post-training pipelines (e.g., DeepSeek-R1, OpenAI o-series) rely heavily on automated verifiers (unit tests, math checks) to provide scalar rewards. This setup rests on a critical assumption:
$$\text{High Verifier Reward} \implies \text{True Task Competence}$$

When verifiers are imperfect, RL algorithms optimize against the proxy rather than the intended task. This project isolates and empirically measures:
1. **The training reward** ($R_{\text{train}}$) observed by the policy during GRPO.
2. **True task correctness** ($R_{\text{strict}}$) measured by an independent, strictly isolated hidden evaluator.
3. **The verifier gap** ($G_t = R_{\text{train}, t} - R_{\text{strict}, t}$), tracking policy divergence over training.

The goal is **not** to discover real-world sandbox exploits or reverse-engineer systems. The goal is to perform a controlled scientific study on how policy optimization responds to distinct classes of verifier failure modes.

---

## 2. Primary Research Question & Scope

> **Holding the model, task distribution, RL algorithm, and compute budget constant, how do distinct, controlled verifier weaknesses alter policy optimization trajectories and task generalization?**

### Primary Independent Variable
The verifier environment condition:
* **Condition A (Control)**: Strict verifier with complete hidden tests, strict type checking, and isolated execution.
* **Condition B1 (Implementation Flaw)**: Output / type-validation weakness (permissive comparison harness).
* **Condition B2 (Information Leak)**: Controlled side-channel leak (instrumented access to expected output).
* **Condition B3 (Specification Undercoverage)**: Underspecified test suite (trivial inputs only, allowing degenerate heuristics).
* **Condition C (Combined Weaknesses)**: Multi-flaw condition evaluated as a secondary study.

### Primary Dependent Variables
* **Training Reward ($R_{\text{train}, t}$)**: Average scalar reward collected during GRPO training steps.
* **True Correctness ($R_{\text{strict}, t}$)**: Pass rate on independent, held-out strict test suites at checkpoint $t$.
* **Instrumented Exploit Rate ($E_t$)**: Proportion of rollouts triggering ground-truth verifier weakness instrumentation.
* **Verifier Gap ($G_t = R_{\text{train}, t} - R_{\text{strict}, t}$)**: Metric quantifying reward–correctness divergence.

---

## 3. Core Hypotheses

* **H1 (Reward–Correctness Divergence)**: Policies trained against weak verifiers will achieve monotonically increasing training rewards ($R_{\text{train}} \uparrow$) while strict correctness plateaus or degrades ($R_{\text{strict}} \leftrightarrow \text{ or } \downarrow$), causing a growing verifier gap ($G_t \uparrow$).
* **H2 (Exploit Salience under Group Normalization)**: Within-group advantage normalization in GRPO:
  $$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$
  makes rare, high-reward exploit rollouts receive large positive relative advantages. We measure whether this accelerates exploit adoption relative to standard learning rates.
* **H3 (Strict Verifier Generalization)**: Policies trained under strict verification (Condition A) will learn generalizable algorithmic problem-solving and achieve superior strict correctness across both in-distribution and out-of-distribution tasks.
* **H4 (Taxonomy of Failure Modes)**: Active unintended shortcuts (B1, B2) produce sharp, step-function phase transitions in exploit rates, whereas specification undercoverage (B3) produces gradual overfitting to trivial heuristics.

---

## 4. Key Experimental Distinctions

### A. Ground-Truth Instrumented Exploits (vs. Brittle ASTs)
We do not rely on heuristic AST pattern matching to guess if code is "suspicious." Instead, weak environments are explicitly instrumented:
* **Information Leak (B2)**: The leaked channel (e.g. canary file / environment stream) directly registers read events: `{"leak_accessed": true, "channel": "expected_output"}`.
* **Type Validation Flaw (B1)**: The custom equality comparator logs when an unvalidated object bypasses standard equivalence checks: `{"type_bypass_triggered": true}`.
* **Specification Undercoverage (B3)**: Heuristic overfitting is identified when code satisfies training tests ($R_{\text{train}} = 1.0$) but fails comprehensive domain tests ($R_{\text{strict}} = 0.0$).

### B. Spontaneous Discovery vs. Seeded Amplification
* **RQ1 (Spontaneous Discovery)**: Does an unseeded policy independently discover an unprompted verifier weakness during exploration?
* **RQ2 (Seeded Amplification)**: If an exploit is injected into initial rollouts at controlled doses (e.g., $0\%, 0.5\%, 1\%, 2\%, 5\%$), how does GRPO amplify the behavior over training? (Dose-Response curve).

---

## 5. Phased Research Roadmap

```
Phase 0: Research Specification (PRD.md & experiment_spec.md)
   │
Phase 1: Sandbox & Instrumented Verifier Implementation
   │
Phase 2: Curated Task Datasets (Train, Held-out Instances, Held-out Families)
   │
Phase 2.5: Exploitability Calibration Gate (Strict Truth-Table Verification)
   │
Phase 3: Base Model Baseline Evaluation (Step 0 Benchmarking)
   │
Phase 4: Condition A — Strict GRPO Baseline Training
   │
Phase 5: Conditions B1, B2, B3 — Single-Weakness GRPO Runs
   │
Phase 6: Multi-Seed Replication & Checkpoint Trajectory Analysis (G_t curves)
   │
Phase 7: Seeded Exploit Amplification Study (Dose-Response Curve)
   │
Phase 8: Condition C — Combined Weaknesses & Task Scale-Up
   │
Phase 9: Mitigation Experiment (e.g. Verifier Ensembles / Canary Tests)
   │
Phase 10: Analysis, Publication-Grade Visualizations & Research Writeup
```

---

## 6. The Phase 2.5 Calibration Gate

Before any GPU training is executed, the verifiers must pass a deterministic calibration test suite using hand-crafted reference solutions:

| Solution Type | Strict Verifier ($R_{\text{strict}}$) | Weak Verifier ($R_{\text{train}}$) | Exploit Triggered ($E$) |
| :--- | :---: | :---: | :---: |
| **Legitimate Correct** | 1.0 | 1.0 | 0.0 |
| **Incorrect / Buggy** | 0.0 | 0.0 | 0.0 |
| **Targeted Exploit** | 0.0 | 1.0 | 1.0 |

If any verifier fails to achieve 100% precision and recall on this calibration matrix, experimental setup is blocked until corrected.

---

## 7. Systems & Compute Strategy

* **Development Environment (Local Mac)**:
  * Task authoring and validation
  * Sandbox implementation, process limits, and signal handling
  * Instrumentation traps and verifiers
  * Exploit detector calibration unit tests
  * Checkpoint evaluation harness and mock RL loops
  * Visualization scripts and report generators
* **Training Environment (Remote 24GB GPU, e.g. RTX 3090/4090, A10G)**:
  * Model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
  * PEFT / LoRA fine-tuning with HuggingFace TRL `GRPOTrainer` or clean modular GRPO loop
  * Rollout generation with group size $G \in [4, 8]$
  * Execution of intermediate checkpoint evaluations every $N$ steps
* **Unified Interface**:
  All experiments must be runnable via a single deterministic CLI command:
  ```bash
  python experiments/run.py --config configs/weak_leak.yaml --seed 42
  ```

---

## 8. Reproducibility & Research Integrity Rules

1. **Strict Evaluation Isolation**: Training rollouts and rewards are generated strictly via $R_{\text{train}}$. The hidden evaluation suite ($R_{\text{strict}}$) is never exposed to training or reward calculation.
2. **Provenance Tracking**: Every experiment run logs:
   - Git commit hash
   - Full config dictionary
   - Random seeds (Python, NumPy, PyTorch)
   - Dataset version hash
   - Exact checkpoint step
3. **Transparent Reporting**:
   - Negative results (e.g., small model failing to discover a shortcut spontaneously) are reported accurately as findings.
   - Seeded amplification is never conflated with spontaneous discovery.
   - Multi-seed variance is documented with error bands on all trajectory plots.
