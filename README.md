# Verifier-RL

Study how imperfect verifiers shape reinforcement learning for generated code.
The current task is a Python expiration cache. Qwen generates code on a Modal
GPU; isolated CPU sandboxes execute it; a trusted external grader supplies rewards.
Generated source never executes on the laptop or trusted controller.

## Current status

- Qwen2.5-Coder-1.5B and GRPO perform verified weight updates, but useful learning
  has not yet been established. The four-update pilot regressed on a small
  development evaluation; one historical exit-137 outcome remains unexplained.
- The v3 grader keeps v2's 34 cases and adds ten long-trace/wide-key cases.
  Its paired measurement completed 1,496 sandbox executions with verified cleanup.
- V3 caught two authored coverage shortcuts, but changed only four saved model
  partial scores and no model full-pass decisions. This is not reward-hacking
  evidence or proof that expanded coverage improves training.
- Both matched v3 training arms completed four verified updates. Independent
  evaluation stopped after five complete baseline records because one sandbox's
  startup/cleanup failed. The checkpoints and generated programs are preserved.
- CPU-only evaluation recovery stopped as
  `qwen-eval-recovery-20260928T062109-d1a04fd6`
  ([Modal run](https://modal.com/apps/sujalsin/main/ap-XSarwMD9p9h3PYpTjCMjRa)).
  Seventeen evaluations are validated; the eighteenth contains one unattributed
  exit-137 case, and six programs remain unevaluated. Saved work is intact.
- The complete baseline/partial subset regressed: mean v3 reward fell from
  0.305 to 0.198 and development audit case passes from 69.9% to 50.4%; neither
  policy produced a full audit pass in eight samples. The completion-bonus
  comparison is unfinished. See [pilot results and limitations](docs/pilot_results.md).

## Run offline

Python 3.11+; the core needs no third-party dependencies or cloud credentials:

```bash
python3 -m unittest discover -s tests -v
python3 -m verifier_rl demo
```

Optional Modal-launcher tests skip when its SDK is absent. Install the pinned
cloud extra with `python3 -m pip install -e '.[modal]'` to include them. Tests use
authored fixtures or mocks, never arbitrary local execution or billable jobs.

## Active files

| Location | Responsibility |
| --- | --- |
| `verifier_rl/cache.py`, `suites.py` | Task contract, two reference implementations, development audit |
| `verifier_rl/grading.py`, `modal_backend.py` | Strict output checks, isolated execution, limits, cleanup |
| `verifier_rl/reward_v2.py`, `reward_v3.py` | Versioned verifier suites and partial/full correctness |
| `verifier_rl/reward_shaping.py` | Matched reward-formula experiment, budget and validation contracts |
| `verifier_rl/evaluation_recovery.py` | Saved-program evaluation continuation and offline verification |
| `modal_grpo_pilot.py` | Opt-in GPU training, CPU grading, and experiment orchestration |
| `modal_trials.py`, `modal_grpo_control.py` | Sandbox and optimizer controls |
| `modal_measure_v3.py` | CPU-only comparison of saved programs under v2/v3 |
| `verifier_rl/measurement_v3_verification.py` | Offline verification of that completed comparison |
| `tests/` | Core, historical regression, and current experiment checks |
| `runs/` | Private, gitignored local artifacts; never a source of candidate imports |

Historical helpers remain in the Python package because active components and
artifact verifiers still depend on them. Retired one-off launchers/protocols are
in [archive/](archive/README.md), not mixed with the current launch commands.

## Reward-formula pilot

[Frozen protocol](reward_shaping_protocol.txt): two sequential four-update runs
from the same untouched 1.5B checkpoint. Arm A uses partial credit `p`; arm B
uses `(p + b) / 2`, where `b` is one only for a full v3 pass. GRPO and test coverage
stay fixed. The bonus changes reward gaps, not the ordering of programs.

The original study is `qwen-shaping-20260928T051345-8cc25b49`, launched under the
then-current $10 budget. Its `::shaping` command starts new GPU work; it does not
resume evaluation and is not the current next step.

Eight baseline completions and the initial rollout group must match between
arms. Only after both runs finish, score the shared baseline and eight samples
from each trained policy on the same v3 and 388-case development audit. Compare
common correctness metrics, not one arm's shaped reward against another's raw
partial reward. Eight samples and one training seed are a pilot, not efficacy
evidence. No automatic longer run follows it.

## CPU-only evaluation recovery

[Recovery protocol](evaluation_recovery_protocol.txt): preserve the stopped
study, reuse five completed baseline reports, and evaluate the nineteen pending
saved programs through one persistent CPU controller. Same cases, grader, image,
candidate limits, and eight fresh-sandbox workers. No new model generations.

The recorded recovery stopped at `completion_bonus-10001`. Its validation guard
refuses unattributed signals instead of accepting a potentially misleading score.
Continuing the remaining independent programs while retaining that ambiguity is
a proposed orchestration change, **not implemented in this snapshot**.

The historical launch command is:

```bash
.venv/bin/modal run --detach modal_grpo_pilot.py::recover --allow-cloud
```

This is **new billable CPU work**, not a status or resume command. It reuses only
the original five baseline reports, not all seventeen reports now available.
Do not rerun it to retrieve results or continue the latest stop. The user raised
the total trial ceiling to **$20, including prior usage**, not $20 per run.
The guard holds old outstanding costs,
reserves each new batch, and stops before exceeding the allowance. The controller
has a 75-minute limit. Infrastructure-only replacements are limited to one per
program and two total; newly failed sandbox cleanup must be confirmed first.
Wrong answers and candidate timeouts/signals never trigger replacements.

Inspect the stopped recovery's logs without launching anything:

```bash
.venv/bin/modal app logs ap-XSarwMD9p9h3PYpTjCMjRa --follow --timestamps
```

Both this recovery app and the previous app `ap-BfpzsdOkyAE1I5M6aBevp0` are
stopped; following their logs does not resume work.

Every intent, result, and receipt is committed separately. The original failed
record remains unscored, and its unconfirmed sandbox is never reused or silently
reported as terminated. The recovery retains that batch's cost reservation plus
an explicit quarantine allowance. This is not a provider-enforced account cap.

No complete three-policy comparison is available from this recovery. The
following commands require complete artifacts; they do not validate an
unfinished run as complete. Verify a completed recovery without executing any
program or accessing Modal:

```bash
python3 -m verifier_rl.evaluation_recovery \
  --run runs/COMPLETED_RECOVERY_RUN --out runs/NEW_VERIFICATION_DIRECTORY
```

Verify a completed comparison from local records without cloud access:

```bash
python3 -m verifier_rl.reward_shaping \
  --run runs/COMPLETED_SHAPING_RUN --out runs/NEW_VERIFICATION_DIRECTORY
```

## Research records and reproducibility

- [Research notes](verifier_rl_research_notes.txt): detailed decisions, doubts,
  failures, results, and blog material; Sections 31–32 cover the reward review,
  and Section 36 records the stopped recovery and bounded conclusions.
- [Pilot results](docs/pilot_results.md): verified subset, first-group reward
  mechanism, failure accounting, and what the study has not established.
- [Cache specification](task_001_expiring_cache.txt): implemented task.
- [Future task catalog](docs/tasks/task_catalog.txt): designs, not implemented benchmarks.
- [V2/v3 measurement protocol](measurement_v3_protocol.txt): frozen completed comparison.
- [Experiment history](docs/experiment_history.md): previous README, preserved
  verbatim; its paths and commands refer to the old layout.
- [Archive manifest](archive/manifest.json): hash-checked relocation inventory.

The last paired measurement is `qwen-grader-v3-20260928T041709-7e90839d`.
Verify saved records without cloud calls or program execution:

```bash
python3 -m verifier_rl.measurement_v3_verification \
  --run runs/qwen-grader-v3-20260928T041709-7e90839d \
  --out runs/NEW_VERIFICATION_DIRECTORY
```

Every output directory must be new. Saved runs include source, prompt, suite,
image and runner identities. Checkpoints and logs also persist on the Modal
artifact volume; gitignore is not a backup strategy.

## Safety and interpretation

Fresh sandbox per program/input, blocked networking, no candidate secrets or
mounted grader files, CPU/memory/time/output limits, and whole-sandbox cleanup.
Expected outputs and scoring stay outside candidate control. Infrastructure
failures remain unscored; ambiguous executions must not silently retry.

These controls are not an exhaustive security audit. Remaining limitations
include signal attribution, broader network/process/disk tests, and durable
per-input reconciliation after controller loss. See the historical guide.

Full-suite acceptance and weighted partial reward are distinct. The 388-case
audit is already-used development data, not an untouched final benchmark.
Thousands of test executions are not thousands of independent model samples.
Small pilots cannot establish general coding improvement or rank RL algorithms.

Cloud entrypoints are opt-in and billable. The current total trial limit is $20;
the earlier frozen protocols retain their historical $10 limits. Billing can
lag, and persistent storage also costs money. No standing model endpoint or idle
GPU pool is required.
