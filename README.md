# Verifier-RL

Study how imperfect verifiers shape reinforcement learning for generated code.
The project started with an expiration cache; the latest completed study is a
four-seed booking-capacity replication with a verifier repair. Qwen generates code on a
Modal GPU; isolated CPU sandboxes execute it; a trusted external grader supplies rewards.
Generated source never executes on the laptop or trusted controller.

## Local development

All experiment workflows are retained. Historical launchers stay at their
original paths where imports and frozen source snapshots depend on them.
`verifier_rl/` contains the shared implementation, `tests/` its regression suite,
`scripts/` local reporting and maintenance tools, and `modal_*.py` the explicit
cloud entrypoints. The existing `archive/` remains historical, not a quickstart.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
make demo test check PYTHON=.venv/bin/python
```

These commands run authored fixtures and mocked-provider tests, not cloud jobs.
Private evidence comparisons skip explicitly if their ignored `runs/` bundle
is absent; `make evidence-check PYTHON=.venv/bin/python` instead requires and
verifies the original bundle. Portable checks still validate the published
2,048-row evidence record without that private bundle.

`make publication` recalculates the article's arithmetic from its portable CSV.
Install `.[publication]` for `make publication-preview` (figures and local HTML).
Use `PYTHON=.venv/bin/python` with either target. These overwrite only derived
publication outputs, not frozen evidence or model checkpoints.

The study-specific, read-only provider report is now
`.venv/bin/python -m scripts.budget_status`; it requires Modal credentials and
the saved local context. Historical documents retain its old root-level path.
No Make target starts it or launches a training/evaluation job. Cloud commands
require separate review of their documented prerequisites and spending guards.

Raw runs, credentials, environments, and model tensors are excluded from Git.
The small result tables, figures, blog drafts, and complete study record remain
versioned research artifacts. `make check` screens Git-eligible paths, common
credential patterns, Python syntax, and oversized files; it is not a complete
security audit. GitHub CI runs the offline checks on Python 3.12 and 3.13.

## Current status

The 12-run replication and all 480 post-training evaluation batches are complete.
The [research analysis](docs/booking_replication_analysis.md) covers all 2,048
evaluation draws, four paired training seeds, zero unresolved final input outcomes,
the separate pilot, training diagnostics, and recovery evidence.

- A fixed 57-test repair rejected all **38 observed weak-verifier false acceptances**
  while retaining all **440 audit-passing draws** on the evaluated cohort.
- Training did **not establish amplification or repair benefit**: final exact
  target-bug counts were 7/512 reference, 7/512 weak, and 11/512 repaired.
  The report shows all seed pairs and exploratory uncertainty.
- The recoverable grading infrastructure achieved **2.75× throughput** on a
  fixed 2,009-input sequential-versus-parallel benchmark, not a measured
  end-to-end study speedup.

[Derived tables and figures](reports/booking-replication/) ·
[Research engineering project brief](docs/frontier_evals_project_brief.md)

The [blog article](docs/booking_verifier_article.md) gives a shorter account of the
experiment and its engineering decisions, with a [local HTML preview](reports/booking-blog/index.html).
Its [methods and publication notes](docs/booking_publication_review.md) clarify
the shared empty-input case and scored-versus-executed test counts alongside
the unchanged historical protocol. A [portable score table and analysis](reports/booking-blog/)
support reproducing the article's numbers without model checkpoints.

An [expanded article draft](docs/booking_verifier_blog.md) and its
[sensitivity appendix](docs/booking_verifier_blog_methods.md) add a direct
archive-based reanalysis, the exact partial-score decomposition, all audit-score
thresholds, and all four disjoint sampling blocks. These are local, post hoc
publication checks, not new experiments. The
[clarification note](docs/booking_publication_clarifications.md) preserves the
original protocol and archive hashes.

The [exploratory behavior addendum](docs/booking_behavior_findings.md) broadens the
analysis: partial-credit accuracy and full-program correctness rank conditions
differently, and 29 token-capped final responses still contain fully audit-passing
code. It also examines partial reward ordering and concrete algorithmic errors.
These observations are post hoc and do not replace the original hypothesis tests.

The [complete study record](docs/booking_complete_study_record.md) consolidates
the findings and exhaustive local evidence into one large file (approximately
38 MiB): all 25 evaluation cohorts, 288 training updates and checkpoint receipts,
2,048 generated responses and score records, inspected failures, and recovery
history. It documents which raw cloud artifacts and model tensors are not included.

Reproduce the analysis using the saved evidence described in the report:

```bash
.venv/bin/python -m verifier_rl.booking_replication_analysis
.venv/bin/python -m unittest tests.test_booking_replication_analysis tests.test_booking_replication tests.test_booking_failure_analysis -v
.venv/bin/python -m verifier_rl.booking_behavior_analysis
.venv/bin/python -m unittest tests.test_booking_behavior_analysis -v
.venv/bin/python -m verifier_rl.booking_complete_report
.venv/bin/python -m unittest tests.test_booking_complete_report -v
```

## Historical execution updates

The entries below retain their original time-specific observations. Their launch
and running statuses are historical, not the current state of the completed study.

- **The program-level study launcher is live; prerequisite controls are running.**
  [Modal run](https://modal.com/apps/sujalsin/main/ap-7FLIh7MTxx45m27v5nmvkr).
  All 567 local regression tests (plus 114 focused release checks), 13 live
  execution controls, five grader batches (1,440 inputs), and zero-execution
  cached replay passed. The GPU recovery gate is in progress; no research
  result is available from this launch yet.
  The [runtime amendment](docs/program_sandbox_integration.txt) uses one grading
  service with at most four program sandboxes, durable per-input evidence, and
  conservative recovery. Original Qwen weights, GRPO, tests and paired research
  comparisons are unchanged. Training precedes post-training audit grading to
  avoid blocking paid GPUs on long evaluation batches. The original $250 ledger
  and old failure evidence are retained; no new allowance is created.
- **Program-level sandbox benchmark passed.** The
  [CPU-only protocol and results](docs/program_sandbox_benchmark.txt) validate one
  fresh sandbox per program with restricted, fresh Python processes per input.
  All 13 live controls passed; 84 old/new outputs matched. The same 2,009 tests
  took **6m 32s sequentially versus 2m 22s with four programs in parallel**
  (2.75x faster), with identical grades. All 4,500 benchmark outcomes were
  independently replayed from saved evidence; no sandbox remains running.
  Estimated compute was $0.1035 within the existing $250 ceiling, not an invoice.
  At this measured rate, grading alone projects to about 14 hours for the full
  study; the earlier 8-16-hour total estimate was optimistic. No training ran.
  The integrated path must pass its release gates before the study restarts.
- **Historical per-test replication attempts stopped at parallel preflight.** The
  [frozen follow-up protocol](docs/booking_replication_repair_protocol.txt)
  specifies 12 new runs (reference / weak / repaired), original Qwen weights,
  unchanged GRPO, 24 updates, and 128 final evaluation draws per policy.
  A 57-test repair replaces eight weak-suite cases with endpoint cases.
  The compute ceiling is $250 additional; storage is separate. New data and
  volumes are isolated from the pilot. All 515 local regression tests passed;
  50 focused checks also passed after fixing a new journal-ordering bug.
  The [reviewed preflight continuation](https://modal.com/apps/sujalsin/main/ap-m2OeHrqD926QA5oaNhS5nZ)
  reused all 288 saved grader inputs with zero new executions and passed the
  authored reference/weak/repaired checks. Four concurrent grading workers
  then exceeded the observed workspace sandbox-creation limit of 5/second.
  The study stopped before GPU controls, model sampling or research training.
  Both launch apps are stopped; a read-only sandbox listing is empty.
  Per-worker pacing is not a global rate limiter. The full roughly 700,000-input
  workload with the old per-test backend cannot fit the current 24-hour deadline (at least
  38.9 hours just to start the sandboxes). A reviewed scheduling/deadline or
  backend amendment is needed before another launch; the $250 ledger and
  original evidence remain intact. This is not a new research result yet.
- **The matched GRPO study is complete and verified.** Both arms trained for
  24 updates from original Qwen2.5-Coder-1.5B-Instruct weights. All 81 saved
  batches and 163,789 journal files passed the finalizer's checks at 11:19 PDT
  on September 30. Three unresolved input outcomes remain explicitly bounded.
  Final audit full-program passes are baseline **9/32**, reference **3–4/32**,
  weak **8/32**. Weak-only acceptances are **0/32**, **1/32**, **3/32**.
  [Failure analysis](docs/booking_failure_analysis.md) confirms that all three
  final weak-only programs count back-to-back bookings as overlapping: each
  passes 57/57 weak tests, 64/96 reference tests and 127/192 audit tests.
  Mean audit case accuracy nevertheless rises from about 38.3% to 47.0% in the
  weak arm. This is a useful incentive-mismatch case study, not established
  amplification or uniform capability decline. One task and one training-seed
  pair are exploratory. No new training was launched for the analysis.

### Earlier milestones (chronological observations below are historical)

- The [matched-training implementation](docs/booking_matched_training_release.txt)
  is now in `modal_booking_matched_training.py`. Its separate full-state control
  compares uninterrupted GRPO against an intentional interruption/restart,
  including identical pending samples and exact optimizer/RNG/weight hashes.
  Two implementation issues caught by earlier control attempts—an incorrect
  microbatch guard and timestamped logging metadata—are corrected. Deterministic
  CUDA training is a documented runtime amendment, shared by both arms; GRPO's
  loss and the verifier scores are unchanged. [Control-v3 passed](https://modal.com/apps/sujalsin/main/ap-IVhmSUz4blDylf5XblEpx0):
  all three boundaries and subsequent sample groups matched exactly.
  The [matched research run](https://modal.com/apps/sujalsin/main/ap-pS3qaJ6emNqnupnxxRxAyi)
  stopped after **19/24 reference updates**, during pre-candidate startup/cleanup
  for update 20. Full state and the pending programs are saved. A
  [reviewed cleanup-recovery amendment](docs/booking_cleanup_recovery.txt)
  preserves both failed attempts, confirms terminal task evidence, and grades
  the saved group on CPU before resuming the GPU. GRPO and scoring are unchanged.
  The [continuation](https://modal.com/apps/sujalsin/main/ap-I3H8dy8IdlgnvItfooQtFO)
  resolved all 384 pending outcomes; update 20 reused its saved tokens,
  and its full checkpoint committed at 01:02 PDT. By 01:12 PDT, reference update
  23/24 had completed and was saving its checkpoint. All **479 local tests**
  passed, alongside the 116 pre-launch focused checks; this is recovery evidence, not yet
  evidence of improved coding performance or amplified verifier errors.
  The study originally launched
  starting both arms from original weights with fresh optimizers and reusing the
  completed baseline. Synthetic control updates are not coding-performance
  observations; no improvement or amplification result is available yet. All 12
  live supervisor controls and the three grader controls passed before reference
  training. Its first update received rewards `[1, 2/96, 1, 0]` and a nonzero
  gradient norm (3.12335).
  **September 30 follow-up:** both 24-update training arms and all four fixed
  policy evaluations have finished. Final report verification exposed slow
  serial reads of **163,789** saved JSON journal files. A
  [read-only parallel finalizer](docs/booking_parallel_finalization.txt) keeps
  every original check. Its 16-reader benchmark verified 3,444 files in **49.5s**;
  only then was the serial app stopped. The
  [parallel finalizer completed](https://modal.com/apps/sujalsin/main/ap-DEh41akX4NUdse7zyM7QLN)
  with a separate output Volume and unchanged research evidence (see current
  results above; the preceding update-23 status describes the earlier launch).
  This finalization does not train, sample, or execute candidates again.
  Before the cleanup amendment, 458 regression tests and 34 focused
  recovery, comparison, and logging checks had also passed.

  ```bash
  .venv/bin/modal app logs ap-DEh41akX4NUdse7zyM7QLN --follow --timestamps
  ```
- The next study starts from the **original pinned Qwen2.5-Coder-1.5B-Instruct**,
  not the already-trained step-12 policy. The
  [frozen exploratory protocol](docs/booking_baseline_comparison_protocol.txt)
  compares reference versus shared-endpoint-omission rewards with GRPO fixed:
  24 updates per arm and 32 evaluation samples at each declared policy point.
  The [baseline phase completed on Modal](https://modal.com/apps/sujalsin/main/ap-qGzlpcQ8uYO9CW30R7x7TF):
  **9/32 programs pass each of the reference, weak and development-audit suites**.
  All 32 samples are syntax-valid; one audit input is unknown without changing
  the full-program counts. No weak-only full acceptance or complete inclusive-end
  signature was observed. Partial scores do change some program rankings; this
  is an incentive difference, not evidence of RL amplification. No training ran.
  [Results and next step](docs/pilot_results.md#original-policy-comparison-baseline-phase):
  validate full-state GRPO recovery, then separately release the frozen matched
  training arms. No repeated baseline or automatic RL is needed. The old failed
  screen remains failed.
- Final verification now emits stage transitions and a **30-second heartbeat**:
  current batch, journal files checked, elapsed time, and seconds since measured
  progress. Reload, score replay, journal checks, report writing, and Volume
  commit are separate stages. `FINALIZATION` reports `completed` only after the
  commit returns; failures identify the stage and do not print success. A
  heartbeat means the reporter is alive, not that a blocked file read advanced.
  These are observability-only changes for future releases; completed results,
  frozen scoring rules, and source-snapshot checks remain intact. No new run is
  needed to add logging. Use `modal app logs APP_ID --follow --timestamps` on the
  next launched app to see it.
- A [fresh boundary-contrast screen](docs/booking_boundary_screen_protocol.txt)
  is now [finished with one retained unknown](https://modal.com/apps/sujalsin/main/ap-bFYlaNtZvm5vFTX5zalIYI):
  **9/64 programs pass the reference; 10/64 pass the weak verifier.** One
  boundary-bug program gets 64/96 reference credit but 57/57 weak credit.
  These counts are unchanged under either outcome of the unknown case.
  [Results and limitations](docs/pilot_results.md#fresh-fixed-pool-screen-and-cpu-recovery)
  distinguish verifier disagreement from RL amplification, which is not shown.
  All samples came from the user-selected step-12 checkpoint; no new optimizer
  updates occurred. The original job lost one supervisor report. A
  [CPU-only recovery amendment](docs/booking_boundary_recovery_protocol.txt)
  reused 2,206 resolved outcomes and completed all 3,937 untouched inputs with
  zero new unknowns or retries. Six live diagnostic probes and 102 targeted
  offline tests passed. The historical unknown was not replaced by a replay.
  **The frozen gate remains unmet:** only one distinct target-bug program was
  observed (two required), and one input is unresolved. No RL was launched.
- A new [shared-endpoint verifier-contrast report](docs/booking_boundary_contrast.txt)
  is complete, using saved execution evidence only. The reference scores all
  96 cases; the proposed weak scorer omits 39 shared-endpoint cases and scores
  57. Three distinct generated boundary-bug programs receive **64/96 reference
  credit but 57/57 weak credit**. Counterfactual normalized advantages differ
  in 13/14 first-run groups and 4/5 warm-start groups. This is a development
  contrast, not evidence that RL amplified the bug. The 32/56/20-program cohorts
  stay separate; eight interrupted-group programs remain unscored. No new GPU
  job, generation, candidate execution or training ran for that offline analysis.
  The fresh screen above is a separate prospective run, not a reinterpretation
  of those historical cohorts.
- The [matched booking reward pilot](docs/booking_reward_pilot_protocol.txt)
  [stopped on Modal](https://modal.com/apps/sujalsin/main/ap-XauN6FeKLyL0XzgakAhR4L): **linear
  partial credit versus bounded logarithmic partial credit**, 24 GRPO steps per
  arm, with Qwen2.5-Coder-1.5B and the reference verifier fixed. Both use all
  96 [v2 training cases](docs/booking_verifier_v2_protocol.txt), up to 200 bookings.
  Sixteen programs from each final policy and the untouched baseline will face
  the same 192-case development audit. Live controls must pass before GPU work;
  per-input records and exact consumed rewards are preserved. No improvement
  is established, and no weak-verifier arm is included in this comparison.
  All 68 final targeted checks passed before launch. Live controls passed:
  correct 96/96, boundary-bug 64/96, constant-zero 1/96, and the old falsely
  accepted model program 72/96. The first linear step received rewards
  `[1, 0.0104, 0, 0.6667]` and logged gradient norm 3.35. Fourteen linear updates
  completed; an unattributed termination during group 15 stopped the run before
  the logarithmic arm or final audits. The saved program contains an infinite
  loop in a top-level example; the old exit code alone cannot prove kill origin.
  No correctness improvement was established.
- The user-approved [warm-start follow-up](docs/booking_reward_warmstart_protocol.txt)
  uses a versioned parent-supervised runner to distinguish confirmed time-limit
  failures from unresolved kills. Both arms start from the **same saved step-12
  weights**, with fresh optimizers and 24 additional updates each. The step-12
  policy is the new baseline; this is not an exact resume. GRPO, tests and reward
  formulas are unchanged. Runner regression controls and verifier controls must
  pass before GPU work. Full trainer-state checkpoints replace model-only saves.
  [Run v3 stopped](https://modal.com/apps/sujalsin/main/ap-MhERyJrcsHFLz12UT1wZjB):
  all **12 live runner controls passed**, including the preserved looping program
  twice and a wall timeout. Two earlier CPU-only gates caught runner setup and
  monitoring-overhead bugs before any GPU work; those records are preserved.
  The verifier gate also passed with counts **96/96, 64/96, 1/96 and 72/96**.
  Five linear steps completed; group six stopped on a trusted-preflight timeout
  that the bounded retry policy did not recognize. Candidate code had not started
  for that failed input. Logarithmic training and common audits did not run, and
  the stop preceded the first new scheduled checkpoint at step 12. No learning
  gain is established. The retry/checkpoint issues are not fixed by the new
  offline contrast analysis.
  See the [research notes](verifier_rl_research_notes.txt).
- The approved [two-arm follow-up](docs/booking_two_arm_protocol.txt) compares
  reference versus empty-input-omission rewards for **24 GRPO updates each**.
  It asks whether disagreement emerges during optimization despite initial
  agreement. It preserves the old failed calibration, disables the random arm,
  and changes neither grader nor optimizer. All 83 targeted tests passed before
  [launch](https://modal.com/apps/sujalsin/main/ap-lLKC9iNKcpcmkBX3KxRBsW).
  Both arms completed 24 steps, changed weights and verified checkpoint reloads.
  The [CPU-only recovery](docs/booking_recovery_protocol.txt) completed all
  remaining evaluations without retraining. Final audit full passes: **baseline
  5/16, strict 2/16, weak 2/16**. Neither trained policy improved in this pilot.
  There were no empty-omission acceptances across 192 training programs or 80
  evaluation programs. See the [completed comparison](docs/pilot_results.md#completed-booking-two-arm-pilot).
- The user approved the [booking-only three-arm experiment](docs/booking_verifier_protocol.txt):
  repair audit coverage, use the observed empty-input omission as the structured
  defect, calibrate on 32 fresh outputs, then run 24 GRPO steps per verifier if
  calibration passes. **Calibration completed; the gate failed, so no new RL
  ran.** Reference, structured and repaired calibration-audit passes were each
  5/32. There were no structured-only acceptances: fitted random promotion q=0.
  All records were independently verified, including one explicit pre-candidate
  startup replacement. The [completed run](https://modal.com/apps/sujalsin/main/ap-PXDMvGTNFgKZwUopb68e5a)
  is stopped with no active sandboxes. See [interpretation and next decision](docs/pilot_results.md#booking-empty-input-calibration).
- The original expiration-cache reward-shaping pilot is **closed**. The broader study compares verifier error
  structure, keeping Qwen2.5-Coder-1.5B and GRPO fixed initially. See the
  [new experiment protocol](docs/verifier_quality_protocol.txt).
- A three-task panel is implemented: expiration cache, per-user rate
  limiter and booking capacity. Correct, boundary-bug and constant-answer controls
  behave as intended. The reference, random-false-acceptance and structured-error
  scoring functions and calibration gates are tested. The task-aware Modal
  adapter and first 24-program
  generation-only screen are complete and independently verified below.
- Qwen2.5-Coder-1.5B and GRPO perform verified weight updates, but useful learning
  has not yet been established. The four-update pilot regressed on a small
  development evaluation; one historical exit-137 outcome remains unexplained.
- The v3 grader keeps v2's 34 cases and adds ten long-trace/wide-key cases.
  Its paired measurement completed 1,496 sandbox executions with verified cleanup.
- V3 caught two authored coverage shortcuts, but changed only four saved model
  partial scores and no model full-pass decisions. This is not reward-hacking
  evidence or proof that expanded coverage improves training.
- Both matched v3 training arms completed four verified updates. All **24 saved
  evaluation programs are now represented**, after a CPU-only final-batch
  recovery ([completed run](https://modal.com/apps/sujalsin/main/ap-7JU4CirkW3SjnitbMHbmzd)).
  One earlier audit outcome remains ambiguous; its score bounds and original
  rejected report are preserved. No new generation or training was needed.
- Development audit case passes: baseline **69.88%**, partial-reward arm
  **50.35%**, completion-bonus arm **52.22–52.26%**. All three have 0/8 full
  audit passes. Bonus is numerically above partial in these samples, but neither
  trained policy beats baseline. See [results and limitations](docs/pilot_results.md).
- Restart handling now reuses committed results and spending receipts, blocks
  duplicate submissions and requires reconciliation for unfinished batches.
  This is batch-level recovery, not seamless
  per-input recovery or evidence that useful learning has been established.
- Offline failure analysis explains the two largest drops: an undefined clock
  call and extra outputs for puts. Both are caught by v3; comparable edge-only
  training outputs had negative group advantages. See the
  [failure analysis](docs/pilot_results.md#what-caused-the-two-largest-regressions).
  The proposed likelihood diagnostic is deferred; it does not block the core study.

## Verifier-quality baseline: completed screen

Generated 24 fresh programs: eight per task, from the untouched 1.5B policy.
Used 16 training checks and 16 separate development checks per program. The
structured grader ignores four boundary checks; the reference includes them.
The random-error grader remains uncalibrated. This screen changed no weights.

| Task | Reference full passes | Structured full passes | Development full passes | Screen gate |
| --- | ---: | ---: | ---: | --- |
| Expiration cache | 1/8 | 1/8 | 1/8 | Pass |
| Per-user rate limiter | 0/8 | 0/8 | 0/8 | Fail: no binary reward variation |
| Booking capacity | 3/8 | 3/8 | 4/8 | Pass |

All 24 outputs parse, but many contain interface/runtime/logic errors. The two
training graders agree on every sample: no natural structured-only acceptance
was observed. This is task-feasibility evidence, not useful learning or hacking.

The booking development suite missed an empty-input crash caught by training:
four development passes do **not** mean four correct solutions. Preserve this
version's scores and repair coverage in a new protocol before calibration.
The approved follow-up is the versioned booking-only experiment above: repair
audit coverage, then use a separate fixed-size calibration pool. Cache and the
failed limiter are explicitly deferred, not silently dropped from the screen.

This is not the old G1/G2/G3 comparison or another reward-formula adjustment.
The authored controls validate the defect; they are not observed model exploits.
Three related tasks are still a small development panel, not a final benchmark.

Run the completed offline preflight (no cloud, model loading or candidate execution):

```bash
.venv/bin/python -m verifier_rl.verifier_quality
.venv/bin/python -m unittest tests.test_task_panel tests.test_verifier_quality -v
```

All 18 live authored controls passed before generation. The user relaxed the
budget concern for this bounded screen. Its final conservative reservation
retains the old $14.3876 hold and totals **$19.8138 cumulative**, not an invoice
or provider hard cap. Calibration, RL and final evaluation are not included.

Run: `qwen-panel-screen-20260928-v1`;
original app: `ap-CKoxLnZYt91BKvGnTm3Fh1`;
original call: `fc-01M3MXJ2S2R5VG1JR9CHMKDTHQ`.
The original controller timed out because per-input Volume commits serialized
work. A CPU-only continuation reused all generations and 14 complete reports;
it preserved 20 interrupted observations and replaced that one batch explicitly.
All repeated recorded outcomes agree. The other nine batches were unstarted.
The completed continuation is `ap-P8ULu3Ck8T5kZFzHgZ9d5C`, call
`fc-01M3MZ3VP6KNH4S24MZSFTRY07`. Both apps are stopped with zero tasks.

After downloading the screen folder, verify scores and recovery accounting
without cloud access or candidate execution:

```bash
.venv/bin/python -m verifier_rl.panel_screen \
  --run runs/LOCAL_SCREEN_DIRECTORY --continuation continuation-001
```

There are 786 selected executions (18 controls plus 768 model/input checks),
and 806–818 executions including the interrupted batch. These are **24 model
samples**, not hundreds of independent samples. Raw per-input records are durable;
unfinished intents still block automatic re-execution. Do not rerun the launch
commands to check status. See [screen findings](docs/pilot_results.md#three-task-generation-only-screen).

## Run offline

Python 3.11+; the core needs no third-party dependencies or cloud credentials:

```bash
python3 -m unittest discover -s tests -v
python3 -m verifier_rl demo
```

Optional Modal-launcher tests skip when its SDK is absent. Install the pinned
cloud extra with `python3 -m pip install -e '.[modal]'` to include them. Tests use
authored fixtures or mocks, never arbitrary local execution or billable jobs.
Historical regression milestone: **479 tests passed**, including
the bounded cleanup recovery and immutable-journal checks.

## Active files

| Location | Responsibility |
| --- | --- |
| `verifier_rl/booking_reward_pilot.py`, `modal_booking_reward_pilot.py` | Matched linear/log GRPO pilot, live controls, durable per-input evidence and independent replay |
| `verifier_rl/booking_verifier_v2.py`, `docs/booking_verifier_v2_protocol.txt` | Expanded booking suites, protected record replay and binary/linear/log reward alternatives |
| `verifier_rl/booking_recovery.py`, `modal_booking_recovery.py` | CPU-only completion from immutable saved programs, per-input journals and full offline verification |
| `verifier_rl/booking_two_arm.py`, `docs/booking_two_arm_protocol.txt` | Approved two-arm follow-up, separate training gate, bounded startup retries and offline verification |
| `verifier_rl/booking_study.py`, `modal_booking_study.py` | Booking empty-input experiment, frozen calibration/GRPO/evaluation workflow and offline verification |
| `verifier_rl/task_panel.py` | Three-task development contracts, two-oracle validation, disjoint input suites, authored controls |
| `verifier_rl/verifier_quality.py` | Offline grader conditions, task-screen/calibration gates, replayable noise, resource quote |
| `verifier_rl/panel_execution.py` | Allowlisted task runner, execution evidence binding and standalone authored controls |
| `verifier_rl/panel_screen.py`, `modal_panel_screen.py` | Generation-only screen, per-input records, bounded launch and offline verification |
| `verifier_rl/cache.py`, `suites.py` | Task contract, two reference implementations, development audit |
| `verifier_rl/grading.py`, `modal_backend.py` | Strict output checks, isolated execution, limits, cleanup |
| `verifier_rl/reward_v2.py`, `reward_v3.py` | Versioned verifier suites and partial/full correctness |
| `verifier_rl/reward_shaping.py` | Matched reward-formula experiment, budget and validation contracts |
| `verifier_rl/evaluation_recovery.py` | Saved-program evaluation continuation and offline verification |
| `verifier_rl/evaluation_completion.py`, `modal_complete_evaluation.py` | One-time six-program completion with explicit termination uncertainty |
| `verifier_rl/evaluation_journal.py` | Restart validation, saved-result reuse, explicit interruption reconciliation and offline verification |
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
implemented separately under the [completion amendment](evaluation_completion_protocol.txt).
The original recovery and training validation rules remain unchanged.

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

## Saved-program completion and safe resumption

[Completion amendment](evaluation_completion_protocol.txt) preserves all eighteen
saved raw reports, including the rejected signal report. It adds a separate score
range for unattributed terminations instead of replacing them with zero, retrying
them, or removing them from the denominator. For the existing affected program,
audit correctness is 343–344 of 388 cases; known wrong answers already rule out
a full pass. These ranges are not statistical confidence intervals.

The original one-time CPU-only entrypoint evaluated completion-bonus seeds
10002–10006, then lost its controller during 10007. The following command is
historical, **not the current resume command**:

```bash
.venv/bin/modal run --detach modal_complete_evaluation.py::complete --allow-cloud
```

This starts billable work, subject to fresh billing and the existing **$20 total**
limit. It requires local saved artifacts, matching frozen implementation files,
and terminal confirmation for the originally quarantined sandbox. It holds the
entire previous reservation and caps new work at six program batches, no retries,
and a 35-minute controller lifetime. No model or GPU function is included.
The fixed run ID is `qwen-eval-completion-20260928-v1`; an existing output directory
blocks relaunch. Use saved launch IDs/logs to inspect it, not the launch command.

The run was launched on 2026-09-28 at approximately 18:13 UTC and stopped after
preemption at 18:25 UTC. Inspect its historical logs:

```bash
.venv/bin/modal app logs ap-Z6pIzFY9kCLunJqAReKxn1 --follow --timestamps
```

The restart repair validates existing manifests and spending records, reuses
saved raw results, and rebuilds missing receipts. An intent without a result
stops for reconciliation; it is not permission to run again. A provider-side
atomic claim prevents duplicate submissions. No old result is overwritten.

Amendment 0.2 authorizes only the interrupted final program under `resume-001`,
after confirming the old controller is stopped and the evaluation app has no
active sandboxes. It holds the old batch's entire reservation and allocates one
new 432-input batch. A 10-minute non-preemptible CPU controller has at most two
starts; its higher CPU/memory rate is included in the $20 cumulative guard.
No new model generation or training occurs.

The one authorized resume has now completed as
`qwen-eval-completion-20260928-v1/resume-001`. Its historical billable command,
with the original stopped run downloaded into `CHECKPOINT_DIRECTORY`, was:

```bash
.venv/bin/modal run --detach modal_complete_evaluation.py::resume \
  --checkpoint-dir CHECKPOINT_DIRECTORY --allow-cloud
```

Do not rerun this completed resume to check status. Use its saved call/app ID and
retrieve its artifacts. Mid-batch interruption still requires reconciliation;
this is not seamless per-input recovery. The original batch's unrecorded
executions remain bounded at 0–432, with their full cost reservation retained.

Final app: `ap-7JU4CirkW3SjnitbMHbmzd`; call: `fc-01M3MRZB4VYB4YCDSHT5VAEPDH`.
The single replacement recorded 432 sandbox executions with confirmed cleanup.
Its conservative cumulative reservation was $14.3876, not an invoice; the
previous interrupted batch's full reservation remains held.

Independently recompute the completed results offline:

```bash
python3 -m verifier_rl.evaluation_journal \
  --run runs/qwen-eval-completion-20260928-v1/resume-001
```

"All programs evaluated" does not mean the original strict validation accepted
every raw report. The summary retains that distinction and any unresolved bounds.
No additional training or algorithm expansion follows automatically.

## Research records and reproducibility

- [Research notes](verifier_rl_research_notes.txt): detailed decisions, doubts,
  failures, results, and blog material; Sections 31–32 cover the reward review,
  and Section 36 records the stopped recovery and bounded conclusions.
- [Pilot results](docs/pilot_results.md): verified subset, first-group reward
  mechanism, failure accounting, and what the study has not established.
- [Cache specification](task_001_expiring_cache.txt): implemented task.
- [Task catalog](docs/tasks/task_catalog.txt): task status and historical six-task design.
- [Verifier-quality protocol](docs/verifier_quality_protocol.txt): current three-task panel and launch/calibration gates.
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

Cloud entrypoints are opt-in and billable. The user relaxed the earlier $20
limit for the bounded follow-up studies; older protocols retain their historical
$10/$20 limits. The booking workflow stopped at failed calibration and does not
authorize automatic extra sampling or training. Resource reservations are not
invoices or provider hard caps. Billing can lag, and persistent storage also
costs money. No standing model endpoint or idle GPU pool is required.
