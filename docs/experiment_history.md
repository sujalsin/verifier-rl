# Verifier-RL

A development evaluator for studying how test coverage affects rewards for
generated code. **The cache task is implemented; the other five tasks remain
specifications. CPU checks, the Qwen/GRPO smoke trial, and a 32-sample
generation-only baseline have completed.**
The first model trial had four zero rewards and unchanged weights: the pipeline
ran, but it did not demonstrate learning. The prompt-reminder pilot also produced
no complete cache solutions. A 16-step SFT pilot changed the 0.5B model's weights
and output format, but its eight after samples all failed on nonempty inputs.
The balanced partial-reward readiness gate failed, so no subsequent RL comparison
was launched. See [the protocol](sft_partial_protocol.txt) and results below.
An untouched **1.5B** diagnostic subsequently produced useful partial-reward
variation in both sampled groups, but still no fully correct program (0/8).
It motivated a bounded partial-reward RL trial; that separate 1.5B pilot is
described below. The earlier generation-only run itself did not train weights.
A subsequent coverage review found that deletion on read could imitate expiry
in the partial probes. A separately versioned v2 suite and strict-output
diagnostic reports now address that specific weakness; see the measurement
repair section below. Reward variation alone is not a sufficient validation.
The subsequent real-reward **1.5B GRPO pilot completed four nonzero-gradient
updates and verified checkpoint reload**, but did not improve the matched
evaluation: official full-audit passes fell from 4/8 to 0/8. One failed audit
includes an unexplained exit-137 termination of unchanged code; the observed
drop must not be attributed entirely to the learned policy. Details below.

The grader runs on the trusted controller. Candidate programs run only through
the opt-in Modal backend. There is deliberately no local arbitrary-code backend.
The offline demo calls a fixed set of author-written fixture functions directly.

## Run locally, without dependencies or cloud access

Requires Python 3.11 or newer. From this directory:

```bash
python3 -m unittest discover -s tests -v
python3 -m verifier_rl demo
python3 -m verifier_rl demo --out runs/my-first-fixture-check
python3 -m verifier_rl export-suites --out runs/my-first-suite-export
```

Four mocked Modal-launcher tests skip when the optional Modal SDK is absent;
install `.[modal]` to include them. They make no cloud calls either way.

Output directories must be new: existing runs are never overwritten. Generated
directories are private and gitignored. Keep durable copies of research artifacts;
gitignore is not a backup strategy. Suite manifests contain development inputs
and expected answers and belong on the controller, not in candidate images.

The demo prints **test cases passed / test cases assigned**, not fractional RL
rewards. Existing G1/G2/G3 reward is 1 only for a completely passing suite; otherwise 0.
The separately versioned balanced suite supports an explicitly named partial
score; it does not silently replace historical reward fields.
An unresolved infrastructure error makes the suite unscored (`reward: null`).
Audit suites always have `reward: null`, even when all their tests pass.

## What is implemented

| File | Responsibility |
| --- | --- |
| `verifier_rl/cache.py` | Exact input validation, dictionary reference, reverse-scan oracle |
| `verifier_rl/suites.py` | G1/G2/G3, separate development audit, seeds, manifests, coverage |
| `verifier_rl/grading.py` | Output comparison, bounded worker pool, retries, per-case records |
| `verifier_rl/fixtures.py` | One correct fixture and nine deliberately faulty fixtures |
| `verifier_rl/modal_backend.py` | Remote execution adapter; contract tests and live CPU smoke checks |
| `verifier_rl/cli.py` | Offline demo, suite export, opt-in cloud evaluation and artifact saving |
| `verifier_rl/cloud_setup.py` | Explicitly opted-in clean sandbox image/app setup |
| `verifier_rl/smoke.py`, `modal_trials.py` | Bounded live CPU conformance trial and persistent records |
| `verifier_rl/model_trial.py`, `modal_model_trial.py` | Bounded 1–4-step Qwen/GRPO trial; extraction-aware CPU rewards and uniform-reward early stop |
| `modal_diagnostic_replay.py` | CPU-only replay of five saved completions on a shared diagnostic input |
| `verifier_rl/prompt_pilot.py`, `modal_prompt_pilot.py` | Frozen eight-seed comparison of the original prompt and explicit interface reminders |
| `verifier_rl/prompt_recovery.py`, `modal_finish_prompt_pilot.py` | CPU-only recovery of saved pilot evidence, limited to confirmed preflight timeouts |
| `modal_grading.py` | Preferred Modal-hosted G3 grader and correct/faulty controls |
| `verifier_rl/training.py`, `modal_grpo_control.py` | Shared GRPO settings and synthetic zero/mixed-reward optimizer controls |
| `verifier_rl/partial_reward.py` | Balanced 24-probe suite, binary/partial scoring of identical outcomes, authored shortcut controls |
| `verifier_rl/sft_data.py` | Fixed non-cache dataset, target validation, assistant-only labels and padding |
| `verifier_rl/sft_trial.py`, `modal_sft_pilot.py` | Bounded SFT, actual checkpoint reload, paired generation and protected evaluation |
| `verifier_rl/checkpoint_pilot.py`, `modal_checkpoint_pilot.py` | Generation-only pinned 1.5B comparison against saved untouched 0.5B outputs |
| `modal_finish_checkpoint_pilot.py` | CPU-only recovery of at most one missing checkpoint-pilot report; never regenerates code |
| `verifier_rl/reward_v2.py` | New 34-case behavior suite and 22 single/compound authored controls; preserves v1 |
| `verifier_rl/diagnostics.py` | Non-executing sidecar reports separating interface, execution and semantic observations |
| `verifier_rl/measurement_v2.py`, `modal_measure_v2.py` | CPU-only v2 remeasurement of the eight saved 1.5B programs; no generation or training |
| `verifier_rl/grpo_pilot.py`, `modal_grpo_pilot.py` | Four-update 1.5B/v2 real-reward pilot, checkpoint reload, and matched independent development audit |

Every suite input is validated, and its expected answer is cross-checked by both
references before execution. Test cases store canonical JSON and expose fresh
input copies. SHA-256 fingerprints cover the task, suite version, seed, inputs,
expected answers, and case labels. The same candidate output is reused for an
identical input shared by suites within one evaluation; no cross-candidate cache
or untrusted environment is reused. This is appropriate for the specified
deterministic functions, not evidence about stochastic programs.

## Grader definitions and reproducibility

Version `cache-coverage-0.1`, default seed `20260924`:

| Suite | Construction | Cases | Operations at default seed |
| --- | --- | ---: | ---: |
| G1 | Fixed ordinary inputs, no overwrites or exact expiry reads | 5 | 15 |
| G2 | 16 shared random inputs + 16 additional random inputs | 32 | 183 |
| G3 | Same 16 random inputs + 16 targeted inputs | 32 | 153 |
| Audit | Separate random seeds, boundary families, tiny-domain enumeration | 388 | 3526 |

Audit removes exact input duplicates with training suites and within itself.
Its count can change with the seed. This is development auditing, **not** an
untouched final evaluation or proof of all-input correctness.

Random lengths are uniform from 0 through 12, initial times 0 through 20,
time increments 0 through 4, write probability 0.55, keys from `a/b/c`, values
from -10 through 10, and lifetimes from 1 through 12. Each draw is documented
in code. Audit random sequences allow lengths through 80 and use a separate
seed; targeted cases add large bounds. No filtering forces random graders to
miss boundaries. Coverage reports count actual input features, not detected bugs.

G2 and G3 have equal case counts but not identical operation counts. G1 has a
different case budget. These are measured confounds, not a perfectly matched
coverage-only causal experiment. Execution costs still need a live benchmark.
The suites are fixed for this pilot; changing sampling during RL requires a
separate protocol decision and version. Save manifests as well as seeds and
record Python/SDK versions; do not rely on seeds alone across runtime changes.

The tests cross-check the references on the hand examples, all valid short
sequences from an explicit eight-operation alphabet, and 1,000 additional
seeded sequences. This increases confidence but does not prove either oracle.

## First fixture observations

At the default seed, the correct fixture passes all four suites. Each of the
nine faulty fixtures fails G3 and the audit. The `zero_missing` fixture passes
all 32 G2 inputs but fails a targeted G3 input; the wrong-expiration-equality
fixture passes G1 but fails both G2 and G3. These observations are reproducible
with `demo`, not model behavior, training improvement, or spontaneous exploits.

## Modal adapter: initial live checks passed

The 2026-09-27 UTC CPU run passed **15 checks / 23 sandbox executions**.
Local evidence: `runs/cpu-20260927T044216-621962c1/summary.json`; durable copy:
Modal volume `verifier-rl-cache-artifacts`, under the same run ID.
SDK: `modal==1.5.5`; clean candidate image: `im-GYYHaMJMKETlbgCvh62lOr`.
Mean end-to-end sandbox invocation was 1.885 seconds (43.366 seconds summed).
These sequential diagnostic probes are not a throughput benchmark or security audit.

The first run caught a real integration bug: SDK 1.5.5 returns `-1` when
`ContainerProcess.wait()` catches an execution timeout. The adapter now records
that as `timeout`, not a generic candidate error. The failed evidence is retained
under `runs/cpu-20260926T051912-ed363f68/`. A local regression test covers it.

The adapter follows Modal's documented [sandbox API](https://modal.com/docs/sdk/py/latest/Sandbox),
[network controls](https://modal.com/docs/guide/sandbox-networking), and
[command execution API](https://modal.com/docs/guide/sandbox-spawn).
It requests network blocking, no mounted volumes/secrets or identity token,
one CPU with a hard cap, and 256 MiB requested/capped memory.

For a first cloud smoke test, an operator must:

1. Install the optional dependency (`python3 -m pip install -e '.[modal]'`) in
   a virtual environment and authenticate Modal locally. SDK 1.5.5 is pinned.
   Never put account tokens in source code, logs, or chat.
2. Supply an existing Modal app and an approved, resolved image ID (`im-...`).
   The adapter neither builds an image nor creates the app. The image must be
   a clean Linux/Python 3.11+ environment starting as root, with `python3` and
   `sleep` on PATH. Do not use an image containing project files, grader inputs,
   references, model credentials, or other secrets. Pin its base image/dependencies
   during image construction and retain the resolved Modal image ID.
3. Run the correct fixture on G1, then perform the live conformance checklist
   below before submitting generated or intentionally hostile programs.

The low-level CLI below runs the trusted controller on the laptop. In the
2026-09-27 diagnostic, sandbox creation worked but the command-start RPC timed
out before candidate execution. The Modal-hosted controller passed the same
runtime checks. **Use the hosted route in the next section for current trials.**
The underlying laptop transport issue remains unresolved; fail-fast handling
limits repeated attempts but is not a network fix.

Low-level diagnostic example, after exporting the fixture demo:

```bash
python3 -m verifier_rl grade runs/my-first-fixture-check/fixture_correct.py \
  --app YOUR_EXISTING_APP --image-id im-YOUR_APPROVED_IMAGE \
  --suites g1 --concurrency 2 --out runs/modal-cache-smoke --allow-cloud
```

This command creates billable remote sandboxes. Default cloud evaluation selects only G1, not the
hundreds of audit cases. Use `--suites g1 g2 g3 audit` deliberately after measuring
cost and reliability. The CLI saves source, configuration, suite manifests,
and a report. Exit code 2 means an infrastructure/configuration failure;
ordinary candidate failure is a valid recorded evaluation, not a CLI crash.

### Reproduce the bounded cloud trials

Use new output directories; these commands create billable resources. Start
with CPU checks and inspect their report before launching any GPU job.

```bash
.venv/bin/python -m verifier_rl.cloud_setup --allow-cloud --out runs/NEW_SETUP
.venv/bin/modal run modal_trials.py --setup-file runs/NEW_SETUP/setup.json
.venv/bin/modal run --detach modal_grading.py \
  --setup-file runs/NEW_SETUP/setup.json \
  --cpu-report runs/CPU_RUN_ID/summary.json --controls --allow-cloud
.venv/bin/modal run --detach modal_grpo_control.py \
  --grader-report runs/GRADER_CONTROL_RUN_ID/summary.json --allow-cloud
```

The hosted grader checks known-correct and known-faulty programs on every G3
case. To evaluate one candidate instead, replace `--controls` with
`--candidate path/to/program.py`. The laptop only submits the job and retrieves
records; a trusted Modal CPU controller manages the isolated candidate sandboxes.
Infrastructure failure stops new submissions, allows in-flight cleanup to finish,
and leaves unsubmitted inputs explicitly `not_executed`. It never becomes reward 0.

The optimizer control uses one L4, a 600-second deadline, a short non-code prompt,
and synthetic rewards: one all-zero step followed by two mixed-reward steps,
both conditions starting from identical weights. It checks gradient norms,
parameter changes, and save/reload identity. This validates optimizer wiring,
**not coding improvement**. Do not use its synthetic checkpoints as a warm-start.
The frozen debugging protocol is [training_validation_protocol.txt](training_validation_protocol.txt).

The cache model trial remains available for a later justified experiment:

```bash
.venv/bin/modal run modal_model_trial.py \
  --setup-file runs/NEW_SETUP/setup.json \
  --cpu-report runs/CPU_RUN_ID/summary.json --max-steps 1
```

It checks the CPU report's image identity, uses one L4 with a
900-second function deadline and no application retries, and permits a bounded
1–4 GRPO steps (default 1; four rollouts per group, binary G3 rewards). It stops
after a uniform-reward group instead of spending further steps without a relative
signal. Each group has its own `train-0000` through `train-0003` records.
The current launcher still starts from the pinned public checkpoint, not SFT;
SFT checkpoint loading is part of the next implementation. No serving endpoint
is required. It saves and reloads the actual checkpoint, checks parameter hashes,
and samples two before/two after outputs using matched seeds. After GPU training
returns, a separate CPU call evaluates these outputs against G1/G2/G3 and all
388 development-audit inputs. The audit is never a reward source. The CPU worker
pool is capped at four; any infrastructure error stops the batch unscored.

Files persist on `verifier-rl-cache-artifacts` in distinct GPU/grading directories.
Raw completions, minimally extracted source, model revision, package versions,
trainer settings, rewards and per-case outcomes are retained. Extraction v0.2
accepts raw Python or exactly one complete Python/py/unlabelled fenced block,
allowing surrounding prose. Multiple, incomplete, malformed or non-Python blocks
are rejected. Code inside the block, including demonstration calls, is preserved;
invalid or truncated code is not repaired. Extraction status/version are logged.
The original enclosing-fence-only v0.1 policy is still available explicitly
through `extract_source(..., version=LEGACY_EXTRACTION_VERSION)` for reproducing
the first trial. No old trial artifact has been changed or rescored. The model
checkpoint has weights/tokenizer, not the optimizer state needed for exact resume.
Top-level ML packages are pinned; resolved transitive versions are recorded.
Two evaluation samples and at most four steps are **only an integration test**,
not learning evidence. The expanded cache path has local regression coverage;
a multi-step cache-learning run has not been demonstrated.

The initial user budget is about $10. Resource/time caps bound exposure, but
they are not an account billing limit. Actual billed cost must be checked in
Modal; local timing estimates do not include every charge or existing usage.

### First Qwen trial: completed, no learning signal

This historical run used extraction v0.1, not the subsequently added v0.2.

Run `qwen-20260927T044818-f52df1bf` used model revision
`ea3f2471cf1b1f0db85067f1ef93848e38e88c25` on one L4. The GPU function took
202.99 seconds; one optimizer step completed with rewards `[0, 0, 0, 0]`, zero
loss/gradient norm, and identical before/after parameter hashes. Reloading the
saved checkpoint reproduced its hash and the same seeded sample texts.

Each of the two baseline and two post-checkpoint samples passed **0/5 G1,
0/32 G2, 0/32 G3, and 0/388 audit**. All four contained prose around code,
rejected under the frozen extraction contract. The training group additionally
showed runtime/type errors and demonstration output contaminating JSON. Two
training outputs hit the 512-token cap. No candidate source was repaired.

The trial made 124 training and 1,756 evaluation sandbox executions, with zero
infrastructure errors and acknowledged termination throughout. Evaluation mean
invocation time was 1.736 seconds at concurrency four; this is not pure execution
time or a sustained load benchmark. The result is a format/execution floor in
this small sample, **not reward hacking, an algorithm comparison, or proof that
the model cannot solve the task**. Do not scale RL before a baseline produces
valid programs and within-group reward variation.

Evidence lives in `runs/qwen-20260927T044818-f52df1bf/` and the same run directory
on the artifact volume. `model.json` contains outputs and training metadata;
`evaluation.json` contains per-case records. Summarize without running code:

```bash
.venv/bin/python -m verifier_rl.trial_report runs/qwen-20260927T044818-f52df1bf
```

The billing query returned approximately $0.0796 in project usage for its
reported intervals. It omitted the still-current hour, so this is **not the
final trial cost**. Recheck `modal billing report --start 2026-09-26 --resolution h`
after reporting catches up. No always-on endpoint or scheduled job was deployed.

### Follow-up controls: grader and optimizer checks passed

Grader run `qwen-grader-control-20260927T222645-b625f8d4` used the hosted
controller: the correct fixture passed 32/32 G3 cases (reward 1), while the
inclusive-expiry bug passed 22/32 (reward 0). All 62 distinct candidate/input
executions completed without infrastructure errors and with confirmed cleanup.

Optimizer run `qwen-optimizer-control-20260927T223422-219f0044` exercised the
actual pinned Qwen/TRL stack on one L4:

| Control | Steps | Reported gradient norms | Weights changed | Checkpoint reload |
| --- | ---: | --- | --- | --- |
| All-zero rewards | 1 | 0 | No | Matches |
| Synthetic mixed rewards | 2 | 44.95, 59.41 | Yes | Matches |

Both conditions started from identical weights. The GPU function took 81.95
seconds. The positive control used short non-code completions, not cache
programs. It demonstrates working optimizer/checkpoint wiring under those
conditions, **not cache learning or reward hacking**. Near-zero logged loss
coexisted with nonzero gradients in the mixed condition; loss alone is not an
update check. No longer cache RL run was launched.

Ninety local tests passed at that stage. Research notes Section 26 records the fixes, the
extracted-source hash gate incident, evidence, limitations, and the user-selected
SFT design. Raw run records are gitignored and persist on the artifact volume;
they are not included in a fresh GitHub clone.

### Where the model runs and where weights live

`modal_model_trial.py` requests an NVIDIA L4 for the remote `train_smoke` function.
Inside that function, Hugging Face `from_pretrained` loads the public model's
weights and tokenizer at the recorded revision into the remote environment;
`.to("cuda")` places the model on the GPU. This is direct execution of downloaded
model weights, not an inference API call or a model running on the laptop.
The laptop launches the job and saves returned JSON records.

The initial Hugging Face download cache is not mounted on a persistent volume
in this implementation. The explicit saved checkpoint IS persistent:
`verifier-rl-cache-artifacts/qwen-20260927T044818-f52df1bf/gpu/checkpoint/`.
It contains weights and tokenizer files; the model was reloaded from there to
verify saving worked. This run's weights did not change because all rewards
were zero. Candidate CPU sandboxes receive source and test inputs, never the
model weights, checkpoint volume, or reference answers.

### Balanced partial rewards and the SFT warm-start pilot

The earlier full-suite reward discarded partial successes. A flat fraction is
not automatically a better signal: the saved always-`None` program passes 15/32
G3 cases despite never retrieving a value. The new `cache-balanced-probes-0.1`
suite keeps G3 unchanged and adds 24 separately versioned probes. Retrieval,
expiry, overwrites, multiple keys and ordering each receive 19% of reward;
edge cases collectively receive 5%. Credit requires each probe's complete
expected output, not isolated matching list elements.

Offline controls at three fixed seeds all pass their pre-specified checks.
At the training seed, correct code scores 1, always-`None` .0375, always-empty
.025, and a dictionary that ignores expiration .2275. These are authored controls,
not model improvement or guarantees against arbitrary shortcuts.

The SFT dataset contains eight training families and two held-out development
families, each with four prompt variants (32/8 examples). Its ten fixed functions
have 40 passing target checks. There is no cache/TTL solution or audit content.
Training uses assistant-only labels, keeps real EOS supervision, masks padding,
and rejects template misalignment or truncation. Only the GPU container loads
the model; no arbitrary candidate code is run on the laptop.

```bash
.venv/bin/modal run --detach modal_sft_pilot.py \
  --setup-file runs/CPU_RUN_ID/setup.json \
  --cpu-report runs/CPU_RUN_ID/summary.json \
  --optimizer-report runs/OPTIMIZER_CONTROL_RUN_ID/summary.json --allow-cloud
```

This is a billable, bounded run: 16 SFT steps, eight before/eight after samples,
one 900-second L4 function, followed by hosted CPU grading with four workers
and no retries. G3 plus the balanced suite has 54 distinct inputs per candidate,
at most 864 sandbox invocations. Checkpoint reload must match the actual trained
parameter hash. The launch saves the dataset, reward checks, frozen plan and
source snapshot locally; per-sample records and the checkpoint also persist on
the artifact volume. Do not rerun a completed pilot just to inspect its results.

The gate for a subsequent binary/partial GRPO comparison requires a mixed group
of SFT outputs with at least one balanced partial score above .05, beyond the
entire edge-case allowance. Complete solutions are not required for partial
reward variation. If this gate fails, no longer RL run starts automatically.
See [the frozen protocol](sft_partial_protocol.txt) and research notes Section 27.

Completed run: `qwen-sft-pilot-20260927T230314-ac2f8eeb`. All 16 SFT optimizer
steps had finite nonzero gradients, parameter hashes changed, and checkpoint
reload matched. Training loss fell .5364 → .0123; development-family loss was
.3734 → .3780. The GPU function took 265.35 seconds. CPU grading completed all
864 distinct candidate/input executions without infrastructure errors.

| Cache measurement | Before SFT | After SFT |
| --- | ---: | ---: |
| Syntactically valid programs | 8/8 | 8/8 |
| Valid returned lists on every input | 1/8 | 0/8 |
| Full G3 / balanced-suite passes | 0/8 each | 0/8 each |
| Distinct source strings | 8 | 2 |
| Mean balanced partial score | .00625 | .0125 |

Every after program uses undefined `key`/`value` variables. They only succeed
on empty input; all eight rewards are .0125, with no within-group variation.
Shorter, cleaner output is not a functioning cache, and the slight increase in
partial score is not meaningful capability evidence. This was SFT, not reward
optimization or reward-hacking behavior. The original-policy samples reproduce
the previous reminder-prompt seeds; they are not new independent baseline draws.

**The readiness gate failed.** The planned binary/partial GRPO comparison and
its independent development audit were not launched. Another SFT experiment
would need a revised warm-start dataset/curriculum. At the user's request, the
next diagnostic instead tests the untouched 1.5B checkpoint before more training.
SFT-stage verification: 101 tests passed; saved reports and stdout
were independently checked. See Section 27 and the run's `verification.json`.

### Untouched 1.5B checkpoint pilot

This generation-only diagnostic compares eight samples from public
`Qwen/Qwen2.5-Coder-1.5B-Instruct`, revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`, with the saved **before-SFT 0.5B**
samples. The cache prompt, decoding, seeds, extraction, G3, balanced probes and
sandbox environment remain fixed. Matching chat templates/effective generation
defaults and unchanged parameter hashes are checked. No training is performed.

```bash
.venv/bin/modal run --detach modal_checkpoint_pilot.py \
  --setup-file runs/CPU_RUN_ID/setup.json \
  --cpu-report runs/CPU_RUN_ID/summary.json \
  --reference-run runs/COMPLETED_SFT_RUN_ID --allow-cloud
```

One 600-second L4 function generates eight outputs, then a 1200-second hosted
controller manages up to 432 new sandbox evaluations with four paced workers
and no retries. This is billable; do not rerun completed jobs to inspect them.
Meaningful mixed rewards can justify a later bounded RL experiment; uniform
success calls for a broader task distribution, while uniform failure calls for
further diagnosis. No branch launches more training automatically. See the
[frozen protocol](checkpoint_1_5b_protocol.txt) and research notes Section 28.
The existing 0.5B GRPO launcher is not automatically converted to 1.5B.

The first preflight caught different native repetition penalties (0.5B: 1.05;
1.5B: 1.1). The corrected protocol explicitly uses 1.05 to match the existing
baseline. It generated all eight samples, but the grading controller was
preempted after seven reports. Completed reports and original samples persist;
this bounded CPU-only recovery handles the one missing report:

```bash
.venv/bin/modal run --detach modal_finish_checkpoint_pilot.py \
  --parent-run qwen-checkpoint-pilot-20260927T234301-fc4ee586 --allow-cloud
```

It allows at most 54 additional sandbox calls and no new model samples. Unknown
work interrupted before its report was saved is recorded separately; do not
count it as a model failure or claim exactly 432 total attempts across recovery.
The current existing-directory guard prevents accidental regeneration after a
platform restart, but does not provide automatic per-input recovery.

Completed comparison: `qwen-checkpoint-recovery-20260927T235622-8129215d`, using
the original eight generated samples and seven reused reports. Recovery only
evaluated saved seed 4007. Key results:

| Measurement | Original 0.5B | Original 1.5B |
| --- | ---: | ---: |
| Valid returned lists on every evaluated input | 1/8 | 3/8 |
| Full G3 / balanced-suite passes | 0/8 each | 0/8 each |
| Mean balanced partial reward | .00625 | .12 |
| Meaningful mixed partial-reward groups | 0/2 | 2/2 |
| Mixed binary-reward groups | 0/2 | 0/2 |

The 1.5B groups score `[.2275, .025, .025, 0]` and
`[.62, .0375, 0, .025]`. Its best sample passes 30/32 G3 and 16/24 balanced
cases, but incorrectly deletes keys after reading them. Binary scoring assigns
all eight programs zero; balanced partial scoring distinguishes useful behavior.
This is a better starting-policy signal, **not evidence of RL learning**.

Verified 432 recorded executions / 448 assigned outcomes, all with acknowledged
cleanup; up to 54 unrecorded interrupted attempts remain outside that claim.
One recorded G3-only execution timeout had no runner-stage output; its cause is
unresolved and it does not affect the balanced reward groups. The original
0.5B samples are reused evidence, not new independent draws. Research notes
Section 28 and the recovery run's `verification.json` record the complete checks.
Current local verification: 108 tests pass.

### Measurement repair: v2 probes and separated diagnostics

The original balanced expiry cases always read a value before checking its
expiration. A deliberately faulty control that **ignores expiry and consumes
values on read** earns .62 overall, including all expiry-probe credit. The exact
answers/comparator are not wrong; the test patterns admit a different mechanism.
This is a grader-coverage finding, not observed policy reward hacking.

`cache-balanced-probes-0.2` is a new suite named `balanced_v2`: six structurally
different probes per substantive family plus four edge cases, 34 cases total.
Every expiry probe includes a first read of an already-expired write alongside
a live value. Other probes cover repeated reads and state/order interactions.
The weights stay .19 per substantive family and .05 for all edges. No historical
suite, prompt, parser, runner, or saved score is replaced.

```bash
.venv/bin/python -m verifier_rl.reward_v2 --out runs/NEW_V2_CONTROLS
.venv/bin/python -m verifier_rl.diagnostics \
  --run runs/EXISTING_GENERATION_AND_REPORTS --out runs/NEW_DIAGNOSTIC_SIDECAR
```

These commands do not execute candidate code, call a model, or use cloud
resources. The first runs fixed authored controls only and exports manifests.
The second requires saved `generation.json` and `reports.json`, checks identities
and complete stdout evidence, and refuses to overwrite an existing output path.
It separates valid wrong answers, malformed output, candidate execution errors,
unattributed timeouts and infrastructure failures. It never accepts the last
stdout line as an alternative answer. Valid-output-only accuracy is explicitly
conditional and must not replace whole-program correctness.

The 66 control/seed comparisons pass: correct code receives 1; all 63 faulty
control/seed combinations fail full acceptance. At the default seed:

| Authored control | v1 partial | v2 partial |
| --- | ---: | ---: |
| Consume on read, ignoring expiry | .62 | .24 |
| Consume on read, correct expiry | .62 | .5567 |
| Correct implementation | 1 | 1 |
| Always `None` | .0375 | .0375 |
| Refresh lifetime on read | .4175 | .7658 |
| Ignore expiry without consuming | .2275 | .4175 |

The increases are retained deliberately: this fixes a specific confound, not
every possible weakness or partial-score ranking. Case count and structures
changed; this is not a matched coverage-only causal comparison. Family labels
describe probes, not proven learned skills. The complete control matrix is
saved under `runs/behavior-v2-controls-20260928/` and is reproducible locally.

The frozen [measurement protocol](measurement_v2_protocol.txt) specifies a
CPU-only re-evaluation of the existing eight untouched 1.5B samples:

```bash
.venv/bin/modal run --detach modal_measure_v2.py \
  --setup-file runs/CPU_RUN_ID/setup.json \
  --cpu-report runs/CPU_RUN_ID/summary.json --allow-cloud
```

This is billable: at most 272 fresh sandbox executions, no GPU, no generation,
four paced workers, no application retries. Candidate intent is committed before
execution and its report immediately afterward. Complete reports can be reused
on a controller restart; an intent without a result stops to avoid silently
repeating unrecorded work. The legacy and new scores belong to different test
distributions, so their difference is not model improvement. No branch starts
training automatically. The existing GRPO launcher remains unchanged.

Completed CPU-only run: `qwen-behavior-v2-20260928T003755-28724af6`. All eight
unchanged programs were evaluated on all 34 cases, with 272 distinct sandbox IDs,
one attempt per input, and no recorded infrastructure failures or timeouts.
The best program passes **20/34** cases (weighted partial reward **.5567**),
but **0/8 programs pass the full suite**. The two fixed groups have rewards:

```text
4000–4003: [.3225, .025,  .025, 0]
4004–4007: [.5567, .0375, 0,    .025]
```

The execution breakdown is 41 correct outputs, 68 valid wrong answers,
129 candidate execution errors, and 34 invalid-JSON outputs. All eight programs
parse; syntax success alone does not imply that the interface or behavior works.
This is eight saved model samples, not 272 independent samples. The mean partial
score changes from .12 on v1 to .123958 on v2 only because the evaluator changed.
`verification.json` in the run directory records summary recomputation, strict
stdout checks, resource/cleanup evidence, and unchanged source identities.
No weights were updated and no independent strict audit was run in this step.
The separate 1.5B/v2 training integration and frozen before/after audit protocol
below follow this measurement step; mixed rewards alone do not establish learning.

Verification at the measurement-repair step: **128 tests passed**. Research notes Section 29 records
the diagnostic findings, repair rationale, limitations, and measurement outcome.

### Bounded 1.5B real-reward GRPO pilot

The [frozen protocol](grpo_1_5b_protocol.txt) preserves the model revision,
prompt, extraction and strict comparator. It uses real `balanced_v2` partial
rewards, at most four fresh groups of four rollouts, full-parameter GRPO at
learning rate `1e-6`, and an early stop after a uniform reward group. It does
not replay the previous successful candidates or use audit scores as rewards.
The legacy 0.5B launcher is unchanged.

Eight fresh samples before training and eight after checkpoint reload use paired
seeds 8000–8007 and matched decoding. After training and sample generation finish,
all sixteen face the 34 reward probes and the full 388-case development audit.
The two suites have no identical input hashes. This is still one task and eight
samples per arm, not a final untouched benchmark or reliable efficacy estimate.

```bash
.venv/bin/modal run --detach modal_grpo_pilot.py \
  --setup-file runs/CPU_RUN_ID/setup.json \
  --cpu-report runs/CPU_RUN_ID/summary.json \
  --optimizer-report runs/OPTIMIZER_CONTROL_RUN_ID/summary.json --allow-cloud
```

This is billable. One L40S GPU function is capped at 1200 seconds; the independent
CPU audit controller at 3600 seconds. Full FP32 parameters, gradients and Adam
moments require about 23 GiB before activations and workspace, so the previous
L4 inference fit is insufficient justification for training on an L4. There are
at most 7296 fresh sandbox executions, eight paced workers, and no application
retries. Interrupted candidate intents without complete reports stop the run.
An interrupted GPU stage cannot silently repeat training. Detached mode does
not turn the laptop entrypoint into a durable workflow: keep it connected through
the handoff from GPU completion to CPU audit. Do not launch a second run to
recover an existing one without inspecting its saved state and spending first.

Report nonzero finite gradients, parameter changes and exact checkpoint reload
separately from full-audit success and partial rewards. Unchanged or degraded
audit performance remains a negative result. At pilot completion,
**136 tests passed**. Research notes Section 30 tracks this pilot and its outcome.

Completed run: `qwen-grpo-partial-20260928T010255-4a0e6d52`.

| Measure | Before | After |
| --- | ---: | ---: |
| Full 388-case audit passes | 4/8 | 0/8 |
| Full 34-case reward-suite passes | 4/8 | 1/8 |
| Mean weighted partial reward | .50625 | .25542 |
| Audit case passes (diagnostic only) | 1650/3104 | 944/3104 |

All four training groups had reward variation; pre-clipping gradient norms were
3.23, 3.16, 3.74 and 3.15. Parameters changed and the saved checkpoint reloaded
with an exactly matching parameter hash. Peak allocation was 26.20 GiB, peak
reservation 27.73 GiB; the GPU function used 437.60 seconds including reward
waits. Training is operational, but useful learning was **not demonstrated**.

There is an important execution caveat: paired seed 8005 has identical extracted
source in both arms, yet went from 388/388 to 387/388 audit cases. Its single
failure (`exhaustive/0035`) returned 137 with the last stage hint at module load
and no output. CPU-limit, memory, or platform termination is not distinguished
by this record. The official failure is retained, not re-run or silently rescued.
The other two identical-source pairs reproduce all their recorded outcomes.
This anomaly warrants investigation before interpreting small full-pass changes.

Other failures include real semantic and interface errors: expiry decremented
on reads, missing-key outputs omitted, incorrect dictionary unpacking, undefined
`time`, and demonstration text contaminating stdout. Both measured proxy reward
and strict accuracy decreased, so this is **not evidence of reward hacking**.
One training seed and eight evaluation samples cannot establish a general
negative effect of GRPO. The 7296 assigned candidate-input pairs resulted in
6874 unique sandboxes; one rejected completion's 422 evaluations were assigned
failure without execution. All 5579 completed stdout records were independently
rechecked. `training-verification.json` and `verification.json` retain the checks
and caveats. No automatic follow-up training was launched.

### Reward-first review (offline; v3 is not enabled for training)

The [reward review protocol](reward_review_protocol.txt) separates test coverage
from reward shaping. Reproduce the saved-record audit and authored controls with:

```bash
.venv/bin/python -m verifier_rl.reward_review \
  --run runs/qwen-grpo-partial-20260928T010255-4a0e6d52 \
  --out runs/NEW_REWARD_REVIEW_DIRECTORY
```

This makes no cloud calls and executes no generated source. It validates saved
outputs, recomputes all sixteen training rewards, and writes a new sidecar;
existing run files cannot be overwritten. `reward_v2.py`, the strict comparator,
and all training defaults remain unchanged.

Completed sidecar: `runs/reward-review-20260928-v1/reward_review.json`.
The existing reward agrees with development-audit case-accuracy ordering on all
94 pairs where both values differ; zero reversed pairs, four reward-only ties,
and 22 ties on both measures. These are **16 program records, 13 unique sources**,
not 120 independent samples. The four reward-only ties involve the unresolved
exit-137 record, which remains an official failure. This is no evidence of broad
misranking in this small pool, not proof that the verifier generalizes.

Two authored incorrect programs receive full v2 credit: one only handles inputs
of length at most six, and one only handles single-character keys. V2 never tests
outside those domains. These are constructed counterexamples, not observed model
exploitation. The opt-in `reward_v3.py` proposal preserves all 34 v2 cases and adds
ten longer/wide-key probes, reaching 200 operations and eight-character keys.
Its 44 cases use the same family-weighted exact-case score. Both controls now fail
full acceptance but retain .88125 partial credit; correct behavior still gets 1.
All 25 authored controls were checked at three seeds (75 combinations). At the
time of this offline review, v3 had not been executed on model samples; see the
paired measurement below. It is not universally lower-scoring for faulty
programs: the decrement-on-read control still receives .905.

The review also compares existing partial rewards, binary rewards, and an
**illustrative, inactive** `(partial + full_suite_pass) / 2` bonus on the same
saved rollouts. In groups 1 and 2, this changes one incomplete program's positive
advantage to negative while keeping the full-pass program positive. In group 4,
all programs are incomplete: halving their rewards mostly cancels under group
normalization, and the same incomplete winner remains positive. Binary rewards
instead give that group no signal. These are scalar counterfactuals, not new
training runs or proof that a bonus improves learning.

This review motivates comparing frozen v2/v3 on the same saved programs before
another training run, keeping coverage and reward-formula changes separate.
The offline review used no GPU/Modal execution, changed no weights,
and did not establish the cause of the earlier regression. Research notes
Section 31 preserves the negative findings and the rationale. Verification after
this review: **155 tests passed**, including nineteen new reward-review/coverage
tests.

### Paired v2/v3 grader measurement (CPU only)

[The frozen protocol](measurement_v3_protocol.txt) evaluates the 32 saved pilot
program records (29 distinct extracted sources) and three authored controls.
Each program gets one execution per input in the 44-input union. The same first
34 outcomes supply v2; all 44 supply v3. Historical v2 is also compared with fresh
v2 to detect execution variation separately from changed test coverage.

```bash
.venv/bin/modal run --detach modal_measure_v3.py --allow-cloud
```

This is a **billable new measurement**, not a command needed to read an existing
result. It creates no model samples, uses no GPU, and changes no weights or
training defaults. The cap is 1,496 fresh CPU sandboxes; the previously rejected
completion is not executed. Current metered spending, maximum resource lifetimes,
and an overhead reserve must fit the original $10 total trial budget before
candidate execution. This check is not a provider-enforced account spending cap.

Every candidate intent and result is saved durably. Complete matching records
can be reused; incomplete intents cannot silently restart. There are zero
retries. Unexpected control behavior, infrastructure/cleanup failures,
signal-style exits, timeouts, or changed historical-v2 outcomes stop progression.
The original audit and its exit-137 outcome are never overwritten or rescored.

Per-program reports include partial/full acceptance, family rates, new failure
reasons and original-group reward/advantage comparisons. Higher partial scores
can reflect dilution of old failures, not better programs. Authored control
failures are not observed model exploitation. Research notes Section 32 explains
the design and its limits.

Once a measurement completes, verify it locally without cloud calls or source
execution, using a new output directory:

```bash
.venv/bin/python -m verifier_rl.measurement_v3_verification \
  --run runs/COMPLETED_MEASUREMENT_RUN \
  --out runs/NEW_VERIFICATION_DIRECTORY
```

This checks the frozen implementation, source/suite/image identities, complete
stdout evidence, fresh sandbox IDs, cleanup, counts, and exact recomputation of
the summary. It does not rerun programs or establish learning efficacy.

Completed run: `qwen-grader-v3-20260928T041709-7e90839d`, with the local
verification sidecar in `runs/qwen-grader-v3-20260928T041709-7e90839d-verification/`.
All **1,496 actual sandboxes** have distinct IDs and confirmed termination;
all 1,198 completed stdout records were independently rechecked. There were no
infrastructure errors, timeouts, or signal-style exits. Ordinary candidate
exceptions and invalid outputs remain failures, not infrastructure successes.
Every saved program reproduced its historical v2 outcomes.

| Saved population | Full passes: v2 / v3 | Mean partial: v2 → v3 |
| --- | --- | --- |
| 16 training rollouts | 3 / 3 | .34318 → .34070 |
| 8 before-evaluation records | 4 / 4 | .50625 → .50625 |
| 8 after-evaluation records | 1 / 1 | .25542 → .25344 |

Both intentionally limited controls changed from v2 full acceptance to v3
rejection (.88125 partial), but **no model program changed full-pass status**.
Only four model partial scores decreased; the other 28 were unchanged. All four
reward groups retained variation, and the same incomplete programs retained
positive relative advantages. Therefore, the coverage checks work, but this
experiment does **not** show that v3 fixes the earlier learning regression or
that the model discovered the controls' shortcuts.

The new runner and offline verification are covered by **172 passing local tests**.
No weights or training defaults changed. The immediate post-run workspace billing
snapshot was $1.09557826 metered ($0 after credits), but reporting lag prevents
treating its increase as the final experiment cost. Both apps reported zero tasks.
Section 32 of the research notes records the per-program changes, negative
findings, budget evidence, and recommendation to isolate reward shaping next.

### Generation-only baseline

`modal_baseline.py` runs 32 fresh samples (seeds 4000–4031), no optimizer or
training. It pins the previous model revision and checks the unchanged prompt
hash. Decoding remains 512 tokens, temperature 0.8, top-p 0.95, top-k disabled.
It records v0.2 extraction status, static syntax/interface diagnostics, token
cap/EOS status, package versions and unchanged full parameter hashes. Static
checks never execute candidate source and do not remove samples from grading.

```bash
.venv/bin/modal run modal_baseline.py \
  --setup-file runs/modal-setup-20260925/setup.json \
  --cpu-report runs/cpu-20260927T044216-621962c1/summary.json \
  --previous-trial runs/qwen-20260927T044818-f52df1bf/model.json
```

After the GPU function returns, up to 16 sequential CPU batches evaluate at most
two samples each on G1/G2/G3 and the full development audit. Each candidate uses up
to eight parallel fresh sandboxes with creation requests spaced at least 0.26
seconds apart (identical per-sandbox permissions/resource limits); there is one grading
controller, no retries, a 600-second deadline per batch and a 900-second GPU
deadline. Full-case grading is retained even for format failures. Evidence is
saved per sample remotely and per completed batch locally. Logical reward groups
remain four samples each. The baseline reports
format/syntax counts, execution status, whole-suite success, diagnostic case
pass counts, audit disagreements and mixed-reward groups of four.

No model weight checkpoint is duplicated for generation-only runs. The public
model revision identifies the policy; generated text and evaluation records
persist on the existing volume. A completed run is not proof of learning or
task generalization. Check billing before rerunning; run limits are not an
account-wide spend cap.

The initial 16-worker baseline hit the account's five-creations/second limit.
Generation succeeded, but the first candidate had two infrastructure failures
and the evaluator stopped. Its records are retained. Resume the same saved
samples, without another GPU call, by adding:

```bash
  --resume-generation runs/qwen-baseline-20260927T053930-ed10bec3/generation.json
```

Resumption creates a new evaluation run directory and preserves the generation
source run ID. Pacing is per backend, not an account-wide limiter: do not run
multiple such controllers concurrently without a shared rate limiter.

The first throttled four-candidate batch reached its 600-second controller
deadline after three complete candidates. Execution batches were reduced to
at most two; per-candidate limits and tests were not changed. Add
`--recover-evaluation-runs qwen-baseline-20260927T055451-bc301730` to recover
its three completed reports from the volume. Recovery checks seed/source,
suite fingerprints, image identity and termination; infrastructure-failed
reports are retained but not reused. Named recovery runs can be comma-separated.
Recovery does not restore untrusted sandbox state: remaining cases still run
in fresh sandboxes. The original attempt remains available for analysis.

If the laptop connection is lost after at least 24 candidates finish, use the
bounded cloud finisher. Its scheduling loop and final summary run on Modal,
not in the local entrypoint. `--detach` lets the job continue if the local
process disconnects; it is still a finite job, not a scheduled service.

```bash
.venv/bin/modal run --detach modal_finish_baseline.py::finish \
  --setup-file runs/modal-setup-20260925/setup.json \
  --cpu-report runs/cpu-20260927T044216-621962c1/summary.json \
  --saved-generation runs/qwen-baseline-20260927T053930-ed10bec3/generation.json \
  --recovery-runs qwen-baseline-20260927T055451-bc301730,qwen-baseline-20260927T062526-61de8ec6
```

The finisher refuses more than eight pending candidates, makes no GPU call,
and writes all recovered records, logical batches and `summary.json` to its
new run directory on the artifact volume. A local `launch.json` records that
remote summary path if the client disconnects before receiving the result.

### Completed 32-sample baseline

Final run: `qwen-baseline-20260927T155618-1f544ea0`. It recovered 25 completed
evaluations and finished seven; all used the original 32 saved samples from
`qwen-baseline-20260927T053930-ed10bec3`. No second generation or training job.

| Measurement | Result |
| --- | ---: |
| Extractable, syntactically valid Python | 30/32 |
| Statically visible required function | 29/32 |
| At least one valid JSON-list output | 5/32 |
| Entire G1 / G2 / G3 passed | 0/32 for each |
| Entire 388-case development audit passed | 0/32 |
| Mixed binary-reward groups | 0/8 for each training grader |

Seventeen responses hit the 512-token limit without EOS, but 15 of those still
had complete parseable Python blocks. The best audit result was seed 4005,
49/388 cases. Aggregate diagnostic case passes were G1 0/160, G2 12/1024,
G3 12/1024 and audit 49/12416; these are not fractional RL rewards.
There are 14,048 unique candidate/input execution records in the completed
dataset, excluding failed/interrupted attempts. Model weights stayed unchanged.

Three original executions exited 137 with runtime EOF messages; we do not know
their cause. Targeted fresh-sandbox rechecks returned ordinary Python exceptions
and code 1 for all three. Both attempts are retained in
`runs/qwen-recheck-20260927T161516-de3ea55e/summary.json`; do not present the original
exits as confirmed OOMs or confidently model-caused errors. No original evidence
was overwritten. The rechecks do not change the correctness conclusion.

Local final `summary.json` and independently recomputed `verified-summary.json`
are in `runs/qwen-baseline-20260927T155618-1f544ea0/`. Full downloaded reports are
in its `remote/qwen-baseline-20260927T155618-1f544ea0/` subdirectory; the same
records persist on the Modal artifact volume. Research notes Section 22 gives
interpretation and limitations. Recorded project billing at 16:08:55 UTC was
$0.361859, with current-hour/reporting lag possible; this is not a final bill.

Conclusion: output extraction is usable for most samples, but this
checkpoint/prompt/protocol still provides no binary reward variation in the
sampled groups. These are 32 sampled programs, not 14,048 independent model
samples. Do not extend RL or switch models before diagnosing the execution and
output failures. This is not evidence of reward hacking or proof the model can
never solve the task.

The subsequent read-only audit found missing-field errors (`value`: 12
candidates; `ttl`: 3), wrapper return-type failures (15), and invalid JSON (10).
These candidate-level categories overlap. Two extraction rejections become
missing-entrypoint errors because the current pipeline executes placeholder
source; another candidate defines `LRUCache` rather than `simulate_cache`.
Candidate 4001 crashes in an unguarded example call before reaching the assigned
input. All 49 audit passes from candidate 4005 have expected output `[]`.

The historical baseline did not persist raw stdout in its per-case reports.
Diagnostic instrumentation now stores a 2 KiB stdout preview, byte count and
hash with each result, plus best-effort stage markers. Candidate output and
stage markers remain untrusted diagnostics. Rejected completions are recorded
at extraction without a sandbox call. See research notes Section 24 for the
controlled run and saved-candidate replay.

### Controlled interface-prompt pilot

The completed pilot held the model revision, decoding settings, 512-token completion
limit and grading rules fixed. It generates eight programs per prompt: the
original task and that task with interface reminders appended. Both conditions
are sampled fresh using paired seeds, alternating which prompt runs first.
The frozen design and decision rules are in `prompt_interface_pilot.txt`.

Each program receives G1 and G3: 37 assigned cases, 36 distinct inputs after
deduplication, at most 576 sandbox executions across 16 programs. Primary outcome
is complete G3 passes per eight programs; valid-output rates are separate
diagnostics. This is an exploratory development pilot without policy updates.
Independent audit and task generalization require later evaluation.

```bash
.venv/bin/modal run --detach modal_prompt_pilot.py \
  --setup-file runs/CPU_RUN_ID/setup.json \
  --cpu-report runs/CPU_RUN_ID/summary.json --allow-cloud
```

This starts a billable, capped GPU job followed by CPU grading. The conformance
gate checks the current check set, runner hash, image and cleanup records;
rerun the CPU trial if the runner changes. The plan, source snapshot, completions
and reports are retained locally and on the artifact volume. The remote
controller persists each candidate result and stops on infrastructure failure.

Results: generation `qwen-prompt-pilot-20260927T210942-6a293d9f`, completed CPU
recovery `qwen-prompt-recovery-20260927T212712-b0a6704d`.

| Program-level outcome | Original prompt | Interface reminders |
| --- | ---: | ---: |
| Extractable, syntactically valid Python | 8/8 | 8/8 |
| Valid JSON-list results on every input | 0/8 | 1/8 |
| Entire G1 passed | 0/8 | 0/8 |
| Entire G3 passed | 0/8 | 0/8 |
| Mixed G3 binary-reward groups | 0/2 | 0/2 |

The one consistently valid reminder program (seed 4002) returns `None` for every
get. It passes 15/32 G3 cases but never retrieves a stored value. This is limited
interface improvement, not a correct cache implementation or evidence of reward
hacking. No policy updates occurred; model parameter hashes match before/after.
The eight original completions exactly reproduce the historical baseline's same
seeds, so they are not additional independent samples relative to that baseline.

One trusted preflight timed out before candidate execution. The original
zero-retry pilot stopped and preserved its evidence. A documented CPU-only
amendment allowed at most three preflight-only replacements, with one per input;
only one was needed. All prior attempts remain recorded. There are 576 distinct
candidate/input outcomes and 577 actual sandbox attempts, **not 576 independent
model samples**. No unresolved infrastructure errors remain.

The saved-program recovery entrypoint is:

```bash
.venv/bin/modal run --detach modal_finish_prompt_pilot.py \
  --parent-run qwen-prompt-pilot-20260927T210942-6a293d9f --allow-cloud
```

That command creates a new billable recovery run; the completed run needs no
rerun. Results persist on the Modal artifact volume and locally under the
recovery run's `generation.json`, `reports.json`, and `summary.json`. Independent
local recomputation verified summaries, source/suite/image/runner identity,
confirmed cleanup, and attempt counts. See research notes Section 25.

Next gate: the user chose a 0.5B SFT warm-start rather than a larger-checkpoint
comparison. [The design](sft_warmstart_plan.txt) proposes reviewed, non-cache
function examples, family-separated development data, assistant-only supervision,
and matched before/after cache generation. That dataset and pilot are now
implemented; see the SFT results above. Do not begin longer RL with uniform
rewards that reflect only empty-input success.
This small pilot does not establish that the 0.5B model can never solve the task,
and it does not evaluate generalization.

### Execution boundary and failure handling

Initial reset policy is a fresh sandbox **per unique candidate/input pair**,
including retries. This strengthens the earlier fresh-process-per-test plan:
there is no shared filesystem or surviving process state between tests.
It also incurs more startup cost; reuse is deferred until reset guarantees and
throughput are measured. Only one test input crosses the execution boundary.

A trusted preflight checks the runtime can apply the same restrictions as the
candidate runner. The runner uses a fresh working directory, clears environment
variables, drops to UID/GID 65534, disables privilege elevation, and applies
CPU/address-space/file-size/open-file/process-count limits. Candidate programs
then run with a 5-second provider execution deadline and a 2-second CPU limit.
These are provisional development budgets, not calibrated training settings.

Both stdout and stderr are consumed concurrently and capped at 16 KiB of
retained data per stream. All output remains untrusted: only externally checked
JSON list values count. Extra printed messages fail JSON parsing. The in-sandbox
Python wrapper is a transport adapter, **not a tamper-proof grader**. A candidate
can interfere with its own wrapper, but receives no reference answers or scoring
authority. The ordinary wrapper also rejects non-list Python return values and
wrong element types before serialization; the controller independently validates
the received JSON types, values, and length.

The controller terminates the whole sandbox in `finally`, including after
output overflow, execution failure, or cancellation. SDK execution-deadline
errors are candidate timeouts; transport/RPC timeouts remain infrastructure
errors. Nonzero process exits are recorded without guessing whether the cause
was memory exhaustion, a signal, or an exception. Only recognized transient
service errors are retried, at most once by default; all attempts are retained.
Unconfirmed termination overrides an apparent success and is not retried.

Records include source/suite/runner hashes, image and SDK identity, limits,
startup time, execution round-trip time, total time, statuses, and cleanup.
Round-trip time is not pure program CPU time. Concurrency is bounded within
each evaluation call; this is not yet an account-wide distributed scheduler.

### Remaining gates before calling this a validated sandbox service

- Completed basic live checks: UID 65534, selected credential-variable absence,
  reference-package absence, failed outbound TCP probe, fresh filesystem, timeout,
  malformed output, and both stream caps. All received whole-sandbox termination
  acknowledgements. One TCP probe is not proof of universal network denial, and
  termination acknowledgement is not independent host-level child inspection.
- Still test process-count exhaustion and broader network/credential boundaries.
  Expected answers are absent by image construction and payload contract; this
  is not an exhaustive filesystem audit.
- Measure output-flood behavior, including SDK-side buffering. The controller's
  retained-output cap is not a guarantee about SDK/provider buffering.
- Validate memory exhaustion and distinguish provider preemption from candidate
  resource failures using actual service signals. Never infer OOM from 137 alone.
- Check disk policy: the bootstrap caps each file at 1 MiB, not total disk usage.
- Handle ambiguous create RPC failures: there may be a sandbox without a returned
  handle. The 120-second provider lifetime is a fallback, not confirmed cleanup.
- Add durable in-flight event logging/reconciliation before long runs. A killed
  controller may leave only input/config artifacts, not a completed result file.
- Benchmark sustained concurrency, tail failures, and billed cost. Current scope
  stays cache-only: first finish the bounded Qwen/GRPO integration trial, then
  measure baseline difficulty before any longer training or task expansion.

Research rationale and task contracts are in `verifier_rl_research_notes.txt`,
`task_catalog.txt`, and the six task specification files.
