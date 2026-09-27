# Verifier-RL

A development evaluator for studying how test coverage affects rewards for
generated code. **The cache task is implemented; the other five tasks remain
specifications. CPU checks, the Qwen/GRPO smoke trial, and a 32-sample
generation-only baseline have completed.**
The first model trial had four zero rewards and unchanged weights: the pipeline
ran, but it did not demonstrate learning. See the trial results below.

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

Output directories must be new: existing runs are never overwritten. Generated
directories are private and gitignored. Keep durable copies of research artifacts;
gitignore is not a backup strategy. Suite manifests contain development inputs
and expected answers and belong on the controller, not in candidate images.

The demo prints **test cases passed / test cases assigned**, not fractional RL
rewards. Training reward is 1 only for a completely passing suite; otherwise 0.
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
| `verifier_rl/model_trial.py`, `modal_model_trial.py` | Bounded Qwen/GRPO integration trial; protected CPU rewards and separate audit |

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

For example, after exporting the fixture demo:

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
.venv/bin/modal run modal_model_trial.py \
  --setup-file runs/NEW_SETUP/setup.json \
  --cpu-report runs/CPU_RUN_ID/summary.json
```

The model trial checks the CPU report's image identity, uses one L4 with a
900-second function deadline and no application retries, and runs exactly one
GRPO step (four rollouts, binary G3 rewards). No SFT or public serving endpoint
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
Two samples and one step are **only an integration test**, not learning evidence.

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

Raw stdout is consumed for scoring but is not persisted in the per-case report.
Demonstration prints can contaminate the JSON channel, but individual invalid-JSON
failures need captured output to establish their cause. The next diagnostic
milestone is bounded stdout/stderr persistence and explicit failure stages,
followed by controlled correct/faulty programs and a small replay of saved
candidates. This work is planned, not implemented. Preserve the original
baseline and version any protocol changes. See research notes Section 23.

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
