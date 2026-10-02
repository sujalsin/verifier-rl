# Reproduction guide

The public checkout supports offline reanalysis of the completed study. Start here to verify the published numbers, then use the [code map](code_map.md) to inspect the experiment and execution system.

## Quickstart

Use Python 3.11 or later. CI checks Python 3.12 and 3.13. Git is needed to obtain the checkout; the reproduction script itself only uses the Python standard library.

```bash
git clone https://github.com/sujalsin/verifier-rl.git
cd verifier-rl
python3 scripts/reproduce.py
```

On Windows, use `py -3 -X utf8 scripts/reproduce.py`. From an existing checkout, `make reproduce` is equivalent. The script also works when invoked by its absolute path from another directory.

A successful run verifies:

1. The score CSV's original byte checksum and all 2,048 expected sample identities.
2. The complete public study archive's byte count and checksum, embedded source bindings, case inventory, 288 checkpoint receipts, and 12 training logs.
3. Agreement between the CSV and archive on each sample's identity, source hash, generation metadata, and shared score fields.
4. Agreement of both recomputed analyses with the checked-in JSON results, including the paired effects and sensitivity analyses. Only floating-point roundoff is tolerated.

All inputs are read-only. Outputs are written to the ignored `build/reproduction/` directory after validation succeeds:

| Output | Contents |
| --- | --- |
| `README.md` | Headline counts and links to the generated files |
| `verification.json` | Verification scope, record counts, Python version, and SHA-256 fingerprints of the inputs and implementation |
| `portable-analysis.json` | Portable arithmetic, per-seed results, and publication follow-up checks |
| `archive/analysis.json` | Reanalysis of the public archive, uncertainty, and limitations |
| `archive/*.csv` | Program rows, paired sensitivity, sampling blocks, and audit thresholds |
| `archive/*.svg` | Paired effects, output distribution, and audit thresholds |
| `archive/manifest.json` | Analyzer and generated-output fingerprints |

The final full-audit counts are **120/512 reference, 130/512 weak, and 113/512 repaired**. Target-bug counts are **7/512, 7/512, and 11/512**, respectively. Across all 2,048 draws, the repair rejects the 38 observed weak false acceptances and retains all 440 audit-passing draws. The four paired training seeds do not establish amplification or a training benefit from the repair.

## Run the full offline suite

The public result check above has no third-party dependencies. The broader regression suite imports the pinned Modal SDK to test provider integrations with authored mocks. Installation requires network access; running the checks requires neither provider credentials nor model weights.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
make verify PYTHON=.venv/bin/python
```

`make verify` runs repository hygiene and published-file checksum checks, an authored demo, the regression suite, and public result reproduction. The full suite can take several minutes. Tests requiring private run artifacts report explicit skips when those artifacts are absent; the public reproduction step must still pass.

Without Make, run these equivalent commands from the repository root:

```bash
.venv/bin/python scripts/check_repository.py
git diff --check
.venv/bin/python -m verifier_rl demo
.venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python scripts/reproduce.py
```

On Windows, create the environment with `py -3 -m venv .venv`, then substitute `.venv\Scripts\python.exe -X utf8` for `.venv/bin/python` in the installation and check commands. UTF-8 mode keeps report text consistent on systems with a different default encoding. No environment activation is required. CI runs the result check with `python -I -S` before installing packages, then runs the demo and full suite on Linux.

## What each level establishes

| Level | Public checkout sufficient? | What it checks |
| --- | --- | --- |
| `python3 scripts/reproduce.py` | Yes; standard library only | Consistency of saved public evidence and published arithmetic |
| `make verify` | Yes, after installing `.[test]` | Reproduction plus authored implementation and mocked-provider regression checks |
| `make evidence-check` | No; original private `runs/` bundle required | Byte fingerprints and deeper validation of the original local evidence inventory |
| A new training run | No; cloud configuration, compute budget, model access, and fresh run configuration required | A new experimental replication, not a replay of the saved analysis |

The public archive includes generated source, saved responses, scores, receipts, and logs. Reanalysis treats model-generated source as data. It does not execute those programs, inspect checkpoint tensors, or independently rerun training. Reading a checkpoint receipt is distinct from validating the underlying model state. The audit is a fixed development suite, and the shared empty-input case is documented in [publication clarifications](booking_publication_clarifications.md).

The original launchers use historical run IDs, frozen source checks, and recovery prerequisites. Their protocols are retained as research records. A fresh replication needs a separately configured run and the documented execution controls; rerunning a historical launch command is not the public quickstart.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| `Missing public input` | Use the complete Git checkout, including `docs/booking_complete_study_record.md` and `reports/`. |
| `score CSV differs from its export manifest` | The expected CSV uses CRLF line endings. Use a fresh checkout with the repository's `.gitattributes`; do not resave the CSV or edit its checksum to make it pass. |
| `archive differs from its recorded hash` or a result mismatch | Check local edits and compare with a fresh checkout of the same commit. Keep the original manifest and investigate the mismatch. |
| `No module named modal` during tests | Install `.[test]` using the same Python interpreter that runs the suite. This dependency is unnecessary for `scripts/reproduce.py`. |
| Private-evidence tests are skipped | Expected in a public-only checkout. `make evidence-check` deliberately fails when the required bundle is missing. |
| Permission errors in async or subprocess tests | Run the offline suite in a normal local development shell or the provided CI job; a restricted notebook/container may block even local socket primitives. |

For the optional article HTML preview, install `.[publication]` and run `make publication-preview PYTHON=.venv/bin/python`. Open `build/publication-preview/index.html`. This uses the original article renderer and stores its generated files under `build/`. The dependency-free reproduction already produces the research SVG figures.
