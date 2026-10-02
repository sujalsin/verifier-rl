# Verifier-RL

An execution-based study of how incomplete test suites affect reinforcement learning for generated code. I built the task and verifier interventions, integrated their rewards with GRPO, and developed recoverable sandbox execution and evidence validation around the training loop.

**[Research report](docs/booking_replication_analysis.md) · [Article](docs/booking_verifier_article.md) · [Methods](docs/booking_publication_review.md) · [Implementation guide](docs/code_map.md)**

The completed booking-capacity study used Qwen2.5-Coder-1.5B-Instruct, three training conditions, four paired seeds, and 2,048 evaluation draws.

- **Grader result:** replacing eight tests rejected all 38 observed weak-verifier false acceptances while retaining all 440 audit-passing draws.
- **Training result:** the four-seed comparison did not establish amplification of the target bug or a training benefit from the repair.
- **Systems result:** a controlled grading benchmark measured 2.75× throughput; fault-injection and restart controls checked that recovery preserved completed observations and training state.

The [publication clarifications](docs/booking_publication_clarifications.md) document the shared empty-input case and the distinction between scored and executed tests. The repair keeps 57 scored tests; every condition still executes the 96-input training universe.

## Reproduce the published results

Requires **Python 3.11+ and Git**. No package installation, GPU, cloud credentials, or private files are needed.

```bash
git clone https://github.com/sujalsin/verifier-rl.git
cd verifier-rl
python3 scripts/reproduce.py
```

This verifies the CSV checksum, parses the public study archive, cross-checks all 2,048 observations, and compares both recomputed analyses with the published results. It also reads the 288 checkpoint receipts and 12 training logs. Generated tables, SVG figures, and a concise report go to **`build/reproduction/README.md`**; tracked evidence and reports remain unchanged.

Expected final counts:

| Training condition | Full audit passes | Target-bug draws |
| --- | ---: | ---: |
| Reference | 120/512 | 7/512 |
| Weak | 130/512 | 7/512 |
| Repaired | 113/512 | 11/512 |

These are generated-program draws from four paired training seeds, not 512 independent training runs. Reproduction checks published saved evidence; it does not rerun training or execute generated programs. See the [reproduction guide](docs/reproduction.md) for the evidence boundary and troubleshooting.

## Inspect the implementation

| Question | Start here |
| --- | --- |
| How are the conditions, seeds, and fixed repair defined? | [Experiment definition](verifier_rl/booking_replication.py) |
| How does sandbox execution become a reward? | [Executor](verifier_rl/program_execution.py) and [reward validation](verifier_rl/program_grading.py) |
| How are completed observations preserved through failures? | [Storage recovery](verifier_rl/program_storage.py) and [tests](tests/test_program_storage.py) |
| What is saved and restored during training? | [GRPO recovery hooks](verifier_rl/grpo_recovery.py) and [tests](tests/test_grpo_recovery.py) |
| What code did the model produce? | [Failure report](docs/booking_replication_analysis.md#two-observed-failure-mechanisms), [failure catalog](reports/booking-replication/analysis.json), and [source-bound examples](reports/booking-behavior/findings.json) |
| How are the public records analyzed? | [Archive analyzer](verifier_rl/booking_publication_analysis.py), [portable analyzer](scripts/booking_publication.py), and [reproduction entry point](scripts/reproduce.py) |

TRL provides GRPO; the trainer wrapper adds recovery hooks without replacing its loss or advantages. Modal provides GPU compute and sandbox primitives. The [code map](docs/code_map.md) explains the project-specific components and their connections.

## Run the offline checks

The full regression suite uses the pinned Modal SDK for mocked-provider tests. It needs no provider credentials.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
make verify PYTHON=.venv/bin/python
```

`verify` runs repository checks, the authored demo, regression tests, and public result reproduction. Checks requiring the original private `runs/` bundle are explicitly skipped when it is absent. Public-archive reproduction is still required. CI runs the same checks on Python 3.12 and 3.13.

For Windows without Make, use `.venv\Scripts\python.exe -X utf8` in the equivalent commands listed in the [reproduction guide](docs/reproduction.md).

## Repository layout

- `verifier_rl/`: task definitions, grading, recovery, and analysis modules.
- `tests/`: authored fixtures, mocked-provider checks, and evidence validation.
- `scripts/`: local reproduction, reporting, and repository maintenance.
- `docs/`: study reports, methods, preserved protocols, and the approximately 38 MiB public study archive.
- `reports/`: published score tables, manifests, and figures.
- `modal_*.py`: historical and study-specific cloud entry points.
- `build/`: ignored, regenerable local outputs.

Historical source paths remain stable because imports and frozen-source validation depend on them. Earlier cache, task-screen, and pilot work is documented in [repository history](docs/repository_history.md) and [archive/](archive/README.md).

## Evidence and execution scope

The public checkout contains the complete published study record and portable analyses. Model tensors and the original private run bundle are not included. `make evidence-check` requires that private bundle and fails if it is absent or altered; it is separate from public reproduction.

Cloud launchers are opt-in, billable research operations with documented prerequisites. They are not required by any quickstart above. Historical commands in preserved protocols describe past runs; they are not instructions to repeat a completed job.
