# Code map

Start with the [research report](booking_replication_analysis.md) for the question and results. The completed four-seed booking study is the main path through this repository.

The [project brief](frontier_evals_project_brief.md) summarizes the research engineering work. The [expanded article](booking_verifier_blog.md) and [sensitivity appendix](booking_verifier_blog_methods.md) explain the additional archive-based analyses.

## From a generated program to an update

1. [`booking_replication.py`](../verifier_rl/booking_replication.py) fixes the arms, seeds, repair cases, evaluation panels, and workload. Its eight-case replacement changes which tests contribute to reward.
2. [`modal_booking_replication.py`](../modal_booking_replication.py) constructs the model and TRL trainer, coordinates grading, and saves experiment state. This is a cloud launcher, not a local reproduction command.
3. [`program_execution.py`](../verifier_rl/program_execution.py) runs candidate programs in Modal sandboxes with restricted per-input child processes. Model-generated source remains data in local analysis.
4. [`program_grading.py`](../verifier_rl/program_grading.py) validates identities and execution results, then selects the condition's scoring subset. An unknown input outcome blocks the reward update.
5. [`grpo_recovery.py`](../verifier_rl/grpo_recovery.py) wraps generation and checkpoint hooks to preserve pending rollouts and optimizer/RNG state. TRL retains responsibility for the GRPO objective.

## Recovery and controls

| Concern | Implementation | Evidence or tests |
| --- | --- | --- |
| Saved observations and bounded storage retries | [program_storage.py](../verifier_rl/program_storage.py) | [continuation tests](../tests/test_program_storage.py), [recovery tests](../tests/test_program_storage_recovery.py), [historical control record](program_storage_recovery.txt) |
| Full-state training resume | [grpo_recovery.py](../verifier_rl/grpo_recovery.py) | [recovery tests](../tests/test_grpo_recovery.py), [release record](booking_matched_training_release.txt) |
| Program-level grading and batching | [program_execution.py](../verifier_rl/program_execution.py), [program_grading.py](../verifier_rl/program_grading.py) | [execution tests](../tests/test_program_execution.py), [grading tests](../tests/test_program_grading.py), [benchmark record](program_sandbox_benchmark.txt) |
| Permits and resource accounting | [permit recovery record](permit_clock_recovery.txt) | [runtime integration record](program_sandbox_integration.txt) |

These controls validate the documented execution scope; they are not a general-purpose sandbox security proof.

## Analysis paths

- **Public reproduction:** [`scripts/reproduce.py`](../scripts/reproduce.py) connects the two public analyzers, compares their rows and published outputs, and writes fresh results under `build/reproduction/`.
- **Portable score arithmetic:** [`scripts/booking_publication.py`](../scripts/booking_publication.py) reads the 2,048-row score CSV and its checksum manifest.
- **Public archive reanalysis:** [`booking_publication_analysis.py`](../verifier_rl/booking_publication_analysis.py) reads the complete published study record, checks its hash and source bindings, and computes results and post hoc sensitivity checks.
- **Original private-bundle validation:** [`booking_replication_analysis.py`](../verifier_rl/booking_replication_analysis.py), [`booking_behavior_analysis.py`](../verifier_rl/booking_behavior_analysis.py), and [`booking_complete_report.py`](../verifier_rl/booking_complete_report.py) have additional prerequisites in the original ignored `runs/` bundle.

The [reproduction guide](reproduction.md) explains which commands require those private inputs. The public path does not skip its archive checks when private inputs are missing.

## Historical work

The repository retains the original cache study, task screening, booking pilots, recovery amendments, and frozen protocols. Some active modules import those earlier helpers, and source-validation tests use their original paths. They remain in place for traceability. The [historical README material](repository_history.md) records their original statuses and commands; the current entry point is the [root README](../README.md).
