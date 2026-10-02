# Publication clarifications for the booking verifier study

This note corrects two presentation ambiguities without modifying the frozen protocol, original results, or complete evidence record. It is an annotation for readers of the forthcoming public article, not a scientific or execution amendment. No experiments were rerun.

## The training and audit suites share one input

The [historical protocol](booking_replication_repair_protocol.txt) says:

> There is no audit input in the training reward.

Read literally as a claim of disjoint input sets, that sentence is incorrect. The precise statement is:

> The audit suite was not used to compute training rewards; the training and audit suites share the mandatory empty-input case.

The training suite has 96 inputs and the development audit has 192. Their intersection consists of `{"bookings": []}`, so the union has 287 unique inputs, not 288. The shared input is present in reference, weak, and repaired training rewards. Training uses the training-suite version of that case; no audit score is supplied to GRPO. This does not make the audit an untouched benchmark: it is the existing development suite, previously used during the pilot.

The two logical case identifiers are:

- `booking-coverage-reward-0.2/training/interaction/00`
- `booking-coverage-reward-0.2/audit/interaction/00`

Both bind to input SHA256 `acae2e7b7d6adf6cd11723f933e7fb5fecf3a909dfe00c688c860c119c57ef68`. The publication analyzer verifies that this is the only overlap using every case definition embedded in the [complete record](booking_complete_study_record.md).

Any unqualified statement in the historical narrative that audit *inputs* never contribute to reward should be read with this correction. The correction concerns input overlap, not a change to the scoring code or saved results.

## Matching scored test count is not a demonstrated compute saving

The weak and repaired verifiers each score 57 training tests. Repair replaces eight ordinary tests with eight shared-endpoint tests; it does not add eight tests on top of 57. Reference scores 96.

However, every condition executes the same 96-input training universe and the trusted grader selects the appropriate subset for reward computation. The count-matched repair therefore does not demonstrate reduced execution cost. It also does not match test difficulty, information content, or the distribution of rewards. Public descriptions should say **the same number of scored tests**, not “the same compute” or “39 fewer executions.”

The separate 2.75× throughput result comes from a controlled execution benchmark with equal workloads. It is unrelated to reducing the number of scored tests.

## Preserved originals

| Original file | SHA256 before this annotation |
| --- | --- |
| `docs/booking_replication_repair_protocol.txt` | `29511096263b54ac52f458f933525aca69c84127d1d931072ac08ef8e576f4a5` |
| `docs/booking_complete_study_record.md` | `deff92ce68edfb264247c315c092cc7ca32168bec1c9d0669a638fe6968f06c4` |

Those files remain unchanged. The article and methods appendix link this note visibly so a reader need not discover the correction by comparing implementation files.
