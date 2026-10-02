# Methods and publication notes for the booking verifier study

This companion to [the blog post](booking_verifier_article.md) records exact results, the scope of the evidence checks, and publication-stage sensitivity analyses. The original primary outcomes remain full audit pass rate and inclusive-endpoint signature frequency at update 24. The new checks do not convert the inconclusive training result into a positive or negative causal finding.

## Corrections alongside the preserved historical protocol

The [historical replication protocol](booking_replication_repair_protocol.txt) says, “There is no audit input in the training reward.” That wording is too broad. Comparing the actual input hashes finds exactly one overlap between the 96 training and 192 audit cases: `{"bookings": []}`.

The precise statement is:

> The audit suite was not used to compute training rewards; the training and audit suites share the mandatory empty-input case.

Its training name is `booking-coverage-reward-0.2/training/interaction/00`; its audit name is `booking-coverage-reward-0.2/audit/interaction/00`. Their shared input hash is `acae2e7b7d6adf6cd11723f933e7fb5fecf3a909dfe00c688c860c119c57ef68`. This accounts for the 287 unique inputs in the combined 288 suite entries. The audit is also an existing development suite used in the pilot, independently of this overlap.

The repair matched the **number of tests scored**: 57 for weak and repaired rewards. All three arms executed the same 96 training inputs for each extracted training program. Nonselected outcomes did not contribute to GRPO reward. There is no measured compute saving attributable to using the repaired reward subset.

The original protocol and complete archive remain unchanged. Their hashes at the time of this analysis are retained in the [source manifest](../reports/booking-blog/source_manifest.json). This is a visible clarification alongside the historical record, not a retrospective alteration of its frozen text.

## All four final seed pairs

The primary target signature is behavioral agreement with the inclusive-endpoint oracle across the prescribed 96 training inputs. The broader 287-input signature was a separate validation measure. A weak false acceptance means full weak-suite acceptance despite failing the audit; on this cohort, reference and repaired full-pass decisions both agreed with the audit. The 38 weak errors are 7.95% of 478 weak acceptances, or 2.36% of 1,608 audit failures. Those percentages answer different questions.

Each count is out of 128 draws from the specified final policy.

| Training seed | Reference full audit | Weak full audit | Repaired full audit | Reference target | Weak target | Repaired target |
|---|---:|---:|---:|---:|---:|---:|
| 20261011 | 29 | 21 | 24 | 2 | 2 | 2 |
| 20261012 | 32 | 45 | 33 | 3 | 0 | 1 |
| 20261013 | 27 | 31 | 22 | 1 | 4 | 5 |
| 20261014 | 32 | 33 | 34 | 1 | 1 | 3 |

For each contrast, first subtract the two policy rates within each training seed. With four differences `d`, the interval is `mean(d) ± 3.1824463053 × sample_sd(d) / sqrt(4)`. Percentages below are percentage-point differences, not relative percentage changes. This follows the [paired-observation calculation](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm).

| Contrast | Metric | Mean difference | Exploratory 95% t interval |
|---|---|---:|---:|
| Weak minus reference | Full audit pass | +1.95 | −8.81 to +12.72 |
| Weak minus reference | Target signature | 0.00 | −3.05 to +3.05 |
| Repaired minus weak | Full audit pass | −3.32 | −12.48 to +5.84 |
| Repaired minus weak | Target signature | +0.78 | −0.23 to +1.80 |
| Repaired minus weak | Mean audit case accuracy | +3.17 | −12.43 to +18.76 |

The last row is secondary. Every interval is conditional on this task and fixed evaluation panel. The calculation assumes independent, approximately normal seed differences; four observations cannot make that assumption convincing, particularly for rare errors. The method was selected during analysis, with no multiplicity correction. There is no equivalence margin or equivalence test. The shared baseline is counted once.

Evaluation draws use common random seeds across policies. That pairing can help comparisons, but matching a seed does not make two different policies produce the same source or turn draw pairs into independent training replications. The concerns about few-run RL evaluation also appear in [Agarwal et al.](https://arxiv.org/abs/2108.13264); that broader work is context, not validation of these particular interval assumptions.

## Additional sensitivity checks

These checks were added while preparing the publication. They are exploratory and did not determine which seeds, checkpoints, or draws were reported.

**Leaving out one training seed.** Recomputing each mean after omitting each seed in turn gives the ranges below. These are four descriptive recomputations, not confidence intervals or four new experiments.

| Contrast and metric | Range of means after omitting one seed |
|---|---:|
| Weak minus reference full audit pass | −0.78 to +4.69 points |
| Weak minus reference target signature | −0.78 to +0.78 points |
| Repaired minus weak full audit pass | −5.21 to −1.30 points |
| Repaired minus weak target signature | +0.52 to +1.04 points |
| Repaired minus weak mean case accuracy | −0.03 to +6.56 points |

The repaired-minus-weak primary point estimates retain their directions in these checks. That does not establish harm: the full four-seed intervals still span zero and the study remains small. The secondary mean-accuracy improvement is less stable under seed omission, which is another reason not to promote it to the main success criterion.

**Repeated source strings.** The 2,017 draws with extracted code contain 1,569 distinct exact source hashes. The remaining 31 draws failed extraction and stay in the primary denominator as failures. Among distinct extracted sources, 321 passed the audit and 33 were weak false acceptances. The repair retained all 321 audit-passing sources and rejected all 33 false-accepting sources. Thus the observed grading correction is not solely a repetition of one source string. Exact source diversity still is not semantic diversity or independent replication; deduplicating would change the estimand from draw frequency to source coverage.

**The fixed evaluation prefix.** At update 24, the first 32 draws per policy yielded full-audit totals of 20/128 reference, 26/128 weak, and 28/128 repaired, with target totals of 2, 1, and 0. The complete 128-draw panels yielded full-audit totals of 120/512, 130/512, and 113/512, with target totals of 7, 7, and 11. These nested panels must not be treated as independent samples.

**Partial-score ordering.** A separate arithmetic implementation reproduces 996,421 eligible unordered pairs in the final cohort. Pairs tied on the audit or any compared verifier are excluded from the common denominator. Reference, weak, and repaired scores disagree with the audit ordering on 5,969, 16,050, and 18,829 pairs respectively. Programs appear in many pairs, so those counts cannot support a test that assumes independent pairs. This is a comparison of graders on the same outputs, not a comparison of training conditions.

All unrounded values are in [analysis.json](../reports/booking-blog/analysis.json).

## Why full-score rejection does not determine a GRPO advantage

This is an explanatory calculation, not evidence about an observed training group. The configured reward was fractional test accuracy, normalized within each group. In the idealized calculation, `A_i = (r_i − mean(r)) / sample_sd(r)`, with zero advantages for a tied group. The actual [TRL implementation](https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/grpo_trainer.py) adds a small denominator stabilizer; omitting it here does not change the signs.

Consider four illustrative completions: a correct solution, the inclusive bug, a constant-zero program, and an extraction failure. The corresponding weak rewards are `[1, 1, 1/57, 0]`; repaired rewards are `[1, 49/57, 1/57, 0]`.

| Hypothetical group | Inclusive bug advantage with weak rewards | Inclusive bug advantage with repaired rewards |
|---|---:|---:|
| Correct, inclusive, constant zero, extraction failure | +0.87 | +0.73 |
| Three correct programs and one inclusive program | 0.00 | −1.50 |

The repair lowers the buggy program's reward in both examples. In the first group it remains above average; in the second it falls below average. Full-score rejection therefore does not guarantee a negative advantage. A positive advantage also does not, by itself, prove that the frequency of a semantic bug will increase after an optimizer update.

The compact archive contains training aggregates and initial token groups, not every original training completion and per-group reward vector. This analysis consequently cannot identify how often target programs were sampled during training or measure the resulting advantage distribution. A mechanism study would require those records. Cross-policy comparisons of final outputs cannot substitute for them.

## What was verified for publication

The existing analysis tests passed: **47 tests** across replication analysis, replication protocol, pilot failure analysis, and exploratory behavior analysis. These included fixed-population checks, source-to-score bindings, and revalidation of all 10,906 saved protected input outcomes for the 38 false acceptances. The local evidence test disables socket creation.

The publication script independently recomputes headline counts, paired intervals, outcome distributions, pair-ordering counts, and the fixed-prefix comparison from an exported table. It also adds the seed-omission and source-deduplication checks. Source hashes identify the frozen inputs used for that export.

All three figures were visually inspected. Local article and methods links, generated HTML links, SVG syntax, and source/output hashes were checked. The HTML preview is generated locally; rendering within the future website remains to be checked when that site exists.

This is an offline reanalysis. It does not rerun training, execute model-generated code, rehash downloaded model tensors, or independently audit the entire raw execution archive. Matching summaries and hashes establishes internal consistency and provenance within the available records; it is not a second laboratory replication.

## Reproduction paths

To reproduce the article's arithmetic from a repository checkout containing the [score CSV](../reports/booking-blog/program_scores.csv), [source manifest](../reports/booking-blog/source_manifest.json), and [script](../scripts/booking_publication.py), use Python 3.11 or newer:

```bash
python3 scripts/booking_publication.py
```

This path uses only the standard library. It checks the CSV hash, the fixed 2,048 sample identities, and score consistency before writing the derived analysis. It requires neither Modal credentials nor checkpoints. This checks calculations conditional on the exported scores; it does not independently reproduce the scores themselves.

For the figures, install Matplotlib in your chosen environment. For the local HTML article preview, also install `markdown-it-py`, then run:

```bash
python3 scripts/booking_publication.py --figures --preview
```

To rebuild the score export from the original evidence, first obtain the compact bundle and targeted failure records under `runs/qwen-booking-replication-repair-20260930-v1/local-analysis/`, together with the existing primary analysis and source checkout whose scientific hashes match that bundle. Then run:

```bash
python3 scripts/booking_publication.py --export-evidence
python3 -m unittest tests.test_booking_replication_analysis tests.test_booking_replication tests.test_booking_failure_analysis tests.test_booking_behavior_analysis -v
```

The complete test command additionally needs the pilot and benchmark artifacts documented in the [technical report](booking_replication_analysis.md). These are explicit evidence prerequisites, not files obtained automatically by the publication script.

A fresh training reproduction is a separate, paid experiment. The recorded runtime used Python 3.12, Modal 1.5.5, PyTorch 2.8.0, Transformers 4.57.1, TRL 0.28.0, datasets 3.5.1, accelerate 1.12.0, and an L40S GPU. It requires the pinned original model, the protocol's sampling and optimizer settings, the execution amendments, and validated recovery controls. Historical resume commands target existing run state and must not be presented as a fresh-run quickstart. [Launcher](../modal_booking_replication.py), [frozen protocol](booking_replication_repair_protocol.txt), [runtime amendment](program_sandbox_integration.txt).

## Website release status

The article, HTML preview, figures, and score table are local publication materials. They have not been deployed. The personal website is not yet available, and the study-specific files are currently untracked in the local repository; these local links are not evidence of a publicly accessible release.

When the website exists, import the article, serve its figures, and replace repository-relative evidence links with links to an accessible, fixed code/data release. The short score table supports numerical inspection. Replaying the deeper evidence requires publishing or otherwise providing the additional bundles described above. The complete archive is approximately 38 MiB and is best offered as an optional evidence download rather than the main article.
