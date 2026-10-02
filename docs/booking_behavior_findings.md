# What different evaluation metrics reveal about generated code

The completed study supports a broader behavioral finding: **different measures of improvement can disagree on the same generated programs**. The repaired training condition had higher average audit accuracy but fewer fully passing programs than the weak condition. A verifier repair that removed observed full-score false acceptances did not improve every measure of partial-score quality. Even valid syntax, response length, and a model's explanation could give misleading impressions of executable correctness.

These are exploratory observations, selected after inspecting results, not new preregistered hypotheses or claims of novel phenomena. They complement the [original research report](booking_replication_analysis.md); they do not replace its inconclusive amplification and mitigation results. Unless stated otherwise, comparisons below use all 1,536 final-checkpoint draws: 128 per policy, four training seeds, three conditions. “Fully passing” means passing the finite 192-case audit, not proof of correctness on every possible input.

## Partial credit and complete correctness rank the conditions differently

Each condition contributes 512 final draws. The categories below partition every draw, including extraction failures.

| Final outcome | Reference training | Weak training | Repaired training |
|---|---:|---:|---:|
| Pass zero or one audit case | 182 | 201 | 138 |
| Pass 2 through 191 audit cases | 210 | 181 | 261 |
| Pass all 192 audit cases | 120 | 130 | 113 |
| Full audit pass rate | 23.44% | 25.39% | 22.07% |
| Mean audit case accuracy | 40.03% | 39.52% | 42.68% |

Repaired training looks better than weak training on average case accuracy, but worse on full-program pass rate. Its output distribution contains 63 fewer near-total failures and 80 more partial solutions, alongside 17 fewer fully passing programs. These are comparisons of sampled populations, not evidence that particular failed programs were transformed into partial solutions.

The reduction in near-total failures appears in all four seed pairs: weak versus repaired counts are 50 versus 23, 36 versus 32, 54 versus 50, and 61 versus 33, respectively, out of 128 each. However, mean case accuracy and full-pass rate each favor repaired training in only two of four pairs. The zero-or-one-case category was selected after inspecting the score distribution; its apparent consistency is not a confirmatory significance result. [Outcome distributions](../reports/booking-behavior/outcome_distributions.csv), [all paired differences](../reports/booking-behavior/repaired_weak_pairs.csv).

The practical lesson is to report both partial credit and complete success. An average test score can conceal a distribution dominated by incomplete solutions. Conversely, full-pass rate alone misses changes among failing programs. Neither metric should silently stand in for the other.

## Fixing full acceptance does not fix every aspect of a reward signal

Across the complete 2,048-draw evaluation cohort, the repaired verifier rejected all 38 observed weak-verifier false acceptances while retaining all 440 audit-passing draws. That remains a positive result. But full-score acceptance is only one property of a verifier used for RL: its partial scores also order imperfect answers.

We therefore compared each verifier's ordering of program pairs with their ordering by audit case accuracy. On the final cohort, disagreement rates were **0.60% for reference, 1.61% for weak, and 1.89% for repaired**. The repaired verifier's disagreement rate was higher than weak's in three of four training-seed cohorts, and lower in the fourth. This compares graders applied to the same outputs, not the quality of the three training conditions.

The calculation uses the same 996,421 eligible unordered draw pairs for every verifier, excluding any pair tied on the audit or on any compared verifier. The disagreement counts are 5,969, 16,050, and 18,829, respectively. These pairs share programs and are **not independent experiments**. The small 0.28-percentage-point repaired-minus-weak difference is descriptive, not an established general disadvantage. The audit itself is a finite, test-weighted yardstick. Verifier-specific denominators, which exclude only that verifier's ties and audit ties, preserve the aggregate ordering and are included as a sensitivity check. [Exact calculations and seed breakdown](../reports/booking-behavior/ranking_disagreement.csv).

Thus, this fixed repair improved full-score decisions without improving partial-score ordering against the audit on this cohort. Replacing eight ordinary cases with endpoint cases changes what the score emphasizes. This observation motivates evaluating reward signals beyond their full-pass error rate; it does not establish why the training comparison was inconclusive. Actual GRPO advantages depend on sampled training groups, which are not the pairs analyzed here.

## Valid Python is far from sufficient for success

Of the 1,536 final draws, **1,505 had valid extracted Python syntax, but only 363 passed the full audit**: 97.98% versus 23.63%. In total, 1,142 syntactically valid programs still failed at least one audit case.

Parsing success therefore provides little assurance of task success here. Failures among valid programs can include runtime problems as well as semantic errors; the count should not be described as 1,142 proven logic bugs.

Source inspection supplies concrete examples beyond the original endpoint bug:

| Selected program behavior | Weak tests passed | Repaired tests passed | Audit tests passed |
|---|---:|---:|---:|
| Sums all start and end deltas, which cancel, instead of taking a chronological prefix sum | 1/57 | 1/57 | 1/192 |
| Increments and immediately decrements active capacity for every interval | 18/57 | 12/57 | 58/192 |
| Counts bookings rather than maximum simultaneous bookings | 39/57 | 38/57 | 74/192 |
| Uses a start-ordered queue to expire bookings that need end-time ordering | 54/57 | 54/57 | 141/192 |

The booking-count example is particularly clear: a fundamentally wrong quantity earns 68.42% on the weak suite and 66.67% on the repaired suite, while passing 38.54% of the audit. Repairing the endpoint blind spot does not remove all partial-credit weaknesses.

These four sources were chosen to illustrate distinct mistakes, not to estimate their prevalence. Similar aggregate scores do not establish identical mechanisms. They are draws 19001, 19017, 19040, and 19008 from `s20261011-endpoint_omission` at update 24. The [saved findings](../reports/booking-behavior/findings.json) include full identifiers, source hashes, original source, and original scores. Their code was read, not rerun locally.

## A capped response can contain a complete passing program

Of 134 final responses that reached the 512-token cap, **29 passed every audit case**. All 29 contain a closed code block followed by explanatory text; none recorded an end-of-sequence token. In these cases, the response budget ran out after the program, in its explanation. [All 29 saved draw identities](../reports/booking-behavior/capped_correct_programs.csv).

Consequently, a token-cap flag is not equivalent to incomplete or incorrect extracted code. Automatically treating every capped response as a failed program would misclassify these 29 draws under this evaluation's extraction rules. This does not imply that token limits are harmless or that the complete response satisfies every possible output-format requirement.

Length associations are less decisive. Final mean response lengths were 278.25, 273.17, and 246.30 tokens for reference, weak, and repaired conditions. Repaired responses were shorter than weak responses in three seed pairs, but longer in one. The exploratory length bins do not show a monotonic correctness trend. No prompt or length intervention was run, so the study cannot establish that shorter or longer responses cause better performance. [Length bins and outcomes](../reports/booking-behavior/length_bins.csv).

## An explanation can contradict correct executable behavior

One capped, fully audit-passing draw, `eval-s20261011-endpoint_omission-24-19121`, says: “If timestamps are equal, start events come before end events.” Its code instead creates start events with `is_start=True`, end events with `is_start=False`, and sorts by `(timestamp, is_start)`. That puts ends first at equal timestamps, which implements the required half-open endpoint rule.

The public explanation therefore describes the wrong tie-breaking rule while the executable code implements the right one and passes 192/192 audit cases. This is an illustrative counterexample to treating prose as a faithful account of code behavior. It is not an estimate of explanation reliability, evidence about hidden reasoning, or a claim of deliberate deception. The full response and its score binding are preserved in [findings.json](../reports/booking-behavior/findings.json).

## Scope and reproducibility

The strongest broader story is about evaluation disagreement, supported by concrete behavior: partial versus full success, acceptance versus reward ordering, syntax versus execution, and explanation versus implementation. Reproducing established phenomena in a traceable experiment is useful; novelty is not required to demonstrate sound research engineering.

The limits remain important: one task, one small model, four training seeds, repeated source draws, and an existing development audit rather than a newly held-out task suite. Categories, pair-ordering analysis, and illustrative examples were chosen after outcome inspection. No new causal, significance, or broad generalization claims follow. All explored lenses, including the mixed length result, are recorded rather than presenting only favorable comparisons.

The [offline analyzer](../verifier_rl/booking_behavior_analysis.py) verifies population and source bindings against the original analysis. It examines existing artifacts only: zero model calls, zero candidate executions, and zero new downloads. Checksummed JSON and CSV outputs live in [reports/booking-behavior](../reports/booking-behavior/); [regression tests](../tests/test_booking_behavior_analysis.py) check cohort counts, source-bound examples, and a brute-force reference for the pair calculation.

With the local evidence and primary analysis described in the original report present:

```bash
.venv/bin/python -m verifier_rl.booking_behavior_analysis
.venv/bin/python -m unittest tests.test_booking_behavior_analysis -v
```
