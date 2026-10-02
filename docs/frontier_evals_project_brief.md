# Verifier reliability project brief for research engineering

The strongest positioning for this project is a controlled study of verifier reliability backed by a recoverable RL evaluation system. Its contribution is not a demonstrated solution to reward hacking. It connects a concrete grading defect, a fixed repair, paired training experiments, failure analysis, and the infrastructure needed to preserve trustworthy measurements when execution fails.

This framing is relevant to the [Research Engineer Frontier Evals and Environments role](https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/), particularly its emphasis on measurement variance, reliability, scalable evaluation, and carrying behavioral questions through experiments and analysis. It is evidence of relevant skills, not a guarantee of an interview or proof of frontier-scale experience.

## Resume entry

**Verifier Reliability and GRPO Evaluation Infrastructure**  
Python, PyTorch, TRL, Modal, RL post-training, evaluation methodology

- Designed and ran a four-seed, 12-run GRPO study of Qwen2.5-Coder-1.5B, evaluating 2,048 program draws with paired seed-level effect estimates and reproducible failure analysis.
- Built a test-count-matched verifier repair that rejected all 38 observed false acceptances while retaining all 440 audit-passing draws on a fixed evaluation cohort; distinguished offline grading gains from inconclusive training effects.
- Engineered resumable evaluation with durable execution evidence, bounded storage retries, and full-state GRPO recovery; measured 2.75× grading throughput on a controlled 2,009-input benchmark with matching outcomes.

If space permits only two bullets, keep the experiment and infrastructure bullets, and put the exact repair result in the linked project description. Keep “observed,” “fixed cohort,” and “benchmark”: those qualifiers make the numbers accurate. Use ownership verbs for decisions and implementation you can explain and defend personally.

## What the project demonstrates

| Capability | Concrete evidence | Boundary of the claim |
|---|---|---|
| Experimental design | Frozen reference, weak, and repaired arms; matched starts; four paired seeds; fixed endpoint | One task and model, not a broad capability benchmark |
| Measurement judgment | All seed differences, exploratory uncertainty, common-draw sensitivity, separate pilot | Four seeds give limited precision; no equivalence claim |
| Behavioral investigation | Original-result validation for all 38 false-acceptance draws; two endpoint mechanisms | Saved finite-input behavior, not proof of intent or universal semantics |
| Evaluation systems | Bounded parallel scheduling, program sandboxing, protected grading, checksummed evidence | Authored controls, not a security certification or production-scale deployment |
| Failure recovery | Injected publication failures; no reexecution on storage retry; checkpoint and clock-skew recovery | Explicitly documented incidents and tested recovery paths, not outage immunity |

These are transferable research-engineering skills. The role also involves ambitious environments and frontier-model work. This project does not yet demonstrate long-horizon agent environments, multi-task generalization, or shipping model improvements to users. Do not imply that it does.

## A concise interview explanation

“I studied whether a code verifier's blind spot becomes a training incentive. The task was interval booking capacity: omitting shared-endpoint tests lets incorrect code earn full reward. I froze a reference verifier, a weak verifier, and a repair with the same test count, then trained a 1.5B model across four paired seeds.

“The repair rejected all 38 observed false acceptances on the complete 2,048-draw evaluation cohort and retained all 440 audit-passing draws. But the weak and reference training conditions each produced seven exact target bugs in their final 512 draws, and the repaired condition produced eleven. I reported that we had improved the verifier without establishing improved learned behavior.

“The infrastructure was part of making that conclusion credible: I separated candidate execution from storage publication, tested recovery with execution disabled, preserved full training state across interruptions, and benchmarked parallel grading at 2.75× the sequential throughput on the same workload.”

The [technical report](booking_replication_analysis.md) contains every denominator, all four seed pairs, uncertainty assumptions, and recovery limitations behind that explanation.

## A broader behavioral research story

The [exploratory behavior analysis](booking_behavior_findings.md) offers a second, complementary angle: choosing an evaluation metric changes what improvement looks like. Repaired training had higher mean audit case accuracy than weak training, 42.68% versus 39.52%, but fewer fully passing programs, 22.07% versus 25.39%. It produced fewer near-total failures in all four seed pairs, without a consistent improvement in full correctness. A successful full-acceptance repair also did not improve every measure of partial-score ordering on saved outputs.

For a behavior-focused resume, an optional replacement for the repair bullet is:

- Audited 1,536 final code generations across four training seeds, identifying disagreement between partial-credit accuracy and full-program success; traced algorithmic errors and 29 token-capped responses containing fully audit-passing code.

Describe these as exploratory observations on one task, not newly discovered universal model properties. In an interview, a useful concrete example is a response whose prose gives the wrong endpoint tie-breaking rule while its code implements the correct rule and passes every audit case. That illustrates why executable behavior and an explanation should be evaluated separately; it does not establish deception or hidden reasoning failure.

## Questions to be ready to answer

**Why does a null or inconclusive training result matter?** The exploratory pilot suggested amplification, but the new seed pairs did not reproduce a positive average target-bug effect. The pipeline let the evidence change the conclusion instead of promoting a favorable pilot. The offline repair finding remains real and separately measured.

**Did the repaired verifier make the model worse?** Its observed final audit full-pass mean was 3.32 percentage points below weak training, while target-bug frequency was 0.78 points higher. With four seeds and wide uncertainty, that is not enough to claim a general harmful effect. It is enough to say the declared repair-benefit criterion was not demonstrated.

**Why not count every test execution as an independent sample?** Hundreds of tests on one generated program measure that program. Many program draws still come from one trained policy. Training comparisons therefore use paired training seeds, not input count, as the replication unit. Even the 2,048 draws include repeated sources and shared sampling seeds.

**What did broader failure analysis add?** Four of 38 false-acceptance draws had order-sensitive endpoint ties instead of the exact inclusive-endpoint behavior. The broader audit exposed failures the narrow signature missed. Neither metric was silently substituted for the other.

**Why did a few zero-gradient updates still change weights?** Identical rewards remove the within-group advantage signal; retained AdamW moments can still move parameters. All three zero-gradient steps had zero reward standard deviation and changed saved parameter hashes. This is why gradients, weights and evaluation matter more than a rounded scalar loss alone.

**What was the hardest systems bug?** A per-input storage callback shared the candidate-transport error path. Storage overload could therefore interrupt collection and make execution outcomes ambiguous. The correction durably saves evidence before publishing an index, limits storage retries, and refuses to interpret storage failure as a wrong answer or permission to rerun code. Another incident exposed an invalid cross-container wall-clock comparison in permit admission.

**What would you change in a follow-up?** Separate training-seed variance from evaluation-draw uncertainty, define a smallest effect worth detecting, select the replication budget before outcomes, and use a fresh audit plus additional tasks. A randomized count-matched repair comparator could test whether the targeted endpoint selection matters beyond changing test composition. Those experiments have not been run.

## Blog and portfolio presentation

A suitable title is **Repairing a code verifier without establishing a training benefit**. Open with the two adjacent bookings that should share one unit of capacity, then show how an incomplete test suite accepts the wrong rule. Follow with the count-matched repair, all-seed training results, and the recovery invariant: retry storage, not ambiguous candidate execution.

Use the [final policy figure](../reports/booking-replication/policy_outcomes.svg) and [paired-effects figure](../reports/booking-replication/paired_effects.svg). The [full report](booking_replication_analysis.md) is the technical narrative; the [CSV and JSON outputs](../reports/booking-replication/) support inspection. For deeper systems questions, link the [storage fault-injection record](program_storage_recovery.txt), [sandbox benchmark](program_sandbox_benchmark.txt), and [permit recovery](permit_clock_recovery.txt).

For a broader blog, use **What different evaluation metrics reveal about generated code**. Lead with the partial-score versus full-correctness reversal, then use the concrete source and capped-response examples from the [exploratory addendum](booking_behavior_findings.md). Keep the original hypothesis result visible and identify the broader analyses as post hoc.

Avoid headlines such as “eliminated reward hacking,” “proved weak verifiers degrade models,” “700,000 independent experiments,” or “frontier-scale eval platform.” The accurate headline is narrower and more useful: the study separates an observable verifier defect from an unproven training consequence, with evidence that can be traced and checked.

The declared study is complete. The immediate portfolio step is to review and publish a curated report, figures, code, and a shareable evidence bundle with clear reproduction prerequisites—not to keep adding seeds until the result looks favorable. No public release or job application has been made by preparing these local materials.
