from copy import deepcopy
import json
import unittest

from verifier_rl.baseline import inspect_completion
from verifier_rl.grading import ExecutionResult, Status, rejected_extraction_report, score_suite
from verifier_rl.prompt_pilot import ARMS, SEEDS, pilot_suites, protocol_plan, schedule, summarize
from verifier_rl.suites import canonical_json, digest


class PromptPilotTests(unittest.TestCase):
    def evidence(self):
        plan = protocol_plan("Original task specification")
        suites = pilot_suites()
        executions = {c.input_hash: (ExecutionResult(Status.COMPLETED,
                        canonical_json(c.expected).encode()),) for s in suites for c in s.cases}
        samples, reports = [], []
        for arm, seed in schedule():
            sample = inspect_completion("def simulate_cache(operations): return []")
            sample.update(arm=arm, seed=seed, prompt_hash=plan["prompt_hashes"][arm], hit_token_cap=False)
            samples.append(sample)
            reports.append({"arm": arm, "seed": seed, "candidate_hash": digest(sample["source"]),
                            "execution_config": {"attempts": len(executions)},
                            "suites": [score_suite(suite, executions) for suite in suites]})
        return {"plan": plan, "samples": samples, "run_id": "qwen-test",
                "parameters_unchanged": True, "gpu_function_seconds": 1}, reports

    def test_freezes_paired_schedule_and_only_adds_interface_reminders(self):
        plan = protocol_plan("Original task specification")
        self.assertEqual(plan["prompts"]["original"], "Original task specification")
        self.assertTrue(plan["prompts"]["interface_reminders"].startswith("Original task specification\n\n"))
        self.assertEqual(set(schedule()), {(arm, seed) for arm in ARMS for seed in SEEDS})
        self.assertEqual(schedule()[:4], [("original", 4000), ("interface_reminders", 4000),
                                         ("interface_reminders", 4001), ("original", 4001)])
        self.assertEqual(plan["max_sandbox_executions"], 576)

    def test_summary_roundtrip_counts_programs_and_repeated_inputs_separately(self):
        generation, reports = self.evidence()
        result = summarize(json.loads(json.dumps(generation)), reports)
        for arm in ARMS:
            row = result["arms"][arm]
            self.assertEqual(row["sample_count"], 8)
            self.assertEqual(row["unique_source_count"], 1)
            self.assertEqual(row["full_suite_pass_counts"], {"g1": 8, "g3": 8})
            self.assertEqual(row["actual_sandbox_executions"], 288)
            self.assertEqual(row["mixed_g3_groups_of_four"], 0)

    def test_rejection_has_zero_executions_and_zero_full_suite_passes(self):
        generation, reports = self.evidence()
        sample = inspect_completion("```python\ndef incomplete(")
        sample.update(arm="original", seed=4000, hit_token_cap=True,
                      prompt_hash=generation["plan"]["prompt_hashes"]["original"])
        generation["samples"][0] = sample
        report = rejected_extraction_report(sample["source"], sample["extraction_status"], pilot_suites())
        report.update(arm="original", seed=4000)
        reports[0] = report
        row = summarize(generation, reports)["arms"]["original"]
        self.assertEqual(row["actual_sandbox_executions"], 252)
        self.assertEqual(row["full_suite_pass_counts"], {"g1": 7, "g3": 7})
        self.assertEqual(row["candidates_with_all_outputs_valid"], 7)
        self.assertEqual(row["mixed_g3_groups_of_four"], 1)

    def test_incomplete_corrupt_or_unscored_evidence_is_rejected(self):
        generation, reports = self.evidence()
        for mutation in ("missing", "duplicate", "hash", "suite", "infra", "weights", "plan"):
            g, r = deepcopy(generation), deepcopy(reports)
            if mutation == "missing": r.pop()
            elif mutation == "duplicate": r[0] = r[1]
            elif mutation == "hash": r[0]["candidate_hash"] = "bad"
            elif mutation == "suite": r[0]["suites"][0]["suite_hash"] = "bad"
            elif mutation == "infra": r[0]["suites"][0]["infrastructure_errors"] = 1
            elif mutation == "weights": g["parameters_unchanged"] = False
            elif mutation == "plan": g["plan"]["temperature"] = 0.2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize(g, r)
