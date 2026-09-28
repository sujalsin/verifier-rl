from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from verifier_rl.diagnostics import diagnose_run, diagnose_sample
from verifier_rl.grading import ExecutionResult, Status, rejected_extraction_report, score_suite
from verifier_rl.model_trial import submission_from_completion
from verifier_rl.suites import Suite, canonical_json, case, get, put


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.suite = Suite("diagnostic_test", "training", (
            case("live", [put(0), get(0)]), case("missing", [get(0)]), case("empty", [])), 123)
        self.sample = {**submission_from_completion("def simulate_cache(operations):\n    return []"),
                       "seed": 4000, "prompt_hash": "prompt"}

    def report(self, executions=None):
        executions = executions or {c.input_hash: (ExecutionResult(Status.COMPLETED,
                     canonical_json(c.expected).encode()),) for c in self.suite.cases}
        from verifier_rl.suites import digest
        return {"seed": 4000, "candidate_hash": digest(self.sample["source"]),
                "execution_config": {"attempts": sum(len(a) for a in executions.values())},
                "suites": [score_suite(self.suite, executions, allow_unexecuted=True)]}

    def diagnose(self, report):
        return diagnose_sample(self.sample, report, (self.suite,))

    def test_separates_wrong_answers_from_invalid_output_without_salvage(self):
        cases = self.suite.cases
        executions = {
            cases[0].input_hash: (ExecutionResult(Status.COMPLETED, b"[8]"),),
            # Last line is perfectly correct but must never earn semantic credit.
            cases[1].input_hash: (ExecutionResult(Status.COMPLETED, b"demo log\n[null]"),),
            cases[2].input_hash: (ExecutionResult(Status.COMPLETED, b"[]"),),
        }
        report = self.report(executions)
        original = deepcopy(report)
        result = self.diagnose(report)
        self.assertEqual(result["outcome_counts"], {"wrong_answer": 1, "output_protocol_failure": 1, "correct": 1})
        self.assertEqual(result["semantic_observations"]["valid_output_cases"], 2)
        self.assertEqual(result["semantic_observations"]["unobservable_cases"], 1)
        self.assertEqual(result["semantic_observations"]["conditional_accuracy"], .5)
        self.assertEqual(report, original)
        self.assertEqual(result["official_suite_scores_unchanged"]["diagnostic_test"]["reward"], 0)

    def test_errors_timeouts_and_infrastructure_have_separate_buckets(self):
        statuses = (Status.CANDIDATE_ERROR, Status.TIMEOUT, Status.INFRASTRUCTURE_ERROR)
        executions = {c.input_hash: (ExecutionResult(status, metadata={"runner_stage": "unknown"}),)
                      for c, status in zip(self.suite.cases, statuses)}
        result = self.diagnose(self.report(executions))
        self.assertEqual(result["outcome_counts"], {"candidate_execution_error": 1,
                         "execution_timeout_unattributed": 1, "infrastructure_failure": 1})
        self.assertIsNone(result["semantic_observations"]["conditional_accuracy"])
        self.assertIsNone(result["official_suite_scores_unchanged"]["diagnostic_test"]["reward"])
        executions[self.suite.cases[2].input_hash] = ()
        self.assertEqual(self.diagnose(self.report(executions))["outcome_counts"]["not_executed"], 1)

    def test_missing_and_truncated_stdout_are_not_recovered_from_last_line(self):
        report = self.report()
        outcome = report["suites"][0]["outcomes"][0]
        outcome["attempts"][0]["metadata"]["stdout_preview"] = "["
        self.assertEqual(self.diagnose(report)["stdout_evidence"]["preview_incomplete"], 1)
        del outcome["attempts"][0]["metadata"]["stdout_preview"]
        self.assertEqual(self.diagnose(report)["stdout_evidence"]["unavailable"], 1)

    def test_corrupt_hashes_case_identity_and_aggregates_are_rejected(self):
        for mutation in ("source", "seed", "suite", "case", "flag", "count", "reward", "attempts", "stdout"):
            bad = self.report()
            outcome = bad["suites"][0]["outcomes"][0]
            if mutation == "source": bad["candidate_hash"] = "bad"
            elif mutation == "seed": bad["seed"] = 999
            elif mutation == "suite": bad["suites"][0]["suite_hash"] = "bad"
            elif mutation == "case": outcome["expected"] = [8]
            elif mutation == "flag": outcome["passed"] = 1
            elif mutation == "count": bad["suites"][0]["passed_count"] = 0
            elif mutation == "reward": bad["suites"][0]["reward"] = 0
            elif mutation == "attempts": bad["execution_config"]["attempts"] = 99
            else: outcome["attempts"][0]["metadata"]["stdout_sha256"] = "bad"
            with self.assertRaises(ValueError, msg=mutation): self.diagnose(bad)

    def test_shared_inputs_are_deduplicated_but_contradictions_rejected(self):
        report = self.report()
        other = Suite("other", "training", self.suite.cases, 123)
        second = deepcopy(report["suites"][0])
        second.update(suite=other.name, suite_hash=other.fingerprint)
        report["suites"].append(second)
        result = diagnose_sample(self.sample, report, (self.suite, other))
        self.assertEqual(result["unique_inputs"], 3)
        self.assertEqual(result["outcome_counts"]["correct"], 3)
        second["outcomes"][0]["attempts"][0]["metadata"]["different"] = True
        with self.assertRaises(ValueError): diagnose_sample(self.sample, report, (self.suite, other))

    def test_rejected_extraction_has_no_execution_or_syntax_credit(self):
        sample = {**submission_from_completion("```python\nincomplete"), "seed": 4000, "prompt_hash": "prompt"}
        report = rejected_extraction_report(sample["source"], sample["extraction_status"], (self.suite,))
        report["seed"] = 4000
        result = diagnose_sample(sample, report, (self.suite,))
        self.assertFalse(result["extraction_accepted"])
        self.assertFalse(result["syntax_valid"])
        self.assertEqual(result["recorded_execution_attempts"], 0)
        self.assertEqual(result["outcome_counts"], {"extraction_failure": 3})

    def test_run_matching_and_no_execution_of_source(self):
        # If the reporter executed this source, the test would fail immediately.
        sample = {**submission_from_completion("raise RuntimeError('must never execute')"),
                  "seed": 4000, "prompt_hash": "prompt"}
        report = self.report()
        from verifier_rl.suites import digest
        report["candidate_hash"] = digest(sample["source"])
        generation = {"run_id": "qwen-test", "plan": {"prompt_hash": "prompt"}, "samples": [sample]}
        before = deepcopy((generation, report))
        result = diagnose_run(generation, [report], (self.suite,))
        self.assertFalse(result["candidate_source_executed"])
        self.assertFalse(result["official_scores_changed"])
        self.assertEqual((generation, report), before)
        with self.assertRaises(ValueError): diagnose_run(generation, [report, report], (self.suite,))
        with self.assertRaises(ValueError): diagnose_run(generation, [], (self.suite,))
        generation["samples"][0]["prompt_hash"] = "different"
        with self.assertRaises(ValueError): diagnose_run(generation, [report], (self.suite,))

    def test_cli_writes_new_sidecar_and_refuses_overwrite(self):
        from verifier_rl.reward_v2 import behavior_suite
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = behavior_suite()
            executions = {c.input_hash: (ExecutionResult(Status.COMPLETED, canonical_json(c.expected).encode()),)
                          for c in suite.cases}
            report = self.report()
            report.update(execution_config={"attempts": len(executions)}, suites=[score_suite(suite, executions)])
            generation = {"run_id": "qwen-test", "plan": {"prompt_hash": "prompt"}, "samples": [self.sample]}
            # Test fixtures only; not project or historical run files.
            (root / "generation.json").write_text(canonical_json(generation))
            (root / "reports.json").write_text(canonical_json([report]))
            command = [sys.executable, "-m", "verifier_rl.diagnostics", "--run", str(root), "--out", str(root / "sidecar")]
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(json.loads((root / "sidecar" / "diagnostics.json").read_text())["official_scores_changed"])
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
