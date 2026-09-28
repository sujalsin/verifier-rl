from pathlib import Path
import unittest

from verifier_rl.model_trial import (EXTRACTION_VERSION, LEGACY_EXTRACTION_VERSION,
                                    canonical_prompt, extract_completion, extract_source,
                                    require_scored_rewards, validate_run_id, evaluate_submission,
                                    submission_from_completion, validate_submission)
from verifier_rl.suites import build_suites
from tests.test_grading import FakeBackend


class SubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_training_rejection_never_submits_placeholder_source(self):
        backend = FakeBackend()
        submission = submission_from_completion("```python\ndef incomplete(")
        result = await evaluate_submission(submission, (build_suites()[2],), backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual(result["suites"][0]["reward"], 0)
        self.assertEqual(result["execution_config"]["attempts"], 0)

    async def test_extraction_metadata_cannot_disagree_with_raw_completion(self):
        submission = submission_from_completion("```python\ndef simulate_cache(x): return []\n```")
        self.assertEqual(validate_submission(submission), submission)
        for key in ("source", "extraction_status", "extraction_version"):
            bad = dict(submission, **{key: "changed"})
            backend = FakeBackend()
            with self.assertRaises(ValueError):
                await evaluate_submission(bad, build_suites()[:1], backend)
            self.assertEqual(backend.calls, [])

    async def test_valid_submission_preserves_code_and_reward(self):
        backend = FakeBackend()
        source = "def simulate_cache(x): return []\nprint('demo')"
        result = await evaluate_submission(submission_from_completion(source), build_suites()[:1], backend)
        self.assertEqual(backend.calls[0].source, source)
        self.assertTrue(result["extraction"]["accepted"])
        self.assertTrue(result["execution_config"]["stop_on_infrastructure_error"])


class ModelTrialTests(unittest.TestCase):
    def test_prompt_excludes_researcher_grader_definitions(self):
        prompt = canonical_prompt(Path("task_001_expiring_cache.txt").read_text())
        self.assertTrue(prompt.startswith("Implement this Python function:"))
        self.assertIn("strictly less", prompt)
        self.assertNotIn("G1", prompt)
        self.assertNotIn("DEVELOPMENT CASES", prompt)

    def test_extraction_never_repairs_code(self):
        source = "def simulate_cache(operations): return []"
        self.assertEqual(extract_source(source), source)
        self.assertEqual(extract_source(f"```python\n{source}\n```"), source)
        bad = f"Here is code:\n```python\n{source}\n```"
        self.assertEqual(extract_source(bad), source)
        self.assertEqual(extract_source(bad, version=LEGACY_EXTRACTION_VERSION), bad)
        self.assertIn("required function absent", extract_source(""))
        with self.assertRaises(ValueError):
            extract_source("x" * 32769)

    def test_prose_is_ignored_but_candidate_code_is_unchanged(self):
        code = "import os\n\ndef simulate_cache(operations):\n    return 5\n\nprint('demo')"
        for label in ("python", "py", ""):
            result = extract_completion(f"Here is the solution:\n```{label}\n{code}\n```\nExplanation.")
            self.assertEqual(result.source, code)
            self.assertEqual(result.status, "python_block")
            self.assertEqual(result.version, EXTRACTION_VERSION)

    def test_ambiguous_incomplete_and_non_python_blocks_are_rejected(self):
        for raw in (
            "```python\nx = 1", "```python\nx = 1\n```\n```python\nx = 2\n```",
            "```javascript\nx = 1\n```", "```python\nx = 1\n```python",
            "````python\nx = 1\n````", "```python\n\n```", "  ",
        ):
            result = extract_completion(raw)
            self.assertTrue(result.status.startswith("rejected_"), raw)
            self.assertNotIn("x = 1", result.source)
            self.assertIn("required function absent", result.source)

    def test_plain_source_and_invalid_code_are_not_repaired(self):
        for code in ("def simulate_cache(:\n    return []", "print('```')", "Prose without a code block"):
            self.assertEqual(extract_source(code), code)
            self.assertEqual(extract_source(f"```python\n{code}\n```"), code)

    def test_legacy_version_preserves_original_behavior(self):
        self.assertIn("Empty model completion", extract_source("", version=LEGACY_EXTRACTION_VERSION))
        self.assertEqual(extract_source("```py\nx = 1\n```", version=LEGACY_EXTRACTION_VERSION), "x = 1")
        with self.assertRaises(ValueError):
            extract_source("x = 1", version="unknown")

    def test_run_paths_are_bounded(self):
        validate_run_id("qwen-20260926-abc123")
        for run_id in ("../other", "qwen-../../bad", "qwen-", "cpu-test"):
            with self.assertRaises(ValueError):
                validate_run_id(run_id)

    def test_infrastructure_errors_never_become_training_reward(self):
        def report(reward, infra=0, suite="g3"):
            return {"suites": [{"suite": suite, "reward": reward, "infrastructure_errors": infra}]}
        self.assertEqual(require_scored_rewards([report(1), report(0)]), [1.0, 0.0])
        for row in (report(None), report(0, 1), report(None, 0, "audit")):
            with self.assertRaises((ValueError, RuntimeError)):
                require_scored_rewards([row])
