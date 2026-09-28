import asyncio
from copy import deepcopy
from dataclasses import asdict
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from verifier_rl.fixtures import fixture_impl, source_for
from verifier_rl.grading import ExecutionResult, Status, rejected_extraction_report, score_suite
from verifier_rl.measurement_v3 import (SOURCE_RUN, budget_check, control_samples, entry_id,
    measurement_plan, measurement_summary, repeatability_changes, require_measurement_report,
    saved_samples, suites)
from verifier_rl.modal_backend import Limits, RUNNER
from verifier_rl.model_trial import evaluate_submission, submission_from_completion
from verifier_rl.reward_review import authored_output
from verifier_rl.reward_v2 import behavior_scores
from verifier_rl.reward_v3 import coverage_scores
from verifier_rl.suites import build_suites, canonical_json, digest

HAS_MODAL = find_spec("modal") is not None


def report_for(sample, selected, fault="correct", status=None, returncode=None):
    if sample["extraction_status"].startswith("rejected_"):
        report = rejected_extraction_report(sample["source"], sample["extraction_status"], selected)
    else:
        executions = {}
        for suite in selected:
            for case in suite.cases:
                output = (authored_output(case.operations, sample["control"]) if sample["arm"] == "control"
                          else fixture_impl(case.operations, fault))
                code = 0 if status is None else returncode
                metadata = {"backend": "modal", "image_id": "im-test", "runner_hash": digest(RUNNER),
                    "limits": asdict(Limits()), "reset": "fresh_sandbox_per_input", "block_network": True,
                    "creation_interval_seconds": .26, "sdk_version": "1.5.5", "cleanup": "terminated",
                    "sandbox_id": f"sb-{entry_id(sample)}-{case.input_hash}", "preflight_returncode": 0,
                    "returncode": code}
                executions[case.input_hash] = (ExecutionResult(status or Status.COMPLETED,
                    canonical_json(output).encode() if status is None else b"", metadata=metadata),)
        report = {"candidate_hash": digest(sample["source"]),
                  "execution_config": {"attempts": len(executions), "max_retries": 0},
                  "suites": [score_suite(s, executions) for s in selected]}
    report.update(arm=sample["arm"], seed=sample["seed"])
    return report


class MeasurementV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old, new = suites()
        audit = build_suites()[-1]
        def sample(arm, seed, fault="correct"):
            return {**submission_from_completion(source_for(fault)), "arm": arm, "seed": seed,
                    "prompt_hash": "test-prompt"}
        faults = ("correct", "stale_expiry", "inclusive_expiry", "extra_put_output")
        records = []
        for i in range(4):
            samples = [sample("training", 9000 + i * 4 + j, f) for j, f in enumerate(faults)]
            reports = [report_for(s, (old,), f) for s, f in zip(samples, faults)]
            records.append({"samples": samples, "reports": reports,
                            "rewards": [behavior_scores(r["suites"][0])["partial"] for r in reports]})
        evaluations = [sample(arm, seed) for arm in ("before", "after") for seed in range(8000, 8008)]
        evaluations[6].update(submission_from_completion("```python\npass\n```\n```python\npass\n```"))
        cls.generation = {"run_id": SOURCE_RUN, "plan": {"prompt_hash": "test-prompt"},
            "samples": evaluations, "rollout_records": records,
            "metrics": {"global_step": 4, "rewards": [r["rewards"] for r in records]}}
        cls.previous = [report_for(s, (old, audit)) for s in evaluations]
        cls.hashes = {"generation": digest(canonical_json(cls.generation)),
                      "reports": digest(canonical_json(cls.previous))}
        cls.entries = control_samples() + saved_samples(cls.generation)
        cls.fresh = [report_for(s, (old, new), faults[(s["seed"] - 9000) % 4]
                               if s["arm"] == "training" else "correct") for s in cls.entries]
        with patch("verifier_rl.measurement_v3.SOURCE_HASHES", cls.hashes):
            cls.plan = measurement_plan(cls.generation, cls.previous)

    def test_frozen_population_and_deduplicated_budget(self):
        p = self.plan
        self.assertEqual(p["model_records"], 32)
        self.assertEqual(p["max_sandbox_executions"], 1496)
        self.assertEqual(p["unique_inputs_per_entry"], 44)
        self.assertEqual(p["suite_cases"], [34, 44])
        self.assertEqual(p["entry_order"][:3], ["control-0", "control-1", "control-2"])
        self.assertFalse(p["training"])
        self.assertEqual(p["new_model_samples"], 0)
        self.assertEqual(p["max_retries"], 0)

    def test_rejects_changed_artifacts_before_execution(self):
        with patch("verifier_rl.measurement_v3.SOURCE_HASHES", self.hashes):
            for field in ("run_id", "samples", "metrics"):
                bad = deepcopy(self.generation)
                bad[field] = "changed"
                with self.assertRaises(ValueError): measurement_plan(bad, self.previous)
            with self.assertRaises(ValueError): measurement_plan(self.generation, self.previous[:-1])

    def test_cloud_control_expectations_include_known_blind_spots(self):
        for s, r in zip(self.entries[:3], self.fresh[:3]):
            require_measurement_report(s, r, "im-test")
            self.assertEqual(behavior_scores(r["suites"][0])["partial"], 1)
            result = coverage_scores(r["suites"][1])
            self.assertEqual(result["partial"], 1 if s["control"] == "correct" else .88125)
            self.assertEqual(result["binary"], int(s["control"] == "correct"))
        bad = deepcopy(self.entries[1])
        bad["control"] = "correct"  # Do not accept wrong behavior just because its score looks plausible.
        with self.assertRaises(ValueError): require_measurement_report(bad, self.fresh[1], "im-test")

    def test_paired_scores_share_exact_execution_evidence(self):
        bad = deepcopy(self.fresh[0])
        bad["suites"][1]["outcomes"][0]["attempts"][0]["metadata"]["stdout_sha256"] = "changed"
        with self.assertRaises(ValueError): require_measurement_report(self.entries[0], bad, "im-test")

    def test_environment_gates(self):
        for key, value in (("cleanup", "unknown"), ("limits", {}), ("block_network", False),
                           ("runner_hash", "wrong"), ("sdk_version", "wrong"), ("preflight_returncode", 1),
                           ("image_id", "wrong"), ("sandbox_id", ""), ("creation_interval_seconds", 0)):
            bad = deepcopy(self.fresh[0])
            for saved in bad["suites"]:
                saved["outcomes"][0]["attempts"][0]["metadata"][key] = value
            with self.assertRaises(ValueError, msg=key): require_measurement_report(self.entries[0], bad, "im-test")

    def test_infra_timeout_and_signal_fail_closed_but_ordinary_errors_remain_visible(self):
        s = self.entries[3]
        for status, code in ((Status.INFRASTRUCTURE_ERROR, None), (Status.TIMEOUT, -1),
                             (Status.CANDIDATE_ERROR, 137)):
            r = report_for(s, suites(), status=status, returncode=code)
            before = deepcopy(r)
            with self.assertRaises(ValueError): require_measurement_report(s, r, "im-test")
            self.assertEqual(before, r)
        r = report_for(s, suites(), status=Status.CANDIDATE_ERROR, returncode=1)
        row = require_measurement_report(s, r, "im-test")
        self.assertEqual(row["outcome_counts"], {"candidate_execution_error": 44})

    def test_rejected_completion_never_reaches_backend(self):
        s = next(s for s in self.entries if s["extraction_status"].startswith("rejected_"))
        class NeverBackend:
            async def execute(self, request):
                raise AssertionError("backend must not receive a rejected completion")
        r = asyncio.run(evaluate_submission(s, suites(), NeverBackend(), concurrency=8))
        r.update(arm=s["arm"], seed=s["seed"])
        require_measurement_report(s, r, "im-test")
        self.assertEqual(r["execution_config"]["attempts"], 0)

    def test_union_uses_44_calls_not_78_without_executing_source(self):
        class AuthoredBackend:
            calls = 0
            async def execute(self, request):
                self.calls += 1
                # Only a known authored function sees the data. Never eval request.source.
                return ExecutionResult(Status.COMPLETED, canonical_json(fixture_impl(json.loads(request.input_json))).encode())
        backend = AuthoredBackend()
        s = {**self.entries[0], **submission_from_completion("raise RuntimeError('must never execute locally')")}
        r = asyncio.run(evaluate_submission(s, suites(), backend, concurrency=8))
        self.assertEqual(backend.calls, 44)
        self.assertEqual(r["execution_config"]["attempts"], 44)
        self.assertTrue(all(result["all_passed"] for result in r["suites"]))

    def test_repeatability_tracks_outputs_separately_from_test_expansion(self):
        prior = self.generation["rollout_records"][0]["reports"][0]
        fresh = self.fresh[3]
        self.assertEqual(repeatability_changes(prior, fresh), [])
        bad = deepcopy(fresh)
        bad["suites"][0]["outcomes"][0]["attempts"][0]["metadata"]["stdout_sha256"] = "changed"
        self.assertEqual(len(repeatability_changes(prior, bad)), 1)

    def test_summary_preserves_partial_score_increases_and_counts_real_executions(self):
        old = deepcopy((self.generation, self.previous, self.fresh))
        with patch("verifier_rl.measurement_v3.SOURCE_HASHES", self.hashes):
            summary = measurement_summary(self.generation, self.previous, self.fresh, "im-test")
        self.assertEqual(old, (self.generation, self.previous, self.fresh))
        self.assertTrue(summary["controls_passed"])
        self.assertTrue(summary["v2_repeatability_passed"])
        self.assertEqual(summary["recorded_sandbox_executions"], 1496)
        stale = summary["rows"][4]
        self.assertGreater(stale["partial_delta"], 0)  # Do not erase inconvenient dilution.
        self.assertEqual(len(summary["groups"]), 4)
        self.assertFalse(summary["training"])

    def test_missing_duplicate_or_reordered_report_rejected(self):
        with patch("verifier_rl.measurement_v3.SOURCE_HASHES", self.hashes):
            for reports in (self.fresh[:-1], self.fresh + self.fresh[:1], list(reversed(self.fresh))):
                with self.assertRaises(ValueError):
                    measurement_summary(self.generation, self.previous, reports, "im-test")

    def test_budget_uses_metered_usage_not_credit_discount_and_preserves_total_limit(self):
        rates = {"cpu_hour_cost_sandbox": ".1419", "mem_gib_hour_cost_sandbox": ".024",
                 "cpu_hour_cost": ".0473", "mem_gib_hour_cost": ".008"}
        budget = budget_check(self.plan, {"metered_cost": "1.07557826", "billed_cost": "0"}, rates)
        self.assertLess(float(budget["total_metered_envelope_usd"]), 10)
        self.assertGreater(float(budget["sandbox_resource_envelope_usd"]), 7)
        self.assertFalse(budget["provider_hard_spending_cap"])
        for cost in ("2", "NaN", "-1", "Infinity"):
            with self.assertRaises(ValueError): budget_check(self.plan, {"metered_cost": cost}, rates)

    @unittest.skipUnless(HAS_MODAL, "optional Modal SDK required for mocked launcher checks")
    def test_artifacts_written_exclusively_and_not_overwritten(self):
        from modal_measure_v3 import persist_equal
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            persist_equal(path, {"evidence": 1})
            persist_equal(path, {"evidence": 1})
            with self.assertRaises(ValueError): persist_equal(path, {"evidence": 2})
            self.assertEqual(json.loads(path.read_text()), {"evidence": 1})

    @unittest.skipUnless(HAS_MODAL, "optional Modal SDK required for mocked launcher checks")
    def test_complete_cloud_record_reused_without_resubmission(self):
        import modal_measure_v3 as launcher
        s, report = self.entries[0], deepcopy(self.fresh[0])
        setup = {"app_name": "test-app", "sandbox_image_id": "im-test"}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "candidate"
            with (patch.object(launcher, "Path", return_value=directory),
                  patch.object(launcher, "artifacts"), patch.object(launcher, "ModalBackend"),
                  patch.object(launcher, "evaluate_submission", new=AsyncMock(return_value=report)) as evaluate):
                first = launcher.evaluate_one.local("qwen-test", s, setup, self.plan)
                second = launcher.evaluate_one.local("qwen-test", s, setup, self.plan)
                self.assertEqual(first, second)
                evaluate.assert_awaited_once()
                with self.assertRaises(ValueError):
                    launcher.evaluate_one.local("qwen-test", s, dict(setup, app_name="changed"), self.plan)
                evaluate.assert_awaited_once()

    @unittest.skipUnless(HAS_MODAL, "optional Modal SDK required for mocked launcher checks")
    def test_incomplete_intent_cannot_silently_rerun(self):
        import modal_measure_v3 as launcher
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "candidate"
            directory.mkdir()  # Existing intent directory with no completed result.
            with (patch.object(launcher, "Path", return_value=directory), patch.object(launcher, "artifacts"),
                  patch.object(launcher, "evaluate_submission", new=AsyncMock()) as evaluate):
                with self.assertRaises(FileExistsError):
                    launcher.evaluate_one.local("qwen-test", self.entries[0],
                        {"app_name": "test", "sandbox_image_id": "im-test"}, self.plan)
                evaluate.assert_not_awaited()

    @unittest.skipUnless(HAS_MODAL, "optional Modal SDK required for mocked launcher checks")
    def test_cloud_failure_saved_before_gate_without_retry(self):
        import modal_measure_v3 as launcher
        s = self.entries[0]
        report = report_for(s, suites(), status=Status.CANDIDATE_ERROR, returncode=137)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "candidate"
            with (patch.object(launcher, "Path", return_value=directory), patch.object(launcher, "artifacts"),
                  patch.object(launcher, "ModalBackend"),
                  patch.object(launcher, "evaluate_submission", new=AsyncMock(return_value=report)) as evaluate):
                for _ in range(2):
                    with self.assertRaises(ValueError):
                        launcher.evaluate_one.local("qwen-test", s,
                            {"app_name": "test", "sandbox_image_id": "im-test"}, self.plan)
                self.assertEqual(json.loads((directory / "result.json").read_text()), report)
                evaluate.assert_awaited_once()

    def test_offline_verification_checks_snapshots_and_recomputed_summary(self):
        from verifier_rl.measurement_v3_verification import verify_documents
        with (tempfile.TemporaryDirectory() as temp,
              patch("verifier_rl.measurement_v3.SOURCE_HASHES", self.hashes),
              patch("verifier_rl.measurement_v3_verification.require_current_conformance")):
            root = Path(temp)
            (root / "verifier_rl").mkdir()
            names = ["verifier_rl/" + name for name in
                     ("measurement_v3.py", "reward_v2.py", "reward_v3.py", "grading.py", "modal_backend.py")]
            names += ["modal_measure_v3.py", "measurement_v3_protocol.txt"]
            snapshot = {name: "raise RuntimeError('snapshot is data, never execute')" for name in names}
            for name, text in snapshot.items(): (root / name).write_text(text)
            summary = measurement_summary(self.generation, self.previous, self.fresh, "im-test")
            summary["run_id"] = "qwen-test"
            rates = {"cpu_hour_cost_sandbox": ".1419", "mem_gib_hour_cost_sandbox": ".024",
                     "cpu_hour_cost": ".0473", "mem_gib_hour_cost": ".008"}
            documents = {"generation": self.generation, "prior_reports": self.previous, "reports": self.fresh,
                "setup": {"sandbox_image_id": "im-test"}, "conformance": {}, "plan": self.plan,
                "summary": summary, "source_snapshot": snapshot,
                "budget": budget_check(self.plan, {"metered_cost": "1.07557826"}, rates)}
            result = verify_documents(documents, root)
            self.assertTrue(result["summary_recomputed_exactly"])
            self.assertEqual(result["recorded_sandbox_executions"], 1496)
            self.assertEqual(result["cleanup"], {"terminated": 1496})
            self.assertFalse(result["candidate_source_executed"])
            self.assertEqual(result["cloud_calls"], 0)
            bad = deepcopy(documents)
            bad["summary"]["controls_passed"] = False
            with self.assertRaises(ValueError): verify_documents(bad, root)
            (root / names[0]).write_text("changed implementation")
            with self.assertRaises(ValueError): verify_documents(documents, root)
