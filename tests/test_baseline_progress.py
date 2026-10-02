from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import modal_booking_baseline_comparison as launcher
from verifier_rl import booking_baseline_comparison as study
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import canonical_json, digest


class ReplayProgressTests(unittest.TestCase):
    """Test replay orchestration with authored records; no candidate execution.

    Per-batch scoring is stubbed here and independently exercised by
    test_booking_baseline_comparison. Journal equality checks remain real.
    """

    @classmethod
    def setUpClass(cls):
        cls.plan = study.make_plan(Path(__file__).resolve().parents[1])

    def setUp(self):
        self.directory = Path("/fixture")
        self.records, self.reads = [], []
        samples = [study.original.sample_from_text(study.original.controls()["correct"],
                   sid=sid, seed=seed, plan=self.plan, tokens=90, eos=True) for sid, seed in study.identities()]
        generated = {"samples": samples, "parameter_hash": self.plan["initial_parameter_hash"],
                     "parameters_unchanged": True, "initial_checkpoint": None, "optimizer_updates": 0,
                     "plan_hash": digest(canonical_json(self.plan))}
        self.documents = {"plan.json": self.plan, "setup.json": {"sandbox_image_id": "im-test"},
                          "supervisor_controls.json": {}, "generation/result.json": generated,
                          "preflight.json": {"passed": True}, "source_snapshot.json": {}}
        self.checked = {}
        batches = [("controls", study.controls(self.plan))]
        batches += [(f"eval-baseline-00-{i:02d}", samples[4*i:4*i+4]) for i in range(8)]
        for key, group in batches:
            entries = {s["sample_id"]: {"input-hash": {"selected": {"passed": None}}} for s in group}
            raw = {"entries": entries}
            checked = {"rows": [{"sample_id": s["sample_id"]} for s in group],
                       "sandbox_ids": [s["sample_id"] for s in group], "submitted_attempts": len(group)}
            self.checked[key] = checked
            self.documents[f"grading/{key}/raw.json"] = raw
            self.documents[f"grading/{key}/result.json"] = {"raw": raw, "checked": checked}
            for sample in group:
                self.documents[f"grading/{key}/inputs/{sample['sample_id']}/input-hash/selected.json"] = {"passed": None}
        for sample in samples:
            sid = sample["sample_id"]
            self.documents[f"generation/samples/{sid}.json"] = sample
            self.documents[f"generation/sample-intents/{sid}.json"] = {
                "sample_id": sid, "seed": sample["seed"], "parameter_hash": self.plan["initial_parameter_hash"]}

    def replay(self, *, quiet=False):
        def read(path):
            relative = path.relative_to(self.directory).as_posix()
            self.reads.append(relative)
            return deepcopy(self.documents[relative])

        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher, "read_json", side_effect=read))
            stack.enter_context(patch.object(launcher.supervisor_controls, "validate_controls", return_value=["supervisor"]))
            stack.enter_context(patch.object(study, "verify_raw", side_effect=lambda raw, samples, key, *args: self.checked[key]))
            stack.enter_context(patch.object(study, "validate_controls", return_value={"passed": True}))
            summary = stack.enter_context(patch.object(study, "policy_summary", side_effect=lambda rows: {
                "programs": len(rows), "unknown_input_outcomes": 1, "audit": {"case_pass_bounds": [2355, 2356]}}))
            stack.enter_context(patch.object(launcher, "ProgressLog", side_effect=lambda run_id, **kw:
                ProgressLog(run_id, emit=(lambda record: None) if quiet else self.records.append, **kw)))
            result = launcher.verify_baseline(self.directory)
            summary.assert_called_once()
            return result

    def test_logs_counts_and_keeps_read_order_result_and_unknown_bounds(self):
        quiet = self.replay(quiet=True)
        original_reads = self.reads[:]
        self.reads.clear()
        observed = self.replay()
        self.assertEqual(canonical_json(observed), canonical_json(quiet))
        self.assertEqual(self.reads, original_reads)
        self.assertEqual(observed["status"], "completed_with_uncertainty")
        self.assertEqual(observed["summary"]["audit"]["case_pass_bounds"], [2355, 2356])
        journals = [r for r in self.records if r["phase"] == "verify_input_journals" and r["event"] == "stage_completed"]
        self.assertEqual([r["batch_index"] for r in journals], list(range(1, 10)))
        self.assertEqual([r["completed"] for r in journals], [3] + [4]*8)
        self.assertTrue(all(r["completed"] == r["total"] and r["batches_total"] == 9 for r in journals))
        generation = next(r for r in self.records if r["phase"] == "verify_generation_journals" and r["event"] == "stage_completed")
        self.assertEqual(generation["completed"], 64)
        self.assertEqual(self.records[-1]["event"], "completed")
        self.assertNotIn(study.original.controls()["correct"], str(self.records))

    def test_journal_mismatch_still_fails_at_exact_file_without_success(self):
        path = "grading/controls/inputs/control-correct/input-hash/selected.json"
        self.documents[path] = {"passed": True}
        with self.assertRaisesRegex(ValueError, "input journal differs"):
            self.replay()
        self.assertEqual(self.records[-1]["event"], "failed")
        self.assertEqual(self.records[-1]["current_item"], path)
        self.assertEqual(self.records[-1]["completed"], 0)
        self.assertNotIn("completed", [r["event"] for r in self.records])


class PublicationProgressTests(unittest.TestCase):
    def run_finalizer(self, failure_stage=None):
        records, calls = [], []
        result = {"status": "completed_with_uncertainty", "summary": {"unknown_input_outcomes": 1}}

        def operation(name, output=None):
            def call(*args, **kwargs):
                calls.append(name)
                if failure_stage == name:
                    raise RuntimeError("failure in " + name)
                if name == "replay":
                    kwargs["progress"].stage("aggregate_summary")
                if name == "commit":
                    self.assertNotIn("completed", [r["event"] for r in records])
                return output
            return call

        artifacts = Mock(reload=Mock(side_effect=operation("reload")), commit=Mock(side_effect=operation("commit")))
        with patch.object(launcher, "artifacts", artifacts), patch.object(launcher, "persist", side_effect=operation("write")), \
             patch.object(launcher, "verify_baseline", side_effect=operation("replay", result)), \
             patch.object(launcher, "ProgressLog", side_effect=lambda run_id: ProgressLog(run_id, emit=records.append)):
            if failure_stage:
                with self.assertRaisesRegex(RuntimeError, "failure in " + failure_stage):
                    launcher.finalize_baseline("/fixture")
            else:
                self.assertIs(launcher.finalize_baseline("/fixture"), result)
        return calls, records

    def test_complete_only_after_reload_replay_write_and_commit(self):
        calls, records = self.run_finalizer()
        self.assertEqual(calls, ["reload", "replay", "write", "commit"])
        self.assertEqual([r["phase"] for r in records if r["event"] == "stage_started"],
                         ["reload_artifacts", "aggregate_summary", "write_report", "commit_artifacts"])
        self.assertEqual(records[-1]["event"], "completed")

    def test_failures_never_publish_false_completion_or_continue(self):
        operations = ["reload", "replay", "write", "commit"]
        for i, stage in enumerate(operations):
            with self.subTest(stage=stage):
                calls, records = self.run_finalizer(stage)
                self.assertEqual(calls, operations[:i+1])
                self.assertEqual(records[-1]["event"], "failed")
                self.assertNotIn("completed", [r["event"] for r in records])


if __name__ == "__main__":
    unittest.main()
