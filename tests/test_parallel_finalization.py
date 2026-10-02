from contextlib import ExitStack
from copy import deepcopy
import ast
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

import modal_booking_matched_training as frozen
import modal_booking_parallel_finalize as launcher
from verifier_rl import supervisor_controls
from verifier_rl.evaluation_journal import persist
from verifier_rl.parallel_finalization import PrefetchJSONReader, journal_records, parallel_reader, read_record
from verifier_rl.progress import ProgressLog
from verifier_rl.suites import canonical_json, digest


def fixture(root, key="test", count=30):
    raw = {"entries": {"sample": {f"input-{i:03d}": {
        "intent-1": {"input": i}, "attempt-1": {"stdout": f"answer-{i}"}, "selected": {"value": i}}
        for i in range(count)}}}
    persist(root / "grading" / key, {"raw": raw})
    # Match the ordering of the parsed on-disk raw JSON, not insertion order.
    raw = frozen.recovery.read_json(root / "grading" / key / "raw.json")
    for relative, value in journal_records(key, raw):
        path = root / relative
        persist(path.parent, {path.stem: value})
    return raw


class ReaderTests(unittest.TestCase):
    def test_values_order_and_no_skipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = fixture(root)
            with PrefetchJSONReader(root) as reader:
                self.assertEqual(reader(root / "grading/test/raw.json"), raw)
                for relative, value in journal_records("test", raw):
                    self.assertEqual(reader(root / relative), value)
            receipt = reader.receipt()
            self.assertEqual(receipt["journal_files"], 90)
            self.assertEqual(receipt["max_pending"], 32)
            self.assertEqual(len(receipt["batches"]), 1)

    def test_actual_concurrency_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = fixture(root)
            lock, active, peak = threading.Lock(), 0, 0
            def read(path):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(.004)
                    return read_record(path)
                finally:
                    with lock:
                        active -= 1
            with PrefetchJSONReader(root, workers=4, read=read) as reader:
                reader(root / "grading/test/raw.json")
                for relative, _ in journal_records("test", raw):
                    reader(root / relative)
            self.assertGreater(peak, 1)
            self.assertLessEqual(peak, 4)
            self.assertLessEqual(reader.max_pending, 8)

    def test_missing_record_is_not_replaced_with_raw_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = fixture(root)
            relative = next(journal_records("test", raw))[0]
            (root / relative).unlink()
            with self.assertRaises(FileNotFoundError):
                with PrefetchJSONReader(root) as reader:
                    reader(root / "grading/test/raw.json")
                    reader(root / relative)
            with self.assertRaises(RuntimeError):
                reader.receipt()

    def test_malformed_json_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = fixture(root)
            relative = next(journal_records("test", raw))[0]
            (root / relative).write_text("{broken")
            with self.assertRaises(json.JSONDecodeError):
                with PrefetchJSONReader(root) as reader:
                    reader(root / "grading/test/raw.json")
                    reader(root / relative)

    def test_incomplete_consumption_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            with self.assertRaisesRegex(ValueError, "not all"):
                with PrefetchJSONReader(root) as reader:
                    reader(root / "grading/test/raw.json")

    def test_changed_order_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = fixture(root)
            relative = list(journal_records("test", raw))[1][0]
            with self.assertRaisesRegex(ValueError, "order differs"):
                with PrefetchJSONReader(root) as reader:
                    reader(root / "grading/test/raw.json")
                    reader(root / relative)

    def test_empty_then_nonempty_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, "empty", count=0)
            raw = fixture(root, "full", count=1)
            with PrefetchJSONReader(root) as reader:
                reader(root / "grading/empty/raw.json")
                reader(root / "grading/full/raw.json")
                for relative, _ in journal_records("full", raw):
                    reader(root / relative)
            self.assertEqual([b["files"] for b in reader.receipt()["batches"]], [0, 3])

    def test_reader_restored_after_error(self):
        original = frozen.read_json
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "fixture"):
            with parallel_reader(frozen, Path(tmp)):
                raise ValueError("fixture")
        self.assertIs(frozen.read_json, original)

    def test_workers_and_unsafe_paths_rejected(self):
        for count in (0, 17, True, 1.5):
            with self.assertRaises(ValueError):
                PrefetchJSONReader(Path("/fixture"), workers=count)
        raw = {"entries": {"sample": {"input": {"../escape": {}}}}}
        with self.assertRaises(ValueError):
            list(journal_records("test", raw))


class FrozenVerifierTests(unittest.TestCase):
    """Exercise the actual frozen 81-batch loop with small authored evidence.

Scoring/policy fixtures are stubs here; their real implementations have their
own regression tests. The original verification loop is not rewritten/mocked.
"""

    def prepare(self, root, stack):
        plan = {"experiment": {"max_startup_retries": 16}, "max_sandbox_starts": 1000,
                "baseline_sha256": "baseline", "control_id": "control"}
        arms = {arm: {"first_tokens": [1], "evaluations": {
            str(step): {"samples": [f"sample-{i}" for i in range(32)]} for step in (12,24)}}
            for arm in frozen.study.ARMS}
        persist(root, {"supervisor_controls": {}, "preflight": {"passed": True}})
        stack.enter_context(patch.object(frozen.release, "validate_arm"))
        stack.enter_context(patch.object(frozen, "verify_policy_journals"))
        stack.enter_context(patch.object(supervisor_controls, "validate_controls", side_effect=lambda *a: []))
        stack.enter_context(patch.object(frozen.study, "validate_controls", return_value={"passed": True}))
        stack.enter_context(patch.object(frozen.study, "controls", return_value=["fixture"]))
        stack.enter_context(patch.object(frozen.study, "rewards_from_raw", return_value=[.5]))
        stack.enter_context(patch.object(frozen.study, "verify_raw", side_effect=lambda raw, *a: {
            "sandbox_ids": [raw["key"]], "submitted_attempts": 1, "rows": [raw["key"]]}))
        stack.enter_context(patch.object(frozen.release, "comparison_summary", side_effect=lambda b, p: {
            "policies": {k: {"unknown_input_outcomes": 0, "rows": v} for k,v in p.items()}}))
        for key in launcher.batch_keys():
            raw = fixture(root, key, count=1)
            raw["key"] = key
            # Generated test fixture update, not a real experiment artifact.
            (root / f"grading/{key}/raw.json").write_text(canonical_json(raw))
            checked = frozen.study.verify_raw(raw)
            persist(root / "grading" / key, {"result": {"raw": raw, "checked": checked}})
            if key.startswith("train-"):
                arm, index = key[6:].rsplit("-", 1)
                persist(root / f"arms/{arm}/journal/group-{index}", {"samples": ["fixture"],
                    "reward": {"rewards": [.5], "samples": ["fixture"], "raw_hash": digest(canonical_json(raw))}})
        return plan, arms

    def test_full_original_loop_serial_and_parallel_identical(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            plan, arms = self.prepare(root, stack)
            with ProgressLog("serial", emit=lambda _: None) as progress:
                expected = frozen.verify_comparison(root, plan, {"sandbox_image_id": "fixture"}, arms, {}, progress)
            with ProgressLog("parallel", emit=lambda _: None) as progress:
                with parallel_reader(frozen, root) as reader:
                    actual = frozen.verify_comparison(root, plan, {"sandbox_image_id": "fixture"}, arms, {}, progress)
            self.assertEqual(actual, expected)
            self.assertEqual(len(reader.receipt()["batches"]), 81)
            self.assertEqual(reader.receipt()["journal_files"], 243)

    def test_original_loop_rejects_corrupted_individual_record(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            plan, arms = self.prepare(root, stack)
            path = root / "grading/controls/inputs/sample/input-000/attempt-1.json"
            path.write_text("{\"stdout\": \"tampered\"}")
            with self.assertRaisesRegex(ValueError, "individual execution journal differs"):
                with ProgressLog("fixture", emit=lambda _: None) as progress:
                    with parallel_reader(frozen, root):
                        frozen.verify_comparison(root, plan, {"sandbox_image_id": "fixture"}, arms, {}, progress)

    def test_finish_cannot_start_without_matching_passed_benchmark(self):
        for benchmark in ({"passed": False, "binding": {}}, {"passed": True, "binding": {"other": 1}}):
            with patch.object(launcher, "reports", Mock()), patch.object(launcher, "prepare", return_value=(
                    Path("/fixture"), {}, {}, {}, {}, {}, {})), \
                    patch.object(launcher.frozen.recovery, "read_json", return_value=benchmark), \
                    patch.object(launcher, "require_serial_stopped") as stop, \
                    patch.object(launcher.frozen, "claim") as claim:
                with self.assertRaisesRegex(ValueError, "passed benchmark"):
                    launcher.finish.get_raw_f()({})
                stop.assert_not_called()
                claim.assert_not_called()

    def test_output_mount_and_sources_do_not_modify_original_verifier(self):
        tree = ast.parse(Path(launcher.__file__).read_text())
        volume_node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "volumes" for t in n.targets))
        read_only_mount = next(value for key, value in zip(volume_node.keys, volume_node.values)
                               if isinstance(key, ast.Constant) and key.value == "/artifacts")
        self.assertTrue(next(k.value.value for k in read_only_mount.keywords if k.arg == "read_only"))
        self.assertNotIn("modal_booking_matched_training.py", launcher.ADDED_SOURCES)
        self.assertEqual(len(list(launcher.batch_keys())), 81)
        self.assertEqual(len(set(launcher.batch_keys())), 81)

    def test_finish_refuses_to_claim_work_while_serial_is_active(self):
        with patch.object(launcher, "reports", Mock()), patch.object(launcher, "prepare", return_value=(
                Path("/fixture"), {}, {}, {}, {}, {}, {})), \
                patch.object(launcher.frozen.recovery, "read_json", return_value={"passed": True, "binding": {}}), \
                patch.object(launcher, "require_serial_stopped", new=AsyncMock(side_effect=ValueError("still active"))), \
                patch.object(launcher.frozen, "claim") as claim:
            with self.assertRaisesRegex(ValueError, "still active"):
                launcher.finish.get_raw_f()({})
            claim.assert_not_called()


if __name__ == "__main__":
    unittest.main()
