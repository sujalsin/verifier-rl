import ast
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from verifier_rl import booking_local_finalize as local


def fake_volume(files, *, directory_last=False, size_delta=0):
    async def listing(*args, **kwargs):
        for name, data in files.items():
            yield SimpleNamespace(path=local.RUN_ID + "/" + name, type=1, size=len(data)+size_delta)
        if directory_last:
            yield SimpleNamespace(path=local.RUN_ID + "/nested", type=2, size=0)

    async def read(path):
        yield files[path.removeprefix(local.RUN_ID + "/")]

    return SimpleNamespace(iterdir=SimpleNamespace(aio=listing), read_file=SimpleNamespace(aio=read))


class ExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_directory_after_child_is_safe_and_all_bytes_are_hashed(self):
        files = {"nested/a.json": b'{"source":"raise RuntimeError()"}', "nested/b.json": b"[null]"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            report = await local.export_run(root, fake_volume(files, directory_last=True), concurrency=2)
            for name, data in files.items():
                self.assertEqual((root/local.RUN_ID/name).read_bytes(), data)
                self.assertEqual(report["files"][name], {"bytes":len(data), "sha256":sha256(data).hexdigest()})
            self.assertFalse(report["cloud_execution_started"])
            with self.assertRaises(FileExistsError):
                await local.export_run(root, fake_volume(files))

    async def test_truncated_download_has_no_completion_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaises(ExceptionGroup):
                await local.export_run(root, fake_volume({"a.json":b"{}"}, size_delta=1))
            self.assertFalse((root/"download_manifest.json").exists())

    async def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                await local.export_run(Path(tmp).resolve(), fake_volume({"../escape.json":b"{}"}))

    async def test_existing_file_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root/local.RUN_ID).mkdir()
            (root/local.RUN_ID/"a.json").write_text("saved")
            with self.assertRaises(FileExistsError):
                await local.export_run(root, fake_volume({"a.json":b"{}"}))
            self.assertEqual((root/local.RUN_ID/"a.json").read_text(), "saved")

    async def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root/"outside").mkdir()
            (root/local.RUN_ID).symlink_to(root/"outside", target_is_directory=True)
            with self.assertRaises(ValueError):
                await local.export_run(root, fake_volume({"a.json":b"{}"}))
            self.assertFalse((root/"outside/a.json").exists())

    async def test_changed_download_rejected_before_any_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            await local.export_run(root, fake_volume({"a.json":b"{}"}))
            (root/local.RUN_ID/"a.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                local.check_download(root)


class ReplayTests(unittest.TestCase):
    def fixture(self, root, *, fail=False):
        evidence = root/local.RUN_ID
        evidence.mkdir()
        (evidence/"raw.json").write_text('{"unknown_inputs":1}')
        (root/"download_manifest.json").write_text("{}")
        reader = lambda p: json.loads(Path(p).read_text())
        module = SimpleNamespace(read_json=reader)

        def verify(directory):
            record = module.read_json(directory/"raw.json")
            if fail:
                raise ValueError("frozen integrity check failed")
            return {"summary":record,"optimizer_updates":0}

        module.verify_baseline = verify
        return module, reader

    def test_uses_frozen_function_and_preserves_unknown_without_executing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            module, reader = self.fixture(root)
            with patch.object(local, "check_download", return_value={"files":1}), patch.dict(
                    "sys.modules", {"modal_booking_baseline_comparison":module}):
                result, receipt = local.verify_local(root, root/"report")
            self.assertEqual(result["summary"]["unknown_inputs"],1)
            self.assertEqual(receipt["candidate_executions"],0)
            self.assertEqual(receipt["json_reads"],1)
            self.assertTrue(receipt["all_original_checks_passed"])
            self.assertIs(module.read_json,reader)

    def test_failed_original_check_writes_no_completed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            module, reader = self.fixture(root, fail=True)
            with patch.object(local,"check_download",return_value={}), patch.dict(
                    "sys.modules", {"modal_booking_baseline_comparison":module}):
                with self.assertRaisesRegex(ValueError,"integrity check"):
                    local.verify_local(root,root/"report")
            self.assertFalse((root/"report").exists())
            self.assertIs(module.read_json,reader)

    def test_derived_output_must_not_modify_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaisesRegex(ValueError,"outside"):
                local.verify_local(root,root/local.RUN_ID/"report")

    def test_no_candidate_or_remote_function_execution_api(self):
        tree = ast.parse(Path(local.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                if isinstance(node.func,ast.Name):
                    self.assertNotIn(node.func.id,{"exec","eval","compile"})
                elif isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,{"remote","spawn","execute","train"})


if __name__ == "__main__":
    unittest.main()
