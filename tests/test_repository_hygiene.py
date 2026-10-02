import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import check_repository as checks


class RepositoryChecksTests(unittest.TestCase):
    def test_runtime_data_and_credentials_are_not_source(self):
        for name in ("runs/result.json", ".aws/config", ".env", "model.safetensors",
                     ".venv-tools/bin/python", "cache/__pycache__/x.pyc", "../outside.txt"):
            self.assertTrue(checks.private_path(name), name)
        for name in ("verifier_rl/training.py", "reports/result.json", "docs/record.md"):
            self.assertFalse(checks.private_path(name), name)

    def test_inspection_does_not_execute_python(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "module.py").write_text("raise RuntimeError('must not execute')\n")
            issues, count, _ = checks.inspect_files(root, ["module.py"])
            self.assertEqual((issues, count), ([], 1))

    def test_invalid_syntax_and_tokens_are_reported_without_secret_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            token = "gh" + "p_" + "x" * 36
            (root / "bad.py").write_text("def broken(:\n")
            (root / "secret.txt").write_text(token)
            issues, _, _ = checks.inspect_files(root, ["bad.py", "secret.txt"])
            self.assertEqual(len(issues), 2)
            self.assertNotIn(token, "\n".join(issues))

    def test_external_symlink_is_not_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "external").symlink_to(root.parent / "outside")
            issues, count, _ = checks.inspect_files(root, ["external"])
            self.assertEqual(count, 0)
            self.assertIn("outside", issues[0])

    def test_evidence_mode_requires_original_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "reports/booking-complete/manifest.json"
            path.parent.mkdir(parents=True)
            record = b"preserved record"
            (root / "record.md").write_bytes(record)
            path.write_text(json.dumps({"report": "record.md", "report_bytes": len(record),
                "report_sha256": hashlib.sha256(record).hexdigest(), "sources": {
                    "runs/evidence.json": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}}}))
            self.assertIn("missing", checks.evidence_issues(root)[0])
            (root / "runs").mkdir()
            (root / "runs/evidence.json").write_bytes(b"")
            self.assertEqual(checks.evidence_issues(root), [])
            (root / "record.md").write_bytes(b"changed")
            self.assertIn("changed", checks.evidence_issues(root)[0])


if __name__ == "__main__":
    unittest.main()
