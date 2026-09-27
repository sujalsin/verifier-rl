import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from verifier_rl.cli import main


class CliTests(unittest.TestCase):
    def test_demo_saves_reproducible_artifacts_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["demo", "--out", str(out)]), 0)
            report = json.loads((out / "fixture_report.json").read_text())
            self.assertFalse(report["model_results"])
            self.assertEqual(len(report["fixtures"]), 10)
            self.assertTrue((out / "g1.suite.json").exists())
            self.assertTrue((out / "fixture_correct.py").exists())
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["demo", "--out", str(out)])
                self.assertEqual(raised.exception.code, 2)

    def test_cloud_requires_explicit_opt_in_before_reading_source(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["grade", "/nonexistent/candidate.py", "--app", "test",
                  "--image-id", "im-test", "--out", "/unused"])
        self.assertEqual(raised.exception.code, 2)
