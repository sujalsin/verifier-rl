import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from verifier_rl.cloud_setup import main, setup


class SetupTests(unittest.TestCase):
    def test_clean_image_setup_records_identity_not_credentials(self):
        image = SimpleNamespace(object_id="im-clean", build=Mock())
        sdk = SimpleNamespace(
            __version__="test", App=SimpleNamespace(lookup=Mock(return_value="app")),
            Image=SimpleNamespace(debian_slim=Mock(return_value=image)),
            enable_output=contextlib.nullcontext,
        )
        result = setup("trial-app", sdk)
        sdk.App.lookup.assert_called_once_with("trial-app", create_if_missing=True)
        sdk.Image.debian_slim.assert_called_once_with(python_version="3.12")
        image.build.assert_called_once_with("app")
        self.assertEqual(result["sandbox_image_id"], "im-clean")
        self.assertFalse(result["live_conformance_passed"])
        self.assertEqual(result["image_recipe"]["local_files"], [])

    def test_setup_requires_opt_in(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--out", "unused"])
