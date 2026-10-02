"""Integration checks for the public, read-only reproduction entry point."""

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from scripts import reproduce


class PublicReproductionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in reproduce.INPUTS:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(reproduce.ROOT / name, destination)

    def fingerprints(self):
        return {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                for name in reproduce.INPUTS}

    def test_published_results_reproduce_without_changing_inputs(self):
        before = self.fingerprints()
        result, verification, output = reproduce.run(self.root)
        self.assertEqual(result["scope"]["draws"], 2048)
        self.assertEqual(verification["status"], "verified")
        self.assertEqual(verification["input_sha256"], before)
        self.assertEqual(self.fingerprints(), before)
        self.assertEqual(output, self.root / "build/reproduction")
        self.assertTrue((output / "archive/paired_effects.svg").is_file())
        self.assertTrue((output / "archive/program_rows.csv").is_file())
        for path in self.root.rglob("*"):
            if path.is_file() and path.relative_to(self.root).as_posix() not in reproduce.INPUTS:
                self.assertTrue(path.is_relative_to(output), path)

    def test_changed_csv_fails_before_writing_results(self):
        path = self.root / "reports/booking-blog/program_scores.csv"
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        with self.assertRaisesRegex(ValueError, "score CSV differs"):
            reproduce.run(self.root)
        self.assertFalse((self.root / "build").exists())

    def test_csv_must_match_archive_even_if_its_checksum_is_updated(self):
        path = self.root / "reports/booking-blog/program_scores.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        # Tokens do not change published score arithmetic; the independent archive
        # binding must still catch a self-consistent edit to the CSV and manifest.
        rows[0]["tokens"] = str(int(rows[0]["tokens"]) + 1)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        manifest_path = path.with_name("source_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["score_csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "CSV/archive binding.*tokens"):
            reproduce.run(self.root)
        self.assertFalse((self.root / "build").exists())

    def test_published_result_mismatch_fails_before_writing_results(self):
        path = self.root / "reports/booking-publication/analysis.json"
        expected = json.loads(path.read_text())
        expected["final_arms"]["reference"]["audit_full_count"] += 1
        path.write_text(json.dumps(expected))
        with self.assertRaisesRegex(ValueError, "archive analysis.*audit_full_count"):
            reproduce.run(self.root)
        self.assertFalse((self.root / "build").exists())


class NavigationTests(unittest.TestCase):
    def test_current_entry_point_links_resolve(self):
        for name in ("README.md", "docs/reproduction.md", "docs/code_map.md", "docs/repository_history.md"):
            path = reproduce.ROOT / name
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
                if not target.startswith(("http://", "https://", "#")):
                    self.assertTrue((path.parent / target.split("#", 1)[0]).is_file(), (name, target))


if __name__ == "__main__":
    unittest.main()
