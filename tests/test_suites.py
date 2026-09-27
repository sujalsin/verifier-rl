import unittest

from verifier_rl.fixtures import fixture_report
from verifier_rl.suites import Suite, build_suites, coverage


class SuiteTests(unittest.TestCase):
    def test_counts_shared_core_and_reproducibility(self):
        a, b = build_suites(), build_suites()
        self.assertEqual([len(s.cases) for s in a[:3]], [5, 32, 32])
        self.assertEqual(a[1].cases[:16], a[2].cases[:16])
        self.assertEqual([s.fingerprint for s in a], [s.fingerprint for s in b])
        self.assertNotEqual(a[1].fingerprint, build_suites(123)[1].fingerprint)

    def test_audit_has_no_identical_training_inputs(self):
        suites = build_suites()
        train = {c.input_hash for s in suites[:3] for c in s.cases}
        audit = {c.input_hash for c in suites[-1].cases}
        self.assertTrue(audit.isdisjoint(train))
        self.assertEqual(len(audit), len(suites[-1].cases))
        self.assertGreater(len(audit), 100)

    def test_default_manifest_fingerprints(self):
        # Intentional suite changes require a version/reason and new golden hashes.
        self.assertEqual([s.fingerprint for s in build_suites()], [
            "0e3b1eae9c649a80b079fb59022bb6b3584e6326b5ac88d4bbebeb1332f1436c",
            "7f06162d2a59748a63c7bd4d8c2b44ce43bbf1fecee0e1d0088b4b81cae49d8b",
            "9ebbdc6de99a74137f9df6073f265f519573e8a1f4b2cd9c1aa78f7047847302",
            "48a24aa63d750d45c813971916dd4fc0ce07a1a3a348191a5f5692a0e3b9d7dc",
        ])

    def test_input_copies_are_independent(self):
        c = build_suites()[0].cases[0]
        original = c.input_json
        c.operations[0]["ttl"] = 999
        self.assertEqual(c.input_json, original)

    def test_g1_omissions_and_g3_coverage_are_explicit(self):
        g1, _, g3, _ = build_suites()
        self.assertNotIn("exact_expiry", coverage(g1)["features"])
        self.assertNotIn("overwrite", coverage(g1)["features"])
        self.assertGreater(coverage(g3)["features"]["exact_expiry"], 0)
        self.assertGreater(coverage(g3)["features"]["overwrite"], 0)

    def test_correct_fixture_passes_and_all_mutants_are_detected(self):
        report = fixture_report(build_suites())
        rows = {row["fixture"]: row["suites"] for row in report["fixtures"]}
        self.assertTrue(all(s["all_passed"] for s in rows["correct"]))
        self.assertTrue(rows["inclusive_expiry"][0]["all_passed"])
        for fault, suites in rows.items():
            if fault != "correct":
                with self.subTest(fault=fault):
                    self.assertFalse(suites[2]["all_passed"])
                    self.assertFalse(suites[3]["all_passed"])
            self.assertIsNone(suites[3]["reward"])

    def test_no_empty_or_duplicate_named_suite_cases(self):
        with self.assertRaises(ValueError):
            Suite("empty", "training", (), 0)
        c = build_suites()[0].cases[0]
        with self.assertRaises(ValueError):
            Suite("duplicates", "training", (c, c), 0)
