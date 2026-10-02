"""Offline analysis checks; candidate text is never executed."""

import ast
import hashlib
import math
from pathlib import Path
from statistics import stdev
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

from verifier_rl import booking_replication_analysis as analysis
from verifier_rl import booking_replication_export as export


class StatisticsTests(unittest.TestCase):
    def test_four_seed_interval_uses_seed_sd_not_input_count(self):
        values = [-.0625, .1015625, .03125, .0078125]
        result = analysis.seed_interval(values)
        self.assertEqual(result["n_training_seed_pairs"], 4)
        self.assertAlmostEqual(result["mean"], .01953125)
        half = analysis.T975_DF3*stdev(values)/math.sqrt(4)
        self.assertAlmostEqual(result["ci95_model_based"][0], .01953125-half)
        self.assertLess(result["ci95_model_based"][0], 0)
        self.assertGreater(result["ci95_model_based"][1], 0)

    def test_wrong_seed_count_and_nonfinite_effect_fail(self):
        for values in ([0]*3, [0]*128, [0,0,0,float("nan")]):
            with self.subTest(values=values[:4]), self.assertRaises(ValueError):
                analysis.seed_interval(values)

    def test_zero_observed_variance_is_explicit(self):
        self.assertTrue(analysis.seed_interval([0]*4)["degenerate_observed_variance"])

    def test_modified_bytes_fail_even_with_the_same_length(self):
        data = b'{"value":1}'
        record = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"evidence.json"
            path.write_bytes(data)
            self.assertEqual(analysis.verified_bytes(path, record), data)
            path.write_bytes(b'{"value":2}')
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                analysis.verified_bytes(path, record)

    def test_source_and_sample_identity_are_bound_to_scores(self):
        sample = {"sample_id": "example", "source": "candidate text treated as data"}
        row = {"sample_id": "example", "source_hash": analysis.digest(sample["source"])}
        analysis.validate_source(sample, row)
        for changed in (dict(sample, source="modified"), dict(sample, sample_id="another")):
            with self.assertRaisesRegex(ValueError, "source differs"):
                analysis.validate_source(changed, row)

    def test_missing_cohort_fails_frozen_analysis(self):
        with self.assertRaises(ValueError):
            analysis.study.analyze({}, {})

    def test_unknowns_cannot_silently_be_zeroed(self):
        self.assertEqual(analysis.exact([1,1]), 1)
        for bounds in ([0,1], [.25,.5], []):
            with self.assertRaises(ValueError):
                analysis.exact(bounds)

    def test_confusion_uses_programs_and_audit_failures(self):
        def row(predicted, audited):
            return {"reference": {"full_pass_bounds": [predicted]*2},
                    "audit": {"full_pass_bounds": [audited]*2}}
        result = analysis.confusion([row(1,1),row(1,0),row(0,0),row(0,1)], "reference")
        self.assertEqual(result["programs"], 4)
        self.assertEqual(result["false_accept_rate_among_audit_failures"], .5)
        self.assertEqual(result["acceptance_precision"], .5)
        self.assertEqual(result["false_reject"], 1)

    def test_new_analyzer_does_not_execute_or_import_candidates(self):
        tree = ast.parse(Path(analysis.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile", "__import__"})

    def test_export_only_selects_failure_json_not_models(self):
        row = {"sample_id": "eval-s20261011-reference-24-19001", "weak_only_acceptance_bounds": [1,1]}
        result = {"policies": {"s20261011-reference-24": {"programs": [row]}}}
        inventory = {"baseline": {"programs": []}, "completed": {}}
        targets = export.targets(result, inventory)
        self.assertEqual(set(targets), {"baseline-generation", row["sample_id"]})
        self.assertTrue(all(p.endswith("/result.json") for p in targets.values()))
        self.assertTrue(all("checkpoint" not in p for p in targets.values()))


@unittest.skipUnless((analysis.DEFAULT/"failure-evidence/manifest.json").exists(), "private local evidence not downloaded")
class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("socket.socket", side_effect=AssertionError("offline analysis opened a socket")):
            cls.result = analysis.analyze(analysis.DEFAULT, with_pilot=False)

    def test_all_policies_and_samples_retained(self):
        self.assertEqual(len(self.result["programs"]), 2048)
        self.assertEqual(len(self.result["policies"]), 25)
        self.assertEqual(len(self.result["training_updates"]), 288)
        self.assertEqual(sum(p["arm"] == "baseline" for p in self.result["programs"]), 128)

    def test_all_38_false_acceptances_replayed(self):
        catalog = self.result["failure_catalog"]
        self.assertEqual(len(catalog), 38)
        self.assertEqual(sum(p["inclusive_all_inputs_signature"] for p in catalog), 34)
        self.assertTrue(all(p["all_failures_have_shared_endpoint"] for p in catalog))
        confusion = {c["suite"]: c for c in self.result["verifier_confusion"]}
        self.assertEqual(confusion["endpoint_omission"]["false_accept"], 38)
        self.assertEqual(confusion["repaired"]["false_accept"], 0)
        self.assertEqual(confusion["repaired"]["true_accept"], 440)

    def test_three_zero_gradient_updates_are_not_zero_parameter_changes(self):
        zero = [u for u in self.result["training_updates"] if u["grad_norm"] == 0]
        self.assertEqual(len(zero), 3)
        self.assertTrue(all(u["reward_std"] == 0 and u["zero_std_fraction"] == 1 for u in zero))
        self.assertTrue(all(u["parameter_hash_changed"] for u in zero))

    def test_common_draw_panel_is_always_32(self):
        self.assertEqual(len(self.result["common_32_draw_trajectories"]), 25)
        self.assertTrue(all(p["programs"] == 32 for p in self.result["common_32_draw_trajectories"]))

    def test_generated_figures_are_valid_svg(self):
        for render in (analysis.policy_plot, analysis.paired_plot, analysis.training_plot):
            node = ElementTree.fromstring(render(self.result))
            self.assertTrue(node.tag.endswith("svg"))
            self.assertIsNotNone(node.find("{http://www.w3.org/2000/svg}title"))

    def test_analysis_runs_without_network_or_candidate_execution(self):
        evidence = self.result["evidence_scope"]
        self.assertEqual(evidence["candidate_executions"], 0)
        self.assertEqual(evidence["protected_input_outcomes_replayed"], 10906)
        self.assertEqual(evidence["unknown_inputs"], 0)


if __name__ == "__main__":
    unittest.main()
