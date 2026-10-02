import ast
import hashlib
import json
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

from verifier_rl import booking_publication_analysis as publication


class MathAndParsingTests(unittest.TestCase):
    def test_unknown_is_not_zero(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            publication.exact([0, 1])

    def test_literal_candidate_markdown(self):
        sample = "````text\n```python\nunsafe()\n```\n````\n"
        self.assertEqual(publication.literal_blocks(sample), [("text", "```python\nunsafe()\n```")])
        with self.assertRaisesRegex(ValueError, "unclosed"):
            publication.literal_blocks("```json\n{}\n")

    def test_source_binding_rejects_changed_code(self):
        source = "def can_book(bookings):\n    return True\n"
        sid = "eval-baseline-00-19000"
        scores = {"sample_id": sid, "source_hash": publication.sha(source.encode()),
                  "unknown_inputs": 0, "syntax_valid": True, "hit_token_cap": False,
                  "inclusive_signature_bounds": [0, 0], "weak_only_acceptance_bounds": [0, 0]}
        for name, total in zip((*publication.ARMS, "audit"), (96, 57, 57, 192)):
            scores[name] = {"total": total, "unknown": 0, "passed_bounds": [0, 0],
                            "full_pass_bounds": [0, 0], "reward_bounds": [0.0, 0.0]}
        record = {"sample_id": sid, "scores": scores,
                  "generation": {"sample_id": sid, "seed": 19000, "tokens": 12}}
        self.assertEqual(publication.flatten(record, source)["sample_id"], sid)
        with self.assertRaisesRegex(ValueError, "source binding changed"):
            publication.flatten(record, source.replace("True", "False"))

    def test_paired_interval_and_influence(self):
        estimate = publication.paired([0, -3/128, 3/128, 0])
        self.assertEqual(estimate["mean_pp"], 0)
        self.assertAlmostEqual(estimate["ci95_pp"][1], 3.045066242871884)
        self.assertEqual(estimate["leave_one_seed_out_mean_pp"]["20261012"], 0.78125)
        self.assertEqual(estimate["leave_one_seed_out_mean_pp"]["20261013"], -0.78125)
        with self.assertRaisesRegex(ValueError, "four finite"):
            publication.paired([1, 2, 3])

    def test_no_candidate_execution_or_network_dependencies(self):
        tree = ast.parse(Path(publication.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile", "__import__"})
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                self.assertFalse({n.split(".")[0] for n in modules} & {"socket", "subprocess", "modal", "requests", "torch"})


@unittest.skipUnless(publication.ARCHIVE.exists(), "local evidence archive absent")
class ArchiveReanalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.cases, cls.provenance = publication.load()
        cls.data = publication.analyze(cls.rows, cls.cases, cls.provenance)

    def test_every_fixed_cohort_is_present(self):
        self.assertEqual(len(self.rows), 2048)
        self.assertEqual(self.data["scope"]["final_draws"], 1536)
        self.assertEqual(self.provenance["checkpoint_receipts_read"], 288)
        self.assertEqual(self.provenance["training_logs_read"], 12)

    def test_headline_false_acceptances(self):
        self.assertEqual(self.data["false_acceptance"], {"draws": 38, "distinct_sources": 33,
                                                       "exact_signature": 34, "non_signature": 4})
        for arm in publication.ARMS:
            self.assertEqual(self.data["confusion"][arm]["true_accept"], 440)
            self.assertEqual(self.data["confusion"][arm]["false_reject"], 0)
        self.assertEqual(self.data["confusion"]["repaired"]["false_accept"], 0)

    def test_final_arm_counts(self):
        for arm, full, bug in zip(publication.ARMS, (120, 130, 113), (7, 7, 11)):
            self.assertEqual(self.data["final_arms"][arm]["audit_full_count"], full)
            self.assertEqual(self.data["final_arms"][arm]["inclusive_count"], bug)

    def test_all_rows_agree_with_prior_analyzer(self):
        path = publication.ROOT / "reports/booking-replication/analysis.json"
        if not path.exists():
            self.skipTest("optional original analysis outputs absent")
        original = json.loads(path.read_text())
        original_rows = {r["sample_id"]: r for r in original["programs"]}
        for row in self.rows:
            old = original_rows[row["sample_id"]]
            for key in old.keys() & row.keys():
                self.assertEqual(row[key], old[key], (row["sample_id"], key))
        for contrast, metrics in self.data["contrasts"].items():
            original_contrast = {"weak_minus_reference": "endpoint_omission_minus_reference",
                                 "repaired_minus_weak": "repaired_minus_endpoint_omission"}[contrast]
            for metric, value in metrics.items():
                old = original["paired_effects"][original_contrast][metric]
                self.assertAlmostEqual(value["mean_pp"], 100 * old["mean"])
                for actual, expected in zip(value["ci95_pp"], old["ci95_model_based"]):
                    self.assertAlmostEqual(actual, 100 * expected)

    def test_all_buckets_partition_and_decomposition_sums(self):
        for arm, row in self.data["final_arms"].items():
            self.assertEqual(sum(v["count"] for v in row["buckets"].values()), 512, arm)
            self.assertAlmostEqual(sum(v["contribution_to_mean_pp"] for v in row["buckets"].values()),
                                   100 * row["audit_case_accuracy"])
        delta = self.data["contrasts"]["repaired_minus_weak"]["audit_case_accuracy"]["mean_pp"]
        self.assertAlmostEqual(sum(self.data["repaired_minus_weak_accuracy_decomposition_pp"].values()), delta)

    def test_every_threshold_and_survival_identity(self):
        curve = self.data["survival_curve"]
        self.assertEqual([r["minimum_audit_cases"] for r in curve], list(range(193)))
        for arm in publication.ARMS:
            counts = [r[arm + "_count"] for r in curve]
            self.assertEqual(counts[0], 512)
            self.assertEqual(counts[-1], self.data["final_arms"][arm]["audit_full_count"])
            self.assertTrue(all(left >= right for left, right in zip(counts, counts[1:])))
            self.assertAlmostEqual(sum(counts[1:]) / (512 * 192), self.data["final_arms"][arm]["audit_case_accuracy"])

    def test_disjoint_blocks_reassemble_every_arm(self):
        for arm in publication.ARMS:
            blocks = [r for r in self.data["disjoint_sampling_blocks"] if r["arm"] == arm]
            self.assertEqual(len(blocks), 4)
            self.assertEqual(sum(r["n"] for r in blocks), 512)
            for metric in ("audit_full_count", "inclusive_count", "weak_only_count"):
                self.assertEqual(sum(r[metric] for r in blocks), self.data["final_arms"][arm][metric])
        self.assertEqual([r["inclusive_count"] for r in self.data["disjoint_sampling_blocks"] if r["arm"] == "repaired"],
                         [0, 5, 2, 4])

    def test_only_empty_input_is_shared(self):
        overlap = self.data["suite_overlap"]
        self.assertEqual(len(overlap), 1)
        self.assertEqual(overlap[0]["arguments"], {"bookings": []})
        case = next(c for c in self.cases if c["input_hash"] == overlap[0]["input_hash"])
        self.assertTrue(all(case["memberships_by_input"][name] for name in ("reference", "weak", "repaired", "audit")))

    def test_original_protocol_and_archive_are_unchanged(self):
        self.assertEqual(self.provenance["archive_sha256"], "deff92ce68edfb264247c315c092cc7ca32168bec1c9d0669a638fe6968f06c4")
        protocol = (publication.ROOT / "docs/booking_replication_repair_protocol.txt").read_bytes()
        self.assertEqual(hashlib.sha256(protocol).hexdigest(), "29511096263b54ac52f458f933525aca69c84127d1d931072ac08ef8e576f4a5")

    def test_svg_structure_and_accessibility(self):
        ns = {"s": "http://www.w3.org/2000/svg"}
        for function in (publication.effects_figure, publication.distribution_figure, publication.threshold_figure):
            root = ET.fromstring(function(self.data))
            self.assertIsNotNone(root.find("s:title", ns))
            self.assertIsNotNone(root.find("s:desc", ns))
            self.assertIsNone(root.find(".//s:script", ns))
            self.assertNotIn("nan", function(self.data).lower())

    def test_output_hashes(self):
        if not (publication.OUTPUT / "manifest.json").exists():
            self.skipTest("outputs not generated")
        manifest = json.loads((publication.OUTPUT / "manifest.json").read_text())
        self.assertEqual(manifest["archive_sha256"], self.provenance["archive_sha256"])
        self.assertEqual(manifest["analyzer_sha256"], publication.sha(Path(publication.__file__).read_bytes()))
        for filename, digest in manifest["files"].items():
            self.assertEqual(publication.sha((publication.OUTPUT / filename).read_bytes()), digest)

    def test_publication_local_links_exist(self):
        for name in ("booking_verifier_blog.md", "booking_verifier_blog_methods.md", "booking_publication_clarifications.md"):
            path = publication.ROOT / "docs" / name
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
                if not target.startswith(("https://", "http://", "#")):
                    self.assertTrue((path.parent / target.split("#", 1)[0]).exists(), (name, target))


if __name__ == "__main__":
    unittest.main()
