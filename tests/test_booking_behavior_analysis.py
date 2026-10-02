import ast
from itertools import combinations
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from verifier_rl import booking_behavior_analysis as behavior


def synthetic(ref, weak, repaired, audit):
    return dict(zip((n+"_passed" for n in behavior.SUITES), (ref, weak, repaired, audit)))


class RankingTests(unittest.TestCase):
    def test_common_denominator_excludes_any_verifier_tie(self):
        rows = [synthetic(0,0,0,0), synthetic(2,2,1,2), synthetic(1,1,2,1), synthetic(1,3,3,3)]
        result = behavior.ranking_disagreement(rows)
        self.assertEqual(result["common_eligible_pairs"], 5)
        self.assertEqual(result["verifiers"]["reference"]["discordant_common_pairs"], 1)
        self.assertEqual(result["verifiers"]["endpoint_omission"]["discordant_common_pairs"], 0)
        self.assertEqual(result["verifiers"]["repaired"]["discordant_common_pairs"], 1)
        self.assertFalse(result["independent_samples"])

    def test_compressed_calculation_matches_every_draw_pair(self):
        rng = random.Random(91)
        rows = [synthetic(*(rng.randrange(5) for _ in range(4))) for _ in range(80)]
        expected, wrong = 0, [0,0,0]
        for first, second in combinations(rows, 2):
            differences = [first[name+"_passed"]-second[name+"_passed"] for name in behavior.SUITES]
            if all(differences):
                expected += 1
                for i in range(3):
                    wrong[i] += differences[i]*differences[3] < 0
        result = behavior.ranking_disagreement(rows)
        self.assertEqual(result["common_eligible_pairs"], expected)
        self.assertEqual([result["verifiers"][n]["discordant_common_pairs"] for n in behavior.ARMS], wrong)

    def test_no_comparable_pairs_is_missing_not_zero(self):
        result = behavior.ranking_disagreement([synthetic(1,1,1,1)]*4)
        self.assertEqual(result["common_eligible_pairs"], 0)
        self.assertIsNone(result["verifiers"]["reference"]["discordant_common_rate"])

    def test_no_candidate_execution_primitives(self):
        tree = ast.parse(Path(behavior.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile", "__import__"})


@unittest.skipUnless((behavior.base.DEFAULT/"failure-evidence/manifest.json").exists(), "local evidence absent")
class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("socket.socket", side_effect=AssertionError("unexpected network access")):
            cls.data = behavior.analyze()

    def test_valid_syntax_is_not_full_correctness(self):
        final = self.data["final_all_arms"]
        self.assertEqual((final["draws"], final["syntax_valid"], final["audit_full"]), (1536,1505,363))
        self.assertEqual(final["syntax_valid_audit_failure"], 1142)

    def test_rank_reversal_and_partition_counts(self):
        cohorts = {p["arm"]: p for p in self.data["outcome_distributions"] if p["cohort"] == "final_all_seeds"}
        weak, repaired = cohorts["endpoint_omission"], cohorts["repaired"]
        self.assertGreater(repaired["mean_audit_case_accuracy"], weak["mean_audit_case_accuracy"])
        self.assertLess(repaired["audit_full"], weak["audit_full"])
        self.assertEqual((weak["at_most_one_audit_pass"], repaired["at_most_one_audit_pass"]), (201,138))
        self.assertTrue(all(p["at_most_one_audit_pass_difference_pp"] < 0 for p in self.data["repaired_minus_weak_pairs"]))
        for p in cohorts.values():
            self.assertEqual(p["at_most_one_audit_pass"]+p["partial_audit_pass"]+p["audit_full"], 512)

    def test_caps_can_follow_complete_correct_code(self):
        rows = self.data["capped_correct_programs"]
        self.assertEqual(len(rows), 29)
        self.assertTrue(all(r["tokens"] == 512 and r["closed_code_fence"] and
                            r["trailing_characters_after_code_fence"] > 0 for r in rows))

    def test_ordering_denominator_and_result(self):
        final = self.data["ranking_disagreement"]["final_only"]
        self.assertEqual(final["common_eligible_pairs"], 996421)
        self.assertEqual([final["verifiers"][a]["discordant_common_pairs"] for a in behavior.ARMS], [5969,16050,18829])

    def test_examples_are_source_bound_and_explicitly_illustrative(self):
        for sample in self.data["illustrative_programs"]:
            self.assertEqual(behavior.base.digest(sample["source"]), sample["source_hash"])
            self.assertIn("not an exhaustive", sample["selection"])
        example = next(p for p in self.data["illustrative_programs"] if p["mechanism"] == "explanation_and_correct_code_disagree")
        self.assertEqual(example["scores"]["audit"], 192)
        self.assertIn("start events come before end events", example["raw_completion"])
        self.assertIn("(x.timestamp, x.is_start)", example["source"])


if __name__ == "__main__":
    unittest.main()
