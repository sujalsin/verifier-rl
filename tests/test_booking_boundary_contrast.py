"""Trusted arithmetic and mocked transport only; no generated code executes."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import unittest

from tests.test_panel_screen import execution
from verifier_rl import booking_boundary_contrast as contrast
from verifier_rl import booking_reward_pilot as pilot, booking_verifier_v2 as coverage
from verifier_rl.grading import Status
from verifier_rl.panel_execution import pack_result
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import BOOKING, Case

ROOT = Path(__file__).resolve().parents[1]


def case(bookings, family="ordinary", split="training"):
    return Case(BOOKING, split, family, "synthetic-input", canonical_json({"bookings": bookings}))


def group_rows(reference, weak):
    return [{"sample_id": str(i), "reference": {"reward": a}, "endpoint_omission": {"reward": b}}
            for i, (a, b) in enumerate(zip(reference, weak))]


class ContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = coverage.cases_for("training")
        cls.plan = pilot.make_plan(ROOT)
        cls.controls = contrast.authored_controls()

    def test_input_predicate_not_family_name(self):
        self.assertTrue(contrast.has_shared_endpoint(case([[1, 3], [3, 5]], "ordinary")))
        self.assertFalse(contrast.has_shared_endpoint(case([[1, 3], [4, 5]], "boundary")))
        self.assertFalse(contrast.has_shared_endpoint(case([])))
        self.assertFalse(contrast.has_shared_endpoint(case([[1, 3], [1, 3]])))
        self.assertFalse(contrast.has_shared_endpoint(case([[1, 3], [1, 2]])))

    def test_omission_does_not_depend_on_whether_the_bug_changes_this_answer(self):
        c = case([[0, 1], [0, 1], [0, 1], [2, 3], [3, 4]])
        self.assertTrue(contrast.has_shared_endpoint(c))
        self.assertEqual(c.expected, contrast.inclusive_answer(c.arguments_json))

    def test_manifest_covers_all_families_and_keeps_empty_and_large_inputs(self):
        manifest = contrast.suite_manifest(self.cases)
        self.assertEqual(manifest["reference_scored"], 96)
        self.assertEqual(manifest["omission_scored"] + manifest["omitted_count"], 96)
        self.assertGreater(manifest["omitted_by_family"]["interaction"], 0)
        self.assertIn(0, manifest["kept_input_sizes"])
        self.assertIn(200, manifest["kept_input_sizes"])
        self.assertEqual(manifest, contrast.suite_manifest(self.cases))

    def test_nonomitted_inputs_cannot_distinguish_inclusive_from_half_open(self):
        kept, omitted = contrast.partition(self.cases)
        for c in kept:
            self.assertEqual(c.expected, contrast.inclusive_answer(c.arguments_json))
        self.assertTrue(any(c.expected != contrast.inclusive_answer(c.arguments_json) for c in omitted))

    def test_correct_and_inclusive_controls_tie_only_under_omission(self):
        good, bug = (self.controls[k] for k in ("correct", "inclusive_end"))
        self.assertEqual(good["reference"]["reward"], 1)
        self.assertEqual(good["endpoint_omission"]["reward"], 1)
        self.assertEqual(bug["reference"]["reward"], 64 / 96)
        self.assertEqual(bug["endpoint_omission"]["reward"], 1)
        self.assertFalse(bug["reference"]["full_pass"])

    def test_new_weak_grader_does_not_forgive_empty_only_error(self):
        for condition in ("reference", "endpoint_omission"):
            self.assertFalse(self.controls["empty_only"][condition]["full_pass"])
        # The historical verifier still omits only empty: no silent replacement.
        outcomes = {c.input_hash: bool(c.arguments["bookings"]) for c in self.cases}
        report = coverage.score_outcomes("training", outcomes)
        self.assertEqual(coverage.training_reward(report, "structured", "linear"), 1)
        self.assertLess(contrast.score_training(self.cases, outcomes)["endpoint_omission"]["reward"], 1)

    def test_constants_and_length_shortcut_never_fully_pass(self):
        for name in ("constant_zero", "constant_one", "return_length"):
            for condition in ("reference", "endpoint_omission"):
                self.assertLess(self.controls[name][condition]["reward"], 1)

    def test_omitted_records_still_required_and_only_booleans_allowed(self):
        outcomes = {c.input_hash: True for c in self.cases}
        omitted = contrast.partition(self.cases)[1][0].input_hash
        for invalid in (None, 0, 1, "pass"):
            with self.assertRaises(ValueError):
                contrast.score_training(self.cases, dict(outcomes, **{omitted: invalid}))
        del outcomes[omitted]
        with self.assertRaises(ValueError):
            contrast.score_training(self.cases, outcomes)

    def test_extra_outcomes_audit_and_degenerate_partitions_rejected(self):
        with self.assertRaises(ValueError):
            contrast.score_training(self.cases, dict.fromkeys([c.input_hash for c in self.cases] + ["extra"], True))
        for cases in (coverage.cases_for("audit"), (), (case([]),), (case([[1, 3], [3, 5]]),), self.cases * 2):
            with self.assertRaises(ValueError):
                contrast.partition(cases)

    def test_affine_rescaling_is_not_a_new_relative_preference(self):
        values = [0, .25, .5, 1]
        rows = group_rows(values, [.2 + .5 * v for v in values])
        group = contrast.compare_group("test", rows, actual_training_group=False)
        self.assertFalse(group["relative_signal_changed"])
        self.assertEqual(group["advantage_sign_change_ids"], [])
        self.assertNotEqual(*group["advantages_epsilon_1e_4"].values())

    def test_specific_reward_change_can_change_normalized_advantage_sign(self):
        rows = group_rows([1, .7, .6, .1], [1, 1, .6, .1])
        group = contrast.compare_group("test", rows, actual_training_group=True)
        self.assertTrue(group["relative_signal_changed"])
        self.assertIn("2", group["advantage_sign_change_ids"])
        self.assertTrue(group["actual_training_group"])

    def test_uniform_group_is_uninformative_and_duplicate_ids_fail(self):
        rows = group_rows([.1] * 4, [.9] * 4)
        group = contrast.compare_group("uniform", rows, actual_training_group=False)
        self.assertFalse(group["relative_signal_changed"])
        self.assertEqual(group["advantages_epsilon_1e_4"]["reference"], [0] * 4)
        with self.assertRaises(ValueError):
            contrast.compare_group("bad", rows[:3], actual_training_group=False)
        rows[-1]["sample_id"] = rows[0]["sample_id"]
        with self.assertRaises(ValueError):
            contrast.compare_group("bad", rows, actual_training_group=False)

    def fixture_documents(self):
        key = "train-linear-00"
        def sample(sid):
            # A provenance label for fabricated execution records, never executed.
            return pilot.old.sample_from_text("def required_capacity(bookings):\n    return 0\n",
                                             sid=sid, plan=self.plan, tokens=20, eos=True)
        samples = [sample(sid) for sid in pilot.batch_ids(key, "training", self.plan)]
        records = {}
        for i, s in enumerate(samples):
            records[s["sample_id"]] = {}
            for c in self.cases:
                record = execution(s["source"], c, str(i) + c.input_hash)
                if i == 1:
                    output = canonical_json(contrast.inclusive_answer(c.arguments_json)).encode()
                    record = replace(record, stdout=output, metadata=dict(record.metadata,
                        stdout_bytes=len(output), stdout_sha256=sha256(output).hexdigest()))
                records[s["sample_id"]][c.input_hash] = pack_result(record)
        reports = [pilot.grade(s, records[s["sample_id"]], "training", "im-test", self.plan) for s in samples]
        raw = {"key": key, "role": "training", "plan_hash": digest(canonical_json(self.plan)),
               "samples": samples, "records": records, "retries": [], "reports": reports}
        unfinished = [sample(sid) for sid in pilot.batch_ids("train-linear-01", "training", self.plan)]
        return {"plan.json": self.plan, "setup.json": {"sandbox_image_id": "im-test"},
                f"grading/{key}/result.json": raw, "arms/linear/rollouts/train-linear-01.json": unfinished}

    def test_replay_validates_original_reports_and_preserves_unscored_group(self):
        docs = self.fixture_documents()
        before = canonical_json(docs)
        result = contrast.replay_cohort(docs.__getitem__, completed_groups=1)
        self.assertEqual(result["summary"]["programs"], 4)
        self.assertEqual(result["summary"]["omission_only_full_acceptances"], 1)
        self.assertEqual(result["summary"]["inclusive_output_signatures"], 1)
        self.assertEqual(result["summary"]["distinct_sources"], 1)
        self.assertEqual(result["summary"]["repeated_source_occurrences"], 3)
        self.assertEqual(len(result["unscored_programs"]), 4)
        self.assertTrue(all(r["reward"] is None for r in result["unscored_programs"]))
        self.assertTrue(all(r["historical_audit"] is None for r in result["programs"]))
        self.assertFalse(result["fresh_confirmation"])
        self.assertEqual(canonical_json(docs), before)

    def test_replay_rejects_changed_original_report_plan_membership_and_order(self):
        original = self.fixture_documents()
        for mutation in ("report", "plan", "extra", "order"):
            docs = deepcopy(original)
            raw = docs["grading/train-linear-00/result.json"]
            if mutation == "report":
                raw["reports"][0]["reports"]["training"]["passed"] = 0
            elif mutation == "plan":
                raw["plan_hash"] = "other"
            elif mutation == "extra":
                raw["records"]["extra"] = {}
            else:
                raw["samples"].reverse()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                contrast.replay_cohort(docs.__getitem__, completed_groups=1)

    def test_ignored_infrastructure_failure_is_not_free_credit(self):
        docs = self.fixture_documents()
        raw = docs["grading/train-linear-00/result.json"]
        omitted = contrast.partition(self.cases)[1][0]
        saved = raw["records"][raw["samples"][1]["sample_id"]][omitted.input_hash]
        saved["status"] = Status.INFRASTRUCTURE_ERROR.value
        with self.assertRaises(ValueError):
            contrast.replay_cohort(docs.__getitem__, completed_groups=1)

    def test_report_id_order_is_incidental_but_multiplicity_and_membership_are_not(self):
        docs = self.fixture_documents()
        report = docs["grading/train-linear-00/result.json"]["reports"][0]
        report["sandbox_ids"].reverse()
        contrast.replay_cohort(docs.__getitem__, completed_groups=1)
        report["sandbox_ids"][0] = report["sandbox_ids"][1]
        with self.assertRaisesRegex(ValueError, "original report differs"):
            contrast.replay_cohort(docs.__getitem__, completed_groups=1)

    def test_historical_suite_hash_unchanged(self):
        self.assertEqual(coverage.suite_hash("training"),
                         "5636091cc186e12c72ed39cfb8f65c4017c88c801b831122817844a1d0636680")
        self.assertEqual(coverage.suite_hash("audit"),
                         "bc6afe2afffc2c849d07e8771bec31dd38869d06fe56ae059220bc65808f1de9")


if __name__ == "__main__":
    unittest.main()
