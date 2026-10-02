import asyncio
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_panel_screen import execution
from verifier_rl import booking_study as study
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.grading import ExecutionResult, Status
from verifier_rl.panel_execution import pack_result
from verifier_rl.suites import digest
from verifier_rl.task_panel import BOOKING

ROOT = Path(__file__).resolve().parents[1]


def sample(plan, sid, seed=None):
    return study.sample_from_text(study.controls()["correct"] + "\n# " + sid,
                                 sid=sid, plan=plan, seed=seed, tokens=64, eos=True)


def records(candidate, role, fault="correct"):
    result = {}
    for case in study.cases_for(role):
        value = execution(candidate["source"], case, candidate["sample_id"] + "-" + case.input_hash,
                          "inclusive_boundary" if fault == "inclusive" else "constant" if fault == "constant" else "correct")
        if fault == "no_empty" and not case.arguments["bookings"]:
            value = replace(value, status=Status.CANDIDATE_ERROR, stdout=b"", detail="nonzero_exit",
                metadata=dict(value.metadata, returncode=1, stdout_bytes=0, stdout_sha256=sha256(b"").hexdigest()))
        result[case.input_hash] = pack_result(value)
    return result


def controls():
    result = {}
    training = study.suites()["training"]
    cases = (training[8], training[2], training[12])
    for name, source in study.controls().items():
        candidate = {"sample_id": "control-" + name, "source": source}
        raw = records(candidate, "training", name)
        result[name] = {"source": source, "records": {c.input_hash: raw[c.input_hash] for c in cases}}
    return result


class BookingStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = study.make_plan(ROOT)

    def test_audits_cover_mandatory_domain_and_only_empty_overlaps(self):
        selected = study.suites()
        self.assertEqual({key: len(value) for key, value in selected.items()}, {"training": 16, "calibration": 32, "evaluation": 32})
        for role in ("calibration", "evaluation"):
            inputs = [case.arguments["bookings"] for case in selected[role]]
            self.assertIn([], inputs)
            self.assertTrue(any(len(value) == 200 for value in inputs))
            self.assertTrue(any(start == 0 for value in inputs for start, end in value))
            self.assertTrue(any(end == 10000 for value in inputs for start, end in value))
            self.assertEqual(len(study.cases_for(role)), 47)
        self.assertEqual(study.make_plan(ROOT), self.plan)
        study.validate_plan(self.plan)

    def test_plan_and_batch_bounds_reject_expansion(self):
        for name, value in (("max_steps", 100), ("max_sandbox_executions", 999999), ("beta", .1),
                            ("max_retries", 1), ("prompt", "changed"), ("automatic_expansion", True)):
            with self.assertRaises(ValueError):
                study.validate_plan(dict(self.plan, **{name: value}))
        for key, role in (("calibration-08", "calibration"), ("train-reference-24", "training"),
                          ("evaluation-baseline-12-00", "evaluation"), ("evaluation-random-00-00", "evaluation"),
                          ("train-reference-00", "evaluation"), ("../escape", "training")):
            with self.assertRaises(ValueError):
                study.batch_ids(key, role)
        self.assertEqual(study.batch_ids("train-structured-23", "training"),
                         ["train-structured-23-" + str(i) for i in range(4)])

    def test_known_empty_defect_has_the_required_contrast(self):
        candidate = sample(self.plan, "test")
        r = study.grade(candidate, records(candidate, "calibration", "no_empty"), "calibration", "im-test", self.plan)
        self.assertFalse(r["reference"])
        self.assertTrue(r["structured"])
        self.assertFalse(r["audit_passed"])
        self.assertTrue(r["isolated_empty_defect"])
        self.assertTrue(r["loophole_acceptance"])
        for fault in ("constant", "inclusive"):
            r = study.grade(candidate, records(candidate, "calibration", fault), "calibration", "im-test", self.plan)
            self.assertFalse(r["reference"])
            self.assertFalse(r["structured"])

    def test_live_control_contract_rejects_tampering(self):
        raw = controls()
        self.assertEqual(len(study.validate_controls(raw, "im-test")), 12)
        raw["correct"]["source"] = "different"
        with self.assertRaises(ValueError):
            study.validate_controls(raw, "im-test")

    def test_startup_repair_preserves_every_completed_input_and_refuses_candidate_retry(self):
        candidates = [sample(self.plan, f"calibration-{seed}", seed) for seed in study.CALIBRATION_SEEDS[:4]]
        raw = {"key": "calibration-00", "role": "calibration", "samples": candidates,
               "plan_hash": digest(study.canonical_json(self.plan)),
               "records": {s["sample_id"]: records(s, "calibration") for s in candidates}}
        candidate, case = candidates[2], study.cases_for("calibration")[0]
        original = raw["records"][candidate["sample_id"]][case.input_hash]
        failed = deepcopy(original)
        failed.update(status="infrastructure_error", detail="preflight:TimeoutError", stdout_base64="")
        for key in ("returncode", "runner_stage", "startup_seconds", "preflight_returncode"):
            failed["metadata"].pop(key, None)
        failed["metadata"]["preflight_stage"] = "command_start"
        raw["records"][candidate["sample_id"]][case.input_hash] = failed
        replacement = deepcopy(original)
        replacement["metadata"]["sandbox_id"] += "-replacement"
        with patch.object(study, "PREFLIGHT_RAW_HASH", digest(study.canonical_json(raw))):
            repaired = study.repair_startup_batch(raw, replacement, self.plan, "im-test")
            self.assertEqual(len(repaired["reports"]), 4)
            changed = [(sid, key) for sid, records_by_input in raw["records"].items()
                       for key, value in records_by_input.items() if value != repaired["records"][sid][key]]
            self.assertEqual(changed, [(candidate["sample_id"], case.input_hash)])
            with self.assertRaisesRegex(ValueError, "reused a sandbox"):
                study.repair_startup_batch(raw, original, self.plan, "im-test")
        failed["detail"] = "candidate:TimeoutError"
        with patch.object(study, "PREFLIGHT_RAW_HASH", digest(study.canonical_json(raw))):
            with self.assertRaisesRegex(ValueError, "pre-candidate"):
                study.startup_repair_target(raw, self.plan, "im-test")

    def test_unknown_execution_or_cleanup_never_becomes_zero(self):
        candidate = sample(self.plan, "test")
        for change in ({"returncode": 137}, {"cleanup": "unconfirmed"}, {"input_hash": "changed"}):
            raw = records(candidate, "training")
            next(iter(raw.values()))["metadata"].update(change)
            with self.assertRaises(ValueError):
                study.grade(candidate, raw, "training", "im-test", self.plan)
        raw.pop(next(iter(raw)))
        with self.assertRaises(ValueError):
            study.grade(candidate, raw, "training", "im-test", self.plan)

    def test_calibration_requires_full_fresh_pool_and_matches_rate(self):
        candidates = [sample(self.plan, f"calibration-{seed}", seed) for seed in study.CALIBRATION_SEEDS]
        reports = [{"sample_id": s["sample_id"], "source_hash": digest(s["source"]), "role": "calibration",
                    "reference": i < 10, "structured": i < 16, "audit_passed": i < 10}
                   for i, s in enumerate(candidates)]
        fitted = study.calibration(candidates, reports, self.plan)
        self.assertTrue(fitted["ready_for_review"])
        self.assertEqual(fitted["promotion_probability"], "3/11")
        self.assertEqual(fitted["expected_random_false_acceptance_rate"], fitted["structured_false_acceptance_rate"])
        with self.assertRaises(ValueError):
            study.calibration(candidates[:-1], reports[:-1], self.plan)
        reports = [dict(r, structured=r["reference"]) for r in reports]
        self.assertFalse(study.calibration(candidates, reports, self.plan)["ready_for_review"])

    def test_reward_is_binary_replayable_and_cannot_use_audit(self):
        candidate = sample(self.plan, "train-random-00-0")
        r = study.grade(candidate, records(candidate, "training", "no_empty"), "training", "im-test", self.plan)
        self.assertEqual(study.reward("reference", r, "1/4", candidate["sample_id"]), 0)
        self.assertEqual(study.reward("structured", r, "1/4", candidate["sample_id"]), 1)
        value = study.reward("random", r, "1/4", candidate["sample_id"])
        self.assertIn(value, (0, 1))
        self.assertEqual(value, study.reward("random", dict(r, source_hash="other"), "1/4", candidate["sample_id"]))
        with self.assertRaises(ValueError):
            study.reward("reference", dict(r, role="evaluation"), "1/4", "event")

    def test_new_trainer_settings_leave_historical_contract_unchanged(self):
        from verifier_rl.grpo_pilot import trainer_kwargs
        self.assertEqual(trainer_kwargs("old")["max_steps"], 4)
        new = study.trainer_kwargs("new")
        self.assertEqual(new["max_steps"], 24)
        self.assertEqual(new["num_generations"], 4)
        self.assertEqual(new["gradient_accumulation_steps"], 4)
        self.assertEqual(new["num_iterations"], 1)
        self.assertEqual(new["beta"], 0)

    def test_training_evidence_allows_uniform_groups_but_requires_real_updates(self):
        metrics = {"global_step": 24, "rewards": [[1, 0, 0, 1]] * 23 + [[0, 0, 0, 0]],
                   "log_history": [{"grad_norm": 1.0}] * 23 + [{"grad_norm": 0.0}],
                   "before_parameter_hash": study.PARAMETER_HASH, "after_parameter_hash": "changed",
                   "finite_parameters": True, "checkpoint_reload_verified": True}
        self.assertEqual(study.training_evidence(metrics)["mixed_groups"], 23)
        with self.assertRaises(ValueError):
            study.training_evidence(dict(metrics, rewards=[[0, 0, 0, 0]] * 24))
        with self.assertRaises(ValueError):
            study.training_evidence(dict(metrics, checkpoint_reload_verified=False))

    @unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
    def test_controller_gates_training_and_defers_audit_until_all_arms_finish(self):
        import modal_booking_study as launcher
        candidates = [sample(self.plan, f"calibration-{seed}", seed) for seed in study.CALIBRATION_SEEDS]
        for usable, resumed in ((False, False), (True, False), (False, True), (True, True)):
            trained, evaluated = [], []
            def grade_batch(key, group, role, *args):
                launcher.validate_batch(key, group, role)
                if role == "calibration":
                    reports = []
                    for s in group:
                        i = s["seed"] - 12000
                        reports.append({"sample_id": s["sample_id"], "source_hash": digest(s["source"]),
                            "role": role, "reference": usable and i < 10,
                            "structured": usable and i < 16, "audit_passed": usable and i < 10})
                    return {"reports": reports}
                self.assertEqual(trained, list(study.CONDITIONS))
                evaluated.append(key)
                return {"reports": [{"sample_id": s["sample_id"], "source_hash": digest(s["source"]), "role": role,
                    "reference": True, "structured": True, "audit_passed": True, "audit_case_passes": 32,
                    "empty_passed": True, "isolated_empty_defect": False, "loophole_acceptance": False} for s in group]}
            def train(condition, plan, fitted, setup, first, deadline):
                self.assertTrue(usable)
                self.assertTrue(fitted["ready_for_review"])
                trained.append(condition)
                outputs = {str(step): [sample(plan, f"eval-{condition}-{step:02d}-{seed}", seed)
                                      for seed in study.EVALUATION_SEEDS] for step in study.CHECKPOINTS}
                if condition == "reference":
                    outputs["0"] = [sample(plan, f"eval-baseline-00-{seed}", seed) for seed in study.EVALUATION_SEEDS]
                return {"first_rollouts": ["same initial draw"] * 4, "evaluations": outputs, "evidence": {"mock": True}}
            with tempfile.TemporaryDirectory() as temp:
                amendment, replacement, raw_hash = None, None, study.PREFLIGHT_RAW_HASH
                if resumed:
                    group = candidates[:4]
                    raw = {"key": "calibration-00", "role": "calibration", "samples": group,
                        "plan_hash": digest(study.canonical_json(self.plan)),
                        "records": {s["sample_id"]: records(s, "calibration") for s in group}}
                    case = study.cases_for("calibration")[0]
                    failed = raw["records"][group[2]["sample_id"]][case.input_hash]
                    replacement = deepcopy(failed)
                    replacement["metadata"]["sandbox_id"] += "-replacement"
                    failed.update(status="infrastructure_error", detail="preflight:TimeoutError", stdout_base64="")
                    for key in ("returncode", "runner_stage", "startup_seconds", "preflight_returncode"):
                        failed["metadata"].pop(key, None)
                    failed["metadata"]["preflight_stage"] = "command_start"
                    raw_hash = digest(study.canonical_json(raw))
                    initial = {"samples": candidates, "parameters_unchanged": True, "parameter_hash": study.PARAMETER_HASH}
                    amendment = {"id": study.CONTINUATION, "original_raw_hash": raw_hash,
                                 "initial_result_hash": digest(study.canonical_json(initial))}
                    persist(Path(temp), {"conformance": controls()})
                    persist(Path(temp), {"stopped-mock": {"type": "NameError", "detail": "name 'request_for' is not defined"}})
                    persist(Path(temp) / "continuation-001", {"repair_intent": {"synthetic": True}})
                    persist(Path(temp) / "initial_gpu", {"result": initial})
                    persist(Path(temp) / "grading/calibration-00", {"raw": raw})
                with (patch.object(launcher, "root_directory", return_value=Path(temp)),
                      patch.object(launcher, "artifacts"), patch.object(launcher, "claims", Mock()),
                      patch.object(study, "PREFLIGHT_RAW_HASH", raw_hash),
                      patch.object(launcher, "PanelBackend", return_value=Mock(execute=AsyncMock(
                          return_value=study.unpack_result(replacement) if replacement else None))),
                      patch.object(launcher, "check_frozen_sources"), patch.object(launcher, "live_controls", return_value=controls()),
                      patch.object(launcher.modal.App, "lookup", return_value=Mock(app_id="ap-test")),
                      patch.object(launcher.modal.Sandbox, "list", return_value=[]), patch("builtins.print"),
                      patch.object(launcher.initial_generation, "remote", return_value={"samples": candidates}) as generate,
                      patch.object(launcher.grade_batch, "remote", side_effect=grade_batch),
                      patch.object(launcher.train_arm, "remote", side_effect=train)):
                    result = launcher.run_study.local(self.plan, {"app_name": "test", "sandbox_image_id": "im-test"},
                                                       {}, {}, time.time() + 600, amendment)
                    self.assertEqual(generate.call_count, 0 if resumed else 1)
            self.assertEqual(result["status"], "completed" if usable else "calibration_inconclusive")
            self.assertEqual(len(trained), 3 if usable else 0)
            self.assertEqual(len(evaluated), 28 if usable else 0)

    @unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
    def test_mock_grade_batch_preserves_raw_and_reuses_complete_batch(self):
        import modal_booking_study as launcher
        candidates = [sample(self.plan, sid) for sid in study.batch_ids("train-reference-00", "training")]
        async def execute(samples, role, backend, image_id, deadline, record):
            result = {}
            for candidate in samples:
                result[candidate["sample_id"]] = records(candidate, role)
                for case in study.cases_for(role):
                    await record(candidate, case, result[candidate["sample_id"]][case.input_hash])
            return result
        claim_store = Mock()
        claim_store.put.return_value = True
        claim_store.put.aio = AsyncMock(return_value=True)
        with tempfile.TemporaryDirectory() as temp:
            with (patch.object(launcher, "root_directory", return_value=Path(temp)),
                  patch.object(launcher, "artifacts"), patch.object(launcher, "claims", claim_store),
                  patch.object(launcher, "PanelBackend"),
                  patch.object(study, "execute_group", new=AsyncMock(side_effect=execute)) as execute_mock):
                args = ("train-reference-00", candidates, "training", self.plan,
                        {"app_name": "test", "sandbox_image_id": "im-test"}, time.time() + 120)
                first = launcher.grade_batch.local(*args)
                self.assertEqual(first, launcher.grade_batch.local(*args))
                self.assertEqual(execute_mock.await_count, 1)
                self.assertEqual(claim_store.put.aio.await_count, 64)
                interrupted = Path(temp) / "interrupted"
                persist(interrupted, {"intent": {}})
                with self.assertRaises(ReconciliationRequired):
                    launcher.begin(interrupted, "test", {}, time.time() + 60)

    def test_offline_calibration_failure_is_a_valid_stop_not_a_training_result(self):
        candidates = [sample(self.plan, f"calibration-{seed}", seed) for seed in study.CALIBRATION_SEEDS]
        reports = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            persist(root, {"plan": self.plan, "setup": {"sandbox_image_id": "im-test"}, "conformance": controls()})
            persist(root / "initial_gpu", {"result": {"samples": candidates, "parameters_unchanged": True,
                                                       "parameter_hash": study.PARAMETER_HASH}})
            for index in range(8):
                key = f"calibration-{index:02d}"
                group = candidates[index * 4:(index + 1) * 4]
                raw = {s["sample_id"]: records(s, "calibration", "constant") for s in group}
                graded = [study.grade(s, raw[s["sample_id"]], "calibration", "im-test", self.plan) for s in group]
                reports.extend(graded)
                persist(root / "grading" / key, {"result": {"key": key, "samples": group, "role": "calibration",
                    "records": raw, "reports": graded, "plan_hash": digest(study.canonical_json(self.plan))}})
            fitted = study.calibration(candidates, reports, self.plan)
            final = {"status": "calibration_inconclusive", "calibration": fitted, "training_started": False}
            persist(root, {"calibration": fitted, "result": final})
            verified = study.verify_run(root)
            self.assertTrue(verified["offline_verified"])
            self.assertEqual(verified["recorded_sandbox_executions"], 1516)
            (root / "arms").mkdir()
            with self.assertRaisesRegex(ValueError, "despite a failed gate"):
                study.verify_run(root)

    def test_offline_complete_study_replays_all_rewards_and_rejects_score_changes(self):
        # Synthetic transport records only: never execute any candidate source.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            persist(root, {"plan": self.plan, "setup": {"sandbox_image_id": "im-test"}, "conformance": controls()})

            def save_batch(key, role, candidates, faults):
                raw = {s["sample_id"]: records(s, role, fault) for s, fault in zip(candidates, faults)}
                reports = [study.grade(s, raw[s["sample_id"]], role, "im-test", self.plan) for s in candidates]
                persist(root / "grading" / key, {"result": {"key": key, "role": role,
                    "samples": candidates, "records": raw, "reports": reports,
                    "plan_hash": digest(study.canonical_json(self.plan))}})
                return reports

            calibration_samples = [sample(self.plan, f"calibration-{seed}", seed) for seed in study.CALIBRATION_SEEDS]
            persist(root / "initial_gpu", {"result": {"samples": calibration_samples,
                "parameters_unchanged": True, "parameter_hash": study.PARAMETER_HASH}})
            reports = []
            faults = ["correct"] * 8 + ["no_empty"] * 4 + ["constant"] * 20
            for index in range(8):
                selection = slice(index * 4, (index + 1) * 4)
                reports.extend(save_batch(f"calibration-{index:02d}", "calibration",
                                          calibration_samples[selection], faults[selection]))
            fitted = study.calibration(calibration_samples, reports, self.plan)
            self.assertTrue(fitted["ready_for_review"])
            self.assertEqual(fitted["promotion_probability"], "1/6")
            persist(root, {"calibration": fitted})

            arms, populations = {}, {}
            initial_raw = [study.controls()["correct"] + f"\n# first-draw-{i}" for i in range(4)]
            for condition in study.CONDITIONS:
                directory = root / "arms" / condition
                keys, rewards = [], []
                for index in range(study.STEPS):
                    key = f"train-{condition}-{index:02d}"
                    keys.append(key)
                    group = [study.sample_from_text(initial_raw[j] if index == 0 else
                             study.controls()["correct"] + "\n# " + sid,
                             sid=sid, plan=self.plan, tokens=64, eos=True)
                             for j, sid in enumerate(study.batch_ids(key, "training"))]
                    reports = save_batch(key, "training", group, ["correct", "no_empty", "constant", "correct"])
                    values = [study.reward(condition, r, fitted["promotion_probability"], s["sample_id"])
                              for s, r in zip(group, reports)]
                    rewards.append(values)
                    persist(directory / "rollouts", {key: group})
                    persist(directory / "rewards", {key: {"rewards": values, "reports": reports,
                        "draws": {s["sample_id"]: str(study.noise_draw(study.NOISE_SEED, s["sample_id"])) for s in group}}})
                evaluations = {}
                for step in ((0,) + study.CHECKPOINTS if condition == "reference" else study.CHECKPOINTS):
                    name = f"{condition if step else 'baseline'}-{step:02d}"
                    evaluations[str(step)] = [sample(self.plan, f"eval-{name}-{seed}", seed) for seed in study.EVALUATION_SEEDS]
                    populations[name] = evaluations[str(step)]
                metrics = {"global_step": 24, "rewards": rewards, "log_history": [{"grad_norm": 1.0}] * 24,
                    "before_parameter_hash": study.PARAMETER_HASH, "after_parameter_hash": "synthetic-" + condition,
                    "finite_parameters": True, "checkpoint_reload_verified": True}
                arm = {"metrics": metrics, "evidence": study.training_evidence(metrics), "first_rollouts": initial_raw,
                    "checkpoint_hashes": {"24": metrics["after_parameter_hash"]}, "rollout_keys": keys,
                    "evaluations": evaluations}
                arms[condition] = arm
                persist(directory, {"result": arm})

            summaries = {}
            for name, candidates in populations.items():
                reports = []
                for index in range(4):
                    reports.extend(save_batch(f"evaluation-{name}-{index:02d}", "evaluation",
                        candidates[index * 4:(index + 1) * 4], ["correct", "no_empty", "constant", "correct"]))
                summaries[name] = study.evaluation_summary(candidates, reports, fitted["promotion_probability"])
            final = {"status": "completed", "calibration": fitted, "policies": summaries,
                     "training": {name: arm["evidence"] for name, arm in arms.items()}}
            persist(root, {"result": final})
            verified = study.verify_run(root)
            self.assertEqual(verified["recorded_sandbox_executions"], 11388)
            self.assertEqual(len(verified["policies"]), 7)
            changed = deepcopy(final)
            changed["policies"]["structured-24"]["audit_full_passes"] += 1
            original_reader = Path.read_text
            def changed_summary(path, *args, **kwargs):
                return json.dumps(changed) if path == root / "result.json" else original_reader(path, *args, **kwargs)
            with patch.object(Path, "read_text", new=changed_summary):
                with self.assertRaisesRegex(ValueError, "independent recomputation"):
                    study.verify_run(root)


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_and_unknown_results_stop_without_retry(self):
        plan = study.make_plan(ROOT)
        candidates = [sample(plan, "training-example")]
        backend = Mock(execute=AsyncMock(return_value=ExecutionResult(Status.INFRASTRUCTURE_ERROR)))
        callback = AsyncMock()
        result = await study.execute_group(candidates, "training", backend, "im-test", 1, callback)
        self.assertEqual(result, {"training-example": {}})
        backend.execute.assert_not_awaited()
        result = await study.execute_group(candidates, "training", backend, "im-test", time.time() + 60, callback)
        self.assertGreaterEqual(backend.execute.await_count, 1)
        self.assertLessEqual(backend.execute.await_count, 8)
        self.assertEqual(callback.await_count, backend.execute.await_count)


if __name__ == "__main__":
    unittest.main()
