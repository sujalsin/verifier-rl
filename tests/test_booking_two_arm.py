import asyncio
from copy import deepcopy
from dataclasses import replace
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_booking_study import ROOT, controls, records, sample
from verifier_rl import booking_study as old, booking_two_arm as two
from verifier_rl.evaluation_journal import persist
from verifier_rl.grading import Status
from verifier_rl.panel_execution import pack_result, request_for, unpack_result
from verifier_rl.suites import canonical_json, digest


def calibration(parent):
    candidates = [sample(parent, f"calibration-{seed}", seed) for seed in old.CALIBRATION_SEEDS]
    reports = [{"sample_id": s["sample_id"], "source_hash": digest(s["source"]), "role": "calibration",
                "reference": i < 5, "structured": i < 5, "audit_passed": i < 5} for i, s in enumerate(candidates)]
    return old.calibration(candidates, reports, parent)


def startup_failure(result):
    metadata = dict(result.metadata)
    for key in ("returncode", "runner_stage", "startup_seconds", "preflight_returncode"):
        metadata.pop(key, None)
    metadata["preflight_stage"] = "command_start"
    return replace(result, status=Status.INFRASTRUCTURE_ERROR, stdout=b"", detail="preflight:TimeoutError", metadata=metadata)


class TwoArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = old.make_plan(ROOT)
        cls.calibration = calibration(cls.parent)
        cls.plan = two.make_plan(cls.parent, cls.calibration)

    def test_zero_contrast_allows_two_arms_without_rewriting_calibration(self):
        before = deepcopy(self.calibration)
        self.assertTrue(two.training_gate(self.calibration)["training_ready"])
        self.assertFalse(self.calibration["ready_for_review"])
        self.assertEqual(self.calibration["promotion_probability"], "0")
        for condition in two.CONDITIONS:
            two.require_training(self.plan, self.calibration, condition)
        with self.assertRaisesRegex(ValueError, "random-error"):
            two.require_training(self.plan, self.calibration, "random")
        self.assertEqual(before, self.calibration)
        old.validate_plan(self.parent)

    def test_mutated_calibration_and_expanded_plan_are_rejected(self):
        for key, value in (("promotion_probability", "1/10"), ("reference_passes", 0), ("audit_rejected", 28),
                           ("ready_for_review", True), ("reasons", [])):
            with self.assertRaises(ValueError):
                two.training_gate(dict(self.calibration, **{key: value}))
        for key, value in (("conditions", ["reference", "structured", "random"]), ("max_steps", 25),
                           ("run_id", old.RUN_ID), ("max_startup_retries", 9), ("max_sandbox_executions", 999999)):
            with self.assertRaises(ValueError):
                two.validate_plan(dict(self.plan, **{key: value}))
        self.assertEqual(self.plan["max_sandbox_executions"], 12 + 2 * 24 * 4 * 16 + 5 * 16 * 47 + 8)
        with self.assertRaises(ValueError):
            two.validate_batch("calibration-00", "calibration")
        with self.assertRaises(ValueError):
            two.validate_batch("train-random-00", "training")

    def test_score_is_unchanged_and_audit_cannot_be_a_reward(self):
        candidate = sample(self.plan, "train-structured-00-0")
        raw = records(candidate, "training", "no_empty")
        before = old.grade(candidate, raw, "training", "im-test", self.parent)
        after = old.grade(candidate, raw, "training", "im-test", self.plan)
        self.assertEqual(before, after)
        self.assertEqual(old.reward("reference", after, "0", candidate["sample_id"]), 0)
        self.assertEqual(old.reward("structured", after, "0", candidate["sample_id"]), 1)
        with self.assertRaises(ValueError):
            old.reward("structured", dict(after, role="evaluation"), "0", candidate["sample_id"])

    def test_retry_receipt_binds_to_one_selected_execution(self):
        candidate = sample(self.plan, "train-reference-00-0")
        raw = records(candidate, "training")
        key = next(iter(raw))
        first = startup_failure(unpack_result(raw[key]))
        replacement = deepcopy(raw[key])
        replacement["metadata"]["sandbox_id"] += "-replacement"
        raw[key] = replacement
        receipt = {"slot": 0, "first": pack_result(first), "replacement": replacement}
        batch = {"samples": [candidate], "role": "training", "records": {candidate["sample_id"]: raw},
                 "startup_retries": [receipt]}
        ids, slots = two.verify_retries(batch, "im-test")
        self.assertEqual(ids, [first.metadata["sandbox_id"]])
        self.assertEqual(slots, [0])
        with self.assertRaises(ValueError):
            two.verify_retries(dict(batch, startup_retries=[receipt, receipt]), "im-test")
        receipt["first"]["metadata"]["cleanup"] = "unconfirmed"
        with self.assertRaises(ValueError):
            two.verify_retries(batch, "im-test")

    @unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
    def test_controller_trains_exactly_two_arms_before_any_audit(self):
        import modal_booking_study as launcher
        parent = {"calibration": self.calibration, "offline_verified": True}
        trained, evaluated = [], []
        def train(condition, plan, fitted, setup, first, deadline):
            two.require_training(plan, fitted, condition)
            self.assertFalse(fitted["ready_for_review"])
            trained.append(condition)
            outputs = {str(step): [sample(plan, f"eval-{condition}-{step:02d}-{seed}", seed)
                                  for seed in old.EVALUATION_SEEDS] for step in old.CHECKPOINTS}
            if condition == "reference":
                outputs["0"] = [sample(plan, f"eval-baseline-00-{seed}", seed) for seed in old.EVALUATION_SEEDS]
            else:
                self.assertEqual(first, ["same"] * 4)
            return {"evaluations": outputs, "first_rollouts": ["same"] * 4, "evidence": {"mock": True}}
        def grade(key, group, role, *args):
            self.assertEqual(trained, list(two.CONDITIONS))
            self.assertEqual(role, "evaluation")
            two.validate_batch(key, role)
            evaluated.append(key)
            return {"reports": [old.grade(s, records(s, role), role, "im-test", self.plan) for s in group]}
        original_reader = Path.read_text
        setup = {"app_name": "test", "sandbox_image_id": "im-test"}
        def read(path, *args, **kwargs):
            if path == Path("/artifacts") / old.RUN_ID / "plan.json":
                return json.dumps(self.parent)
            if path == Path("/artifacts") / old.RUN_ID / "setup.json":
                return json.dumps(setup)
            return original_reader(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            with (patch.object(launcher, "root_directory", return_value=Path(temp)),
                  patch.object(Path, "read_text", new=read), patch.object(old, "verify_run", return_value=parent),
                  patch.object(launcher, "artifacts"), patch.object(launcher, "claims", Mock()),
                  patch.object(launcher, "check_frozen_sources"), patch.object(launcher, "live_controls", return_value=controls()),
                  patch.object(launcher.modal.App, "lookup", return_value=Mock(app_id="ap-test")),
                  patch.object(launcher.modal.Sandbox, "list", return_value=[]), patch("builtins.print"),
                  patch.object(launcher.initial_generation, "remote") as generate,
                  patch.object(launcher.train_arm, "remote", side_effect=train),
                  patch.object(launcher.grade_batch, "remote", side_effect=grade)):
                result = launcher.run_two_arm.local(self.plan, setup, {}, {}, time.time() + 120)
                generate.assert_not_called()
        self.assertEqual(trained, ["reference", "structured"])
        self.assertEqual(len(evaluated), 20)
        self.assertEqual(len(result["policies"]), 5)
        self.assertFalse(result["random_arm_enabled"])

    @unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
    def test_launcher_is_opt_in_and_claims_are_namespaced(self):
        import modal_booking_study as launcher
        with self.assertRaisesRegex(ValueError, "allow-cloud"):
            launcher.launch_two_arm(False)
        self.assertNotEqual(launcher.root_directory(self.plan), launcher.root_directory(self.parent))
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(launcher, "artifacts"), patch.object(launcher, "claims") as claims:
                claims.put.return_value = True
                launcher.begin(Path(temp), "train/reference", {}, time.time() + 60, run_id=two.RUN_ID)
                self.assertEqual(claims.put.call_args.args[0], two.RUN_ID + "/work/train/reference")

    @unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
    def test_new_grading_path_durably_records_one_startup_retry(self):
        import modal_booking_study as launcher
        group = [sample(self.plan, sid) for sid in two.validate_batch("train-reference-00", "training")]
        cases = {c.arguments_json: c for c in old.cases_for("training")}
        raw = {s["source"]: records(s, "training") for s in group}
        first_source, first_input = group[0]["source"], next(iter(cases))
        injected = False
        async def execute(request):
            nonlocal injected
            result = unpack_result(raw[request.source][cases[request.input_json].input_hash])
            if request.source == first_source and request.input_json == first_input:
                if not injected:
                    injected = True
                    return startup_failure(result)
                return replace(result, metadata=dict(result.metadata, sandbox_id=result.metadata["sandbox_id"] + "-retry"))
            return result
        stored = {}
        def put(key, value, skip_if_exists=False):
            if key in stored:
                return False
            stored[key] = value
            return True
        claims = Mock()
        claims.put.side_effect = put
        claims.put.aio = AsyncMock(side_effect=put)
        with tempfile.TemporaryDirectory() as temp:
            with (patch.object(launcher, "root_directory", return_value=Path(temp)), patch.object(launcher, "artifacts"),
                  patch.object(launcher, "claims", claims), patch.object(launcher, "PanelBackend",
                    return_value=Mock(execute=AsyncMock(side_effect=execute)))):
                result = launcher.grade_batch.local("train-reference-00", group, "training", self.plan,
                    {"app_name": "test", "sandbox_image_id": "im-test"}, time.time() + 120)
        self.assertEqual(len(result["startup_retries"]), 1)
        self.assertEqual(two.verify_retries(result, "im-test")[1], [0])
        self.assertIn(two.RUN_ID + "/startup-retry/slot/0", stored)
        self.assertIn(two.RUN_ID + "/startup-retry/result/0", stored)
        self.assertTrue(all(r["reference"] for r in result["reports"]))
        self.assertTrue(all(key.startswith(two.RUN_ID + "/") for key in stored))

    def test_complete_offline_replay_checks_all_five_policies_and_consumed_rewards(self):
        # Synthetic transport/optimizer records, not locally executed candidate code.
        parent = {"calibration": self.calibration, "offline_verified": True}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            persist(root, {"plan": self.plan, "setup": {"sandbox_image_id": "im-test"}, "conformance": controls(),
                "parent_verification": parent, "calibration": self.calibration})
            def save_batch(key, role, group):
                faults = ["correct", "no_empty", "constant", "correct"]
                raw = {s["sample_id"]: records(s, role, fault) for s, fault in zip(group, faults)}
                reports = [old.grade(s, raw[s["sample_id"]], role, "im-test", self.plan) for s in group]
                persist(root / "grading" / key, {"result": {"key": key, "role": role, "samples": group,
                    "records": raw, "reports": reports, "startup_retries": [], "plan_hash": digest(canonical_json(self.plan))}})
                return reports
            arms, populations = {}, {}
            initial_raw = [old.controls()["correct"] + f"\n# initial-{i}" for i in range(4)]
            for condition in two.CONDITIONS:
                directory = root / "arms" / condition
                keys, rewards = [], []
                for index in range(old.STEPS):
                    key = f"train-{condition}-{index:02d}"
                    keys.append(key)
                    group = [old.sample_from_text(initial_raw[j], sid=sid, plan=self.plan, tokens=64, eos=True)
                             for j, sid in enumerate(two.validate_batch(key, "training"))]
                    reports = save_batch(key, "training", group)
                    values = [old.reward(condition, r, "0", s["sample_id"]) for s, r in zip(group, reports)]
                    rewards.append(values)
                    persist(directory / "rollouts", {key: group})
                    persist(directory / "rewards", {key: {"rewards": values, "reports": reports,
                        "draws": {s["sample_id"]: str(old.noise_draw(old.NOISE_SEED, s["sample_id"])) for s in group}}})
                evaluations = {}
                for step in ((0,) + old.CHECKPOINTS if condition == "reference" else old.CHECKPOINTS):
                    name = f"{condition if step else 'baseline'}-{step:02d}"
                    evaluations[str(step)] = [sample(self.plan, f"eval-{name}-{seed}", seed) for seed in old.EVALUATION_SEEDS]
                    populations[name] = evaluations[str(step)]
                metrics = {"global_step": 24, "rewards": rewards, "log_history": [{"grad_norm": 1.0}] * 24,
                    "before_parameter_hash": old.PARAMETER_HASH, "after_parameter_hash": "synthetic-" + condition,
                    "finite_parameters": True, "checkpoint_reload_verified": True, "generated_rollout_tokens": 24 * 4 * 64}
                arm = {"metrics": metrics, "evidence": old.training_evidence(metrics), "first_rollouts": initial_raw,
                    "checkpoint_hashes": {"0": old.PARAMETER_HASH, "24": metrics["after_parameter_hash"]},
                    "rollout_keys": keys, "evaluations": evaluations}
                arms[condition] = arm
                persist(directory, {"result": arm})
            summaries = {}
            for name, group in populations.items():
                reports = []
                for index in range(4):
                    reports.extend(save_batch(f"evaluation-{name}-{index:02d}", "evaluation", group[index * 4:(index + 1) * 4]))
                summaries[name] = two.summary(group, reports)
            final = {"status": "completed", "version": two.VERSION, "calibration": self.calibration,
                     "policies": summaries, "training": {name: arm["evidence"] for name, arm in arms.items()}}
            persist(root, {"result": final})
            with patch.object(old, "verify_run", return_value=parent):
                verified = two.verify_run(root, "synthetic-parent")
                self.assertEqual(verified["recorded_sandbox_starts"], 6844)
                self.assertEqual(len(verified["policies"]), 5)
                path = root / "arms/reference/rewards/train-reference-00.json"
                changed = json.loads(path.read_text())
                changed["rewards"][0] = 0
                original_reader = Path.read_text
                def tampered(p, *args, **kwargs):
                    return json.dumps(changed) if p == path else original_reader(p, *args, **kwargs)
                with patch.object(Path, "read_text", new=tampered):
                    with self.assertRaisesRegex(ValueError, "consumed reward"):
                        two.verify_run(root, "synthetic-parent")


class StartupRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_one_confirmed_pre_candidate_retry_is_allowed(self):
        parent = old.make_plan(ROOT)
        candidate = sample(parent, "train-reference-00-0")
        case = old.cases_for("training")[0]
        good = unpack_result(records(candidate, "training")[case.input_hash])
        first = startup_failure(good)
        replacement = replace(good, metadata=dict(good.metadata, sandbox_id="sb-replacement"))
        backend = Mock(execute=AsyncMock(side_effect=[first, replacement]))
        reserve, journal = AsyncMock(return_value=0), AsyncMock()
        wrapped = two.StartupRetryBackend(backend, "im-test", reserve, journal)
        self.assertEqual(await wrapped.execute(request_for(candidate["source"], case)), replacement)
        self.assertEqual(backend.execute.await_count, 2)
        self.assertEqual(journal.await_count, 1)
        self.assertEqual(len(wrapped.retries), 1)
        backend.execute = AsyncMock(side_effect=[first, first])
        wrapped = two.StartupRetryBackend(backend, "im-test", reserve, journal)
        self.assertEqual(await wrapped.execute(request_for(candidate["source"], case)), first)
        self.assertEqual(backend.execute.await_count, 2)  # No third attempt.

    async def test_no_retry_for_candidate_errors_unknown_signals_or_exhausted_budget(self):
        parent = old.make_plan(ROOT)
        candidate = sample(parent, "train-reference-00-0")
        case = old.cases_for("training")[0]
        good = unpack_result(records(candidate, "training")[case.input_hash])
        first = startup_failure(good)
        for bad in (replace(first, metadata=dict(first.metadata, cleanup="unconfirmed")),
                    replace(first, detail="candidate:TimeoutError"),
                    replace(good, status=Status.CANDIDATE_ERROR, metadata=dict(good.metadata, returncode=137)),
                    replace(first, metadata=dict(first.metadata, source_hash="wrong"))):
            backend = Mock(execute=AsyncMock(return_value=bad))
            reserve = AsyncMock()
            wrapped = two.StartupRetryBackend(backend, "im-test", reserve, AsyncMock())
            self.assertEqual(await wrapped.execute(request_for(candidate["source"], case)), bad)
            reserve.assert_not_awaited()
            self.assertEqual(backend.execute.await_count, 1)
        backend = Mock(execute=AsyncMock(return_value=first))
        wrapped = two.StartupRetryBackend(backend, "im-test", AsyncMock(return_value=None), AsyncMock())
        self.assertEqual(await wrapped.execute(request_for(candidate["source"], case)), first)
        self.assertEqual(backend.execute.await_count, 1)


if __name__ == "__main__":
    unittest.main()
