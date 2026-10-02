import asyncio
from copy import deepcopy
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_panel_screen import execution
from tests.test_booking_two_arm import startup_failure
from verifier_rl import booking_reward_pilot as pilot, booking_verifier_v2 as verifier
from verifier_rl.evaluation_journal import ReconciliationRequired
from verifier_rl.panel_execution import pack_result, unpack_result
from verifier_rl.suites import digest
from verifier_rl.suites import canonical_json

ROOT = Path(__file__).resolve().parents[1]


def sample(plan, sid, seed=None):
    return pilot.old.sample_from_text(pilot.old.controls()["correct"], sid=sid, plan=plan,
                                     seed=seed, tokens=64, eos=True)


def report(sample, role, passed=96, audit_passed=192):
    scores = {"training": verifier.score_outcomes("training", {c.input_hash: i < passed
              for i, c in enumerate(verifier.cases_for("training"))})}
    if role == "evaluation":
        scores["audit"] = verifier.score_outcomes("audit", {c.input_hash: i < audit_passed
                            for i, c in enumerate(verifier.cases_for("audit"))})
    return {"sample_id": sample["sample_id"], "role": role, "reports": scores, "sandbox_ids": []}


class RewardPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = pilot.make_plan(ROOT)

    def test_fixed_matched_plan_and_resource_bounds(self):
        pilot.validate_plan(self.plan)
        self.assertEqual(self.plan["max_sandbox_executions"], 32608)
        self.assertEqual(self.plan["trainer"]["max_steps"], 24)
        self.assertEqual(len(pilot.cases_for("evaluation")), 287)
        self.assertEqual(len(pilot.cases_for("training")), 96)
        for change in ({"steps": 48}, {"log_strength": 1}, {"verifier_condition": "structured"},
                       {"max_gpu_calls": 3}, {"max_startup_retries": 20}, {"automatic_expansion": True}):
            with self.assertRaises(ValueError):
                pilot.validate_plan(dict(self.plan, **change))

    def test_schedule_rejects_extra_arms_steps_and_old_seeds(self):
        for key, role in (("train-linear-24", "training"), ("train-reference-00", "training"),
                          ("eval-baseline-04", "evaluation"), ("preflight", "evaluation")):
            with self.assertRaises(ValueError):
                pilot.batch_ids(key, role)
        key = "eval-baseline-00"
        group = [sample(self.plan, sid, int(sid.rsplit("-", 1)[1])) for sid in pilot.batch_ids(key, "evaluation")]
        pilot.validate_batch(key, "evaluation", group, self.plan)
        group[0]["seed"] = 14000
        with self.assertRaises(ValueError):
            pilot.validate_batch(key, "evaluation", group, self.plan)

    def test_linear_and_log_use_same_cases_and_only_training_evidence(self):
        s = sample(self.plan, "test")
        r = report(s, "training", 48)
        self.assertEqual(pilot.reward(r, "linear"), .5)
        self.assertEqual(pilot.reward(r, "logarithmic"), verifier.shape_reward(.5, "logarithmic"))
        for role in ("preflight", "evaluation"):
            with self.assertRaises(ValueError):
                pilot.reward(report(s, role), "linear")
        with self.assertRaises(ValueError):
            pilot.reward(r, "structured")

    def test_saved_control_identity_is_pinned(self):
        s = sample(self.plan, pilot.SAVED_ID)
        with self.assertRaises(ValueError):
            pilot.preflight_samples(s, self.plan)
        with patch.object(pilot, "SAVED_SOURCE_HASH", digest(s["source"])):
            group = pilot.preflight_samples(s, self.plan)
            graded = [report(c, "preflight", p) for c, p in zip(group, (96, 64, 1, 48))]
            self.assertTrue(pilot.validate_preflight(group, graded)["passed"])
            graded[-1] = report(group[-1], "preflight", 96)
            with self.assertRaises(ValueError):
                pilot.validate_preflight(group, graded)

    def test_grade_replays_both_suites_without_duplicate_empty_execution(self):
        s = sample(self.plan, "test")
        records = {c.input_hash: pack_result(execution(s["source"], c, c.input_hash))
                   for c in pilot.cases_for("evaluation")}
        r = pilot.grade(s, records, "evaluation", "im-test", self.plan)
        self.assertEqual(len(r["sandbox_ids"]), 287)
        self.assertEqual(r["reports"]["training"]["passed"], 96)
        self.assertEqual(r["reports"]["audit"]["passed"], 192)
        with self.assertRaises(ValueError):
            pilot.grade(s, records, "training", "im-test", self.plan)
        records.pop(next(iter(records)))
        with self.assertRaises(ValueError):
            pilot.grade(s, records, "evaluation", "im-test", self.plan)

    def test_float_optimizer_evidence_and_zero_variation(self):
        metrics = {"global_step": 24, "rewards": [[0, .25, .5, 1]] * 24,
            "log_history": [{"grad_norm": 1.0}] * 24, "before_parameter_hash": pilot.old.PARAMETER_HASH,
            "after_parameter_hash": "a" * 64, "finite_parameters": True, "checkpoint_reload_verified": True}
        self.assertEqual(pilot.training_evidence(metrics)["mixed_groups"], 24)
        self.assertTrue(pilot.training_evidence(metrics)["parameters_changed"])
        for change in ({"checkpoint_reload_verified": False}, {"global_step": 23},
                       {"rewards": [[float("nan")]*4]*24}, {"rewards": [[0]*4]*24}):
            with self.assertRaises(ValueError):
                pilot.training_evidence(dict(metrics, **change))
        uniform = dict(metrics, rewards=[[.5]*4]*24, log_history=[{"grad_norm": 0.0}]*24,
                       after_parameter_hash=pilot.old.PARAMETER_HASH)
        self.assertEqual(pilot.training_evidence(uniform)["mixed_groups"], 0)

    def test_policy_summary_counts_programs_not_tests_as_independent_samples(self):
        samples = [sample(self.plan, f"eval-baseline-{seed}", seed) for seed in pilot.SEEDS]
        reports = [report(s, "evaluation", 96, 192 if i < 4 else 100) for i, s in enumerate(samples)]
        summary = pilot.policy_summary(samples, reports)
        self.assertEqual(summary["programs"], 16)
        self.assertEqual(summary["audit_full_passes"], 4)
        self.assertEqual(summary["audit_cases"], 3072)
        self.assertEqual(summary["audit_case_passes"], 1968)

    def test_budget_keeps_old_holds_and_limits_gpu_calls(self):
        rates = {k: "1" for k in ("cpu_hour_cost", "mem_gib_hour_cost", "cpu_hour_cost_sandbox",
                                  "mem_gib_hour_cost_sandbox", "gpu_hour_cost_l40s")}
        value = pilot.budget_quote(self.plan, rates, {"metered_cost": "1"}, "143.32")
        self.assertEqual(value["prior_reservation_held_usd"], "143.32")
        self.assertEqual(value["gpu_max_usd"], "140")
        self.assertFalse(value["is_invoice"])

    def test_retry_validation_preserves_first_attempt_and_disallows_reuse(self):
        s = sample(self.plan, "test")
        c = pilot.cases_for("training")[0]
        second = execution(s["source"], c, "second")
        first = replace(startup_failure(second), metadata=dict(startup_failure(second).metadata, sandbox_id="sb-first"))
        receipt = {"slot": 0, "sample_id": s["sample_id"], "input_hash": c.input_hash,
                   "first": pack_result(first), "replacement": pack_result(second)}
        raw = {"samples": [s], "role": "training", "records": {"test": {c.input_hash: pack_result(second)}},
               "retries": [receipt]}
        self.assertEqual(pilot.verify_retries(raw, "im-test"), (["sb-first"], [0]))
        raw["retries"].append(receipt)
        with self.assertRaises(ValueError):
            pilot.verify_retries(raw, "im-test")

    def test_full_offline_replay_rejects_changed_consumed_reward(self):
        # Shorten only this synthetic fixture's schedule; no model/code executes.
        with patch.object(pilot, "STEPS", 2):
            plan = pilot.make_plan(ROOT)
            saved = sample(plan, pilot.SAVED_ID)
            with patch.object(pilot, "SAVED_SOURCE_HASH", digest(saved["source"])):
                plan = pilot.make_plan(ROOT)
                docs = {"plan.json": plan, "setup.json": {"sandbox_image_id": "im-test"}}
                def batch(key, role, samples, passes=None):
                    records = {}
                    for j, s in enumerate(samples):
                        rows = {}
                        for i, c in enumerate(pilot.cases_for(role)):
                            r = execution(s["source"], c, key + "-" + str(j) + "-" + c.input_hash)
                            if passes is not None and i >= passes[j]:
                                output = canonical_json(c.expected + 1).encode()
                                from hashlib import sha256
                                r = replace(r, stdout=output, metadata=dict(r.metadata, stdout_bytes=len(output),
                                                    stdout_sha256=sha256(output).hexdigest()))
                            rows[c.input_hash] = pack_result(r)
                        records[s["sample_id"]] = rows
                    reports = [pilot.grade(s, records[s["sample_id"]], role, "im-test", plan) for s in samples]
                    docs[f"grading/{key}/result.json"] = {"key": key, "role": role, "samples": samples,
                        "records": records, "reports": reports, "retries": [], "plan_hash": digest(canonical_json(plan))}
                    return reports
                controls = pilot.preflight_samples(saved, plan)
                preflight = batch("preflight", "preflight", controls, [96, 64, 1, 48])
                docs["preflight_samples.json"] = controls
                docs["preflight.json"] = pilot.validate_preflight(controls, preflight)
                populations = {}
                for arm in pilot.ARMS:
                    rewards = []
                    for step in range(2):
                        key = f"train-{arm}-{step:02d}"
                        group = [sample(plan, sid) for sid in pilot.batch_ids(key, "training")]
                        docs[f"arms/{arm}/rollouts/{key}.json"] = group
                        reports = batch(key, "training", group, [0, 24, 72, 96])
                        values = [pilot.reward(r, arm) for r in reports]
                        rewards.append(values)
                        docs[f"arms/{arm}/rewards/{key}.json"] = {"rewards": values, "reports": reports}
                    evaluation = {}
                    for step in ((0, 24) if arm == "linear" else (24,)):
                        policy = arm if step else "baseline"
                        group = [sample(plan, f"eval-{policy}-{seed}", seed) for seed in pilot.SEEDS]
                        populations[policy] = evaluation[str(step)] = group
                        for s in group:
                            docs[f"arms/{arm}/evaluation-{step:02d}/samples/{s['sample_id']}.json"] = s
                    metrics = {"global_step": 2, "rewards": rewards, "log_history": [{"grad_norm": 1.0}]*2,
                        "before_parameter_hash": pilot.old.PARAMETER_HASH, "after_parameter_hash": "a"*64,
                        "finite_parameters": True, "checkpoint_reload_verified": True, "generated_rollout_tokens": 2*4*64}
                    docs[f"arms/{arm}/result.json"] = {"arm": arm, "metrics": metrics,
                        "evidence": pilot.training_evidence(metrics), "first_rollouts": [s["raw"] for s in group[:4]],
                        "checkpoint_hashes": {"0": pilot.old.PARAMETER_HASH, "24": "a"*64}, "evaluations": evaluation}
                for policy, samples in populations.items():
                    for i in range(4):
                        batch(f"eval-{policy}-{i:02d}", "evaluation", samples[4*i:4*i+4])
                def read(path):
                    if path.endswith("/selected.json"):
                        _, key, _, sid, input_hash, _ = path.split("/")
                        return docs[f"grading/{key}/result.json"]["records"][sid][input_hash]
                    return docs[path]
                result = pilot.verify_with_reader(read)
                self.assertTrue(result["offline_verified"])
                self.assertEqual(result["policies"]["baseline"]["audit_full_passes"], 16)
                docs["arms/linear/rewards/train-linear-00.json"]["rewards"][0] = .123
                with self.assertRaisesRegex(ValueError, "consumed reward"):
                    pilot.verify_with_reader(read)


@unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
class ControllerTests(unittest.TestCase):
    def test_live_gate_blocks_training_and_audit_follows_both_arms(self):
        import modal_booking_reward_pilot as launcher
        plan = pilot.make_plan(ROOT)
        setup = {"sandbox_image_id": "im-test", "app_name": "test"}
        for gate_passes in (False, True):
            with self.subTest(gate_passes=gate_passes):
                trained, evaluated = [], []
                def train(arm, plan, setup, first, deadline):
                    self.assertEqual(first, None if not trained else ["matched"]*4)
                    trained.append(arm)
                    policies = {"24": [sample(plan, f"eval-{arm}-{seed}", seed) for seed in pilot.SEEDS]}
                    if arm == "linear":
                        policies["0"] = [sample(plan, f"eval-baseline-{seed}", seed) for seed in pilot.SEEDS]
                    return {"evaluations": policies, "first_rollouts": ["matched"]*4}
                def grade(key, group, role, *args):
                    if role == "preflight":
                        self.assertEqual(trained, [])
                        return {"reports": []}
                    self.assertEqual(trained, list(pilot.ARMS))
                    pilot.validate_batch(key, role, group, plan)
                    evaluated.append(key)
                    return {"reports": [report(s, role) for s in group]}
                with (patch.object(launcher, "artifacts"), patch.object(launcher, "claims", Mock()),
                      patch.object(launcher, "persist"), patch.object(launcher, "check_snapshot"),
                      patch.object(launcher.modal.App, "lookup", return_value=Mock(app_id="ap-test")),
                      patch.object(launcher.modal.Sandbox, "list", return_value=[]), patch("builtins.print"),
                      patch.object(pilot, "validate_preflight", return_value={"passed": True},
                                   side_effect=None if gate_passes else ValueError("gate failed")),
                      patch.object(pilot, "verify_run", return_value={"offline_verified": True}),
                      patch.object(launcher.train_arm, "remote", side_effect=train),
                      patch.object(launcher.grade_batch, "remote", side_effect=grade)):
                    if gate_passes:
                        self.assertTrue(launcher.run_pilot.local(plan, setup, [], {}, {}, time.time()+60)["offline_verified"])
                    else:
                        with self.assertRaisesRegex(ValueError, "gate failed"):
                            launcher.run_pilot.local(plan, setup, [], {}, {}, time.time()+60)
                self.assertEqual(trained, list(pilot.ARMS) if gate_passes else [])
                self.assertEqual(len(evaluated), 12 if gate_passes else 0)


@unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import modal_booking_reward_pilot as launcher
        self.launcher = launcher
        self.plan = pilot.make_plan(ROOT)
        self.sample = sample(self.plan, "test")
        self.case = pilot.cases_for("training")[0]
        self.setup = {"sandbox_image_id": "im-test", "app_name": "test"}

    async def test_pre_candidate_timeout_retries_once_and_records_both_attempts(self):
        second = execution(self.sample["source"], self.case, "second")
        first = replace(startup_failure(second), metadata=dict(startup_failure(second).metadata, sandbox_id="sb-first"))
        backend = Mock(execute=AsyncMock(side_effect=[first, second]))
        store = Mock()
        store.put.aio = AsyncMock(return_value=True)
        with tempfile.TemporaryDirectory() as temp, patch.object(pilot, "cases_for", return_value=(self.case,)):
            records, retries = await self.launcher.execute_group([self.sample], "training", self.setup,
                Path(temp), time.time()+60, backend, store)
            self.assertEqual(backend.execute.await_count, 2)
            self.assertEqual(len(retries), 1)
            self.assertEqual(records["test"][self.case.input_hash], pack_result(second))
            directory = Path(temp)/"inputs"/"test"/self.case.input_hash
            self.assertTrue((directory/"attempt-1.json").exists())
            self.assertTrue((directory/"attempt-2.json").exists())
            self.assertTrue((directory/"selected.json").exists())

    async def test_candidate_signal_is_not_retried_or_scored_zero(self):
        from verifier_rl.grading import Status
        result = execution(self.sample["source"], self.case, "killed")
        result = replace(result, status=Status.CANDIDATE_ERROR, metadata=dict(result.metadata, returncode=137))
        backend = Mock(execute=AsyncMock(return_value=result))
        store = Mock()
        store.put.aio = AsyncMock(return_value=True)
        with tempfile.TemporaryDirectory() as temp, patch.object(pilot, "cases_for", return_value=(self.case,)):
            with self.assertRaisesRegex(ReconciliationRequired, "unattributed termination"):
                await self.launcher.execute_group([self.sample], "training", self.setup,
                    Path(temp), time.time()+60, backend, store)
            self.assertEqual(backend.execute.await_count, 1)
            self.assertTrue((Path(temp)/"inputs"/"test"/self.case.input_hash/"selected.json").exists())

    async def test_deadline_or_duplicate_intent_never_executes(self):
        for expired in (False, True):
            backend = Mock(execute=AsyncMock())
            store = Mock()
            store.put.aio = AsyncMock(return_value=False)
            with tempfile.TemporaryDirectory() as temp, patch.object(pilot, "cases_for", return_value=(self.case,)):
                with self.assertRaises(ReconciliationRequired):
                    await self.launcher.execute_group([self.sample], "training", self.setup,
                        Path(temp), time.time()-1 if expired else time.time()+60, backend, store)
                backend.execute.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
