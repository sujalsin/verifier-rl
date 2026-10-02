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
from tests.test_booking_two_arm import calibration, startup_failure
from verifier_rl import booking_study as old, booking_two_arm as two, booking_recovery as recovery
from verifier_rl.evaluation_journal import persist, ReconciliationRequired
from verifier_rl.grading import Status
from verifier_rl.panel_execution import pack_result, request_for, unpack_result
from verifier_rl.suites import canonical_json, digest


def preflight_kill(packed):
    value = startup_failure(unpack_result(packed))
    return pack_result(replace(value, detail="preflight:RuntimeError", metadata=dict(value.metadata,
        sandbox_id=recovery.FAILED_SANDBOX, preflight_stage="output_collection", preflight_returncode=137,
        preflight_stderr_preview="")))


def fixture():
    parent_plan = old.make_plan(ROOT)
    fitted = calibration(parent_plan)
    plan = two.make_plan(parent_plan, fitted)
    parent = {"calibration": fitted, "offline_verified": True}
    docs = {"plan.json": plan, "calibration.json": fitted, "parent_verification.json": parent,
        "source_snapshot.json": {},
        "setup.json": {"app_name": "test", "sandbox_image_id": "im-test"}, "conformance.json": controls(),
        "stopped-1.json": {"stage": "independent_evaluation", "type": "ValueError",
                           "detail": "incomplete or extra execution evidence"}}
    def batch(key, role, group, faults=None):
        faults = faults or ["correct"] * 4
        raw = {s["sample_id"]: records(s, role, f) for s,f in zip(group, faults)}
        result = {"key": key, "role": role, "samples": group, "records": raw, "startup_retries": [],
                  "plan_hash": digest(canonical_json(plan))}
        docs[f"grading/{key}/raw.json"] = result
        reports = [old.grade(s, raw[s["sample_id"]], role, "im-test", plan) for s in group]
        docs[f"grading/{key}/result.json"] = dict(result, reports=reports)
        docs[f"grading/{key}/intent.json"] = {"key": key, "samples": group, "role": role,
            "setup": docs["setup.json"], "plan_hash": digest(canonical_json(plan)), "deadline": 10000000000., "retry": False}
        return reports
    populations = {}
    initial = [old.controls()["correct"] + f"\n# first-{i}" for i in range(4)]
    for condition in two.CONDITIONS:
        keys, rewards = [], []
        for index in range(24):
            key = f"train-{condition}-{index:02d}"
            keys.append(key)
            group = [old.sample_from_text(initial[j], sid=sid, plan=plan, tokens=64, eos=True)
                     for j, sid in enumerate(two.validate_batch(key, "training"))]
            reports = batch(key, "training", group, ["correct", "no_empty", "constant", "correct"])
            values = [old.reward(condition, r, "0", s["sample_id"]) for s,r in zip(group,reports)]
            rewards.append(values)
            docs[f"arms/{condition}/rollouts/{key}.json"] = group
            docs[f"arms/{condition}/rewards/{key}.json"] = {"rewards": values, "reports": reports,
                "draws": {s["sample_id"]: str(old.noise_draw(old.NOISE_SEED, s["sample_id"])) for s in group}}
        outputs = {}
        for step in ((0,) + old.CHECKPOINTS if condition == "reference" else old.CHECKPOINTS):
            name = f"{condition if step else 'baseline'}-{step:02d}"
            outputs[str(step)] = populations[name] = [sample(plan, f"eval-{name}-{seed}", seed) for seed in old.EVALUATION_SEEDS]
            for s in outputs[str(step)]:
                docs[f"arms/{condition}/evaluation-{step:02d}/samples/{s['sample_id']}.json"] = s
        metrics = {"global_step": 24, "rewards": rewards, "log_history": [{"grad_norm": 1.0}] * 24,
            "before_parameter_hash": old.PARAMETER_HASH, "after_parameter_hash": "synthetic-" + condition,
            "finite_parameters": True, "checkpoint_reload_verified": True, "generated_rollout_tokens": 24 * 4 * 64}
        docs[f"arms/{condition}/result.json"] = {"condition": condition, "metrics": metrics,
            "evidence": old.training_evidence(metrics), "first_rollouts": initial, "rollout_keys": keys,
            "evaluations": outputs, "checkpoint_hashes": {"0": old.PARAMETER_HASH, "24": metrics["after_parameter_hash"]}}
    for name in ("baseline-00", "reference-12", "reference-24"):
        summaries = []
        for index in range(4 if name != "reference-24" else 3):
            summaries.extend(batch(f"evaluation-{name}-{index:02d}", "evaluation", populations[name][4*index:4*(index+1)]))
        if name != "reference-24":
            docs[f"evaluation_summaries/{name}.json"] = two.summary(populations[name], summaries)
    group = populations["reference-24"][12:]
    batch(recovery.FAILED_BATCH, "evaluation", group)
    del docs[f"grading/{recovery.FAILED_BATCH}/result.json"]
    raw = docs[f"grading/{recovery.FAILED_BATCH}/raw.json"]
    second = raw["records"][group[1]["sample_id"]]
    drop = [k for k in second if k != recovery.FAILED_INPUT][-2:]
    for key in drop:
        del second[key]
    second[recovery.FAILED_INPUT] = preflight_kill(second[recovery.FAILED_INPUT])
    for s in group[2:]:
        raw["records"][s["sample_id"]] = {}
    for s in group:
        for key, value in raw["records"][s["sample_id"]].items():
            docs[f"grading/{recovery.FAILED_BATCH}/inputs/{s['sample_id']}/{key}.json"] = {
                "sample_id": s["sample_id"], "input_hash": key, "execution": value}
    return {"run_id": two.RUN_ID, "documents": docs}, parent


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.parent = fixture()
        cls.state = recovery.validate_source(cls.source, cls.parent)
        cls.plan = recovery.make_plan(cls.state)

    def test_preserves_45_complete_programs_and_only_pending_inputs(self):
        self.assertEqual(len(self.state["samples"]), 80)
        self.assertEqual(len(self.state["reports"]), 45)
        self.assertEqual(len(self.state["pending"]), 1601)
        self.assertEqual(len(self.state["records"][recovery.FAILED_SAMPLE]), 44)
        self.assertNotIn(("eval-reference-24-14012", recovery.FAILED_INPUT), self.state["pending"])
        self.assertIn((recovery.FAILED_SAMPLE, recovery.FAILED_INPUT), self.state["pending"])
        self.assertEqual(self.plan["generation_calls"], 0)
        self.assertEqual(self.plan["training_calls"], 0)
        with self.assertRaises(ValueError):
            recovery.validate_plan(dict(self.plan, max_new_starts=99999), self.state)

    def test_all_consumed_training_rewards_replayed_and_corruption_rejected(self):
        changed = deepcopy(self.source)
        changed["documents"]["arms/reference/rewards/train-reference-00.json"]["rewards"][0] = 0
        with self.assertRaisesRegex(ValueError, "consumed training"):
            recovery.validate_source(changed, self.parent)
        changed = deepcopy(self.source)
        changed["documents"]["grading/evaluation-structured-12-00/intent.json"] = {}
        with self.assertRaisesRegex(ValueError, "unreviewed prior"):
            recovery.validate_source(changed, self.parent)

    def test_preflight_kill_is_not_a_candidate_signal_or_oom_claim(self):
        sid, key = recovery.FAILED_SAMPLE, recovery.FAILED_INPUT
        case = next(c for c in old.cases_for("evaluation") if c.input_hash == key)
        request = request_for(self.state["samples"][sid]["source"], case)
        failed = unpack_result(self.state["failed_execution"])
        self.assertTrue(recovery.pre_candidate_failure(failed, request, "im-test"))
        for updates in ({"returncode": 137}, {"startup_seconds": 1}, {"cleanup": "unconfirmed"},
                        {"preflight_stage": "complete"}, {"source_hash": "wrong"}, {"preflight_returncode": 1}):
            self.assertFalse(recovery.pre_candidate_failure(replace(failed, metadata=dict(failed.metadata, **updates)), request, "im-test"))

    def test_complete_replay_reuses_saved_scores_and_counts_original_failure(self):
        attempts = {}
        good = {sid: records(s, "evaluation") for sid,s in self.state["samples"].items()}
        for sid, key in self.state["pending"]:
            value = deepcopy(good[sid][key])
            value["metadata"]["sandbox_id"] += "-recovery"
            attempts[sid + "/" + key] = [value]
        result = recovery.verify_completion(self.source, self.parent, self.plan, attempts)
        self.assertEqual(len(result["policies"]), 5)
        self.assertTrue(all(v["audit_full_passes"] == 16 for v in result["policies"].values()))
        self.assertEqual(result["new_sandbox_starts"], 1601)
        self.assertEqual(result["source_sandbox_starts"] + result["new_sandbox_starts"], 6845)
        self.assertEqual(result["additional_startup_retries"], 0)
        pair = recovery.FAILED_SAMPLE + "/" + recovery.FAILED_INPUT
        attempts[pair][0]["metadata"]["sandbox_id"] = recovery.FAILED_SANDBOX
        with self.assertRaisesRegex(ValueError, "sandbox reuse"):
            recovery.verify_completion(self.source, self.parent, self.plan, attempts)

    def test_filters_do_not_download_weights_or_allow_path_escape(self):
        self.assertTrue(recovery.selected_path("arms/reference/result.json"))
        self.assertTrue(recovery.selected_path("grading/" + recovery.FAILED_BATCH + "/inputs/sample/hash.json"))
        for path in ("../plan.json", "/plan.json", "arms/reference/checkpoint-24/model.safetensors",
                     "arms/reference/checkpoint-24/tokenizer.json", "grading/train-reference-00/inputs/x/hash.json"):
            self.assertFalse(recovery.selected_path(path))

    def test_budget_retains_prior_hold_without_gpu_allowance(self):
        rates = {"cpu_hour_cost": ".1", "mem_gib_hour_cost": ".02",
                 "cpu_hour_cost_sandbox": ".12", "mem_gib_hour_cost_sandbox": ".04"}
        quote = recovery.budget_quote(self.plan, rates, {"metered_cost": "2"}, "134.65")
        self.assertEqual(quote["prior_reservation_held_usd"], "134.65")
        self.assertEqual(quote["gpu_max_usd"], "0")
        self.assertFalse(quote["provider_hard_cap"])
        with self.assertRaises(ValueError):
            recovery.budget_quote(self.plan, rates, {"metered_cost": "NaN"}, "134.65")

    @unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
    def test_whole_cpu_controller_runs_only_pending_evaluation(self):
        import modal_booking_recovery as launcher
        attempts = {"synthetic": []}
        final = {"status": "completed", "policies": {"mock": True}}
        with tempfile.TemporaryDirectory() as temp:
            def path(value):
                return Path(temp) if value == "/artifacts" else Path(value)
            with (patch.object(launcher, "Path", side_effect=path), patch.object(launcher, "artifacts"),
                  patch.object(launcher, "claims", Mock()), patch.object(launcher, "check_frozen_sources"),
                  patch.object(launcher, "local_source", return_value=self.source),
                  patch.object(old, "verify_run", return_value=self.parent),
                  patch.object(launcher.modal.App, "lookup", return_value=Mock(app_id="ap-test")),
                  patch.object(launcher.modal.Sandbox, "list", return_value=[]),
                  patch.object(launcher, "PanelBackend") as backend,
                  patch.object(launcher, "complete_pending", new=AsyncMock(return_value=attempts)) as complete,
                  patch.object(recovery, "verify_journal", return_value=attempts),
                  patch.object(recovery, "verify_completion", return_value=final), patch("builtins.print")):
                self.assertEqual(launcher.run_recovery.local(self.plan, {}, {}, time.time()+60), final)
                self.assertEqual(complete.await_count, 1)
                self.assertEqual(complete.call_args.args[0]["pending"], self.state["pending"])
                self.assertEqual(backend.call_count, 1)
                saved = json.loads((Path(temp) / recovery.RUN_ID / "source.json").read_text())
                self.assertEqual(saved, self.source)


@unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
class RecoveryExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import modal_booking_recovery as launcher
        self.launcher = launcher
        plan = old.make_plan(ROOT)
        s = sample(plan, "eval-structured-24-14000", 14000)
        self.cases = old.cases_for("evaluation")[:2]
        raw = records(s, "evaluation")
        self.good = [unpack_result(raw[c.input_hash]) for c in self.cases]
        self.state = {"samples": {s["sample_id"]: s}, "setup": {"sandbox_image_id": "im-test"},
                      "pending": [(s["sample_id"], c.input_hash) for c in self.cases]}
        self.store = Mock()
        self.saved = {}
        def put(key, value, skip_if_exists=False):
            if key in self.saved:
                return False
            self.saved[key] = value
            return True
        self.store.put.aio = AsyncMock(side_effect=put)

    async def test_only_confirmed_pre_candidate_failure_retries_and_is_durable(self):
        seen = {}
        async def execute(request):
            index = next(i for i,c in enumerate(self.cases) if c.arguments_json == request.input_json)
            seen[index] = seen.get(index, 0) + 1
            self.assertTrue(any('/intent/' in key for key in self.saved))
            if index == 0 and seen[index] == 1:
                return unpack_result(preflight_kill(pack_result(self.good[index])))
            return self.good[index]
        backend = Mock(execute=AsyncMock(side_effect=execute))
        with tempfile.TemporaryDirectory() as temp:
            deadline = time.time()+60
            value = await self.launcher.complete_pending(self.state, {}, Path(temp), deadline, backend, self.store)
            self.assertEqual(recovery.load_attempts(temp), value)
            self.assertEqual(recovery.verify_journal(temp, self.state, {}, deadline), value)
            self.assertEqual(sorted(len(v) for v in value.values()), [1,2])
            self.assertTrue(any('/startup-slot/0' in key for key in self.saved))
            with self.assertRaises(ReconciliationRequired):
                await self.launcher.complete_pending(self.state, {}, Path(temp), time.time()+60, backend, self.store)
        self.assertEqual(backend.execute.await_count, 3)

    async def test_existing_failed_input_gets_only_one_new_start(self):
        plan = old.make_plan(ROOT)
        s = sample(plan, recovery.FAILED_SAMPLE, 14013)
        case = next(c for c in old.cases_for("evaluation") if c.input_hash == recovery.FAILED_INPUT)
        first = preflight_kill(records(s, "evaluation")[case.input_hash])
        state = {"samples": {s["sample_id"]: s}, "setup": {"sandbox_image_id": "im-test"},
                 "pending": [(s["sample_id"], case.input_hash)]}
        backend = Mock(execute=AsyncMock(return_value=unpack_result(first)))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ReconciliationRequired, "preflight:RuntimeError"):
                await self.launcher.complete_pending(state, {}, Path(temp), time.time()+60, backend, self.store)
        self.assertEqual(backend.execute.await_count, 1)

    async def test_candidate_signal_is_preserved_without_retry(self):
        value = replace(self.good[0], status=Status.CANDIDATE_ERROR, detail="nonzero_exit",
                        metadata=dict(self.good[0].metadata, returncode=137))
        self.state["pending"] = self.state["pending"][:1]
        backend = Mock(execute=AsyncMock(return_value=value))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ReconciliationRequired, "unattributed termination"):
                await self.launcher.complete_pending(self.state, {}, Path(temp), time.time()+60, backend, self.store)
            self.assertEqual(len(recovery.load_attempts(temp)), 1)
        self.assertEqual(backend.execute.await_count, 1)

    async def test_deadline_and_duplicate_claim_prevent_execution(self):
        backend = Mock(execute=AsyncMock())
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ReconciliationRequired, "deadline"):
                await self.launcher.complete_pending(self.state, {}, Path(temp), 0, backend, self.store)
        backend.execute.assert_not_called()
        with self.assertRaisesRegex(ValueError, "allow-cloud"):
            self.launcher.launch(False)


if __name__ == "__main__":
    unittest.main()
