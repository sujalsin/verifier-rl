import ast
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_booking_screen_recovery import Store, intent_for
from tests.test_booking_warmstart import result_for
from tests.test_modal_backend import process
from tests.test_supervised_execution import simulated_result
from verifier_rl import booking_baseline_comparison as study, durable_grading as durable
from verifier_rl import booking_reward_pilot as historical
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.panel_execution import pack_result
from verifier_rl.suites import canonical_json, digest

ROOT = Path(__file__).resolve().parents[1]


def samples_for(plan, key="eval-baseline-00-00"):
    source = study.original.controls()["correct"]
    ids = (study.identities()[:4] if key.startswith("eval-") else [(f"{key}-{i}", None) for i in range(4)])
    return [study.original.sample_from_text(source, sid=sid, plan=plan, seed=seed, tokens=90, eos=True) for sid, seed in ids]


def raw_for(samples, key, role, plan):
    # Authored records, not local candidate execution. result_for uses the trusted
    # oracle to fabricate protected-envelope fixtures for evidence-validation tests.
    entries = {}
    for sample in samples:
        entries[sample["sample_id"]] = {}
        for case in study.cases_for(role):
            record = pack_result(result_for(sample["source"], case, sample["sample_id"]+case.input_hash))
            entries[sample["sample_id"]][case.input_hash] = {
                "intent-1":intent_for(sample,case), "attempt-1":record, "selected":record}
    return {"key":key, "role":role, "samples":samples, "entries":entries, "plan_hash":digest(canonical_json(plan))}


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = study.make_plan(ROOT)

    def test_original_model_no_project_training_and_no_automatic_rl(self):
        study.validate_plan(json.loads(json.dumps(self.plan)))
        self.assertEqual(self.plan["initial_parameter_hash"], study.original.PARAMETER_HASH)
        self.assertNotEqual(self.plan["initial_parameter_hash"], historical.WARM_HASH)
        self.assertIsNone(self.plan["initial_checkpoint"])
        self.assertEqual(self.plan["prior_project_rl_updates"], 0)
        self.assertFalse(self.plan["automatic_training"])
        self.assertFalse(self.plan["historical_screen_gate_passed"])
        self.assertEqual(self.plan["baseline_max_sandbox_starts"], 9500)

    def test_plan_rejects_unannounced_design_changes(self):
        for change in ({"initial_parameter_hash":historical.WARM_HASH}, {"initial_checkpoint":historical.WARM_CHECKPOINT},
                {"automatic_training":True}, {"steps_per_arm":48}, {"evaluation_seeds":[1,2]},
                {"reward_shape":"logarithmic"}, {"scoring_hash":"changed"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                study.validate_plan(dict(self.plan, **change))

    def test_fixed_suites_and_only_one_shared_empty_input(self):
        train, evaluation = study.cases_for("training"), study.cases_for("evaluation")
        audit = study.coverage.cases_for("audit")
        kept, omitted = study.contrast.partition(train)
        self.assertEqual([len(train),len(audit),len(evaluation),len(kept),len(omitted)], [96,192,287,57,39])
        overlap = {c.input_hash for c in train} & {c.input_hash for c in audit}
        self.assertEqual(len(overlap), 1)
        self.assertEqual(next(c for c in train if c.input_hash in overlap).arguments["bookings"], [])
        self.assertEqual(max(len(c.arguments["bookings"]) for c in train), 200)

    def test_grpo_unchanged_except_explicit_seed_and_checkpoint_policy(self):
        old = study.original.trainer_kwargs("output")
        new = study.trainer_kwargs("output")
        difference = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
        self.assertEqual(difference, {"seed","data_seed","save_strategy","save_steps","save_only_model","save_total_limit"})
        self.assertEqual(new["loss_type"], "grpo")
        self.assertEqual(new["max_steps"], 24)
        self.assertFalse(new["save_only_model"])
        self.assertEqual(new["save_steps"], 1)

    def test_exact_sampling_and_policy_schedule(self):
        sample = samples_for(self.plan)
        study.validate_batch("eval-baseline-00-00", sample, "evaluation", self.plan)
        with self.assertRaises(ValueError):
            study.validate_batch("eval-baseline-00-00", sample[::-1], "evaluation", self.plan)
        for policy,step in (("baseline",12),("reference",0),("reference",25),("logarithmic",24)):
            with self.assertRaises(ValueError):
                study.identities(policy,step)
        self.assertEqual(len(study.identities("endpoint_omission",24)),32)

    def test_unknown_scores_retain_denominators(self):
        sample = samples_for(self.plan)[0]
        outcomes = {c.input_hash:{"passed":True,"inclusive_match":False,"reason":"pass"} for c in study.cases_for("evaluation")}
        h = next(c.input_hash for c in study.cases_for("training") if c.arguments["bookings"] == [])
        outcomes[h] = {"passed":None,"inclusive_match":None,"reason":"lost_parent"}
        row = study.program_summary(sample,outcomes,"evaluation")
        self.assertEqual(row["reference"]["passed_bounds"],[95,96])
        self.assertEqual(row["endpoint_omission"]["passed_bounds"],[56,57])
        self.assertEqual(row["audit"]["passed_bounds"],[191,192])
        self.assertEqual(row["unknown_inputs"],1)  # shared input counted once
        self.assertEqual(row["audit"]["full_pass_bounds"],[0,1])

    def test_policy_summary_never_drops_an_unknown_program(self):
        outcomes = {c.input_hash:{"passed":True,"inclusive_match":False,"reason":"pass"} for c in study.cases_for("evaluation")}
        rows=[]
        for sid, seed in study.identities():
            sample=study.original.sample_from_text(study.original.controls()["correct"],sid=sid,seed=seed,plan=self.plan,tokens=90,eos=True)
            rows.append(study.program_summary(sample,outcomes,"evaluation"))
        summary=study.policy_summary(rows)
        self.assertEqual(summary["audit"]["full_pass_bounds"],[32,32])
        self.assertEqual(summary["audit"]["case_pass_bounds"],[6144,6144])
        self.assertFalse(summary["amplification_established"])
        with self.assertRaisesRegex(ValueError,"all 32"):
            study.policy_summary(rows[:31])

    def test_finite_phase_only_cost_bound(self):
        rates={"cpu_hour_cost":".0473","mem_gib_hour_cost":".008","cpu_hour_cost_sandbox":".1419",
               "mem_gib_hour_cost_sandbox":".024","gpu_hour_cost_l40s":"1.95"}
        value=study.budget_quote(self.plan,rates,{"metered_cost":"8.53"})
        self.assertGreater(float(value["additional_resource_envelope_usd"]),46)
        self.assertLess(float(value["additional_resource_envelope_usd"]),55)
        self.assertFalse(value["is_invoice"])
        with self.assertRaises(ValueError):
            study.budget_quote(self.plan,dict(rates,gpu_hour_cost_l40s="NaN"),{})

    def test_launcher_has_no_trainer_call_or_gpu_candidate_execution(self):
        tree=ast.parse((ROOT/"modal_booking_baseline_comparison.py").read_text())
        names={node.name for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
        self.assertNotIn("train_arm",names)
        self.assertNotIn("GRPOTrainer",(ROOT/"modal_booking_baseline_comparison.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                self.assertNotIn(node.func.id,("exec","eval"))

    def test_local_file_mounts_are_last_image_operation(self):
        # Modal validates this only on resolution, not ordinary Python import.
        tree=ast.parse((ROOT/"modal_booking_baseline_comparison.py").read_text())
        assignments={node.targets[0].id:node.value for node in tree.body
                     if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
        for name in ("cpu_image","gpu_image"):
            self.assertIsInstance(assignments[name],ast.Call)
            self.assertIsInstance(assignments[name].func,ast.Name)
            self.assertEqual(assignments[name].func.id,"with_launchers")


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=study.make_plan(ROOT)
        cls.key="train-reference-00"
        cls.samples=samples_for(cls.plan,cls.key)
        cls.raw=raw_for(cls.samples,cls.key,"training",cls.plan)

    def test_reward_recomputed_from_execution_not_asserted_score(self):
        self.assertEqual(study.rewards_from_raw(self.raw,self.samples,self.key,"reference",self.plan,"im-test"),[1]*4)
        with self.assertRaises(ValueError):
            study.rewards_from_raw(self.raw,self.samples,self.key,"endpoint_omission",self.plan,"im-test")

    def test_unknown_training_reward_blocks_update_not_zero_or_censor(self):
        raw=deepcopy(self.raw)
        sample,case=self.samples[0],study.cases_for("training")[0]
        record=pack_result(result_for(sample["source"],case,"unknown",returncode=-9))
        entry=raw["entries"][sample["sample_id"]][case.input_hash]
        entry.update({"attempt-1":record,"selected":record})
        with self.assertRaisesRegex(ReconciliationRequired,"unknown training outcome"):
            study.rewards_from_raw(raw,self.samples,self.key,"reference",self.plan,"im-test")

    def test_missing_input_is_not_successful_subset(self):
        raw=deepcopy(self.raw)
        del raw["entries"][self.samples[0]["sample_id"]][study.cases_for("training")[0].input_hash]
        with self.assertRaisesRegex(ValueError,"missing/extra"):
            study.verify_raw(raw,self.samples,self.key,"training",self.plan,"im-test")

    def test_wrong_input_identity_is_error_not_unknown(self):
        raw=deepcopy(self.raw)
        entry=raw["entries"][self.samples[0]["sample_id"]][study.cases_for("training")[0].input_hash]
        entry["attempt-1"]["metadata"]["source_hash"]="wrong"
        with self.assertRaisesRegex(ValueError,"identity"):
            study.verify_raw(raw,self.samples,self.key,"training",self.plan,"im-test")

    def test_selected_record_cannot_cherry_pick_different_attempt(self):
        raw=deepcopy(self.raw)
        entry=raw["entries"][self.samples[0]["sample_id"]][study.cases_for("training")[0].input_hash]
        entry["selected"]={}
        with self.assertRaisesRegex(ValueError,"selected"):
            study.verify_raw(raw,self.samples,self.key,"training",self.plan,"im-test")

    def test_audit_cannot_supply_training_rewards(self):
        raw=deepcopy(self.raw)
        raw["role"]="evaluation"
        with self.assertRaisesRegex(ValueError,"provenance"):
            study.rewards_from_raw(raw,self.samples,self.key,"reference",self.plan,"im-test")


class DurableWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plan=dict(study.make_plan(ROOT),concurrency=1)
        self.sample=samples_for(self.plan)[0]
        self.cases=study.cases_for("training")[:3]
        self.store,self.commit=Store(),AsyncMock()
        self.deadline=time.time()+1200
        self.key="test-batch"

    async def run_worker(self,tmp,backend):
        return await durable.execute_batch([self.sample],self.cases,Path(tmp),self.key,self.plan,
                "im-test",self.deadline,backend,self.store,self.commit)

    def values(self,unknown=False):
        return [result_for(self.sample["source"],c,str(i),**({"returncode":-9} if unknown and i==1 else {}))
                for i,c in enumerate(self.cases)]

    async def test_unknown_continues_other_inputs_and_reentry_reuses_everything(self):
        backend=Mock(execute=AsyncMock(side_effect=self.values(unknown=True)))
        with tempfile.TemporaryDirectory() as tmp:
            entries,p=await self.run_worker(tmp,backend)
            self.assertEqual(p["new_starts"],3)
            self.assertEqual(p["unknown_inputs"],1)
            self.assertEqual(len(entries[self.sample["sample_id"]]),3)
            backend.execute.reset_mock()
            again,q=await self.run_worker(tmp,backend)
            self.assertEqual(entries,again)
            self.assertEqual(q["new_starts"],0)
            backend.execute.assert_not_awaited()

    async def test_provider_only_completed_result_survives_lost_volume(self):
        backend=Mock(execute=AsyncMock(side_effect=self.values()))
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            entries,_=await self.run_worker(first,backend)
            backend.execute.reset_mock()
            again,p=await self.run_worker(second,backend)
            self.assertEqual(entries,again)
            self.assertEqual(p["new_starts"],0)
            backend.execute.assert_not_awaited()

    async def test_interrupted_intent_not_resubmitted_and_untouched_cases_continue(self):
        backend=Mock(execute=AsyncMock(side_effect=RuntimeError("lost transport")))
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            with self.assertRaisesRegex(RuntimeError,"lost transport"):
                await self.run_worker(first,backend)
            self.assertEqual(backend.execute.await_count,1)
            prefix=f"{self.plan['run_id']}/durable/{self.key}/{self.sample['sample_id']}/{self.cases[0].input_hash}/1/intent"
            self.store.values[prefix]["submitted_at"]-=181
            backend.execute=AsyncMock(side_effect=self.values()[1:])
            _,p=await self.run_worker(second,backend)
            self.assertEqual(p["unknown_inputs"],1)
            self.assertEqual(p["new_starts"],2)

    async def test_live_orphan_halts_until_sandbox_lifetime_expires(self):
        backend=Mock(execute=AsyncMock(side_effect=RuntimeError("lost transport")))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                await self.run_worker(tmp,backend)
            backend.execute.reset_mock()
            with self.assertRaisesRegex(ReconciliationRequired,"may still be active"):
                await self.run_worker(tmp,backend)
            backend.execute.assert_not_awaited()

    async def test_pre_candidate_retry_once_only_and_replayed_without_resubmit(self):
        self.cases=self.cases[:1]
        case=self.cases[0]
        first,_,_=await simulated_result(source=self.sample["source"],case=case,preflight=process(b"",code=-1))
        backend=Mock(execute=AsyncMock(side_effect=[first,result_for(self.sample["source"],case,"replacement")]))
        with tempfile.TemporaryDirectory() as tmp, patch.object(durable.asyncio,"sleep",AsyncMock()):
            entries,p=await self.run_worker(tmp,backend)
            self.assertEqual(p["new_starts"],2)
            self.assertEqual(p["unknown_inputs"],0)
            entry=entries[self.sample["sample_id"]][case.input_hash]
            self.assertIn("intent-2",entry)
            self.assertEqual(entry["intent-2"]["retry_slot"],0)
            backend.execute.reset_mock()
            await self.run_worker(tmp,backend)
            backend.execute.assert_not_awaited()

    async def test_exhausted_retry_budget_retains_unknown(self):
        self.cases=self.cases[:1]
        first,_,_=await simulated_result(source=self.sample["source"],case=self.cases[0],preflight=process(b"",code=-1))
        for i in range(self.plan["max_startup_retries"]):
            self.store.values[self.plan["run_id"]+f"/durable/startup-slot/{i}"]={"old":True}
        backend=Mock(execute=AsyncMock(return_value=first))
        with tempfile.TemporaryDirectory() as tmp:
            _,p=await self.run_worker(tmp,backend)
            self.assertEqual(p["new_starts"],1)
            self.assertEqual(p["unknown_inputs"],1)

    async def test_circuit_persists_across_invocations(self):
        self.plan["unknown_circuit"]=2
        backend=Mock(execute=AsyncMock(side_effect=[result_for(self.sample["source"],c,str(i),returncode=-9) for i,c in enumerate(self.cases)]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ReconciliationRequired,"circuit"):
                await self.run_worker(tmp,backend)
            self.assertEqual(backend.execute.await_count,2)
            backend.execute.reset_mock()
            with self.assertRaisesRegex(ReconciliationRequired,"circuit"):
                await self.run_worker(tmp,backend)
            backend.execute.assert_not_awaited()

    async def test_bad_identity_stops_not_counts_as_unknown(self):
        first=self.values()[0]
        backend=Mock(execute=AsyncMock(return_value=replace(first,metadata=dict(first.metadata,source_hash="wrong"))))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"identity"):
                await self.run_worker(tmp,backend)
            self.assertEqual(backend.execute.await_count,1)

    async def test_cleanup_failure_stops_new_starts(self):
        first,_,_=await simulated_result(source=self.sample["source"],case=self.cases[0],parent_code=137)
        backend=Mock(execute=AsyncMock(return_value=replace(first,metadata=dict(first.metadata,cleanup="unconfirmed"))))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ReconciliationRequired,"cleanup"):
                await self.run_worker(tmp,backend)
            self.assertEqual(backend.execute.await_count,1)

    async def test_expired_deadline_submits_nothing(self):
        self.deadline=time.time()-1
        backend=Mock(execute=AsyncMock())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ReconciliationRequired,"deadline"):
                await self.run_worker(tmp,backend)
            backend.execute.assert_not_awaited()

    async def test_changed_batch_cannot_reuse_old_records(self):
        backend=Mock(execute=AsyncMock(side_effect=self.values()))
        with tempfile.TemporaryDirectory() as tmp:
            await self.run_worker(tmp,backend)
            backend.execute.reset_mock()
            self.deadline+=1
            with self.assertRaisesRegex(ValueError,"binding"):
                await self.run_worker(tmp,backend)
            backend.execute.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
