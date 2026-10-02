import ast
import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_booking_baseline_comparison import raw_for
from tests.test_booking_screen_recovery import Store
from tests.test_booking_warmstart import result_for
from verifier_rl import booking_replication as study, durable_grading
from verifier_rl.evaluation_journal import ReconciliationRequired
from verifier_rl.evaluation_journal import persist
from verifier_rl.suites import canonical_json, digest

ROOT=Path(__file__).resolve().parents[1]


def samples(plan, key="eval-baseline-00-00", policy="baseline", step=0):
    ids=(study.identities(policy,step)[:4] if key.startswith("eval-") else [(key+f"-{i}",None) for i in range(4)])
    return [study.pilot.original.sample_from_text(study.pilot.original.controls()["correct"],
            sid=sid,seed=seed,plan=plan["experiment"],tokens=90,eos=True) for sid,seed in ids]


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=study.make_plan(ROOT)

    def test_four_fresh_seeds_three_conditions_original_policy(self):
        study.validate_plan(deepcopy(self.plan))
        self.assertEqual(len(study.SEEDS),4)
        self.assertNotIn(study.pilot.TRAIN_SEED,study.SEEDS)
        self.assertFalse(set(study.EVAL_SEEDS)&set(study.pilot.SEEDS))
        self.assertIsNone(self.plan["experiment"]["initial_checkpoint"])
        self.assertFalse(self.plan["pilot_pooled_with_replications"])
        self.assertEqual(self.plan["workload"]["research_input_slots"],698368)
        self.assertEqual(self.plan["workload"]["batches_with_grader_controls"],805)
        self.assertEqual(self.plan["workload"]["training_runs"],12)

    def test_protocol_drift_rejected(self):
        for key,value in (("training_seeds",[1,2,3,4]),("additional_compute_ceiling_usd","251"),
                          ("evaluation_counts",{"0":32,"12":32,"24":32}),("automatic_relaunch",True)):
            changed=deepcopy(self.plan)
            changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                study.validate_plan(changed)
        changed=deepcopy(self.plan)
        changed["repair"]["added"].reverse()
        with self.assertRaises(ValueError):
            study.validate_plan(changed)

    def test_only_seed_changes_in_grpo(self):
        old=study.pilot.trainer_kwargs("out")
        for seed in study.SEEDS:
            new=study.trainer_kwargs("out",seed)
            self.assertEqual({k for k in new if new[k]!=old[k]},{"seed","data_seed"})
            self.assertEqual(new["loss_type"],"grpo")
            self.assertEqual(new["num_generations"],4)
            self.assertEqual(new["max_steps"],24)
        with self.assertRaises(ValueError):
            study.trainer_kwargs("out",study.pilot.TRAIN_SEED)

    def test_repair_is_fixed_count_training_only_and_does_not_remove_empty(self):
        removed,added,repaired=study.repair()
        weak,omitted=study.pilot.contrast.partition(study.pilot.cases_for("training"))
        self.assertEqual((len(removed),len(added),len(repaired)),(8,8,57))
        self.assertTrue(set(c.input_hash for c in removed)<=set(c.input_hash for c in weak))
        self.assertTrue(set(c.input_hash for c in added)<=set(c.input_hash for c in omitted))
        self.assertEqual({c.split for c in repaired},{"training"})
        self.assertTrue(any(c.arguments["bookings"]==[] for c in repaired))
        self.assertEqual(max(len(c.arguments["bookings"]) for c in added),200)
        # Only authored test/oracle values, not candidate execution.
        self.assertEqual(sum(study.pilot.contrast.inclusive_answer(c.arguments_json)==c.expected for c in repaired),49)
        self.assertFalse(self.plan["repair"]["selection_uses_audit"])

    def test_fixed_samples_and_wrong_seed_arm_or_checkpoint_rejected(self):
        self.assertEqual(len(study.identities()),128)
        self.assertEqual(len(study.identities(study.label(study.SEEDS[0],"repaired"),12)),32)
        s=samples(self.plan)
        study.validate_batch("eval-baseline-00-00",s,"evaluation",self.plan)
        for key,group,role in (("eval-baseline-00-00",s[::-1],"evaluation"),
                              ("eval-baseline-00-32",s,"evaluation"),("controls",s,"training")):
            with self.assertRaises(ValueError):
                study.validate_batch(key,group,role,self.plan)
        for policy,step in (("baseline",12),("s20261003-reference",24),("s20261011-log",24)):
            with self.assertRaises(ValueError):
                study.identities(policy,step)
        for i in range(4):
            study.validate_batch(f"parallel-controls-{i}",study.parallel_controls(self.plan,i),"training",self.plan)
        with self.assertRaises(ValueError):
            study.validate_batch("parallel-controls-2",study.parallel_controls(self.plan,1),"training",self.plan)

    def test_repair_and_audit_bounds_keep_unknown_denominator(self):
        s=samples(self.plan)[0]
        outcomes={c.input_hash:{"passed":True,"inclusive_match":False,"reason":"pass"}
                  for c in study.pilot.cases_for("evaluation")}
        outcomes[study.repair()[1][0].input_hash]={"passed":None,"inclusive_match":None,"reason":"unknown"}
        row=study.program_summary(s,outcomes,"evaluation")
        self.assertEqual(row["reference"]["passed_bounds"],[95,96])
        self.assertEqual(row["endpoint_omission"]["passed_bounds"],[57,57])
        self.assertEqual(row["repaired"]["passed_bounds"],[56,57])
        self.assertEqual(row["audit"]["passed_bounds"],[192,192])
        self.assertEqual(row["unknown_inputs"],1)

    def test_real_evidence_validator_replays_new_labels_and_reward(self):
        seed=study.SEEDS[0]
        key=f"train-{study.label(seed,'repaired')}-00"
        s=samples(self.plan,key)
        raw=raw_for(s,key,"training",self.plan)
        checked=study.verify_raw(raw,s,key,"training",self.plan,"im-test")
        self.assertEqual([r["repaired"]["passed_bounds"] for r in checked["rows"]],[[57,57]]*4)
        self.assertEqual(study.rewards_from_raw(raw,s,key,seed,"repaired",self.plan,"im-test"),[1.0]*4)
        with self.assertRaises(ValueError):
            study.rewards_from_raw(raw,s,key,seed,"reference",self.plan,"im-test")
        with self.assertRaises(ValueError):
            study.verify_raw(raw,s,key,"evaluation",self.plan,"im-test")
        with self.assertRaises(ValueError):
            study.verify_raw(raw,s,key,"training",self.plan,"im-changed")

    def test_any_unknown_training_input_blocks_all_rewards(self):
        seed=study.SEEDS[0]
        for arm in study.ARMS:
            key=f"train-{study.label(seed,arm)}-00"
            with patch.object(study,"verify_raw",return_value={"unknown_inputs":1}):
                with self.assertRaises(ReconciliationRequired):
                    study.rewards_from_raw({},[],key,seed,arm,self.plan,"im-test")

    def test_analysis_requires_all_seeds_and_reports_seed_level_pairs(self):
        with self.assertRaises(ValueError):
            study.analyze({}, {})
        policies={}
        for seed in study.SEEDS:
            for arm in study.ARMS:
                for step in (12,24):
                    count=32 if step==12 else 128
                    n={"reference":1,"endpoint_omission":3,"repaired":1}[arm]
                    summary={"programs":count,"audit":{"full_pass_rate_bounds":[.25,.25]},
                        "inclusive_signature_bounds":[n,n],"weak_only_acceptance_bounds":[n,n]}
                    policies[f"{study.label(seed,arm)}-{step}"]={"summary":summary}
        result=study.analyze({"summary":{}},policies)
        contrast=result["contrasts"]["inclusive_signature"]["endpoint_omission_minus_reference"]
        self.assertEqual(len(contrast["pairs"]),4)
        self.assertEqual(contrast["mean_difference_bounds"],[2/128,2/128])
        self.assertTrue(result["bounds_are_missing_data_bounds_not_confidence_intervals"])
        self.assertFalse(result["automatic_amplification_claim"])

    def test_launcher_gpu_limit_and_no_grpo_math_override_or_local_exec(self):
        tree=ast.parse((ROOT/"modal_booking_replication.py").read_text())
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name,maximum in (("train_arm",2),("grade_batch",4),("budget",1),("seed_controller",4)):
            call=functions[name].decorator_list[0]
            self.assertEqual(next(k.value.value for k in call.keywords if k.arg=="max_containers"),maximum)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                self.assertNotIn(node.func.id,{"exec","eval"})
            if isinstance(node,ast.FunctionDef):
                self.assertNotIn(node.name,{"compute_loss","training_step","_calculate_rewards"})


class ParallelJournalTests(unittest.IsolatedAsyncioTestCase):
    async def test_disjoint_concurrent_batches_preserve_all_attempts(self):
        plan=study.make_plan(ROOT)
        store=Store()
        cases=study.pilot.cases_for("training")[:3]
        tasks=[]
        for seed in study.SEEDS:
            key=f"train-{study.label(seed,'reference')}-00"
            sample=samples(plan,key)[0]
            tasks.append((key,sample))
        async def execute(request):
            await asyncio.sleep(0)
            case=next(c for c in cases if canonical_json(c.arguments)==request.input_json)
            return result_for(request.source,case,f"parallel-{time.monotonic_ns()}")
        backend=Mock()
        backend.execute=AsyncMock(side_effect=execute)
        commit=AsyncMock()
        with tempfile.TemporaryDirectory() as tmp:
            results=await asyncio.gather(*(durable_grading.execute_batch([s],cases,Path(tmp)/k,k,
                study.execution_plan(plan),"im-test",time.time()+600,backend,store,commit) for k,s in tasks))
            self.assertEqual(backend.execute.await_count,12)
            self.assertEqual(len([k for k in store.values if k.endswith("/batch")]),4)
            for (key,s),(entries,progress) in zip(tasks,results):
                self.assertEqual(progress["inputs"],3)
                self.assertEqual(len(entries[s["sample_id"]]),3)
                self.assertTrue(all("selected" in e for e in entries[s["sample_id"]].values()))
            self.assertEqual(len(list(Path(tmp).glob("*/inputs/*/*/attempt-1.json"))),12)


class InvocationTests(unittest.TestCase):
    def setUp(self):
        import modal_booking_replication as launcher
        from tests.test_compute_budget import RATES
        self.launcher=launcher
        self.context={"plan":{"max_call_starts":2},"rates":RATES,"deadline":time.time()+600}
        self.function=Mock()
        self.function.spawn.return_value.object_id="fc-test"
        self.function.spawn.return_value.get.return_value={"payload":{"done":True},"seconds":1,"sandbox_seconds":"0"}
        self.claims=Mock()
        self.claims.get.return_value=None
        self.claims.put.return_value=True
        self.budget=Mock()
        self.budget.remote.return_value={"committed_usd":"21"}

    def invoke(self):
        return self.launcher.invoke(self.function,"test-call","cpu",60,self.context,"argument")

    def test_reservation_happens_before_worker_and_settlement_after(self):
        events=[]
        self.budget.remote.side_effect=lambda action,*a:(events.append(action) or {"committed_usd":"21"})
        self.function.spawn.side_effect=lambda *a:(events.append("launch") or Mock(
            object_id="fc-test",get=Mock(return_value={"payload":{"done":True},"seconds":1})))
        with patch.object(self.launcher,"claims",self.claims),patch.object(self.launcher,"budget",self.budget):
            self.assertEqual(self.invoke(),{"done":True})
        self.assertEqual(events,["reserve","launch","settle"])

    def test_denied_budget_never_allocates_worker(self):
        from verifier_rl.compute_budget import BudgetReached
        self.budget.remote.side_effect=BudgetReached("no remaining budget")
        with patch.object(self.launcher,"claims",self.claims),patch.object(self.launcher,"budget",self.budget):
            with self.assertRaises(BudgetReached):
                self.invoke()
        self.function.spawn.assert_not_called()
        self.assertTrue(any(call.args[0].endswith("/stop") for call in self.claims.put.call_args_list))

    def test_lost_call_never_refunds_reservation(self):
        self.function.spawn.return_value.get.side_effect=RuntimeError("lost call")
        with patch.object(self.launcher,"claims",self.claims),patch.object(self.launcher,"budget",self.budget):
            with self.assertRaises(RuntimeError):
                self.invoke()
        self.assertEqual([c.args[0] for c in self.budget.remote.call_args_list],["reserve"])

    def test_cleanup_uncertainty_blocks_new_work_globally(self):
        self.function.spawn.return_value.get.side_effect=ReconciliationRequired("cleanup unconfirmed; stop new work")
        with patch.object(self.launcher,"claims",self.claims),patch.object(self.launcher,"budget",self.budget):
            with self.assertRaises(ReconciliationRequired):
                self.invoke()
        self.assertTrue(any(call.args[0].endswith("/stop") for call in self.claims.put.call_args_list))


class JournalOrderTests(unittest.TestCase):
    def test_roundtrip_canonical_order_not_worker_insertion_order(self):
        import modal_booking_replication as launcher
        # Both sample keys and input keys deliberately differ from JSON sort order.
        raw={"entries":{"z-sample":{"z-hash":{"selected":{"v":3},"intent-1":{"v":1},"attempt-1":{"v":2}},
                                  "a-hash":{"selected":{"v":3},"intent-1":{"v":1},"attempt-1":{"v":2}}},
                        "a-sample":{"b-hash":{"selected":{"v":3},"intent-1":{"v":1},"attempt-1":{"v":2}}}}}
        from verifier_rl.parallel_finalization import journal_records
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            persist(directory/"grading/controls",{"raw":raw})
            for name,value in journal_records("controls",raw):
                path=directory/name
                persist(path.parent,{path.stem:value})
            receipt=launcher.verify_batch_journals(directory,"controls",raw)
            self.assertEqual(receipt["journal_files"],9)
            self.assertEqual(receipt["batches"][0]["key"],"controls")
            # Equality alone is order-insensitive; every actual file must still match.
            path=directory/"grading/controls/inputs/z-sample/a-hash/selected.json"
            path.unlink()  # this test's own temporary authored fixture only
            with self.assertRaises(FileNotFoundError):
                launcher.verify_batch_journals(directory,"controls",raw)

    def test_preflight_amendment_preserves_budget_protocol_and_deadline(self):
        import modal_booking_replication as launcher
        original={"modal_booking_replication.py":"def build_trainer():\n    return 1\n"}
        previous={"plan":study.make_plan(ROOT),"source_snapshot":original,"deadline":time.time()+600,
                  "rates":{"unchanged":True},"billing_before":{"unchanged":True}}
        sources={k:v+"def verify_batch_journals():\n    pass\n" for k,v in original.items()}
        apps=[{"app_id":"ap-viF7pPM44WU8sRtjGICMNt","state":"stopped","tasks":"0"}]
        updated=launcher.preflight_resume_context(previous,sources,apps)
        self.assertEqual(updated["deadline"],previous["deadline"])
        self.assertEqual(updated["plan"],previous["plan"])
        self.assertEqual(updated["billing_before"],previous["billing_before"])
        self.assertFalse(updated["runtime_amendment"]["new_budget"])
        self.assertNotIn("runtime_amendment",previous)
        with self.assertRaises(ValueError):
            launcher.preflight_resume_context(previous,
                {"modal_booking_replication.py":sources["modal_booking_replication.py"].replace("return 1","return 2")},apps)
        with self.assertRaises(ReconciliationRequired):
            launcher.preflight_resume_context(previous,sources,[dict(apps[0],tasks="1",state="running")])
