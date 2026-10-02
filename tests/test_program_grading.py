"""Authored JSON, mock transports and trusted oracles only; no local code execution."""

import ast
import asyncio
import base64
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_booking_replication import samples
from tests.test_booking_screen_recovery import Store
from tests.test_compute_budget import RATES
from verifier_rl import booking_replication as study, program_execution as execution, program_grading as grading
from verifier_rl import compute_budget as accounting, program_benchmark as benchmark
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.grading import ExecutionResult, Status
from verifier_rl.modal_backend import Limits
from verifier_rl.panel_execution import pack_result
from verifier_rl.suites import canonical_json, digest

ROOT=Path(__file__).resolve().parents[1]


def document_for(sample,cases,identifier="sb-test",answer=None,unknown=False,cleanup="terminated"):
    source=sample["source"]
    metadata={"version":execution.VERSION,"profile":execution.PROFILE,"runner_hash":digest(execution.runner()),
        "profile_hash":digest(execution.READONLY_BOOTSTRAP),"source_hash":digest(source),"task_id":execution.BOOKING,
        "image_id":"im-test","sdk_version":"1.5.5","limits":asdict(Limits()),
        "sandbox_lifetime_seconds":execution.lifetime_for(len(cases)),"block_network":True,
        "reset":"fresh_sandbox_per_program_fresh_process_per_input","cleanup":cleanup,
        "sandbox_id":identifier,"preflight_returncode":0,"candidate_submission_attempted":True,"total_seconds":1.2}
    records={}
    for i,case in enumerate(cases):
        details={"input_hash":case.input_hash,"source_hash":digest(source),"sandbox_id":identifier,"parent_returncode":0}
        if unknown and i==0:
            value=ExecutionResult(Status.INFRASTRUCTURE_ERROR,detail="authored_unknown",metadata=details)
        else:
            payload=canonical_json({"source":source,"input":case.arguments,"limits":asdict(Limits())})
            actual=case.expected if answer is None else answer(case)
            envelope={"version":execution.VERSION,"payload_hash":digest(payload),"candidate_started":True,
                "returncode":0,"user_cpu_seconds":.01,"system_cpu_seconds":.01,"max_rss_kib":10000,
                "wall_seconds":.1,"enforced_limit":None,"stdout_base64":base64.b64encode(
                    (canonical_json(actual)+"\n").encode()).decode(),"stderr_base64":"",
                "stdout_truncated":False,"stderr_truncated":False}
            status,reason,stdout,_=execution.inspect_envelope(envelope,digest(payload))
            details["supervisor_report"]=envelope
            value=ExecutionResult(status,stdout,reason,False,details)
        records[case.input_hash]=pack_result(value)
    return {"metadata":metadata,"records":records,"input_order":[c.input_hash for c in cases]}


def raw_for_programs(group,key,role,plan,runtime,answers=None,unknown=False):
    cases=study.pilot.cases_for(role)
    entries={}
    for i,sample in enumerate(group):
        if sample["extraction_status"].startswith("rejected_"):
            entries[sample["sample_id"]]={"rejected":True,"sample_hash":grading.fingerprint(sample)}
        else:
            entries[sample["sample_id"]]={"intent":grading.intent_for(sample,cases,runtime),
                "result":document_for(sample,cases,"sb-"+str(i),None if answers is None else answers[i],unknown)}
    return {"version":grading.VERSION,"key":key,"role":role,"samples":group,"programs":entries,
            "plan_hash":grading.fingerprint(plan),"runtime":runtime}


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=study.make_plan(ROOT)
        cls.runtime=grading.binding("im-test",time.time()+600)

    def test_control_scores_identical_with_one_sandbox_per_program(self):
        group=study.pilot.controls(self.plan["experiment"])
        answers=[None,lambda c:study.pilot.contrast.inclusive_answer(c.arguments_json),lambda c:0]
        raw=raw_for_programs(group,"controls","training",self.plan,self.runtime,answers)
        checked=grading.verify_raw(raw,group,"controls","training",self.plan,self.runtime)
        self.assertTrue(study.validate_controls(checked)["passed"])
        self.assertEqual([r["reference"]["passed_bounds"] for r in checked["rows"]],[[96,96],[64,64],[1,1]])
        self.assertEqual([r["endpoint_omission"]["passed_bounds"] for r in checked["rows"]],[[57,57],[57,57],[1,1]])
        self.assertEqual([r["repaired"]["passed_bounds"] for r in checked["rows"]],[[57,57],[49,49],[1,1]])
        self.assertEqual(len(checked["sandbox_ids"]),3)

    def test_partial_rewards_and_arm_binding(self):
        for arm,expected in (("reference",64/96),("endpoint_omission",1),("repaired",49/57)):
            seed=study.SEEDS[0]
            key=f"train-{study.label(seed,arm)}-00"
            group=samples(self.plan,key)
            raw=raw_for_programs(group,key,"training",self.plan,self.runtime,
                [lambda c:study.pilot.contrast.inclusive_answer(c.arguments_json)]*4)
            self.assertEqual(grading.rewards_from_raw(raw,group,key,seed,arm,self.plan,self.runtime),[expected]*4)
            with self.assertRaises(ValueError):
                grading.rewards_from_raw(raw,group,key,study.SEEDS[1],arm,self.plan,self.runtime)

    def test_unknown_retains_denominator_and_blocks_training(self):
        seed=study.SEEDS[0]
        key=f"train-{study.label(seed,'reference')}-00"
        group=samples(self.plan,key)
        raw=raw_for_programs(group,key,"training",self.plan,self.runtime,unknown=True)
        checked=grading.verify_raw(raw,group,key,"training",self.plan,self.runtime)
        self.assertEqual(checked["unknown_inputs"],4)
        self.assertEqual(checked["rows"][0]["reference"]["passed_bounds"],[95,96])
        with self.assertRaises(ReconciliationRequired):
            grading.rewards_from_raw(raw,group,key,seed,"reference",self.plan,self.runtime)

    def test_audit_scoring_keeps_all_fixed_cases(self):
        key="eval-baseline-00-00"
        group=samples(self.plan,key)
        raw=raw_for_programs(group,key,"evaluation",self.plan,self.runtime)
        checked=grading.verify_raw(raw,group,key,"evaluation",self.plan,self.runtime)
        self.assertEqual(checked["rows"][0]["audit"]["passed_bounds"],[192,192])
        self.assertEqual(len(next(iter(raw["programs"].values()))["result"]["records"]),287)
        self.assertEqual(checked["submitted_attempts"],4)

    def test_rejected_programs_remain_in_population(self):
        group=samples(self.plan)
        group[0]=study.pilot.original.sample_from_text("",sid=group[0]["sample_id"],seed=group[0]["seed"],
            plan=self.plan["experiment"],tokens=1,eos=True)
        raw=raw_for_programs(group,"eval-baseline-00-00","evaluation",self.plan,self.runtime)
        checked=grading.verify_raw(raw,group,"eval-baseline-00-00","evaluation",self.plan,self.runtime)
        self.assertEqual(len(checked["rows"]),4)
        self.assertEqual(checked["rows"][0]["audit"]["passed_bounds"],[0,0])
        self.assertEqual(checked["submitted_attempts"],3)

    def test_tampered_runtime_source_input_and_cross_program_sandbox_rejected(self):
        group=samples(self.plan)
        raw=raw_for_programs(group,"eval-baseline-00-00","evaluation",self.plan,self.runtime)
        first=group[0]["sample_id"]
        for field in ("image_id","source_hash","profile","runner_hash"):
            bad=deepcopy(raw)
            bad["programs"][first]["result"]["metadata"][field]="changed"
            with self.subTest(field=field),self.assertRaises(ValueError):
                grading.verify_raw(bad,group,raw["key"],"evaluation",self.plan,self.runtime)
        bad=deepcopy(raw)
        bad["programs"][group[1]["sample_id"]]["result"]=document_for(group[1],study.pilot.cases_for("evaluation"),"sb-0")
        with self.assertRaisesRegex(ValueError,"different programs"):
            grading.verify_raw(bad,group,raw["key"],"evaluation",self.plan,self.runtime)
        bad=deepcopy(raw)
        bad["programs"][first]["result"]["records"].pop(next(iter(bad["programs"][first]["result"]["records"])))
        with self.assertRaises(ValueError):
            grading.verify_raw(bad,group,raw["key"],"evaluation",self.plan,self.runtime)
        with self.assertRaises(ValueError): grading.validate_binding(dict(self.runtime,program_concurrency=8))

    def test_reservation_counts_program_lifetimes_not_per_test_starts(self):
        group=samples(self.plan)
        self.assertEqual(grading.maximum_sandbox_seconds(group,"evaluation"),4*2643)
        self.assertEqual(grading.maximum_sandbox_seconds(group,"training"),4*924)
        group[0]["extraction_status"]="rejected_authored_test"
        self.assertEqual(grading.maximum_sandbox_seconds(group,"evaluation"),3*2643)

    def test_every_individual_journal_read_after_canonical_roundtrip(self):
        group=study.pilot.controls(self.plan["experiment"])
        raw=raw_for_programs(group,"controls","training",self.plan,self.runtime)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            persist(directory/"grading/controls",{"raw":raw})
            for name,value in grading.journal_records("controls",raw):
                path=directory/name
                persist(path.parent,{path.stem:value})
            receipt=grading.verify_journals(directory,"controls",json.loads(canonical_json(raw)))
            self.assertEqual(receipt["journal_files"],3*98)
            self.assertEqual(receipt["read_order_sha256"],grading.verify_journals(directory,"controls",raw)["read_order_sha256"])
            name,_=next(grading.journal_records("controls",raw))
            (directory/name).unlink()  # Own temporary authored fixture only.
            with self.assertRaises(FileNotFoundError): grading.verify_journals(directory,"controls",raw)


class DurableTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plan=study.make_plan(ROOT)
        self.runtime=grading.binding("im-test",time.time()+600)
        self.group=samples(self.plan)
        self.cases=study.pilot.cases_for("training")[:3]
        self.store=Store()
        def write(key,value,skip_if_exists=False):
            if skip_if_exists:
                return self.store.write(key,value,skip_if_exists=True)
            self.store.values[key]=deepcopy(value)
            return True
        self.store.put.aio.side_effect=write
        self.commit=AsyncMock()
        self.backend=Mock()
        self.active,self.peak,self.created=0,0,0
        async def execute(source,cases,*,on_record):
            self.active+=1
            self.peak=max(self.peak,self.active)
            self.created+=1
            doc=document_for({"source":source},cases,"sb-"+str(self.created),answer=lambda c:0)
            for h,value in doc["records"].items():
                await asyncio.sleep(0)
                await on_record(h,value)
            self.active-=1
            return doc
        self.backend.execute_program=AsyncMock(side_effect=execute)

    async def execute(self,tmp,group=None,key="authored-batch"):
        return await grading.execute_programs(self.group if group is None else group,self.cases,Path(tmp),
            key,self.plan,self.runtime,self.backend,self.store,self.commit)

    async def test_four_program_concurrency_and_completed_reuse_no_resubmission(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries,progress,seconds=await self.execute(tmp)
            self.assertEqual(self.peak,4)
            self.assertEqual(self.backend.execute_program.await_count,4)
            self.assertEqual(progress["inputs"],12)
            self.assertEqual(seconds,"8")
            repeated,p,seconds=await self.execute(tmp)
            self.assertEqual(entries,repeated)
            self.assertEqual(p["reused_programs"],4)
            self.assertEqual(seconds,"0")
            self.assertEqual(self.backend.execute_program.await_count,4)

    async def test_restore_from_provider_keeps_wrong_answers_without_executing(self):
        with tempfile.TemporaryDirectory() as original,tempfile.TemporaryDirectory() as restored:
            entries,_,_=await self.execute(original)
            # Historical per-input Dict journals remain readable. New batches
            # publish only final indexes and require their durable Volume copy.
            for sid,entry in entries.items():
                for h,record in entry["result"]["records"].items():
                    self.store.values[self.runtime["namespace"]+"/grading/authored-batch/"+sid+"/input/"+h]=record
            repeated,p,seconds=await self.execute(restored)
            self.assertEqual(entries,repeated)
            self.assertEqual(p["reused_programs"],4)
            self.assertEqual(seconds,"0")
            self.assertEqual(len(list(Path(restored).glob("programs/*/inputs/*.json"))),12)
            self.assertEqual(self.backend.execute_program.await_count,4)

    async def test_pending_intent_never_automatically_retries(self):
        sample=self.group[0]
        key=self.runtime["namespace"]+"/grading/authored-batch/"+sample["sample_id"]+"/intent"
        self.store.values[key]=grading.intent_for(sample,self.cases,self.runtime)
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ReconciliationRequired,"no resubmission"):
            await self.execute(tmp,[sample])
        self.backend.execute_program.assert_not_awaited()

    async def test_provider_record_mutation_or_missing_evidence_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as original,tempfile.TemporaryDirectory() as restored:
            entries,_,_=await self.execute(original,[self.group[0]])
            for sid,entry in entries.items():
                for h,record in entry["result"]["records"].items():
                    self.store.values[self.runtime["namespace"]+"/grading/authored-batch/"+sid+"/input/"+h]=deepcopy(record)
            key=next(k for k in self.store.values if "/input/" in k)
            self.store.values[key]["detail"]="changed"
            with self.assertRaisesRegex(ValueError,"records changed"):
                await self.execute(restored,[self.group[0]])
            del self.store.values[key]
            with self.assertRaisesRegex(ReconciliationRequired,"missing durable"):
                await self.execute(restored,[self.group[0]])
        self.assertEqual(self.backend.execute_program.await_count,1)

    async def test_rejected_source_never_creates_a_sandbox(self):
        group=deepcopy(self.group)
        for sample in group: sample["extraction_status"]="rejected_authored"
        with tempfile.TemporaryDirectory() as tmp:
            entries,p,seconds=await self.execute(tmp,group)
        self.assertEqual(len(entries),4)
        self.assertEqual(p["new_program_starts"],0)
        self.assertEqual(seconds,"0")
        self.backend.execute_program.assert_not_awaited()

    async def test_cleanup_uncertainty_blocks_next_batch_and_preserves_evidence(self):
        async def fail(source,cases,**kwargs):
            return document_for({"source":source},cases,cleanup="unconfirmed:TimeoutError")
        self.backend.execute_program.side_effect=fail
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ReconciliationRequired,"cleanup unconfirmed"):
                await self.execute(tmp,[self.group[0]])
            self.assertTrue((Path(tmp)/"programs"/self.group[0]["sample_id"]/"result.json").exists())
        self.assertIn(self.runtime["namespace"]+"/stop",self.store.values)
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ReconciliationRequired,"scheduler stopped"):
            await self.execute(tmp,[self.group[1]],key="next-batch")
        self.assertEqual(self.backend.execute_program.await_count,1)

    async def test_unknown_circuit_preserved_without_double_counting_reuse(self):
        async def unknown(source,cases,**kwargs):
            return document_for({"source":source},cases,unknown=True)
        self.backend.execute_program.side_effect=unknown
        self.plan=dict(self.plan,unknown_circuit=2)
        with tempfile.TemporaryDirectory() as tmp:
            await self.execute(tmp,[self.group[0]])
            await self.execute(tmp,[self.group[0]])
        from verifier_rl import program_storage
        ledger=self.store.values[self.runtime["namespace"]+"/unknown/"+program_storage.AMENDMENT]
        self.assertEqual(len(ledger["identities"]),1)
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ReconciliationRequired,"circuit reached"):
            await self.execute(tmp,[self.group[1]],key="next-batch")
        self.assertEqual(self.store.values[self.runtime["namespace"]+"/stop"]["reason"],"unknown_circuit")

    async def test_deadline_and_unsafe_path_stop_before_sandbox(self):
        self.runtime=grading.binding("im-test",1)
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ReconciliationRequired): await self.execute(tmp)
        bad=dict(self.group[0],sample_id="../escape")
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError): await self.execute(tmp,[bad])
        self.backend.execute_program.assert_not_awaited()

    async def test_failed_publication_recovers_only_storage_not_execution(self):
        from verifier_rl import program_storage
        actual_write=self.store.put.aio.side_effect
        publications=[]
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"programs"/self.group[0]["sample_id"]
            def unavailable(key,value,**kw):
                if key.endswith("/final"):
                    self.assertTrue((target/"result.json").exists())
                    self.assertGreater(self.commit.await_count,2)
                    publications.append(deepcopy(value))
                    raise OSError("injected Dict overload after durable commit")
                return actual_write(key,value,**kw)
            self.store.put.aio.side_effect=unavailable
            with self.assertRaises(program_storage.StorageFailure):
                await self.execute(tmp,[self.group[0]])
            saved=json.loads((target/"result.json").read_text())
            self.assertEqual(len(publications),program_storage.ATTEMPTS)
            self.assertEqual(self.backend.execute_program.await_count,1)
            self.assertFalse(any("/input/" in k for k in self.store.values))
            self.assertNotIn(self.runtime["namespace"]+"/stop",self.store.values)
            self.store.put.aio.side_effect=actual_write
            recovered,p,seconds=await self.execute(tmp,[self.group[0]])
            self.assertEqual(recovered[self.group[0]["sample_id"]]["result"],saved)
            self.assertEqual(p["reused_programs"],1)
            self.assertEqual(seconds,"0")
            self.assertEqual(self.backend.execute_program.await_count,1)

    async def test_lost_publication_ack_is_idempotent(self):
        actual_write=self.store.put.aio.side_effect
        failed=False
        def lost_ack(key,value,**kw):
            nonlocal failed
            result=actual_write(key,value,**kw)
            if key.endswith("/final") and not failed:
                failed=True
                raise OSError("injected lost acknowledgment")
            return result
        self.store.put.aio.side_effect=lost_ack
        with tempfile.TemporaryDirectory() as tmp:
            await self.execute(tmp,[self.group[0]])
        self.assertTrue(failed)
        self.assertEqual(self.backend.execute_program.await_count,1)

    async def test_commit_failure_retries_storage_only(self):
        count=0
        async def commit():
            nonlocal count
            count+=1
            if count==3:
                raise OSError("injected commit failure")
        self.commit.side_effect=commit
        with tempfile.TemporaryDirectory() as tmp:
            await self.execute(tmp,[self.group[0]])
        self.assertGreater(count,3)
        self.assertEqual(self.backend.execute_program.await_count,1)


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import modal_booking_replication as launcher
        cls.launcher=launcher
        cls.plan=study.make_plan(ROOT)
        source="def build_trainer(): return 1\ndef evaluation_at_boundary(): return 2\ndef runtime(): return 3\n"
        cls.previous={"plan":cls.plan,"source_snapshot":{"modal_booking_replication.py":source,"old_library.py":"unchanged"},
            "setup":{"sandbox_image_id":"im-old"},"rates":RATES,"billing_before":{"total":10},"deadline":100}
        cls.sources={**cls.previous["source_snapshot"],"verifier_rl/program_execution.py":"frozen-runner",
                     "verifier_rl/program_benchmark.py":"frozen-benchmark"}
        cls.bc={"setup":{"program_image_id":"im-test"},"saved_pool":{},"snapshot":cls.sources}
        cls.br={"status":"passed","controls":13,"unique_sandboxes":121,"research_updates":0,
                "model_samples":0,"full_study_started":False,"manifest":{"authored":True}}
        cls.bc["manifest"]=cls.br["manifest"]

    def context(self,**kw):
        with patch.object(benchmark,"manifest",return_value=self.br["manifest"]):
            return self.launcher.program_context(kw.get("previous",self.previous),self.bc,kw.get("result",self.br),
                kw.get("sources",self.sources),kw.get("apps",[]),now=time.time())

    def test_runtime_branch_retains_research_budget_and_historical_evidence(self):
        context=self.context()
        self.assertEqual(context["plan"],self.plan)
        self.assertEqual(context["budget_context"]["deadline"],100)
        self.assertGreater(context["deadline"],time.time())
        self.assertFalse(context["runtime_amendment"]["new_budget"])
        self.assertTrue(context["runtime_amendment"]["historical_stop_retained"])
        self.assertFalse(context["runtime_amendment"]["reuse_historical_grades"])
        self.assertEqual(self.launcher.namespace(context),grading.NAMESPACE)
        self.assertEqual(self.launcher.run_root(context),Path("/artifacts")/study.RUN_ID/grading.RELATIVE)

    def test_unbenchmarked_or_research_changes_block_release(self):
        for sources in (dict(self.sources,**{"old_library.py":"changed"}),
                        dict(self.sources,**{"verifier_rl/program_execution.py":"changed"}),
                        dict(self.sources,**{"modal_booking_replication.py":self.sources["modal_booking_replication.py"].replace("return 1","return 2")})):
            with self.assertRaises(ValueError): self.context(sources=sources)
        with self.assertRaises(ValueError): self.context(result=dict(self.br,status="failed"))
        with self.assertRaises(ReconciliationRequired): self.context(apps=[{"tasks":"1"}])

    def test_one_service_no_concurrent_decorator_and_reuse_only_guard(self):
        tree=ast.parse((ROOT/"modal_booking_replication.py").read_text())
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        worker=functions["grade_program_batch"]
        self.assertEqual(len(worker.decorator_list),1)
        values={k.arg:k.value for k in worker.decorator_list[0].keywords}
        self.assertEqual(ast.literal_eval(values["max_containers"]),1)
        self.assertEqual(ast.literal_eval(values["retries"]),0)
        source=ast.unparse(worker)
        self.assertLess(source.index("if reuse_only:"),source.index("ProgramBackend("))
        self.assertEqual(grading.PARALLEL_PROGRAMS,4)

    def test_dispatch_reserves_program_ttl_and_never_uses_legacy_worker(self):
        context=self.context()
        group=samples(self.plan)
        with patch.object(self.launcher,"invoke",return_value="graded") as call:
            self.assertEqual(self.launcher.grade(context,"eval-baseline-00-00",group,"evaluation"),"graded")
        self.assertIs(call.call_args.args[0],self.launcher.grade_program_batch)
        self.assertEqual(call.call_args.kwargs,{"sandbox_seconds":4*2643})

    def test_original_budget_binding_preserved_and_no_missing_ledger_reset(self):
        context=self.context()
        store=Mock()
        original_binding=grading.fingerprint(context["budget_context"])
        ledger=accounting.reserve(accounting.initialize(),"historical-lost-call","15","old")
        saved={"binding":original_binding,"ledger":ledger}
        def get(key,default=None):
            if key==study.RUN_ID+"/budget": return deepcopy(saved)
            if key==grading.NAMESPACE+"/authorization": return {"context_hash":grading.fingerprint(context)}
            return default
        store.get.side_effect=get
        with patch.object(self.launcher,"claims",store):
            result=self.launcher.budget.get_raw_f()("reserve",context,"fresh","1","new")
        self.assertIn("historical-lost-call",result["ledger"]["items"])
        self.assertIn(grading.AMENDMENT+"/fresh",result["ledger"]["items"])
        self.assertEqual(result["committed_usd"],"36")
        store.get.side_effect=lambda key,default=None:None
        with patch.object(self.launcher,"claims",store),self.assertRaisesRegex(ValueError,"cannot initialize"):
            self.launcher.budget.get_raw_f()("status",context)

    def test_program_invocation_reserves_ttl_and_keeps_history_namespace_separate(self):
        context=self.context()
        store=Mock()
        store.get.return_value=None
        store.put.return_value=True
        budget=Mock()
        budget.remote.return_value={"committed_usd":"40"}
        function=Mock()
        function.spawn.return_value.object_id="fc-authored"
        function.spawn.return_value.get.return_value={"payload":{"ok":True},"seconds":1,"sandbox_seconds":"5"}
        with patch.object(self.launcher,"claims",store),patch.object(self.launcher,"budget",budget):
            self.launcher.invoke(function,"authored","cpu",60,context,sandbox_seconds=100)
        self.assertEqual(budget.remote.call_args_list[0].args[3],str(accounting.cost(RATES,"cpu",120)+accounting.cost(RATES,"sandbox",100)))
        self.assertTrue(all(c.args[0].startswith(grading.NAMESPACE) for c in store.put.call_args_list))
        self.assertEqual([c.args[0] for c in budget.remote.call_args_list],["reserve","settle"])


class TrainingPhaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_twelve_policies_before_audits_with_two_active_calls(self):
        import modal_booking_replication as launcher
        import threading
        lock=threading.Lock()
        active,peak,called=0,0,[]
        plan=study.make_plan(ROOT)
        def invoke(function,key,kind,seconds,context,seed,arm):
            nonlocal active,peak
            with lock:
                active+=1
                peak=max(peak,active)
                called.append((function,key,seed,arm))
            time.sleep(.005)
            with lock: active-=1
            return {"authored":True}
        with patch.object(launcher,"invoke",side_effect=invoke),patch.object(study,"validate_arm"),patch.object(launcher,"grade") as grade:
            result=await launcher.program_training_phase({"plan":plan})
        self.assertEqual(peak,2)
        self.assertEqual(len(result),12)
        self.assertEqual(set(result),{study.label(seed,arm) for seed in study.SEEDS for arm in study.ARMS})
        self.assertTrue(all(c[0] is launcher.train_arm for c in called))
        for seed in study.SEEDS:
            self.assertEqual([c[3] for c in called if c[2]==seed],plan["arm_orders"][str(seed)])
        grade.assert_not_called()

    async def test_failed_training_stops_unstarted_policies(self):
        import modal_booking_replication as launcher
        plan=study.make_plan(ROOT)
        with patch.object(launcher,"invoke",side_effect=ReconciliationRequired("authored interruption")) as call,patch.object(study,"validate_arm"):
            with self.assertRaises(ReconciliationRequired):
                await launcher.program_training_phase({"plan":plan})
        self.assertLessEqual(call.call_count,2)


if __name__=="__main__":
    unittest.main()
