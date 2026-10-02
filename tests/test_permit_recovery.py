import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from tests.test_parallel_evaluation import fixture
from tests.test_program_grading import document_for
from verifier_rl import permit_recovery as repair, parallel_evaluation as parallel
from verifier_rl import booking_replication as study, compute_budget as budget
from verifier_rl.evaluation_journal import ReconciliationRequired
from verifier_rl.suites import digest


class ClockTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        value, jobs, _ = fixture()
        self.job = jobs[0]
        self.state = parallel.initialize(value, jobs)
        self.base = {"key":self.job["key"],"identity":parallel.fingerprint(self.job),"owner":"worker"}
        self.state, _ = parallel.transition(self.state,"claim",self.base,0)
        self.base["sample_id"] = self.job["samples"][0]["sample_id"]
        self.elapsed = 0.0
        self.nonce = 0
        self.delay = 0
        self.calls = 0

    def new_nonce(self):
        self.nonce += 1
        return str(self.nonce)

    async def sleep(self, value):
        self.elapsed += value

    async def rpc(self, action, payload):
        self.calls += 1
        self.state, result = repair.permit_transition(self.state,payload,now=self.elapsed+9000,wall=100,boot="boot-a")
        if "grant" in result and self.delay:
            self.elapsed += self.delay
            self.delay = 0
        return result

    def client(self, offset):
        return repair.PermitRPC(self.rpc,clock=lambda:self.elapsed+offset,sleep=self.sleep,nonce=self.new_nonce)

    async def test_both_clock_offsets_accept_fresh_grant(self):
        for offset in (-1000000,1000000):
            await self.asyncSetUp()
            client = self.client(offset)
            response = await client("permit",self.base)
            self.assertGreaterEqual(response["expires"],self.elapsed+offset)
            self.assertEqual(client.expired_grants,0)
            self.assertEqual(len(self.state["permits"]),1)

    async def test_delayed_response_retires_only_unused_grant(self):
        self.delay = 6
        client = self.client(-3600)
        result = await client("permit",self.base)
        self.assertEqual(client.expired_grants,1)
        self.assertEqual(result["nonce"],"2")
        token = self.base["key"]+"/"+self.base["sample_id"]
        self.assertEqual(len(self.state["retired_permits"][token]),1)
        self.assertEqual(self.state["retired_permits"][token][0]["nonce"],"1")
        self.assertEqual(self.state["permits"][token]["nonce"],"2")
        with self.assertRaisesRegex(ReconciliationRequired,"entered once"):
            await client("permit",self.base)

    async def test_lost_reply_is_not_retried_or_replayed(self):
        client = self.client(0)
        calls = 0
        async def lost(*args):
            nonlocal calls
            calls += 1
            raise TimeoutError("lost grant reply")
        client.rpc = lost
        with self.assertRaises(TimeoutError): await client("permit",self.base)
        self.assertEqual(calls,1)

    async def test_repeated_delays_have_finite_bound(self):
        client = self.client(0)
        original = self.rpc
        async def delayed(*args):
            result = await original(*args)
            if "grant" in result: self.elapsed += 6
            return result
        client.rpc = delayed
        with self.assertRaisesRegex(ReconciliationRequired,"delay exceeded"):
            await client("permit",self.base)
        self.assertEqual(client.expired_grants,3)

    async def test_restart_waits_out_previous_grants_and_preserves_rate(self):
        await self.client(0)("permit",self.base)
        new = dict(self.base,sample_id=self.job["samples"][1]["sample_id"],nonce="new",unused_previous=None)
        restarted, response = repair.permit_transition(self.state,new,now=1,wall=100,boot="boot-b")
        self.assertEqual(response["wait_seconds"],10)  # canary cooldown is 12s
        restarted, response = repair.permit_transition(restarted,new,now=12.9,wall=100,boot="boot-b")
        self.assertIn("wait_seconds",response)
        restarted, response = repair.permit_transition(restarted,new,now=13,wall=100,boot="boot-b")
        self.assertIn("grant",response)
        newer = dict(new,sample_id=self.job["samples"][2]["sample_id"],nonce="third")
        _, response = repair.permit_transition(restarted,newer,now=13.1,wall=100,boot="boot-b")
        self.assertIn("wait_seconds",response)

    async def test_wrong_renewal_and_stopped_coordinator_rejected(self):
        await self.client(0)("permit",self.base)
        with self.assertRaisesRegex(ReconciliationRequired,"ambiguous replay"):
            repair.permit_transition(self.state,dict(self.base,nonce="other"),now=999,wall=100,boot="boot-a")
        self.state["stop"] = {"reason":"test"}
        with self.assertRaisesRegex(ReconciliationRequired,"stop/deadline"):
            repair.permit_transition(self.state,dict(self.base,nonce="other"),now=999,wall=100,boot="boot-a")

    async def test_latest_local_expiration_still_blocks_late_sandbox_start(self):
        response = await self.client(0)("permit",self.base)
        self.elapsed += 6
        self.assertGreater(self.elapsed,response["expires"])


class WorkerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_frozen_worker_runs_once_after_expired_unused_grant(self):
        from unittest.mock import AsyncMock
        value,jobs,cases = fixture()
        manifest = jobs[0]
        state = parallel.initialize(value,jobs)
        elapsed, serial, delayed = 0.0, 0, False
        created = []
        async def sleep(seconds):
            nonlocal elapsed
            elapsed += seconds
        async def rpc(action,payload):
            nonlocal state,elapsed,delayed
            if action == "try_permit":
                state,result = repair.permit_transition(state,payload,now=elapsed+9000,wall=100,boot="test")
                if "grant" in result and not delayed:
                    delayed = True
                    elapsed += 6
                return result
            state,result = parallel.transition(state,action,payload,100)
            return result
        def nonce():
            nonlocal serial
            serial += 1
            return str(serial)
        clock = lambda: elapsed-100000
        permits = repair.PermitRPC(rpc,clock=clock,sleep=sleep,nonce=nonce)
        def backend(gate):
            class Backend:
                async def execute_program(self,source,cases,on_record):
                    await gate()
                    created.append(len(created))
                    doc = document_for({"source":source},cases,f"sb-{len(created)}",answer=lambda c:0)
                    for key,record in doc["records"].items(): await on_record(key,record)
                    return doc
            return Backend()
        with tempfile.TemporaryDirectory() as tmp:
            result = await parallel.execute(manifest,cases,tmp,backend,permits,AsyncMock(),AsyncMock(),owner="one",clock=clock)
        self.assertEqual(len(created),4)
        self.assertGreaterEqual(permits.expired_grants,1)
        self.assertEqual(result["stats"]["unknown"],0)
        self.assertEqual(len(state["batches"]),1)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cases = study.pilot.cases_for("evaluation")
        _,jobs,_ = fixture(cases=self.cases)
        self.manifest = jobs[0]
        self.documents = {s["sample_id"]:document_for(s,self.cases,f"sb-{i}",answer=lambda c:0)
                          for i,s in enumerate(self.manifest["samples"])}

    def receipts(self):
        result = {}
        for sample in self.manifest["samples"]:
            sid = sample["sample_id"]
            doc = self.documents[sid]
            outcomes = parallel.checked_program(sample,doc,self.cases,"im-test")
            result[sid] = {"document_hash":parallel.fingerprint(doc),
                "unknown":[sid+"/"+h for h,v in outcomes.items() if v["passed"] is None],
                "cleanup":doc["metadata"]["cleanup"],"sandbox_ids":repair.storage.sandbox_ids(doc)}
        return result

    def never_started(self, sample):
        doc = self.documents[sample["sample_id"]]
        for key in ("sandbox_id","preflight_returncode","preflight_stderr","runtime_python","startup_seconds"):
            doc["metadata"].pop(key,None)
        doc["metadata"].update(candidate_submission_attempted=False,cleanup="no_handle; creation_may_be_unconfirmed",
            failure={"stage":"create","type":"ReconciliationRequired","detail":"expired permit; do not start a sandbox"})
        doc["records"] = {c.input_hash:{"status":"infrastructure_error","detail":"program_not_executed",
            "stdout_base64":"","retryable":False,"metadata":{"input_hash":c.input_hash,"source_hash":digest(sample["source"])}}
            for c in self.cases}

    def test_complete_peer_evidence_reconstructs_exact_receipt(self):
        before = deepcopy(self.documents)
        result = repair.classify_interrupted(self.manifest,self.documents,self.receipts())
        self.assertEqual(result["status"],"completed")
        self.assertEqual(self.documents,before)
        self.assertEqual(result["raw"]["documents"],before)

    def test_only_precreate_expiry_is_eligible_for_first_execution(self):
        for sample in self.manifest["samples"]: self.never_started(sample)
        result = repair.classify_interrupted(self.manifest,self.documents,self.receipts())
        self.assertEqual(result["status"],"proven_never_started")
        self.assertEqual(len(result["unknown"]),1148)
        sample = self.manifest["samples"][0]
        for changes in ({"candidate_submission_attempted":True},{"sandbox_id":"sb-ambiguous"},
                        {"failure":{"stage":"create","type":"TimeoutError","detail":"ambiguous"}}):
            with self.subTest(changes=changes):
                changed = deepcopy(self.documents[sample["sample_id"]])
                changed["metadata"].update(changes)
                with self.assertRaises(ReconciliationRequired):
                    repair.require_never_started(sample,changed,self.cases,"im-test")

    def test_changed_archive_hash_and_mixed_partial_batch_block_recovery(self):
        receipts = self.receipts()
        receipts[self.manifest["samples"][0]["sample_id"]]["document_hash"] = "changed"
        with self.assertRaisesRegex(ValueError,"published program"):
            repair.classify_interrupted(self.manifest,self.documents,receipts)
        self.never_started(self.manifest["samples"][0])
        with self.assertRaises(ReconciliationRequired):
            repair.classify_interrupted(self.manifest,self.documents,self.receipts())


class BudgetTests(unittest.TestCase):
    def test_inheritance_preserves_spending_and_requires_stopped_owner(self):
        import modal_permit_recovery as cloud
        from tests.test_budget_handoff import Store
        config = {"name":cloud.NAME,"budget_binding":"binding"}
        ledger = budget.reserve(budget.initialize(),"old-interrupted","12","identity")
        old_owner = {"app_id":cloud.OLD_APP,"config_hash":cloud.OLD_HASH}
        store = Store({cloud.RUN+"/budget":{"binding":"binding","ledger":ledger},
                       cloud.cloud.PREFIX+"/"+cloud.OLD_NAME+"/budget-owner":old_owner})
        function = cloud.permit_budget.get_raw_f()
        with tempfile.TemporaryDirectory() as tmp, patch.object(cloud,"validate"), patch.object(cloud,"store",store), \
             patch.object(cloud,"require_stopped") as stopped, patch.object(cloud,"app",SimpleNamespace(app_id="ap-new")), \
             patch.object(cloud,"permit_budget",MagicMock(object_id="fu-new")), patch.object(cloud,"work",MagicMock()), \
             patch.object(cloud,"archive",MagicMock()), patch.object(cloud.cloud,"paths",side_effect=lambda c,base="/artifacts":Path(tmp)/base.strip("/")):
            before = store.get(cloud.RUN+"/budget")
            function(config,"initialize",{})
            self.assertEqual(store.get(cloud.RUN+"/budget"),before)
            function(config,"reserve",{"ticket":"new","amount":"5","identity":"new"})
            after = store.get(cloud.RUN+"/budget")["ledger"]
            self.assertEqual(after["items"]["old-interrupted"],ledger["items"]["old-interrupted"])
            self.assertEqual(after["ceiling"],"250")
            stopped.side_effect = ReconciliationRequired("old owner active")
            with self.assertRaises(ReconciliationRequired): function(config,"status",{})


if __name__ == "__main__":
    unittest.main()
