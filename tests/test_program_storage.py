from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock

from tests.test_program_grading import document_for
from verifier_rl import program_storage as storage, program_execution as execution
from verifier_rl import booking_replication as study, program_grading as grading
from verifier_rl.grading import ExecutionResult, Status
from verifier_rl.panel_execution import pack_result
from verifier_rl.suites import digest

ROOT=Path(__file__).resolve().parents[1]


class StorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_bound_and_conflict_not_retried(self):
        operation=AsyncMock(side_effect=OSError("injected"))
        with self.assertRaises(storage.StorageFailure):
            await storage.retry(operation,label="test",sleep=AsyncMock())
        self.assertEqual(operation.await_count,3)
        operation=AsyncMock(side_effect=ValueError("evidence differs"))
        with self.assertRaises(ValueError):
            await storage.retry(operation,label="test",sleep=AsyncMock())
        self.assertEqual(operation.await_count,1)

    async def test_immutable_atomic_write_and_no_temporary_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"evidence.json"
            await storage.save(path,{"answer":0})
            await storage.save(path,{"answer":0})
            with self.assertRaises(ValueError): await storage.save(path,{"answer":1})
            self.assertEqual(json.loads(path.read_text()),{"answer":0})
            self.assertEqual(len(list(Path(tmp).iterdir())),1)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.cases=study.pilot.cases_for("training")[:4]
        self.sample={"source":"def required_capacity(bookings): return 0"}
        self.old=document_for(self.sample,self.cases,"sb-old",answer=lambda c:0)
        self.old["metadata"]["failure"]={"stage":"candidate","type":"ResourceExhaustedError","detail":"Dict backend is overloaded"}
        for case in self.cases[2:]:
            self.old["records"][case.input_hash]=pack_result(ExecutionResult(Status.INFRASTRUCTURE_ERROR,
                detail="program_not_executed",metadata={"input_hash":case.input_hash,"source_hash":digest(self.sample["source"])}))
        self.new=document_for(self.sample,self.cases[2:],"sb-new",answer=lambda c:0)

    def test_only_never_submitted_suffix_filled_known_wrong_results_unchanged(self):
        self.assertEqual(storage.partial_cases(self.old,self.sample["source"],self.cases,"im-test"),list(self.cases[2:]))
        result=storage.continue_document(self.old,self.new,self.sample["source"],self.cases,"im-test")
        checked=execution.validate_program(result,self.sample["source"],self.cases,"im-test")
        self.assertTrue(all(v["passed"] is not None for v in checked.values()))
        for case in self.cases[:2]:
            self.assertEqual(result["records"][case.input_hash],self.old["records"][case.input_hash])
        self.assertEqual(storage.sandbox_ids(result),["sb-old","sb-new"])
        self.assertIn("continuation_hash",grading.metadata_for(result))
        bad=deepcopy(result)
        bad["records"][self.cases[0].input_hash]["detail"]="better answer"
        with self.assertRaises(ValueError): execution.validate_program(bad,self.sample["source"],self.cases,"im-test")

    def test_submitted_ambiguous_input_never_retried(self):
        bad=deepcopy(self.old)
        bad["records"][self.cases[2].input_hash]["metadata"]["sandbox_id"]="sb-old"
        with self.assertRaises(ValueError): storage.partial_cases(bad,self.sample["source"],self.cases,"im-test")
        bad=deepcopy(self.old)
        bad["metadata"]["failure"]["detail"]="different failure"
        with self.assertRaises(ValueError): storage.partial_cases(bad,self.sample["source"],self.cases,"im-test")

    def test_continuation_cannot_reexecute_known_prefix(self):
        new=document_for(self.sample,self.cases,"sb-new",answer=lambda c:0)
        with self.assertRaises(ValueError): storage.continue_document(self.old,new,self.sample["source"],self.cases,"im-test")


class SourceTests(unittest.TestCase):
    def test_amendment_freezes_research_and_execution_contract(self):
        path=ROOT/"runs"/study.RUN_ID/grading.RELATIVE/"context.json"
        if not path.exists(): self.skipTest("private runtime release artifact not present")
        previous=json.loads(path.read_text())
        names=set(previous["source_snapshot"])|{"verifier_rl/program_storage.py"}
        snapshot={name:(ROOT/name).read_text() for name in names}
        context=storage.context_for(previous,snapshot)
        for name in ("plan","setup","deadline","budget_context","program_runtime"):
            self.assertEqual(context[name],previous[name])
        self.assertEqual(storage.parent_context(context),grading.fingerprint(previous))
        bad=dict(snapshot)
        bad["verifier_rl/booking_replication.py"]+="\n# scientific source edit\n"
        with self.assertRaises(ValueError): storage.validate_sources(previous["source_snapshot"],bad)
        bad=dict(snapshot)
        bad["modal_booking_replication.py"]=bad["modal_booking_replication.py"].replace("weight_decay=0,betas", "weight_decay=1,betas")
        with self.assertRaises(ValueError): storage.validate_sources(previous["source_snapshot"],bad)
