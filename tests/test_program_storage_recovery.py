import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from modal_proto import api_pb2

import modal_program_storage_recovery as recovery
from tests.test_program_grading import document_for
from verifier_rl import program_storage as storage, program_grading as grading, program_execution as execution
from verifier_rl import booking_replication as study
from verifier_rl.evaluation_journal import ReconciliationRequired
from verifier_rl.grading import ExecutionResult, Status
from verifier_rl.panel_execution import pack_result
from verifier_rl.suites import digest

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    cases=study.pilot.cases_for("training")
    runtime=grading.binding("im-test",12345)
    samples,history=[],{}
    for i in range(4):
        sid=recovery.BATCH+f"-{i}"
        sample={"sample_id":sid,"source":"def required_capacity(bookings): return 0"}
        samples.append(sample)
        doc=document_for(sample,cases,recovery.FAILED_SANDBOX if i==2 else f"sb-peer-{i}",answer=lambda c:0)
        if i==2:
            doc["metadata"]["failure"]={"stage":"candidate","type":"ResourceExhaustedError","detail":"Dict backend is overloaded"}
            for c in cases[25:]:
                doc["records"][c.input_hash]=pack_result(ExecutionResult(Status.INFRASTRUCTURE_ERROR,
                    detail="program_not_executed",metadata={"input_hash":c.input_hash,"source_hash":digest(sample["source"])}))
        history[sid]={"intent":grading.intent_for(sample,cases,runtime),"result":doc,"final":grading.metadata_for(doc)}
    return history,samples,cases,runtime


class RecoveryTests(unittest.TestCase):
    def test_review_accepts_only_the_exact_never_submitted_71_inputs(self):
        history,samples,cases,runtime=fixture()
        self.assertEqual(recovery.validate_history(history,samples,cases,"im-test",runtime),list(cases[25:]))
        original=deepcopy(history)
        source=samples[2]["source"]
        new=document_for(samples[2],cases[25:],"sb-continuation",answer=lambda c:0)
        merged=storage.continue_document(history[recovery.FAILED_SAMPLE]["result"],new,source,cases,"im-test")
        checked=execution.validate_program(merged,source,cases,"im-test")
        self.assertTrue(all(o["passed"] is not None for o in checked.values()))
        self.assertEqual(history,original)
        for c in cases[:25]: self.assertEqual(merged["records"][c.input_hash],original[recovery.FAILED_SAMPLE]["result"]["records"][c.input_hash])

    def test_missing_peer_or_altered_receipt_blocks_recovery(self):
        history,samples,cases,runtime=fixture()
        history[samples[0]["sample_id"]]["final"]["metadata"]["cleanup"]="changed"
        # Metadata objects alias the document in this authored fixture, but the
        # unconfirmed lifecycle still must fail the independent validation.
        with self.assertRaises(ValueError): recovery.validate_history(history,samples,cases,"im-test",runtime)
        history,samples,cases,runtime=fixture()
        del history[samples[0]["sample_id"]]
        with self.assertRaises(ValueError): recovery.validate_history(history,samples,cases,"im-test",runtime)

    def test_partially_submitted_unknown_is_not_authorized_for_continuation(self):
        history,samples,cases,runtime=fixture()
        record=history[recovery.FAILED_SAMPLE]["result"]["records"][cases[25].input_hash]
        record["metadata"]["sandbox_id"]=recovery.FAILED_SANDBOX
        history[recovery.FAILED_SAMPLE]["final"]=grading.metadata_for(history[recovery.FAILED_SAMPLE]["result"])
        with self.assertRaises(ValueError): recovery.validate_history(history,samples,cases,"im-test",runtime)

    def test_helper_has_no_training_generation_or_legacy_counter_deletion(self):
        tree=ast.parse((ROOT/"modal_program_storage_recovery.py").read_text())
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertNotIn("trainer.train",calls)
        self.assertNotIn("model.generate",calls)
        self.assertEqual(calls.count("backend.execute_program"),1)  # The 71-input suffix only.
        self.assertNotIn("store.clear",calls)
        self.assertNotIn("os.remove",calls)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_parent_stopped_not_just_absent(self):
        response=api_pb2.AppGetLifecycleResponse(lifecycle=api_pb2.AppLifecycle(app_state=api_pb2.APP_STATE_STOPPED,stopped_at=123))
        client=SimpleNamespace(stub=SimpleNamespace(AppGetLifecycle=AsyncMock(return_value=response)))
        value=await recovery.require_idle_apps([],client)
        self.assertEqual(value["app_id"],recovery.PARENT_APP)
        self.assertEqual(client.stub.AppGetLifecycle.call_args.kwargs,{"timeout":30,"retry":None})
        response.lifecycle.app_state=api_pb2.APP_STATE_DETACHED
        with self.assertRaises(ReconciliationRequired): await recovery.require_idle_apps([],client)


@unittest.skipUnless(
    (ROOT / "runs" / study.RUN_ID / grading.RELATIVE / "context.json").exists(),
    "historical authorization comparison requires the optional private runs/ context",
)
class AuthorizationTests(unittest.TestCase):
    def test_no_new_start_without_completed_recovery_receipt(self):
        import json
        import modal_booking_replication as launch
        previous=json.loads((ROOT/"runs"/study.RUN_ID/grading.RELATIVE/"context.json").read_text())
        sources={name:(ROOT/name).read_text() for name in set(previous["source_snapshot"])|{"verifier_rl/program_storage.py"}}
        context=storage.context_for(previous,sources)
        records={grading.NAMESPACE+"/authorization":{"context_hash":grading.fingerprint(previous)}}
        mock=Mock()
        mock.get.side_effect=lambda key,default=None:records.get(key,default)
        with patch.object(launch,"claims",mock):
            with self.assertRaisesRegex(ValueError,"not yet reviewed"): launch.require_program_authorization(context)
            records[grading.NAMESPACE+"/"+storage.metadata_relative(context)+"/ready"]={"context_hash":grading.fingerprint(context)}
            launch.require_program_authorization(context)
            changed=deepcopy(context)
            changed["deadline"]+=1
            with self.assertRaises(ValueError): launch.require_program_authorization(changed)

    def test_actual_training_control_guard_accepts_only_reviewed_storage_source_delta(self):
        import json
        import modal_booking_replication as launch
        previous=json.loads((ROOT/"runs"/study.RUN_ID/grading.RELATIVE/"context.json").read_text())
        sources={name:(ROOT/name).read_text() for name in set(previous["source_snapshot"])|{"verifier_rl/program_storage.py"}}
        context=storage.context_for(previous,sources)
        result={"passed":True,"version":"fixture"}
        def read(path):
            path=str(path)
            if path.endswith("source_snapshot.json"): return previous["source_snapshot"]
            if path.endswith("recovery-control/result.json"): return result
            if path.endswith("program_reuse_preflight.json"):
                return {"passed":True,"service_concurrency":1,"program_concurrency":4,"new_sandbox_reservation_seconds":0}
            return {"passed":True,"batches":4}
        with patch.object(launch,"read_json",side_effect=read),patch.object(launch.old_release,"validate_control",return_value=result):
            launch.require_controls(context)
            bad=deepcopy(context)
            bad["source_snapshot"]["verifier_rl/booking_replication.py"]+="\n# not allowed\n"
            with self.assertRaises(ValueError): launch.require_controls(bad)
