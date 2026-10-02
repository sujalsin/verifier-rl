import ast
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock, AsyncMock

from tests.test_booking_replication import samples
from tests.test_program_grading import raw_for_programs
from verifier_rl import parallel_handoff as handoff, parallel_evaluation as parallel
from verifier_rl import booking_replication as study, program_grading as old, compute_budget as accounting
from verifier_rl.evaluation_journal import ReconciliationRequired, persist

ROOT=Path(__file__).resolve().parents[1]


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.plan=study.make_plan(ROOT)
        self.parent={"plan":self.plan,"program_runtime":old.binding("im-test",9999999999)}
        self.key="eval-baseline-00-00"
        self.samples=samples(self.plan)
        self.manifest=parallel.job(self.samples,study.pilot.cases_for("evaluation"),self.key,self.plan,
            parallel.runtime(self.parent["program_runtime"],"research",9999999999))
        self.intents={s["sample_id"]:None for s in self.samples}

    def old_result(self):
        raw=raw_for_programs(self.samples,self.key,"evaluation",self.plan,self.parent["program_runtime"],
                             answers=[lambda case:0]*4)
        checked=old.verify_raw(raw,self.samples,self.key,"evaluation",self.plan,self.parent["program_runtime"])
        return {"raw":raw,"checked":checked,"receipt":{"raw_hash":parallel.fingerprint(raw),
            "checked_hash":parallel.fingerprint(checked),"reader":{"version":old.VERSION,"key":self.key,
            "journal_files":sum(1 for _ in old.journal_records(self.key,raw))}}}

    def test_missing_without_any_provider_or_disk_intent_only_is_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(handoff.classify_old(self.manifest,tmp,self.parent,self.intents),{"status":"never_started"})
            bad=dict(self.intents)
            bad[self.samples[0]["sample_id"]]={"intent":True}
            with self.assertRaises(ReconciliationRequired): handoff.classify_old(self.manifest,tmp,self.parent,bad)
            persist(Path(tmp)/"grading"/self.key/"programs"/self.samples[0]["sample_id"],{"intent":{"partial":True}})
            with self.assertRaises(ReconciliationRequired): handoff.classify_old(self.manifest,tmp,self.parent,self.intents)

    def test_completed_wrong_answers_are_retained_not_resampled(self):
        result=self.old_result()
        with tempfile.TemporaryDirectory() as tmp:
            persist(Path(tmp)/"grading"/self.key,{"result":result})
            found=handoff.classify_old(self.manifest,tmp,self.parent,self.intents)
            self.assertEqual(found["status"],"completed")
            self.assertEqual(found["rows"],result["checked"]["rows"])
            self.assertEqual(found["rows"][0]["audit"]["full_pass_bounds"],[False,False])
            bad=deepcopy(result)
            bad["receipt"]["raw_hash"]="tampered"
            with self.assertRaises(ValueError): handoff.check_old(bad,self.samples,self.key,self.parent)

    def test_per_input_receipt_required_and_missing_intent_inspection_blocked(self):
        result=self.old_result()
        result["receipt"]["reader"]["journal_files"]-=1
        with self.assertRaises(ValueError): handoff.check_old(result,self.samples,self.key,self.parent)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): handoff.classify_old(self.manifest,tmp,self.parent,{})

    def test_new_and_old_scoring_equal_and_tampered_receipt_rejected(self):
        legacy=self.old_result()
        documents={sid:v["result"] for sid,v in legacy["raw"]["programs"].items()}
        raw={"manifest":self.manifest,"documents":documents}
        outcomes=[parallel.checked_program(s,documents[s["sample_id"]],study.pilot.cases_for("evaluation"),"im-test") for s in self.samples]
        receipt={"raw_hash":parallel.fingerprint(raw),"rows":legacy["checked"]["rows"],
                 "job_hash":parallel.fingerprint(self.manifest),"outcome_hashes":[parallel.fingerprint(o) for o in outcomes]}
        self.assertEqual(handoff.check_new(self.manifest,raw,receipt),
            {k:legacy["checked"][k] for k in ("rows","sandbox_ids")})
        with self.assertRaises(ValueError): handoff.check_new(self.manifest,raw,receipt|{"rows":[]})

    def test_no_conclusion_from_incomplete_seed_or_batch_population(self):
        with self.assertRaises(ValueError): handoff.assemble({}, {}, {})
        arms={study.label(s,a):{} for s in study.SEEDS for a in study.ARMS}
        with self.assertRaises(ValueError): handoff.assemble({}, {}, arms)


class HandoffGuardTests(unittest.TestCase):
    def test_readonly_default_drain_and_budget_order(self):
        source=(ROOT/"modal_parallel_handoff.py").read_text()
        tree=ast.parse(source)
        nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        self.assertEqual(ast.literal_eval(nodes["handoff_main"].args.defaults[0]),"inspect")
        self.assertNotIn("Sandbox.create",source)
        self.assertNotIn("app stop",source)
        self.assertNotIn(".terminate(",source)
        self.assertNotIn("train_arm",source)
        self.assertNotIn("delete",source)
        self.assertLess(source.index('if lifecycle["state"] == "APP_STATE_STOPPED": break'),
                        source.index('result = asyncio.run(inventory(parent,config))'))
        self.assertLess(source.index('await handed_budget.remote.aio(config,"reserve",key'),
                        source.index('await cloud.grade_parallel.remote.aio(config,manifest)'))

    def test_cannot_transfer_budget_while_old_writer_running(self):
        import modal_parallel_handoff as launch
        from types import SimpleNamespace
        store=Mock()
        config={"budget_binding":"old"}
        store.get.return_value={"config_hash":parallel.fingerprint(config)}
        with patch.object(launch,"qualify"),patch.object(launch,"store",store),patch.object(
                launch,"lifecycle_sync",Mock(return_value={"state":"APP_STATE_DETACHED"})):
            with self.assertRaises(ReconciliationRequired):
                launch.handed_budget.get_raw_f()(config,"reserve","batch","1","identity")
        store.put.assert_not_called()

    def test_new_budget_keeps_old_holds_and_ceiling(self):
        ledger=accounting.reserve(accounting.initialize(),"old-lost-call","100","old")
        ledger=accounting.reserve(ledger,"new-job","1","new")
        ledger=accounting.settle(ledger,"new-job","0.1","new")
        self.assertEqual(ledger["ceiling"],"250")
        self.assertIsNone(ledger["items"]["old-lost-call"]["actual"])
        self.assertEqual(str(accounting.committed(ledger)),"120.1")


if __name__=="__main__": unittest.main()
