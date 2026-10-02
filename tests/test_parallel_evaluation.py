"""No candidate execution: authored protected JSON and asynchronous transports."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from tests.test_program_grading import document_for
from verifier_rl import parallel_evaluation as parallel, program_storage as storage
from verifier_rl import booking_replication as study
from verifier_rl.evaluation_journal import ReconciliationRequired

ROOT = Path(__file__).resolve().parents[1]


def fixture(count=1, cases=None):
    plan = study.make_plan(ROOT)
    value = parallel.runtime({"image_id": "im-test"}, "canary", 9999999999)
    cases = cases or study.pilot.cases_for("training")[:3]
    jobs = []
    for n in range(count):
        samples = [study.pilot.original.sample_from_text("def required_capacity(bookings): return 0",
            sid=f"control-{n}-{i}", seed=None, plan=plan["experiment"], tokens=0, eos=True) for i in range(4)]
        jobs.append(parallel.job(samples, cases, f"batch-{n}", plan, value, control=True))
    return value, jobs, cases


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        value, jobs, self.cases = fixture()
        self.job = jobs[0]
        self.state = parallel.initialize(value, jobs)
        self.base = {"key": self.job["key"], "identity": parallel.fingerprint(self.job), "owner": "one"}
        self.state, _ = parallel.transition(self.state, "claim", self.base, 0)

    def call(self, action, now=0, **kwargs):
        self.state, answer = parallel.transition(self.state, action, dict(self.base, **kwargs), now)
        return answer

    def program(self, i, **kw):
        sid = self.job["samples"][i]["sample_id"]
        self.call("permit", i*7, sample_id=sid)
        args = {"sample_id": sid, "document_hash": str(i), "unknown": [],
                "cleanup": "terminated", "sandbox_ids": [f"sb-{i}"]}
        return self.call("program", **(args | kw))

    def test_no_duplicate_owner_or_start(self):
        with self.assertRaises(ReconciliationRequired): self.call("claim")
        self.call("permit", sample_id="control-0-0")
        with self.assertRaises(ReconciliationRequired): self.call("permit", 10, sample_id="control-0-0")
        with self.assertRaises(ValueError): self.call("permit", 11, sample_id="made-up")
        with self.assertRaises(ValueError): self.call("permit", 11, sample_id="control-0-1", owner="other")

    def test_global_spacing_and_deadline(self):
        self.call("permit", sample_id="control-0-0")
        with self.assertRaises(ValueError): self.call("permit", 6.9, sample_id="control-0-1")
        self.call("permit", 7, sample_id="control-0-1")
        with self.assertRaises(ReconciliationRequired):
            self.call("permit", 9999999999, sample_id="control-0-2")

    def test_results_idempotent_but_not_mutable(self):
        self.program(0)
        self.call("program", sample_id="control-0-0", document_hash="0", unknown=[],
                  cleanup="terminated", sandbox_ids=["sb-0"])
        with self.assertRaises(ValueError):
            self.call("program", sample_id="control-0-0", document_hash="changed", unknown=[],
                      cleanup="terminated", sandbox_ids=["sb-0"])

    def test_no_premature_batch_or_reused_sandbox(self):
        with self.assertRaises(ValueError): self.call("batch", receipt={})
        self.program(0)
        with self.assertRaises(ValueError): self.program(1, sandbox_ids=["sb-0"])

    def test_unknown_circuit_accumulates_across_workers(self):
        self.program(0, unknown=[f"a-{i}" for i in range(40)])
        result = self.program(1, unknown=[f"b-{i}" for i in range(24)])
        self.assertEqual(result["stop"]["reason"], "unknown_circuit")
        with self.assertRaises(ReconciliationRequired): self.call("permit", 20, sample_id="control-0-2")

    def test_cleanup_stops_new_work_but_accepts_existing_results(self):
        self.call("permit", sample_id="control-0-0")
        self.call("permit", 7, sample_id="control-0-1")
        for i, cleanup in ((0, "unconfirmed"), (1, "terminated")):
            self.call("program", sample_id=f"control-0-{i}", document_hash=str(i), unknown=[],
                      cleanup=cleanup, sandbox_ids=[f"sb-{i}"])
        self.assertEqual(len(self.state["programs"]), 2)
        self.assertIsNotNone(self.state["stop"])

    def test_contract_and_duplicate_manifest_rejected(self):
        value, jobs, _ = fixture()
        with self.assertRaises(ValueError): parallel.initialize(value, jobs + jobs)
        with self.assertRaises(ValueError): parallel.validate_runtime(value | {"workers": 100})


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.value, self.jobs, self.cases = fixture(4)
        self.state = parallel.initialize(self.value, self.jobs)
        self.now = 0
        self.created = self.active = self.peak = 0
        self.commit = AsyncMock()
        self.archive = AsyncMock()
        self.failure = False
        self.failures = 0

    async def rpc(self, action, payload):
        if action == "program" and self.failure and payload["sample_id"] == "control-0-0":
            self.failures += 1
            raise OSError("injected index outage")
        if action == "permit": self.now += 7
        self.state, result = parallel.transition(self.state, action, payload, self.now)
        return result

    def backend(self, gate):
        outer = self
        class MockBackend:
            async def execute_program(self, source, cases, on_record):
                await gate()
                outer.created += 1
                outer.active += 1
                outer.peak = max(outer.peak, outer.active)
                document = document_for({"source": source}, cases, f"sb-{outer.created}", answer=lambda c: 0)
                for h, value in document["records"].items():
                    await asyncio.sleep(0.005)
                    await on_record(h, value)
                outer.active -= 1
                return document
        return MockBackend()

    async def run_job(self, tmp, n=0, **kwargs):
        return await parallel.execute(self.jobs[n], self.cases, Path(tmp)/str(n), self.backend,
            self.rpc, self.commit, self.archive, owner=f"owner-{n}", clock=lambda: self.now, **kwargs)

    async def test_four_workers_four_programs_and_no_per_test_rpc(self):
        with tempfile.TemporaryDirectory() as tmp:
            await asyncio.gather(*(self.run_job(tmp, n) for n in range(4)))
            self.assertEqual(self.created, 16)
            self.assertGreater(self.peak, 4)
            self.assertLessEqual(self.peak, 16)
            self.assertEqual(len(self.state["batches"]), 4)
            self.assertEqual(len(self.state["permits"]), 16)

    async def test_publication_failure_recovery_never_reexecutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.failure = True
            with self.assertRaises(storage.StorageFailure): await self.run_job(tmp)
            self.assertEqual(self.failures, 3)
            hashes = [parallel.fingerprint(json.loads(p.read_text())) for p in sorted((Path(tmp)/"0/programs").glob("*/result.json"))]
            self.assertEqual(self.created, 4)
            self.assertEqual(len(self.state["batches"]), 0)
            self.failure = False
            result = await self.run_job(tmp, publish_only=True)
            self.assertEqual(self.created, 4)
            self.assertEqual(result["stats"]["new_programs"], 0)
            self.assertEqual(result["sandbox_seconds"], "0")
            self.assertEqual(hashes, [parallel.fingerprint(json.loads(p.read_text())) for p in sorted((Path(tmp)/"0/programs").glob("*/result.json"))])
            self.assertEqual(len(self.state["batches"]), 1)

    async def test_partial_intent_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"0/programs/control-0-0/intent.json"
            await storage.save(path, {"partial": True})
            with self.assertRaises(ReconciliationRequired): await self.run_job(tmp)
            self.assertLessEqual(self.created, 3)

    async def test_tampered_final_result_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            await self.run_job(tmp)
            path = Path(tmp)/"0/programs/control-0-0/result.json"
            changed = json.loads(path.read_text())
            changed["metadata"]["source_hash"] = "tampered"
            with patch.object(Path, "read_text", autospec=True, side_effect=lambda p,*a,**k:
                    json.dumps(changed) if p == path else original(p,*a,**k)):
                with self.assertRaises(ValueError): await self.run_job(tmp, publish_only=True)
            self.assertEqual(self.created, 4)

    async def test_expired_permit_never_reaches_candidate(self):
        normal = self.rpc
        async def expired(action, payload):
            answer = await normal(action, payload)
            if action == "permit": answer["expires"] = self.now-1
            return answer
        self.rpc = expired
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ReconciliationRequired): await self.run_job(tmp)
        self.assertEqual(self.created, 0)

    async def test_suite_change_prevents_any_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                await parallel.execute(self.jobs[0], self.cases[:1], tmp, self.backend, self.rpc,
                    self.commit, self.archive, owner="one")
        self.assertEqual(self.created, 0)


class DeploymentTests(unittest.TestCase):
    def test_canary_is_separate_and_never_stops_parent(self):
        import ast
        source = (ROOT/"modal_parallel_evaluation.py").read_text()
        tree = ast.parse(source)
        functions = {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        self.assertNotIn("terminate", source)
        self.assertNotIn(".stop(", source)
        for name, expected in (("coordinate",1),("grade_parallel",4),("canary",1)):
            call = functions[name].decorator_list[0]
            settings = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
            self.assertEqual(settings["retries"], "0")
            self.assertEqual(settings["max_containers"], "parallel.WORKERS" if name=="grade_parallel" else "1")
        self.assertLess(source.index('held = asyncio.run(reserve_canary('), source.index('call = canary.spawn'))

    def test_parent_sources_remain_frozen_and_control_size_fixed(self):
        from modal_parallel_evaluation import config_for, PARENT_CONTEXT, control_jobs, validate_config
        path = ROOT/"runs"/PARENT_CONTEXT
        if not path.exists(): self.skipTest("private context unavailable")
        parent = json.loads(path.read_text())
        config = config_for(parent,"canary-001")
        validate_config(config)
        jobs = control_jobs(config)
        self.assertEqual(len(jobs),5)
        self.assertTrue(all(len(j["samples"])==4 and len(j["case_hashes"])==287 for j in jobs))
        self.assertEqual(config["plan"],parent["plan"])


class BudgetAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_exact_existing_service(self):
        from types import SimpleNamespace
        from modal_proto import api_pb2
        from modal_parallel_evaluation import reserve_canary, PARENT_BUDGET
        client = SimpleNamespace(stub=SimpleNamespace(AppGetLayout=AsyncMock(return_value=
            api_pb2.AppGetLayoutResponse(app_layout=api_pb2.AppLayout(function_ids={"budget":PARENT_BUDGET},
                objects=[api_pb2.Object(object_id=PARENT_BUDGET, function_handle_metadata=api_pb2.FunctionHandleMetadata())])))))
        handle = SimpleNamespace(remote=AsyncMock(return_value={"committed_usd":"180"}))
        with patch("modal._functions._Function._new_hydrated", return_value=handle) as hydrate:
            result = await reserve_canary({},"test","identity",client=client)
        hydrate.assert_called_once()
        self.assertEqual(hydrate.call_args.args[0],PARENT_BUDGET)
        handle.remote.assert_awaited_once_with("reserve",{},"test","5","identity")
        self.assertEqual(result["committed_usd"],"180")


original = Path.read_text

if __name__ == "__main__": unittest.main()
