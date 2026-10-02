import ast
import asyncio
from copy import deepcopy
from pathlib import Path
import unittest

from verifier_rl import booking_replication as study, compute_budget as accounting
from verifier_rl import program_benchmark as bench
from verifier_rl.evaluation_journal import ReconciliationRequired

ROOT = Path(__file__).resolve().parents[1]
RATES = {"cpu_hour_cost":"0.0473", "mem_gib_hour_cost":"0.008",
         "cpu_hour_cost_sandbox":"0.1419", "mem_gib_hour_cost_sandbox":"0.024"}


def saved():
    return {"samples":[{"sample_id":f"eval-reference-24-{i}",
                        "source":"def required_capacity(bookings): return 0"}
                       for i in range(18000,18004)]}


class Store:
    def __init__(self):
        self.data = {}
    def get(self, key, default=None):
        return deepcopy(self.data.get(key, default))
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self.data:
            return False
        self.data[key] = deepcopy(value)
        return True


class ProtocolTests(unittest.TestCase):
    def test_fixed_pool_and_manifest_are_not_new_research(self):
        manifest = bench.manifest(saved())
        self.assertEqual(len(manifest["pool"]),7)
        self.assertEqual(len(manifest["controls"]),13)
        self.assertEqual(len(manifest["comparison_inputs"]),12)
        self.assertEqual(len(manifest["benchmark_inputs"]),287)
        self.assertEqual(manifest["research_updates"],0)
        self.assertEqual(manifest["new_model_samples"],0)
        self.assertFalse(manifest["automatic_study_launch"])
        self.assertEqual(manifest["owning_controllers"],1)
        with self.assertRaises(ValueError):
            bench.programs({"samples":list(reversed(saved()["samples"]))})

    def test_controls_compile_but_are_never_executed_locally(self):
        for name,(source,statuses,passes) in bench.controls().items():
            compile(source,name,"exec")
            self.assertEqual(len(statuses),2)
            outcomes={c.input_hash:{"status":status,"passed":passed}
                      for c,status,passed in zip(bench.control_cases(),statuses,passes)}
            bench.validate_control(name,outcomes)
            outcomes[bench.control_cases()[0].input_hash]["status"]="forged"
            with self.assertRaises(ValueError): bench.validate_control(name,outcomes)

    def test_timeout_bound_fits_existing_reservation(self):
        self.assertEqual(bench.maximum_sandbox_seconds(),52044)
        maximum=accounting.cost(RATES,"cpu",bench.FUNCTION_SECONDS+120)
        maximum+=accounting.cost(RATES,"sandbox",bench.maximum_sandbox_seconds())
        self.assertLess(maximum,accounting.number(bench.RESERVATION_USD))

    def test_projection_is_grading_only_not_total_runtime(self):
        result=bench.summarize_timing(100,7,287)
        self.assertEqual(result["inputs"],2009)
        self.assertAlmostEqual(result["inputs_per_second"],20.09)
        self.assertTrue(result["projection_is_not_total_training_runtime"])
        with self.assertRaises(ValueError): bench.summarize_timing(0,7,287)

    def test_cloud_launcher_has_no_gpu_or_training_calls(self):
        tree=ast.parse((ROOT/"modal_program_benchmark.py").read_text())
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        decorator=functions["run_benchmark"].decorator_list[0]
        args={kw.arg:ast.literal_eval(kw.value) for kw in decorator.keywords
              if kw.arg in ("max_containers","retries","nonpreemptible")}
        self.assertEqual(args,{"max_containers":1,"retries":0,"nonpreemptible":True})
        self.assertFalse(any(kw.arg=="gpu" for kw in decorator.keywords))
        # App uses app_id, unlike Sandbox/Image/FunctionCall object_id.
        import modal
        self.assertIsInstance(modal.App.app_id,property)
        owner_attributes={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)
                          and isinstance(n.value,ast.Name) and n.value.id=="owner"}
        self.assertEqual(owner_attributes,{"app_id"})
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                self.assertNotIn(node.func.id,{"exec","eval","GRPOTrainer"})


class BudgetTests(unittest.TestCase):
    def setUp(self):
        # Importing the launcher defines Modal handles; no remote calls occur.
        import modal_program_benchmark as cloud
        self.cloud=cloud
        self.context={"plan":study.make_plan(ROOT),"rates":RATES,
                      "billing_before":21.8,"deadline":9999999999}
        self.store=Store()
        ledger=accounting.initialize("250","20")
        ledger=accounting.reserve(ledger,"old-failed-call","18.417775","old-identity")
        self.key=study.RUN_ID+"/budget"
        self.store.put(self.key,{"binding":cloud.budget_binding(self.context),"ledger":ledger})
        self.store.put(study.RUN_ID+"/stop",{"reason":"cleanup_unconfirmed"})

    def test_preserves_old_holds_cap_and_stop_flag(self):
        previous=self.store.get(self.key)
        reservation=self.cloud.reserve_benchmark(self.store,self.context,bench.manifest(saved()),{})
        current=self.store.get(self.key)
        self.assertEqual(current["binding"],previous["binding"])
        self.assertEqual(current["ledger"]["ceiling"],"250")
        self.assertEqual(current["ledger"]["items"]["old-failed-call"],
                         previous["ledger"]["items"]["old-failed-call"])
        self.assertEqual(reservation["committed_usd"],"43.417775")
        self.assertEqual(self.store.get(study.RUN_ID+"/stop"),{"reason":"cleanup_unconfirmed"})
        with self.assertRaises(ReconciliationRequired):
            self.cloud.reserve_benchmark(self.store,self.context,bench.manifest(saved()),{})
        self.assertEqual(self.store.get(self.key),current)

    def test_insufficient_remaining_budget_blocks_not_resets(self):
        previous=self.store.get(self.key)
        previous["ledger"]=accounting.reserve(previous["ledger"],"other","210","other")
        self.store.put(self.key,previous)
        with self.assertRaises(accounting.BudgetReached):
            self.cloud.reserve_benchmark(self.store,self.context,bench.manifest(saved()),{})
        self.assertEqual(self.store.get(self.key),previous)

    def test_binding_change_blocks(self):
        self.context["deadline"]+=1
        with self.assertRaises(ValueError):
            self.cloud.reserve_benchmark(self.store,self.context,bench.manifest(saved()),{})


class ParallelTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_parallelism_and_order(self):
        active=maximum=0
        async def worker(item):
            nonlocal active,maximum
            active+=1
            maximum=max(maximum,active)
            await asyncio.sleep(.001)
            active-=1
            return item*2
        self.assertEqual(await bench.bounded_map(range(9),worker,4),list(range(0,18,2)))
        self.assertEqual(maximum,4)
        self.assertEqual(active,0)

    async def test_failure_stops_queued_work_and_waits_for_inflight_cleanup(self):
        started,cleaned=[],[]
        async def worker(item):
            started.append(item)
            try:
                await asyncio.sleep(.001 if item==0 else .01)
                if item==0: raise ReconciliationRequired("cleanup unconfirmed")
                return item
            finally:
                cleaned.append(item)
        with self.assertRaises(ReconciliationRequired):
            await bench.bounded_map(range(9),worker,2)
        self.assertEqual(started,[0,1])
        self.assertEqual(sorted(cleaned),[0,1])


if __name__=="__main__":
    unittest.main()
