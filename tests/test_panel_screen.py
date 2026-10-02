import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_modal_backend import fake_sdk, process
from verifier_rl import panel_screen as screen
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.grading import ExecutionResult, Status
from verifier_rl.measurement_v2 import PARAMETER_HASH
from verifier_rl.modal_backend import Limits, RUNNER
from verifier_rl.panel_execution import (PanelBackend, authored_booking, authored_limiter, fixture_source,
                                         pack_result, request_for, require_evidence, runner_for, unpack_result)
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import (BOOKING, CACHE, LIMITER, TASK_IDS, build_panel, build_suite,
                                   control_answer, task_for)
from verifier_rl.verifier_quality import SCREEN_SEEDS

ROOT = Path(__file__).resolve().parents[1]


def execution(source, case, identity, fault="correct"):
    """Synthetic transport evidence from author-written functions, not actual executions."""
    output = canonical_json(control_answer(case.task_id, case.arguments, fault)).encode()
    return ExecutionResult(Status.COMPLETED, output, metadata={
        "backend": "modal", "task_id": case.task_id, "image_id": "im-test",
        "runner_hash": digest(runner_for(case.task_id)), "limits": asdict(Limits()),
        "cleanup": "terminated", "preflight_returncode": 0, "sdk_version": "1.5.5",
        "block_network": True, "reset": "fresh_sandbox_per_input", "creation_interval_seconds": .26,
        "sandbox_id": "sb-" + identity, "total_seconds": 1.0, "returncode": 0,
        "stdout_bytes": len(output), "stdout_sha256": sha256(output).hexdigest(),
        "source_hash": digest(source), "input_hash": case.input_hash})


def model_sample(plan, task_id, seed):
    source = f"def {task_for(task_id).entrypoint}(*args, **kwargs):\n    pass\n"
    sample = screen.inspect_completion(task_id, source)
    sample.update(task_id=task_id, seed=seed, sample_id=screen.sample_id(task_id, seed),
                  prompt_hash=digest(plan["prompts"][task_id]), tokens=16, hit_token_cap=False,
                  ended_with_eos=True)
    return sample


class Claims:
    def __init__(self):
        self.saved = {}
    def put(self, key, value, *, skip_if_exists):
        if key in self.saved:
            return False
        self.saved[key] = value
        return True


class PanelExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_allowlisted_invocations_keep_permissions_and_payload_clean(self):
        for task_id in TASK_IDS:
            case = build_suite(task_id, "training")[0]
            sdk, sandbox = fake_sdk(process(canonical_json(case.expected).encode()))
            sdk.__version__ = "1.5.5"
            backend = PanelBackend("test", "im-test", task_id, sdk=sdk, creation_interval_seconds=.26)
            source = "candidate source is never executed by this fake"
            result = await backend.execute(request_for(source, case))
            require_evidence(result, task_id, "im-test", source=source, case=case)
            calls = sandbox.exec.aio.call_args_list
            payload = json.loads(calls[1].args[4])
            self.assertEqual(payload["input"], case.arguments)
            self.assertEqual(set(payload), {"source", "input", "limits"})
            self.assertEqual(calls[1].args[3], runner_for(task_id))
            self.assertNotIn("expected", payload)
            options = sdk.Sandbox.create.aio.call_args.kwargs
            self.assertTrue(options["block_network"])
            self.assertEqual(options["secrets"], [])
            self.assertEqual(options["volumes"], {})
            self.assertFalse(options["include_oidc_identity_token"])
            sandbox.terminate.aio.assert_awaited_once_with(wait=True)

    async def test_invalid_calls_stop_before_sdk_initialization(self):
        sdk, _ = fake_sdk()
        backend = PanelBackend("test", "im-test", BOOKING, sdk=sdk)
        with self.assertRaises(ValueError):
            await backend.execute(request_for("source", build_suite(CACHE, "training")[0]))
        sdk.App.lookup.aio.assert_not_awaited()
        for task_id in ("os.system", "__import__", "unknown"):
            with self.assertRaises(ValueError):
                PanelBackend("test", "im-test", task_id, sdk=sdk)

    def test_runners_compile_without_local_execution_and_legacy_stays_distinct(self):
        for task_id in TASK_IDS:
            compile(runner_for(task_id), "<runner>", "exec")
            self.assertNotEqual(runner_for(task_id), RUNNER)
            for fault in ("correct", "inclusive_boundary", "constant"):
                compile(fixture_source(task_id, fault), "<authored-fixture>", "exec")

    def test_authored_functions_agree_with_offline_controls(self):
        for task_id, function in ((BOOKING, authored_booking), (LIMITER, authored_limiter)):
            for cases in build_panel()[task_id].values():
                for case in cases:
                    for fault in ("correct", "inclusive_boundary", "constant"):
                        self.assertEqual(function(**case.arguments, fault=fault),
                                         control_answer(task_id, case.arguments, fault))

    def test_raw_serialization_identity_and_uncertainty(self):
        case = build_suite(CACHE, "training")[0]
        result = execution("source", case, "test")
        self.assertEqual(unpack_result(pack_result(result)), result)
        for change in ({"cleanup": "unconfirmed"}, {"input_hash": "changed"},
                       {"source_hash": "changed"}, {"returncode": 137}, {"stdout_sha256": "changed"}):
            changed = replace(result, metadata=dict(result.metadata, **change))
            with self.assertRaises(ValueError):
                require_evidence(changed, CACHE, "im-test", source="source", case=case)
        with self.assertRaises(ValueError):
            require_evidence(ExecutionResult(Status.INFRASTRUCTURE_ERROR), CACHE, "im-test")

    async def test_scheduler_preserves_raw_failure_and_stops_without_retry(self):
        cases = build_suite(CACHE, "training")
        backend = Mock(execute=AsyncMock(return_value=ExecutionResult(Status.INFRASTRUCTURE_ERROR)))
        record = AsyncMock()
        results = await screen.execute_cases("source", cases, backend, "im-test", time.time() + 60, record, concurrency=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(backend.execute.await_count, 1)
        self.assertEqual(record.await_count, 1)
        self.assertEqual(next(iter(results.values())).status, Status.INFRASTRUCTURE_ERROR)

    async def test_deadline_prevents_new_submissions(self):
        backend = Mock(execute=AsyncMock())
        results = await screen.execute_cases("source", build_suite(CACHE, "training"), backend,
                                             "im-test", 1, AsyncMock())
        self.assertEqual(results, {})
        backend.execute.assert_not_awaited()


class ScreenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = screen.make_plan(ROOT)
        cls.panel = screen.validate_plan(cls.plan)
        cls.controls = {}
        for item in screen.control_items(cls.panel):
            cls.controls[item["id"]] = {"task_id": item["task_id"], "source": item["source"],
                "records": {case.input_hash: pack_result(execution(item["source"], case,
                            item["id"] + case.name.replace("/", "-"), item["fault"])) for case in item["cases"]}}
        samples = [model_sample(cls.plan, t, s) for t in TASK_IDS for s in SCREEN_SEEDS]
        cls.generation = {"plan_hash": digest(canonical_json(cls.plan)), "samples": samples,
                          "before_parameter_hash": PARAMETER_HASH, "after_parameter_hash": PARAMETER_HASH,
                          "parameters_unchanged": True}
        cls.records = {sample["sample_id"]: {case.input_hash: pack_result(execution(sample["source"], case,
                      sample["sample_id"] + case.name.replace("/", "-")))
                      for cases in cls.panel[sample["task_id"]].values() for case in cases} for sample in samples}

    def test_complete_synthetic_summary_is_verified_but_saturated(self):
        summary = screen.summarize(self.plan, self.generation, self.controls, self.records, "im-test")
        self.assertTrue(summary["verified"])
        self.assertEqual(summary["model_samples"], 24)
        self.assertEqual(summary["sandbox_executions"], 786)
        self.assertFalse(summary["all_tasks_screen_eligible"])
        self.assertTrue(all(g["reference_passes"] == 8 for g in summary["gates"].values()))

    def test_wrong_model_identity_or_missing_samples_rejected(self):
        for change in ({"after_parameter_hash": "changed"}, {"samples": self.generation["samples"][:-1]}):
            with self.assertRaises(ValueError):
                screen.summarize(self.plan, dict(self.generation, **change), self.controls, self.records, "im-test")

    def test_wrong_case_source_binding_or_incomplete_records_rejected(self):
        sample = self.generation["samples"][0]
        records = deepcopy(self.records[sample["sample_id"]])
        key = next(iter(records))
        records[key]["metadata"]["source_hash"] = "different"
        with self.assertRaises(ValueError):
            screen.program_report(sample, self.panel[CACHE], records, "im-test")
        del records[key]
        with self.assertRaises(ValueError):
            screen.program_report(sample, self.panel[CACHE], records, "im-test")

    def test_extraction_rejection_never_needs_candidate_execution(self):
        sample = self.generation["samples"][0]
        rejected = dict(sample, **screen.inspect_completion(CACHE, ""))
        report = screen.program_report(rejected, self.panel[CACHE], {}, "im-test")
        self.assertEqual(report["executions"], 0)
        self.assertFalse(report["scores"]["training"]["reference_accepted"])
        with self.assertRaises(ValueError):
            screen.program_report(rejected, self.panel[CACHE], self.records[sample["sample_id"]], "im-test")

    def test_bad_control_blocks_generation_gate(self):
        controls = deepcopy(self.controls)
        key = next(iter(controls))
        controls[key]["source"] = "different"
        with self.assertRaises(ValueError):
            screen.validate_controls(self.plan, controls, "im-test")

    def test_decode_or_execution_expansion_rejected(self):
        for field, value in (("max_programs", 25), ("max_sandbox_executions", 1000),
                             ("max_retries", 1), ("training", True), ("max_controller_starts", 99)):
            plan = deepcopy(self.plan)
            plan[field] = value
            with self.assertRaises(ValueError):
                screen.validate_plan(plan)

    @unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
    def test_batch_reuses_result_and_does_not_repeat_interrupted_intent(self):
        import modal_panel_screen as launcher
        item = next(screen.control_items(self.panel))
        async def evaluate(source, cases, backend, image_id, deadline, record, **kwargs):
            results = {case.input_hash: execution(source, case, case.name.replace("/", "-")) for case in cases}
            for case in cases:
                await record(case, results[case.input_hash])
            return results
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "batch"
            with (patch.object(launcher, "artifacts") as volume, patch.object(launcher, "claims", Claims()),
                  patch.object(launcher, "PanelBackend"),
                  patch.object(screen, "execute_cases", new=AsyncMock(side_effect=evaluate)) as execute):
                volume.commit.aio = AsyncMock()
                args = (directory, item["id"], item["source"], item["cases"],
                        {"app_name": "test", "sandbox_image_id": "im-test"}, time.time() + 60)
                first = launcher.evaluate_batch(*args)
                second = launcher.evaluate_batch(*args)
                self.assertEqual(first, second)
                self.assertEqual(execute.await_count, 1)
                self.assertEqual(len(list((directory / "inputs").glob("*.json"))), 2)
                interrupted = Path(temp) / "interrupted"
                persist(interrupted, {"intent": {}})
                with self.assertRaises(ReconciliationRequired):
                    launcher.evaluate_batch(interrupted, *args[1:])
                self.assertEqual(execute.await_count, 1)

    @unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
    def test_controller_start_claims_are_bounded(self):
        import modal_panel_screen as launcher
        with patch.object(launcher, "claims", Claims()):
            launcher.claim_start("controller")
            launcher.claim_start("controller")
            with self.assertRaises(ReconciliationRequired):
                launcher.claim_start("controller")

    @unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
    def test_fast_raw_journal_is_durable_before_batch_volume_commit(self):
        import modal_panel_screen as launcher
        item = next(screen.control_items(self.panel))
        async def evaluate(source, cases, backend, image_id, deadline, record, **kwargs):
            results = {case.input_hash: execution(source, case, case.name.replace("/", "-")) for case in cases}
            await asyncio.gather(*(record(case, results[case.input_hash]) for case in cases))
            return results
        claim_store = Mock()
        claim_store.put.return_value = True
        claim_store.put.aio = AsyncMock(return_value=True)
        with tempfile.TemporaryDirectory() as temp:
            with (patch.object(launcher, "artifacts") as volume, patch.object(launcher, "claims", claim_store),
                  patch.object(launcher, "PanelBackend"),
                  patch.object(screen, "execute_cases", new=AsyncMock(side_effect=evaluate))):
                volume.commit.aio = AsyncMock()
                result = launcher.evaluate_batch(Path(temp), item["id"], item["source"], item["cases"],
                    {"app_name": "test", "sandbox_image_id": "im-test"}, time.time() + 60, journal_to_dict=True)
                self.assertEqual(len(result["records"]), 2)
                self.assertEqual(claim_store.put.aio.await_count, 2)
                self.assertEqual(volume.commit.aio.await_count, 0)
                self.assertEqual(volume.commit.call_count, 2)  # Intent, then full result.

    @unittest.skipUnless(find_spec("modal"), "optional Modal SDK")
    def test_cpu_continuation_reuses_23_and_preserves_partial_batch(self):
        import modal_panel_screen as launcher
        target = self.generation["samples"][-1]
        sid = target["sample_id"]
        reconciliation = {"original_app_id": "ap-CKoxLnZYt91BKvGnTm3Fh1", "original_state": "stopped",
                          "original_tasks": "0", "active_sandbox_ids": [],
                          "original_call_error_type": "FunctionTimeoutError",
                          "reason": "controller_submission_or_function_deadline"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            persist(root, {"plan": self.plan, "deadline": time.time() - 121,
                           "setup": {"sandbox_image_id": "im-test"}})
            persist(root / "gpu", {"generation": self.generation})
            for key, value in self.controls.items():
                persist(root / "controls" / key, {"result": value})
            for sample in self.generation["samples"][:-1]:
                persist(root / "programs" / sample["sample_id"], {"result": {
                    "task_id": sample["task_id"], "source": sample["source"],
                    "records": self.records[sample["sample_id"]]}})
            partial = root / "programs" / sid
            persist(partial, {"intent": {"source_hash": digest(target["source"]), "task_id": BOOKING,
                    "input_hashes": [case.input_hash for cases in self.panel[BOOKING].values() for case in cases]}})
            for case in self.panel[BOOKING]["training"][:3]:
                persist(partial / "inputs", {case.input_hash: {"execution": pack_result(execution(
                    target["source"], case, "old-partial-" + case.name.replace("/", "-")))}})
            def finish_batch(directory, key, source, cases, *args, **kwargs):
                raw = {"task_id": BOOKING, "source": source, "records": {
                    case.input_hash: pack_result(execution(source, case, "new-" + case.name.replace("/", "-")))
                    for case in cases}}
                persist(directory, {"result": raw})
                return raw
            with (patch.object(launcher, "root_directory", return_value=root),
                  patch.object(launcher, "artifacts"), patch.object(launcher, "claims", Claims()),
                  patch.object(launcher.modal.App, "lookup", return_value=Mock(app_id="ap-test")),
                  patch.object(launcher.modal.Sandbox, "list", return_value=[]), patch("builtins.print"),
                  patch.object(launcher, "evaluate_batch", side_effect=finish_batch) as evaluate):
                result = launcher.complete_screen.local(self.plan, {"app_name": "test", "sandbox_image_id": "im-test"},
                                                         reconciliation, time.time() + 60)
                self.assertEqual(evaluate.call_count, 1)
                self.assertTrue(result["summary"]["verified"])
                self.assertEqual(result["accounting"]["reused_complete_programs"], 23)
                self.assertEqual(result["accounting"]["total_execution_bounds"], [789, 818])
                self.assertTrue((root / "continuation-001/original_partial_records.json").exists())
                self.assertFalse((partial / "result.json").exists())  # Original is untouched.
            verified = screen.verify_directory(root, continuation="continuation-001")
            self.assertEqual(verified["continuation_accounting"], result["accounting"])
            accounting_path = root / "continuation-001/accounting.json"
            accounting_path.write_text(json.dumps(dict(result["accounting"], new_model_samples=1)))
            with self.assertRaisesRegex(ValueError, "accounting differs"):
                screen.verify_directory(root, continuation="continuation-001")
            accounting_path.write_text(json.dumps(result["accounting"]))
            paths_path = root / "continuation-001/selected_paths.json"
            paths = result["selected_paths"]
            paths_path.write_text(json.dumps(dict(paths, **{sid: "../outside.json"})))
            with self.assertRaisesRegex(ValueError, "outside declared"):
                screen.verify_directory(root, continuation="continuation-001")
            paths_path.write_text(json.dumps(paths))
            first = self.generation["samples"][0]["sample_id"]
            replacement = f"continuation-001/programs/{first}/result.json"
            persist((root / replacement).parent, {"result": json.loads((root / paths[first]).read_text())})
            paths_path.write_text(json.dumps(dict(paths, **{first: replacement})))
            with self.assertRaisesRegex(ValueError, "completed original result was replaced"):
                screen.verify_directory(root, continuation="continuation-001")

    def test_offline_verifier_recomputes_saved_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = screen.summarize(self.plan, self.generation, self.controls, self.records, "im-test")
            persist(root, {"plan": self.plan, "setup": {"sandbox_image_id": "im-test"}, "summary": summary})
            persist(root / "gpu", {"generation": self.generation})
            for key, result in self.controls.items():
                persist(root / "controls" / key, {"result": result})
            for key, records in self.records.items():
                persist(root / "programs" / key, {"result": {"records": records}})
            self.assertEqual(screen.verify_directory(root), summary)


if __name__ == "__main__":
    unittest.main()
