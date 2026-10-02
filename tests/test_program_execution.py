import ast
import base64
from copy import deepcopy
from dataclasses import asdict
import json
import unittest
from unittest.mock import AsyncMock

from tests.test_modal_backend import fake_sdk, process
from verifier_rl import booking_baseline_comparison as study, program_execution as runner
from verifier_rl.grading import Status
from verifier_rl.modal_backend import Limits
from verifier_rl.suites import canonical_json, digest

CASES = study.cases_for("training")[:2]
SOURCE = "def required_capacity(bookings): return len(bookings)"


def envelope(case, **updates):
    payload = canonical_json({"source":SOURCE,"input":case.arguments,"limits":asdict(Limits())})
    return dict({"version":runner.VERSION,"payload_hash":digest(payload),"candidate_started":True,
        "returncode":0,"user_cpu_seconds":.01,"system_cpu_seconds":.01,"max_rss_kib":10000,
        "wall_seconds":.1,"enforced_limit":None,"stdout_base64":base64.b64encode(
            (str(case.expected)+"\n").encode()).decode(),"stderr_base64":"",
        "stdout_truncated":False,"stderr_truncated":False},**updates)


def fake_program(values=None):
    sdk,sandbox = fake_sdk()
    sdk.__version__ = "1.5.5"
    values = values or [envelope(c) for c in CASES]
    sandbox.exec.aio.side_effect = [process(b'{"ready":true,"uid":0,"python":"3.12.test"}')] + [
        process((canonical_json(v)+"\n").encode()) for v in values]
    gate = AsyncMock()
    return runner.ProgramBackend("test","im-test",sdk=sdk,start_gate=gate),sdk,sandbox,gate


class SourceTests(unittest.TestCase):
    def test_remote_only_source_compiles_and_restricts_before_candidate(self):
        source = runner.runner()
        tree = ast.parse(source)
        child = ast.literal_eval(tree.body[0].value)
        compile(source,"remote-parent","exec")
        compile(child,"remote-child","exec")
        self.assertIn('seccomp_init(0x00050000 | errno.EPERM)',child)
        self.assertIn('lib.seccomp_load(ctx)',child)
        self.assertIn('os.setuid(65534)',child)
        self.assertLess(child.index('prepare(payload["limits"])'),child.index('os.write(ready_fd'))
        self.assertLess(child.index('os.close(ready_fd)'),child.index('exec(compile(payload["source"]'))
        self.assertIn('os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND',child)
        self.assertIn('resource.RLIMIT_NPROC, 1',child)
        self.assertNotIn('allow("clone"',child)
        self.assertNotIn('allow("fork"',child)
        self.assertNotIn('allow("execve"',child)
        self.assertNotIn('allow("socket"',child)
        self.assertNotIn('expected',{n.value for n in ast.walk(ast.parse(child))
                                    if isinstance(n,ast.Constant) and isinstance(n.value,str)})

    def test_lifetime_and_input_bounds(self):
        self.assertEqual(runner.lifetime_for(287),2643)
        for count in (0,301,True,1.5):
            with self.assertRaises(ValueError): runner.lifetime_for(count)
        with self.assertRaises(ValueError): runner.validate_request(SOURCE,(CASES[0],CASES[0]))
        with self.assertRaises(ValueError): runner.validate_request("",CASES)
        with self.assertRaises(ValueError): runner.CreationGate(.1)

    def test_old_envelope_rejected_by_new_contract(self):
        with self.assertRaises(ValueError):
            runner.inspect_envelope(envelope(CASES[0],version="root-supervised-panel-0.3"),"anything")


class ProgramTests(unittest.IsolatedAsyncioTestCase):
    async def test_storage_callback_failure_retains_completed_result_and_is_not_candidate_error(self):
        backend,sdk,sandbox,_=fake_program()
        result=await backend.execute_program(SOURCE,CASES,on_record=AsyncMock(side_effect=OSError("disk storage failed")))
        self.assertEqual(result["metadata"]["failure"]["stage"],"evidence_storage")
        self.assertEqual(result["metadata"]["cleanup"],"terminated")
        outcomes=runner.validate_program(result,SOURCE,CASES,"im-test")
        self.assertTrue(outcomes[CASES[0].input_hash]["passed"])
        self.assertEqual(result["records"][CASES[1].input_hash]["detail"],"program_not_executed")
        self.assertEqual(sdk.Sandbox.create.aio.await_count,1)
        self.assertEqual(sandbox.exec.aio.await_count,2)  # Preflight + first input, no replay.

    async def test_one_sandbox_multiple_inputs_and_external_grading(self):
        backend,sdk,sandbox,gate = fake_program()
        callback = AsyncMock()
        result = await backend.execute_program(SOURCE,CASES,on_record=callback)
        outcomes = runner.validate_program(result,SOURCE,CASES,"im-test")
        self.assertTrue(all(v["passed"] for v in outcomes.values()))
        self.assertEqual(sdk.Sandbox.create.aio.await_count,1)
        self.assertEqual(sandbox.exec.aio.await_count,3)
        self.assertEqual(callback.await_count,2)
        gate.assert_awaited_once()
        sandbox.terminate.aio.assert_awaited_once_with(wait=True)
        sandbox.detach.aio.assert_awaited_once()
        options = sdk.Sandbox.create.aio.call_args.kwargs
        self.assertEqual(options["timeout"],runner.lifetime_for(2))
        self.assertEqual(options["memory"],(256,256))
        self.assertTrue(options["block_network"])
        self.assertEqual(options["secrets"],[])
        self.assertEqual(options["volumes"],{})
        self.assertFalse(options["include_oidc_identity_token"])
        for call,case in zip(sandbox.exec.aio.call_args_list[1:],CASES):
            payload = json.loads(call.args[4])
            self.assertEqual(set(payload),{"source","input","limits"})
            self.assertEqual(payload["input"],case.arguments)
            self.assertEqual(call.kwargs["timeout"],8)

    async def test_different_programs_never_reuse_sandbox(self):
        backend,sdk,one,_ = fake_program()
        _,_,two,_ = fake_program()
        two.object_id = "sb-second"
        sdk.Sandbox.create.aio.side_effect = [one,two]
        a = await backend.execute_program(SOURCE,CASES)
        b = await backend.execute_program(SOURCE,CASES)
        self.assertNotEqual(a["metadata"]["sandbox_id"],b["metadata"]["sandbox_id"])
        self.assertEqual(sdk.Sandbox.create.aio.await_count,2)

    async def test_candidate_timeout_does_not_discard_remaining_inputs(self):
        backend,_,sandbox,_ = fake_program([envelope(CASES[0],returncode=-24),envelope(CASES[1])])
        result = await backend.execute_program(SOURCE,CASES)
        outcomes = runner.validate_program(result,SOURCE,CASES,"im-test")
        self.assertEqual(outcomes[CASES[0].input_hash]["status"],"timeout")
        self.assertIs(outcomes[CASES[0].input_hash]["passed"],False)
        self.assertIs(outcomes[CASES[1].input_hash]["passed"],True)
        self.assertEqual(sandbox.exec.aio.await_count,3)

    async def test_unknown_child_signal_keeps_unknown_not_zero(self):
        backend,_,_,_ = fake_program([envelope(CASES[0],returncode=-9),envelope(CASES[1])])
        result = await backend.execute_program(SOURCE,CASES)
        outcomes = runner.validate_program(result,SOURCE,CASES,"im-test")
        self.assertIsNone(outcomes[CASES[0].input_hash]["passed"])
        self.assertTrue(outcomes[CASES[1].input_hash]["passed"])

    async def test_missing_or_invalid_parent_report_stops_not_retries(self):
        backend,sdk,sandbox,_ = fake_program()
        sandbox.exec.aio.side_effect = [process(b'{"ready":true,"uid":0,"python":"3.12"}'),process(b"forged")]
        result = await backend.execute_program(SOURCE,CASES)
        outcomes = runner.validate_program(result,SOURCE,CASES,"im-test")
        self.assertTrue(all(v["passed"] is None for v in outcomes.values()))
        self.assertEqual(sdk.Sandbox.create.aio.await_count,1)
        self.assertEqual(sandbox.exec.aio.await_count,2)

    async def test_profile_bootstrap_failure_stops_subsequent_cases(self):
        backend,_,sandbox,_ = fake_program([envelope(CASES[0],candidate_started=False,returncode=1)])
        result = await backend.execute_program(SOURCE,CASES)
        self.assertEqual(sandbox.exec.aio.await_count,2)
        self.assertTrue(all(v["passed"] is None for v in runner.validate_program(result,SOURCE,CASES,"im-test").values()))

    async def test_incomplete_cleanup_does_not_produce_rewards(self):
        backend,_,sandbox,_ = fake_program()
        sandbox.terminate.aio.side_effect = RuntimeError("lost cleanup")
        result = await backend.execute_program(SOURCE,CASES)
        self.assertTrue(all(v["passed"] is None for v in runner.validate_program(result,SOURCE,CASES,"im-test").values()))

    async def test_identity_tampering_rejected(self):
        backend,_,_,_ = fake_program()
        result = await backend.execute_program(SOURCE,CASES)
        for field in ("profile","profile_hash","source_hash","runner_hash","image_id","reset"):
            bad = deepcopy(result)
            bad["metadata"][field] = "tampered"
            with self.subTest(field=field),self.assertRaises(ValueError):
                runner.validate_program(bad,SOURCE,CASES,"im-test")
        bad = deepcopy(result)
        bad["records"][CASES[0].input_hash]["status"] = "candidate_error"
        with self.assertRaises(ValueError): runner.validate_program(bad,SOURCE,CASES,"im-test")
        bad = deepcopy(result)
        bad["input_order"].reverse()
        with self.assertRaises(ValueError): runner.validate_program(bad,SOURCE,CASES,"im-test")

    async def test_candidate_stdout_cannot_be_its_own_verdict(self):
        backend,_,_,_ = fake_program([envelope(c,stdout_base64=base64.b64encode(b'{"reward":1}').decode()) for c in CASES])
        result = await backend.execute_program(SOURCE,CASES)
        self.assertTrue(all(v["passed"] is False for v in runner.validate_program(result,SOURCE,CASES,"im-test").values()))


if __name__ == "__main__":
    unittest.main()
