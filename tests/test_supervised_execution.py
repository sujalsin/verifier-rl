import ast
import base64
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest

from tests.test_modal_backend import fake_sdk, process
from verifier_rl import supervised_execution as runner, booking_verifier_v2 as verifier
from verifier_rl.grading import Status, MAX_OUTPUT_BYTES
from verifier_rl.modal_backend import Limits
from verifier_rl.panel_execution import pack_result, request_for, require_evidence as legacy_evidence, unpack_result
from verifier_rl.suites import canonical_json, digest
from verifier_rl.task_panel import BOOKING

CASE = verifier.cases_for("training")[0]
SOURCE = "def required_capacity(bookings): return 0"


def envelope(source=SOURCE, case=CASE, **updates):
    payload = canonical_json({"source": source, "input": case.arguments, "limits": asdict(Limits())})
    value = dict(version=runner.VERSION, payload_hash=digest(payload), candidate_started=True,
        returncode=0, user_cpu_seconds=.01, system_cpu_seconds=.01, max_rss_kib=10000,
        wall_seconds=.1, enforced_limit=None, stdout_base64=base64.b64encode(b"0\n").decode(),
        stderr_base64="", stdout_truncated=False, stderr_truncated=False)
    return dict(value, **updates)


async def simulated_result(value=None, source=SOURCE, case=CASE, *, parent_code=0, preflight=None):
    value = value or envelope(source, case)
    sdk, sandbox = fake_sdk(process((canonical_json(value) + "\n").encode(), code=parent_code), preflight)
    sdk.__version__ = "1.5.5"
    backend = runner.SupervisedPanelBackend("test", "im-test", BOOKING, sdk=sdk, creation_interval_seconds=.26)
    return await backend.execute(request_for(source, case)), sdk, sandbox


class EnvelopeTests(unittest.TestCase):
    def test_runner_compiles_but_is_not_executed_locally(self):
        source = runner.supervised_runner(BOOKING)
        tree = ast.parse(source)
        child = ast.literal_eval(tree.body[0].value)
        compile(source, "remote-supervisor", "exec")
        compile(child, "remote-child", "exec")
        self.assertIn('(resource.RLIMIT_CPU, limits["cpu_seconds"]),', child)
        self.assertIn('resource.setrlimit(name, (value, value))', child)
        self.assertNotIn('limits["cpu_seconds"] + 1', child)
        self.assertLess(child.index('prepare(payload["limits"])'), child.index('os.write(ready_fd'))
        self.assertLess(child.index('os.close(ready_fd)'), child.index('exec(compile(payload["source"]'))
        self.assertNotIn('exec(compile(payload["source"]', runner.PARENT)
        self.assertIn('start_new_session=True', runner.PARENT)
        self.assertIn('os.wait4', runner.PARENT)
        self.assertIn('time.sleep(.05)', runner.PARENT)

    def test_classification_uses_parent_evidence_not_137_heuristic(self):
        for updates, status, detail in (
            ({}, Status.COMPLETED, ""),
            ({"returncode": 1}, Status.CANDIDATE_ERROR, "child_nonzero_exit"),
            ({"returncode": 137}, Status.CANDIDATE_ERROR, "child_nonzero_exit"),
            ({"returncode": -24}, Status.TIMEOUT, "supervisor_cpu_limit_or_sigxcpu"),
            ({"returncode": -9, "user_cpu_seconds": 2.1}, Status.TIMEOUT, "supervisor_cpu_limit_or_sigxcpu"),
            ({"returncode": 0, "user_cpu_seconds": 2.1}, Status.TIMEOUT, "supervisor_cpu_limit_or_sigxcpu"),
            ({"returncode": -9}, Status.INFRASTRUCTURE_ERROR, "unattributed_child_signal"),
            ({"candidate_started": False}, Status.INFRASTRUCTURE_ERROR, "child_bootstrap_failed"),
            ({"returncode": -9, "enforced_limit": "wall", "wall_seconds": 5.01}, Status.TIMEOUT, "supervisor_wall_limit"),
            ({"enforced_limit": "output", "stdout_truncated": True}, Status.OUTPUT_LIMIT, "supervisor_output_limit"),
        ):
            with self.subTest(updates=updates):
                value = envelope(**updates)
                self.assertEqual(runner.inspect_envelope(value, value["payload_hash"])[:2], (status, detail))

    def test_malformed_or_unsubstantiated_reports_rejected(self):
        for updates in ({"returncode": True}, {"returncode": -100}, {"candidate_started": 1},
                        {"user_cpu_seconds": float("nan")}, {"wall_seconds": -1}, {"max_rss_kib": True},
                        {"enforced_limit": "wall"}, {"enforced_limit": "output"}, {"stdout_truncated": True},
                        {"stdout_base64": "!"}, {"stdout_base64": base64.b64encode(b"x"*(MAX_OUTPUT_BYTES+1)).decode()},
                        {"version": "old"}, {"payload_hash": "bad"}, {"extra": 1}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                runner.inspect_envelope(envelope(**updates), envelope()["payload_hash"])

    def test_candidate_printing_success_envelope_is_only_output_data(self):
        fake = canonical_json(envelope()).encode()
        value = envelope(stdout_base64=base64.b64encode(fake).decode(), returncode=1)
        status, _, stdout, _ = runner.inspect_envelope(value, value["payload_hash"])
        self.assertEqual(status, Status.CANDIDATE_ERROR)
        self.assertEqual(stdout, fake)


class SupervisedBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_contract_and_evidence_roundtrip(self):
        result, sdk, sandbox = await simulated_result()
        self.assertEqual(result.stdout, b"0\n")
        self.assertEqual(runner.require_evidence(result, BOOKING, "im-test", source=SOURCE, case=CASE), "sb-test")
        self.assertEqual(unpack_result(pack_result(result)), result)
        calls = sandbox.exec.aio.call_args_list
        self.assertEqual(calls[0].kwargs["timeout"], 5)
        self.assertEqual(calls[1].kwargs["timeout"], 8)
        self.assertEqual(calls[1].args[3], runner.supervised_runner(BOOKING))
        options = sdk.Sandbox.create.aio.call_args.kwargs
        self.assertEqual(options["memory"], (256, 256))
        self.assertEqual(options["volumes"], {})
        self.assertEqual(options["secrets"], [])
        self.assertTrue(options["block_network"])
        self.assertFalse(options["include_oidc_identity_token"])
        sandbox.terminate.aio.assert_awaited_once_with(wait=True)

    async def test_confirmed_timeout_is_gradable_only_with_new_version(self):
        result, _, _ = await simulated_result(envelope(returncode=-24))
        runner.require_evidence(result, BOOKING, "im-test", source=SOURCE, case=CASE)
        with self.assertRaises(ValueError):
            legacy_evidence(result, BOOKING, "im-test", source=SOURCE, case=CASE)

    async def test_unknown_signal_stays_unscored_and_not_retryable(self):
        result, _, _ = await simulated_result(envelope(returncode=-9))
        runner.validate_report(result, BOOKING, "im-test", source=SOURCE, case=CASE)
        with self.assertRaisesRegex(ValueError, "unresolved supervised"):
            runner.require_evidence(result, BOOKING, "im-test", source=SOURCE, case=CASE)
        self.assertFalse(runner.startup_retry_allowed(result, request_for(SOURCE, CASE), "im-test", BOOKING))

    async def test_parent_137_or_sdk_timeout_is_not_candidate_failure(self):
        for code in (137, -1, 1):
            result, _, _ = await simulated_result(parent_code=code)
            self.assertEqual(result.status, Status.INFRASTRUCTURE_ERROR)
            self.assertEqual(result.stdout, b"")
            self.assertFalse(runner.startup_retry_allowed(result, request_for(SOURCE, CASE), "im-test", BOOKING))

    async def test_only_reviewed_pre_candidate_failure_can_retry(self):
        result, _, sandbox = await simulated_result(preflight=process(b"", code=137))
        self.assertEqual(sandbox.exec.aio.await_count, 1)
        self.assertTrue(runner.startup_retry_allowed(result, request_for(SOURCE, CASE), "im-test", BOOKING))

    async def test_tampered_outcome_or_binding_rejected(self):
        result, _, _ = await simulated_result()
        for field, value in (("source_hash", "changed"), ("input_hash", "changed"), ("cleanup", "unconfirmed"),
                             ("supervisor_stdout_sha256", "changed"), ("returncode", 137), ("stdout_sha256", "changed")):
            altered = replace(result, metadata=dict(result.metadata, **{field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.require_evidence(altered, BOOKING, "im-test", source=SOURCE, case=CASE)
        altered = deepcopy(result.metadata)
        altered["supervisor_report"]["returncode"] = 1
        with self.assertRaises(ValueError):
            runner.require_evidence(replace(result, metadata=altered), BOOKING, "im-test", source=SOURCE, case=CASE)

    async def test_bounded_candidate_streams_can_exceed_old_transport_cap(self):
        encoded = base64.b64encode(b"x" * MAX_OUTPUT_BYTES).decode()
        value = envelope(stdout_base64=encoded, stderr_base64=encoded, enforced_limit="output", stdout_truncated=True)
        result, _, _ = await simulated_result(value)
        self.assertEqual(result.status, Status.OUTPUT_LIMIT)
        runner.require_evidence(result, BOOKING, "im-test", source=SOURCE, case=CASE)
        self.assertEqual(len(unpack_result(pack_result(result)).stdout), MAX_OUTPUT_BYTES)

    async def test_wrong_report_payload_is_unscored(self):
        result, _, _ = await simulated_result(envelope(payload_hash="forged"))
        self.assertEqual(result.status, Status.INFRASTRUCTURE_ERROR)
        self.assertIn("invalid_supervisor_report", result.detail)


if __name__ == "__main__":
    unittest.main()
