import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from verifier_rl.grading import ExecutionRequest, MAX_OUTPUT_BYTES, Status
from verifier_rl.modal_backend import (BOOTSTRAP, Limits, ModalBackend, OutputLimitError,
                                       PREFLIGHT, RUNNER, read_bounded)


async def stream(*chunks):
    for chunk in chunks:
        yield chunk


def method(result=None, error=None):
    return SimpleNamespace(aio=AsyncMock(return_value=result, side_effect=error))


def process(output=b"[null]", stderr=b"", code=0, error=None):
    return SimpleNamespace(stdout=stream(output), stderr=stream(stderr),
                           wait=method(code, error))


class ExecTimeoutError(Exception): pass
class ServiceError(Exception): pass
class AuthError(Exception): pass


def fake_sdk(candidate=None, preflight=None):
    ready = preflight or process(b'{"ready":true,"uid":65534,"python":"3.12.test"}')
    sandbox = SimpleNamespace(object_id="sb-test", terminate=method(), detach=method())
    sandbox.exec = SimpleNamespace(aio=AsyncMock(side_effect=[ready, candidate or process()]))
    sdk = SimpleNamespace(
        __version__="test-double",
        App=SimpleNamespace(lookup=method("app-test")),
        Image=SimpleNamespace(from_id=Mock(return_value="image-test")),
        Sandbox=SimpleNamespace(create=method(sandbox)),
        exception=SimpleNamespace(ExecTimeoutError=ExecTimeoutError, ServiceError=ServiceError,
                                  AuthError=AuthError),
    )
    return sdk, sandbox


REQUEST = ExecutionRequest("def simulate_cache(operations): return [None]",
                           '[{"key":"a","op":"get","time":0}]')


class ModalContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_permissions_payload_and_cleanup(self):
        sdk, sandbox = fake_sdk()
        backend = ModalBackend("test", "im-approved", sdk=sdk)
        result = await backend.execute(REQUEST)
        self.assertEqual(result.status, Status.COMPLETED)
        self.assertEqual(result.stdout, b"[null]")
        self.assertEqual(result.metadata["stdout_preview"], "[null]")
        self.assertEqual(result.metadata["stdout_bytes"], 6)
        self.assertEqual(len(result.metadata["stdout_sha256"]), 64)
        self.assertEqual(result.metadata["runner_stage"], "unknown")
        kwargs = sdk.Sandbox.create.aio.call_args.kwargs
        self.assertTrue(kwargs["block_network"])
        self.assertEqual(kwargs["cpu"], (1.0, 1.0))
        self.assertEqual(kwargs["memory"], (256, 256))
        self.assertEqual(kwargs["volumes"], {})
        self.assertEqual(kwargs["secrets"], [])
        self.assertFalse(kwargs["include_oidc_identity_token"])
        sdk.App.lookup.aio.assert_awaited_once_with("test", create_if_missing=False)
        calls = sandbox.exec.aio.call_args_list
        self.assertEqual(calls[0].args[3], PREFLIGHT)
        payload = json.loads(calls[1].args[4])
        self.assertEqual(set(payload), {"source", "input", "limits"})
        self.assertEqual(payload["source"], REQUEST.source)
        self.assertEqual(calls[1].kwargs, {"timeout": 5, "text": False})
        sandbox.terminate.aio.assert_awaited_once_with(wait=True)
        sandbox.detach.aio.assert_awaited_once()
        self.assertEqual(result.metadata["cleanup"], "terminated")
        self.assertIn("runner_hash", result.metadata)

    async def test_fresh_sandbox_for_every_invocation(self):
        sdk, first = fake_sdk()
        _, second = fake_sdk()
        sdk.Sandbox.create.aio.side_effect = [first, second]
        backend = ModalBackend("test", "im-approved", sdk=sdk)
        await backend.execute(REQUEST)
        await backend.execute(REQUEST)
        self.assertEqual(sdk.Sandbox.create.aio.await_count, 2)
        first.terminate.aio.assert_awaited_once()
        second.terminate.aio.assert_awaited_once()

    async def test_creation_pacing_serializes_concurrent_starts(self):
        sdk, _ = fake_sdk()
        started = []
        async def create(*args, **kwargs):
            started.append(asyncio.get_running_loop().time())
            _, sandbox = fake_sdk()
            return sandbox
        sdk.Sandbox.create.aio.side_effect = create
        backend = ModalBackend("test", "im-approved", sdk=sdk, creation_interval_seconds=.025)
        results = await asyncio.gather(*(backend.execute(REQUEST) for _ in range(3)))
        self.assertTrue(all(r.status == Status.COMPLETED for r in results))
        self.assertTrue(all(b - a >= .024 for a, b in zip(started, started[1:])))
        self.assertEqual(results[0].metadata["creation_interval_seconds"], .025)

    def test_creation_pacing_rejects_invalid_intervals(self):
        for interval in (-1, float("nan"), float("inf"), True, "1", 6):
            with self.assertRaises(ValueError):
                ModalBackend("test", "im-approved", creation_interval_seconds=interval)

    async def test_remote_timeout_not_infrastructure_timeout(self):
        sdk, sandbox = fake_sdk(process(error=ExecTimeoutError()))
        result = await ModalBackend("test", "im-approved", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.status, Status.TIMEOUT)
        self.assertFalse(result.retryable)
        sandbox.terminate.aio.assert_awaited_once()

    async def test_nonzero_exit_is_candidate_error(self):
        sdk, sandbox = fake_sdk(process(code=1))
        result = await ModalBackend("test", "im-approved", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.status, Status.CANDIDATE_ERROR)
        sandbox.terminate.aio.assert_awaited_once()

    async def test_failure_stage_is_recorded_as_untrusted_diagnostic(self):
        stderr = b"\x0a".join((b"VERIFIER_RL_STAGE=function_call", b"Traceback (most recent call last):",
                                b"RuntimeError: fixture")) + b"\x0a"
        sdk, _ = fake_sdk(process(b"", stderr=stderr, code=1))
        result = await ModalBackend("test", "im-approved", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.metadata["runner_stage"], "function_call")
        self.assertIn("RuntimeError: fixture", result.metadata["stderr_preview"])
        self.assertEqual(result.status, Status.CANDIDATE_ERROR)

    async def test_sdk_timeout_sentinel_is_timeout(self):
        sdk, sandbox = fake_sdk(process(code=-1))
        result = await ModalBackend("test", "im-approved", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.status, Status.TIMEOUT)
        self.assertEqual(result.metadata["returncode"], -1)
        sandbox.terminate.aio.assert_awaited_once()

    async def test_both_streams_are_bounded(self):
        for output, stderr in ((b"x"*(MAX_OUTPUT_BYTES+1), b""),
                               (b"[]", b"x"*(MAX_OUTPUT_BYTES+1))):
            sdk, sandbox = fake_sdk(process(output, stderr))
            result = await ModalBackend("test", "im-approved", sdk=sdk).execute(REQUEST)
            self.assertEqual(result.status, Status.OUTPUT_LIMIT)
            sandbox.terminate.aio.assert_awaited_once()

    async def test_preflight_failure_is_unscored(self):
        sdk, sandbox = fake_sdk(preflight=process(b"not ready", code=1))
        result = await ModalBackend("test", "im-approved", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.status, Status.INFRASTRUCTURE_ERROR)
        self.assertEqual(sandbox.exec.aio.await_count, 1)
        sandbox.terminate.aio.assert_awaited_once()

    async def test_preflight_deadline_never_launches_candidate(self):
        sdk, sandbox = fake_sdk(preflight=process(b"", code=-1))
        result = await ModalBackend("test", "im-approved", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.status, Status.INFRASTRUCTURE_ERROR)
        self.assertEqual(result.detail, "preflight:TimeoutError")
        self.assertEqual(result.metadata["preflight_returncode"], -1)
        self.assertNotIn("returncode", result.metadata)
        self.assertEqual(result.metadata["cleanup"], "terminated")
        self.assertEqual(sandbox.exec.aio.await_count, 1)
        sandbox.terminate.aio.assert_awaited_once()

    async def test_transient_api_error_but_not_auth_is_retryable(self):
        for error, retryable in ((ServiceError(), True), (AuthError(), False), (TimeoutError(), False)):
            sdk, _ = fake_sdk()
            sdk.Sandbox.create.aio.side_effect = error
            result = await ModalBackend("test", "im-approved", sdk=sdk).execute(REQUEST)
            self.assertEqual(result.status, Status.INFRASTRUCTURE_ERROR)
            self.assertEqual(result.retryable, retryable)

    async def test_cleanup_failure_overrides_apparent_success(self):
        sdk, sandbox = fake_sdk()
        sandbox.terminate.aio.side_effect = ServiceError()
        result = await ModalBackend("test", "im-approved", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.status, Status.INFRASTRUCTURE_ERROR)
        self.assertEqual(result.detail, "cleanup_unconfirmed")
        self.assertFalse(result.retryable)
        sandbox.detach.aio.assert_awaited_once()

    async def test_cancelled_execution_still_terminates(self):
        async def wait_forever():
            await asyncio.Event().wait()
        proc = process()
        proc.wait = method(error=wait_forever)
        sdk, sandbox = fake_sdk(proc)
        task = asyncio.create_task(ModalBackend("test", "im-approved", sdk=sdk).execute(REQUEST))
        for _ in range(100):
            if proc.wait.aio.await_count:
                break
            await asyncio.sleep(.001)
        self.assertTrue(proc.wait.aio.await_count)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        sandbox.terminate.aio.assert_awaited_once()

    async def test_read_limit_counts_chunks_not_lines(self):
        with self.assertRaises(OutputLimitError):
            await read_bounded(stream(b"123", b"456"), limit=5)
        self.assertEqual(await read_bounded(stream(b"12", b"3"), limit=3), b"123")

    def test_runner_compiles_but_is_never_executed_locally(self):
        compile(PREFLIGHT, "<preflight>", "exec")
        compile(RUNNER, "<runner>", "exec")
        self.assertIn("os.setuid(65534)", BOOTSTRAP)
        self.assertIn("resource.RLIMIT_NPROC", BOOTSTRAP)
        self.assertNotIn("expected", RUNNER)

    def test_configuration_guardrails(self):
        for image in ("python:latest", "", "im-../bad"):
            with self.assertRaises(ValueError):
                ModalBackend("test", image)
        with self.assertRaises(ValueError):
            Limits(cpu_seconds=10, wall_seconds=2)
