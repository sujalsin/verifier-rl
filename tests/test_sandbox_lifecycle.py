import asyncio
import base64
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from tests.test_modal_backend import fake_sdk, method, REQUEST, ServiceError
from verifier_rl.grading import Status
from verifier_rl.modal_backend import ModalBackend
from verifier_rl import sandbox_lifecycle as lifecycle


class TerminalTests(unittest.IsolatedAsyncioTestCase):
    def sandbox(self, code=None):
        return SimpleNamespace(object_id="sb-" + "a"*22, poll=method(code))

    def task(self, status=3):
        return {"task_id": "ta-fixture", "sdk_version": "1.5.5", "status": status, "exitcode": 0}

    async def test_public_poll_does_not_use_private_fallback(self):
        lookup = AsyncMock()
        sb = self.sandbox(137)
        result = await lifecycle.confirm_terminal(sb, lookup=lookup)
        self.assertEqual(result["method"], "Sandbox.poll")
        self.assertEqual(result["returncode"], 137)
        lookup.assert_not_awaited()

    async def test_task_result_confirms_termination_when_poll_has_no_result(self):
        sb = self.sandbox()
        lookup = AsyncMock(return_value=self.task())
        result = await lifecycle.confirm_terminal(sb, lookup=lookup)
        self.assertEqual(result["method"], "SandboxGetTaskId.task_result")
        self.assertEqual(result["status"], 3)
        lookup.assert_awaited_once_with(sb.object_id, 10)

    async def test_failed_poll_can_use_explicit_task_terminal_result(self):
        sb = self.sandbox()
        sb.poll.aio.side_effect = TimeoutError()
        result = await lifecycle.confirm_terminal(sb, lookup=AsyncMock(return_value=self.task()))
        self.assertEqual(result["poll_error"], "TimeoutError")

    async def test_missing_or_invalid_task_result_is_not_terminal(self):
        for record in (None, self.task(0), self.task(True), self.task(9),
                       dict(self.task(), sdk_version="different"), dict(self.task(), task_id="")):
            with self.subTest(record=record), self.assertRaises(ValueError):
                await lifecycle.confirm_terminal(self.sandbox(), lookup=AsyncMock(return_value=record))

    async def test_lookup_is_bounded(self):
        async def hangs(*args):
            await asyncio.Event().wait()
        with self.assertRaises(TimeoutError):
            await lifecycle.confirm_terminal(self.sandbox(), timeout=.01, lookup=hangs)

    async def test_terminal_receipt_cannot_be_rebound_or_invented(self):
        sb = self.sandbox(0)
        result = await lifecycle.confirm_terminal(sb)
        for update in ({"sandbox_id": "other"}, {"returncode": None}, {"returncode": True},
                       {"checked_at": float("nan")}, {"method": "active_list_empty"}):
            with self.assertRaises(ValueError):
                lifecycle.validate_terminal(dict(result, **update), sb.object_id)


class BackendRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_and_cleanup_errors_both_survive(self):
        sdk, sb = fake_sdk()
        sb.exec.aio.side_effect = TimeoutError()
        sb.terminate.aio.side_effect = ServiceError()
        with patch.object(lifecycle, "confirm_terminal", AsyncMock(side_effect=ValueError("still unknown"))):
            result = await ModalBackend("test", "im-test", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.status, Status.INFRASTRUCTURE_ERROR)
        self.assertEqual(result.detail, "cleanup_unconfirmed")
        self.assertEqual(result.metadata["cleanup_wait_error"], "ServiceError")
        self.assertEqual(result.metadata["transport_stage"], "preflight")
        self.assertEqual(result.metadata["pre_cleanup_result"]["detail"], "preflight:TimeoutError")
        self.assertIs(result.metadata["candidate_submission_attempted"], False)
        self.assertEqual(sb.exec.aio.await_count, 1)

    async def test_terminal_fallback_preserves_original_startup_failure(self):
        sdk, sb = fake_sdk()
        sb.exec.aio.side_effect = TimeoutError()
        sb.terminate.aio.side_effect = TimeoutError()
        receipt = {"version": lifecycle.VERSION, "method": "Sandbox.poll", "sandbox_id": sb.object_id,
                   "returncode": 137, "checked_at": 1234}
        with patch.object(lifecycle, "confirm_terminal", AsyncMock(return_value=receipt)):
            result = await ModalBackend("test", "im-test", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.detail, "preflight:TimeoutError")
        self.assertEqual(result.metadata["cleanup"], "terminated")
        self.assertEqual(result.metadata["cleanup_terminal_evidence"], receipt)
        self.assertEqual(sb.exec.aio.await_count, 1)

    async def test_lost_launch_ack_is_never_labelled_not_submitted(self):
        sdk, sb = fake_sdk()
        ready = sb.exec.aio.side_effect
        sb.exec.aio.side_effect = [next(ready), TimeoutError()]
        result = await ModalBackend("test", "im-test", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.detail, "launch:TimeoutError")
        self.assertIs(result.metadata["candidate_submission_attempted"], True)

    async def test_output_is_preserved_but_not_scored_when_cleanup_unknown(self):
        sdk, sb = fake_sdk()
        sb.terminate.aio.side_effect = TimeoutError()
        with patch.object(lifecycle, "confirm_terminal", AsyncMock(side_effect=ValueError("unknown"))):
            result = await ModalBackend("test", "im-test", sdk=sdk).execute(REQUEST)
        self.assertEqual(result.status, Status.INFRASTRUCTURE_ERROR)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(base64.b64decode(result.metadata["pre_cleanup_result"]["stdout_base64"]), b"[null]")
        self.assertIs(result.metadata["candidate_submission_attempted"], True)


if __name__ == "__main__":
    unittest.main()
