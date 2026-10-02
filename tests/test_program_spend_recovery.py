import base64
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from modal.exception import NotFoundError
from modal_proto import api_pb2

import modal_program_spend_recovery as recovery
from tests.test_program_execution import CASES, SOURCE, fake_program
from verifier_rl import booking_replication as study, program_grading as grading, program_execution as execution
from verifier_rl.evaluation_journal import ReconciliationRequired, persist
from verifier_rl.grading import ExecutionResult, Status
from verifier_rl.modal_backend import Limits
from verifier_rl.panel_execution import pack_result
from verifier_rl.suites import canonical_json, digest


class AppLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def client(self, state):
        response = api_pb2.AppGetLifecycleResponse(
            lifecycle=api_pb2.AppLifecycle(app_state=state, stopped_at=123))
        return SimpleNamespace(stub=SimpleNamespace(AppGetLifecycle=AsyncMock(return_value=response)))

    async def test_old_app_absent_from_recent_list_still_requires_terminal_proof(self):
        client = self.client(api_pb2.APP_STATE_STOPPED)
        receipt = await recovery.require_idle_apps([{"app_id": "current", "tasks": "0"}], client)
        self.assertEqual(receipt["app_id"], recovery.PARENT_APP)
        self.assertEqual(receipt["state"], "stopped")
        self.assertEqual(client.stub.AppGetLifecycle.call_args.args[0].app_id, recovery.PARENT_APP)
        self.assertEqual(client.stub.AppGetLifecycle.call_args.kwargs, {"timeout": 30, "retry": None})

    async def test_live_or_stopping_old_app_rejected(self):
        for state in (api_pb2.APP_STATE_DETACHED, api_pb2.APP_STATE_STOPPING, api_pb2.APP_STATE_INITIALIZING):
            with self.subTest(state=state), self.assertRaises(ReconciliationRequired):
                await recovery.require_idle_apps([], self.client(state))

    async def test_unknown_parent_is_not_assumed_stopped(self):
        client = self.client(api_pb2.APP_STATE_STOPPED)
        client.stub.AppGetLifecycle.side_effect = NotFoundError("missing")
        with self.assertRaises(NotFoundError):
            await recovery.require_idle_apps([], client)

    async def test_any_active_worker_blocks_maintenance(self):
        client = self.client(api_pb2.APP_STATE_STOPPED)
        with self.assertRaises(ReconciliationRequired):
            await recovery.require_idle_apps([{"app_id": "other", "tasks": "1"}], client)
        client.stub.AppGetLifecycle.assert_not_awaited()


def historical_record(case, sandbox_id):
    stdout = (str(case.expected) + "\n").encode()
    payload = canonical_json({"source": SOURCE, "input": case.arguments, "limits": asdict(Limits())})
    envelope = {"version": execution.VERSION, "payload_hash": digest(payload), "candidate_started": True,
                "returncode": 0, "user_cpu_seconds": .01, "system_cpu_seconds": .01, "max_rss_kib": 10000,
                "wall_seconds": .1, "enforced_limit": None, "stdout_base64": base64.b64encode(stdout).decode(),
                "stderr_base64": "", "stdout_truncated": False, "stderr_truncated": False}
    return pack_result(ExecutionResult(Status.COMPLETED, stdout, "", False,
        {"input_hash": case.input_hash, "source_hash": digest(SOURCE), "sandbox_id": sandbox_id,
         "parent_returncode": 0, "supervisor_report": envelope}))


def history_fixture():
    cases = study.pilot.cases_for("training")
    runtime = grading.binding("im-test", 100)
    history, samples, terminals = {}, [], {}
    for policy, step in recovery.JOBS:
        for index in range(4):
            sid = f"train-{policy}-{step:02d}-{index}"
            sample = {"sample_id": sid, "source": SOURCE}
            samples.append(sample)
            if step == 18:
                sandbox_id = recovery.OLD_SANDBOXES[index]
                records = {c.input_hash: historical_record(c, sandbox_id) for c in cases[:recovery.OLD_COUNTS[index]]}
                final = None
                terminals[sandbox_id] = {"version": "modal-terminal-evidence-0.1", "sandbox_id": sandbox_id,
                                        "method": "Sandbox.poll", "returncode": 137, "checked_at": 10}
            else:
                records = {c.input_hash: pack_result(ExecutionResult(Status.INFRASTRUCTURE_ERROR,
                    detail="program_not_executed", metadata={"input_hash": c.input_hash,
                    "source_hash": digest(SOURCE)})) for c in cases}
                document = {"metadata": {"candidate_submission_attempted": False,
                    "failure": {"stage": "create", "type": "ResourceExhaustedError",
                                "detail": "Workspace test has exceeded its spend limit"}},
                    "records": records, "input_order": [c.input_hash for c in cases]}
                final = grading.metadata_for(document)
            history[sid] = {"intent": grading.intent_for(sample, cases, runtime), "final": final, "records": records}
    return history, samples, cases, runtime, terminals


class HistoryTests(unittest.TestCase):
    def test_accepts_only_scoped_incident(self):
        args = history_fixture()
        self.assertEqual(recovery.validate_history(*args), sum(recovery.OLD_COUNTS))

    def test_extra_or_missing_program_rejected(self):
        args = history_fixture()
        args[0]["unrelated"] = {}
        with self.assertRaises(ValueError): recovery.validate_history(*args)

    def test_completed_program_cannot_be_retried(self):
        args = history_fixture()
        next(iter(args[0].values()))["final"] = {"already": "completed"}
        with self.assertRaises(ValueError): recovery.validate_history(*args)

    def test_unknown_or_running_old_sandbox_rejected(self):
        args = history_fixture()
        args[4][recovery.OLD_SANDBOXES[0]]["returncode"] = None
        with self.assertRaises(ValueError): recovery.validate_history(*args)

    def test_creation_error_must_be_explicit_spend_limit(self):
        args = history_fixture()
        item = args[0]["train-s20261014-endpoint_omission-10-0"]
        item["final"]["metadata"]["failure"]["type"] = "TimeoutError"
        with self.assertRaises(ValueError): recovery.validate_history(*args)

    def test_no_handle_must_precede_candidate_submission(self):
        args = history_fixture()
        item = args[0]["train-s20261014-endpoint_omission-10-0"]
        item["final"]["metadata"]["candidate_submission_attempted"] = True
        with self.assertRaises(ValueError): recovery.validate_history(*args)

    def test_arbitrary_unknown_signal_not_reinterpreted(self):
        args = history_fixture()
        record = next(iter(next(iter(args[0].values()))["records"].values()))
        record.update(status="infrastructure_error", detail="unattributed_child_signal")
        with self.assertRaises(ValueError): recovery.validate_history(*args)

    def test_tampered_supervisor_record_rejected(self):
        args = history_fixture()
        record = next(iter(next(iter(args[0].values()))["records"].values()))
        record["metadata"]["supervisor_report"]["payload_hash"] = "wrong"
        with self.assertRaises(ValueError): recovery.validate_history(*args)

    def test_history_source_identity_required(self):
        args = history_fixture()
        record = next(iter(next(iter(args[0].values()))["records"].values()))
        record["metadata"]["source_hash"] = "wrong"
        with self.assertRaises(ValueError): recovery.validate_history(*args)


class ReplacementTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_must_preserve_every_known_outcome(self):
        backend, _, _, _ = fake_program()
        document = await backend.execute_program(SOURCE, CASES)
        previous = {"records": deepcopy(document["records"])}
        self.assertEqual(recovery.validate_replacement(document, previous, {"source": SOURCE}, CASES, "im-test"), 2)
        document["metadata"]["reviewed_recovery"] = {"additional_attempt": 1}
        self.assertEqual(recovery.validate_replacement(document, previous, {"source": SOURCE}, CASES, "im-test"), 2)

    async def test_better_output_cannot_replace_known_wrong_answer(self):
        backend, _, _, _ = fake_program()
        document = await backend.execute_program(SOURCE, CASES)
        previous = {"records": deepcopy(document["records"])}
        first = previous["records"][CASES[0].input_hash]
        first["stdout_base64"] = base64.b64encode(b"987654321\n").decode()
        first["metadata"]["supervisor_report"]["stdout_base64"] = first["stdout_base64"]
        with self.assertRaises(ReconciliationRequired):
            recovery.validate_replacement(document, previous, {"source": SOURCE}, CASES, "im-test")

    async def test_new_unknown_outcome_cannot_be_published(self):
        backend, _, _, _ = fake_program()
        document = await backend.execute_program(SOURCE, CASES)
        document["metadata"]["cleanup"] = "unconfirmed"
        with self.assertRaises(ReconciliationRequired):
            recovery.validate_replacement(document, {"records": {}}, {"source": SOURCE}, CASES, "im-test")


class PreservationTests(unittest.TestCase):
    def test_sync_cli_bridge_uses_sdk_event_loop(self):
        from modal._utils.async_utils import synchronizer
        response = api_pb2.AppGetLifecycleResponse(
            lifecycle=api_pb2.AppLifecycle(app_state=api_pb2.APP_STATE_STOPPED))
        client = SimpleNamespace(stub=SimpleNamespace(AppGetLifecycle=AsyncMock(return_value=response)))
        receipt = synchronizer.create_blocking(recovery.require_idle_apps)([], client)
        self.assertEqual(receipt["state"], "stopped")
        client.stub.AppGetLifecycle.assert_awaited_once()

    def test_publication_archives_old_files_without_deleting_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, archive = root / "grading/known-batch", root / "recovery/original_files/known-batch"
            persist(source / "programs/old", {"result": {"failed": True}})
            recovery.archived_publish(source, archive, {"programs/new": {"result": {"recovered": True}}})
            self.assertEqual(json.loads((archive / "programs/old/result.json").read_text()), {"failed": True})
            self.assertEqual(json.loads((source / "programs/new/result.json").read_text()), {"recovered": True})
            with self.assertRaises(ReconciliationRequired): recovery.archived_publish(source, archive, {})

    def test_frozen_sources_and_original_deadline_required(self):
        sources = {"some.py": "# frozen"}
        context = {"plan": study.make_plan(Path(__file__).resolve().parents[1]), "source_snapshot": sources, "deadline": 100,
                   "program_runtime": grading.binding("im-test", 100),
                   "runtime_amendment": {"id": grading.AMENDMENT}}
        with patch.object(recovery, "SNAPSHOT_HASH", grading.fingerprint(sources)):
            recovery.validate_context(context, sources, 99)
            with self.assertRaises(ValueError): recovery.validate_context(context, sources, 100)
            with self.assertRaises(ValueError): recovery.validate_context(context, {"some.py": "changed"}, 99)
            bad = deepcopy(context)
            bad["deadline"] = 200
            with self.assertRaises(ValueError): recovery.validate_context(bad, sources, 99)


if __name__ == "__main__":
    unittest.main()
