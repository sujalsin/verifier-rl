from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import modal

import modal_budget_handoff as launch
from tests.test_budget_reconciliation import proof, RATES
from verifier_rl import compute_budget as budget
from verifier_rl.parallel_evaluation import fingerprint


class AppIdentityTests(unittest.TestCase):
    def test_pinned_sdk_hydrated_property_without_client_or_current_call(self):
        from modal.app import _App
        actual_app = _App("identity-unit-test")
        actual_app._app_id = "ap-hydrated-test"
        with patch.object(launch, "app", actual_app), patch.object(
                launch._Client, "from_env", side_effect=modal.exception.NotFoundError("missing call")), patch.object(
                modal, "current_function_call_id", side_effect=AssertionError("unavailable call context")):
            for _ in range(4):
                self.assertEqual(launch.current_app_id(), "ap-hydrated-test")

    def test_resume_qualification_freezes_science_runtime_and_source_bindings(self):
        old = {"name":launch.NAME, "runtime":{"deadline":123}, "plan":{}, "setup":{}, "rates":{},
               "budget_binding":"original", "scientific_sources":{}, "scheduler_sources":{}}
        config = deepcopy(old) | {"name":launch.RESUME_NAME, "previous_config":old,
            "accounting_revision":launch.IDENTITY_REVISION,
            "accounting_sources":{n:launch.digest(Path(n).read_text()) for n in launch.SOURCES}}
        with patch.object(launch.cloud, "validate_config"):
            launch.qualify(config)
            for field in ("runtime", "plan", "setup", "rates", "budget_binding", "scientific_sources", "scheduler_sources"):
                with self.subTest(field=field):
                    changed = deepcopy(config)
                    changed[field] = "changed"
                    with self.assertRaisesRegex(ValueError, "scientific settings"):
                        launch.qualify(changed)
            changed = deepcopy(config)
            changed["accounting_sources"] = {}
            with self.assertRaisesRegex(ValueError, "complete accounting source"):
                launch.qualify(changed)


class Store:
    def __init__(self, values):
        self.values = deepcopy(values)
        self.fail_after_ledger_write = False
        self.ledger_writes = 0

    def get(self, key, default=None):
        return deepcopy(self.values.get(key, default))

    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = deepcopy(value)
        if key == launch.RUN + "/budget":
            self.ledger_writes += 1
            if self.fail_after_ledger_write:
                self.fail_after_ledger_write = False
                raise OSError("injected ambiguous publication after durable transaction")
        return True


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = {"budget_binding":"binding", "rates":RATES}
        self.key = "program-sandbox-001/training/test/attempt-0"
        self.escrow_key = "parallel-evaluation-001/research-001/" + launch.ESCROW_TICKET
        ledger = budget.reserve(budget.initialize(), self.key, "10", "id")
        ledger = budget.reserve(ledger, "unrelated", "3", "unrelated")
        ledger = budget.reserve(ledger, self.escrow_key, "20", "escrow")
        self.store = Store({launch.RUN + "/budget":{"binding":"binding", "ledger":ledger}})
        self.authorization = {"evidence":{self.key:proof()}, "escrow":{
            "key":self.escrow_key, "item":ledger["items"][self.escrow_key],
            "purpose":"reviewed_admission_only_drain", "compute_allocated":False}}
        self.function = launch.reconciled_budget.get_raw_f()
        def blocked(fn):
            if fn is launch.lifecycle:
                return lambda app_id: {"app_id":app_id, "state":"APP_STATE_STOPPED", "stopped_at":1000}
            return lambda: "ap-successor"
        for target, value in (("qualify",lambda _:None), ("store",self.store),
                              ("app",SimpleNamespace(app_id="ap-successor")),
                              ("work",MagicMock()), ("archive",MagicMock()),
                              ("reconciled_budget",MagicMock(object_id="fu-successor"))):
            replacement = patch.object(launch, target, value)
            replacement.start(); self.addCleanup(replacement.stop)
        replacement = patch.object(launch.synchronizer, "create_blocking", side_effect=blocked)
        replacement.start(); self.addCleanup(replacement.stop)
        replacement = patch.object(launch.cloud, "paths", side_effect=lambda config, base="/artifacts":
                                   Path(self.tmp.name) / base.strip("/"))
        replacement.start(); self.addCleanup(replacement.stop)

    def test_ambiguous_publication_recovers_without_double_release(self):
        self.store.fail_after_ledger_write = True
        with self.assertRaises(OSError):
            self.function(self.config, "initialize", self.authorization)
        first = self.store.get(launch.RUN + "/budget")
        self.assertTrue((Path(self.tmp.name) / "artifacts/accounting/transaction.json").exists())
        self.assertTrue((Path(self.tmp.name) / "evidence/accounting/transaction.json").exists())
        result = self.function(self.config, "initialize", self.authorization)
        self.assertEqual(self.store.get(launch.RUN + "/budget"), first)
        self.assertEqual(first["ledger"]["ceiling"], "250")
        self.assertEqual(first["ledger"]["items"][self.escrow_key]["actual"], "0")
        self.assertIsNone(first["ledger"]["items"]["unrelated"]["actual"])
        self.assertGreater(float(result["verified_hold_reduction_usd"]), 9)
        writes = self.store.ledger_writes
        self.assertEqual(self.function(self.config, "initialize", self.authorization), result)
        self.assertEqual(self.store.ledger_writes, writes)

    def test_transaction_refuses_unrelated_intervening_ledger_update(self):
        self.store.fail_after_ledger_write = True
        with self.assertRaises(OSError):
            self.function(self.config, "initialize", self.authorization)
        changed = self.store.values[launch.RUN + "/budget"]
        changed["ledger"] = budget.reserve(changed["ledger"], "intervening", "1", "x")
        with self.assertRaisesRegex(ValueError, "durable transaction"):
            self.function(self.config, "initialize", self.authorization)

    def test_never_writes_while_previous_application_active(self):
        with patch.object(launch.synchronizer, "create_blocking", return_value=lambda _: {"state":"APP_STATE_DETACHED"}):
            with self.assertRaisesRegex(ValueError, "previous writer"):
                self.function(self.config, "initialize", self.authorization)
        self.assertEqual(self.store.ledger_writes, 0)

    def test_new_owner_cannot_replace_another_live_successor(self):
        self.store.put(launch.PREFIX + "/budget-owner", {"function_id":"fu-other", "app_id":"ap-other",
                       "config_hash":fingerprint(self.config)})
        def blocked(fn):
            if fn is launch.lifecycle:
                return lambda app_id: {"state":"APP_STATE_STOPPED" if app_id == launch.OLD_APP else "APP_STATE_DETACHED"}
            return lambda: "ap-successor"
        with patch.object(launch.synchronizer, "create_blocking", side_effect=blocked):
            with self.assertRaisesRegex(ValueError, "live successor"):
                self.function(self.config, "initialize", self.authorization)
        self.assertEqual(self.store.ledger_writes, 0)

    def test_status_is_read_only_and_distinguishes_holds_from_cost(self):
        self.function(self.config, "initialize", self.authorization)
        writes = self.store.ledger_writes
        result = self.function(self.config, "status", {})
        self.assertEqual(self.store.ledger_writes, writes)
        self.assertFalse(result["is_provider_billing"])
        self.assertEqual(result["outstanding_maximum_reservations_usd"], "3")

    def test_real_identity_helper_never_looks_up_function_call_metadata(self):
        with patch.object(launch._Client, "from_env", side_effect=modal.exception.NotFoundError("missing call")) as lookup:
            with patch.object(modal, "current_function_call_id", side_effect=AssertionError("not needed")):
                self.function(self.config, "initialize", self.authorization)
                for _ in range(4):
                    self.function(self.config, "status", {})
            lookup.assert_not_called()
        self.assertEqual(self.store.get(launch.PREFIX + "/budget-owner")["app_id"], "ap-successor")

    def test_unknown_local_identity_does_not_guess_or_write(self):
        with patch.object(launch, "app", SimpleNamespace(app_id=None)):
            with self.assertRaisesRegex(ValueError, "identity is unavailable"):
                self.function(self.config, "initialize", self.authorization)
        self.assertIsNone(self.store.get(launch.PREFIX + "/budget-owner"))
        self.assertEqual(self.store.ledger_writes, 0)

    def resume_config(self):
        old = self.config | {"name":launch.NAME}
        self.store.put(launch.namespace(old) + "/accounting-ready", {
            "config_hash":fingerprint(old), "transaction_hash":"retained-reconciliation"})
        self.store.put(launch.namespace(old) + "/budget-owner", {
            "function_id":"fu-old", "config_hash":fingerprint(old), "app_id":launch.RESUME_APP})
        config = old | {"name":launch.RESUME_NAME, "previous_config":old,
                        "accounting_revision":launch.IDENTITY_REVISION}
        payload = {"revision":launch.IDENTITY_REVISION, "previous_config_hash":fingerprint(old)}
        return config, payload

    def test_identity_resume_preserves_entire_ledger_and_old_namespace(self):
        config, payload = self.resume_config()
        before = self.store.get(launch.RUN + "/budget")
        old_owner = self.store.get(launch.PREFIX + "/budget-owner")
        ready = self.function(config, "initialize", payload)
        self.assertEqual(ready["verified_hold_reduction_usd"], "0")
        self.assertEqual(self.store.get(launch.RUN + "/budget"), before)
        self.function(config, "reserve", {"ticket":"next", "amount":"1", "identity":"next"})
        after = self.store.get(launch.RUN + "/budget")
        for key, value in before["ledger"]["items"].items():
            self.assertEqual(after["ledger"]["items"][key], value)
        self.assertIn("parallel-evaluation-001/research-003/next", after["ledger"]["items"])
        self.assertEqual(after["ledger"]["ceiling"], "250")
        self.assertEqual(self.store.get(launch.PREFIX + "/budget-owner"), old_owner)

    def test_identity_resume_rejects_active_failed_app_and_wrong_predecessor(self):
        config, payload = self.resume_config()
        with patch.object(launch.synchronizer, "create_blocking", return_value=lambda _: {"state":"APP_STATE_DETACHED"}):
            with self.assertRaisesRegex(ValueError, "previous writer"):
                self.function(config, "initialize", payload)
        self.store.values[launch.PREFIX + "/budget-owner"]["app_id"] = "ap-unexpected"
        with self.assertRaisesRegex(ValueError, "predecessor identity"):
            self.function(config, "initialize", payload)
        self.assertEqual(self.store.ledger_writes, 0)

    def test_identity_resume_ambiguous_publication_is_idempotent(self):
        config, payload = self.resume_config()
        before = self.store.get(launch.RUN + "/budget")
        self.store.fail_after_ledger_write = True
        with self.assertRaises(OSError):
            self.function(config, "initialize", payload)
        self.function(config, "initialize", payload)
        self.assertEqual(self.store.get(launch.RUN + "/budget"), before)

    def test_owner_transfer_requires_matching_config_and_stopped_app(self):
        config, payload = self.resume_config()
        self.function(config, "initialize", payload)
        owner_key = launch.namespace(config) + "/budget-owner"
        old = self.store.get(owner_key)
        with patch.object(launch, "app", SimpleNamespace(app_id="ap-new")):
            with patch.object(launch.synchronizer, "create_blocking", return_value=lambda aid: {
                    "state":"APP_STATE_STOPPED" if aid == launch.RESUME_APP else "APP_STATE_DETACHED"}):
                with self.assertRaisesRegex(ValueError, "live successor"):
                    self.function(config, "status", {})
            self.assertEqual(self.store.get(owner_key), old)
            self.function(config, "status", {})
            self.assertEqual(self.store.get(owner_key)["app_id"], "ap-new")


class ResumeInventoryTests(unittest.TestCase):
    def setUp(self):
        from tests.test_parallel_handoff import PartitionTests
        fixture = PartitionTests()
        fixture.setUp()
        legacy = fixture.old_result()
        self.manifest = fixture.manifest
        docs = {sid:v["result"] for sid,v in legacy["raw"]["programs"].items()}
        self.raw = {"manifest":self.manifest, "documents":docs}
        cases = launch.cloud.study.pilot.cases_for("evaluation")
        outcomes = [launch.parallel.checked_program(s, docs[s["sample_id"]], cases, "im-test")
                    for s in self.manifest["samples"]]
        self.receipt = {"raw_hash":fingerprint(self.raw), "rows":legacy["checked"]["rows"],
                        "job_hash":fingerprint(self.manifest), "outcome_hashes":[fingerprint(o) for o in outcomes]}
        self.new = deepcopy(self.manifest)
        self.new["key"] = "never-started"
        self.state = launch.parallel.initialize(self.manifest["runtime"], [self.manifest, self.new])
        self.state["owners"][self.manifest["key"]] = "owner"
        self.state["batches"][self.manifest["key"]] = {k:self.receipt[k] for k in ("raw_hash", "job_hash", "outcome_hashes")}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        launch.persist(self.root, {"retained":{"value":"unchanged"}})
        launch.persist(self.root / "grading" / self.manifest["key"], {"raw":self.raw, "receipt":self.receipt})
        self.snapshot = {"completed":{"retained":{"source_path":str(self.root / "retained.json"),
            "source_hash":fingerprint({"value":"unchanged"}), "rows":[], "sandbox_ids":[]}},
            "missing":[self.manifest, self.new], "baseline":{}, "saved_arms":{}}

    def check(self):
        return launch.resume_inventory(self.snapshot, self.state, self.root, total=3)

    def test_completed_failures_preserved_and_only_never_started_work_selected(self):
        result = self.check()
        self.assertEqual(set(result["completed"]), {"retained", self.manifest["key"]})
        self.assertEqual(result["missing"], [self.new])
        self.assertEqual(result["completed"][self.manifest["key"]]["rows"], self.receipt["rows"])

    def test_claimed_partial_batch_is_not_retried(self):
        self.state["owners"][self.new["key"]] = "partial"
        with self.assertRaisesRegex(ValueError, "partial or unknown"):
            self.check()

    def test_unknown_stop_and_provider_intent_block_resume(self):
        for field, value in (("unknown", ["uncertain"]), ("stop", "cleanup"),
                             ("permits", {self.new["key"] + "/sample":{}}),
                             ("programs", {self.new["key"] + "/sample":{}})):
            with self.subTest(field=field):
                previous = self.state[field]
                self.state[field] = value
                with self.assertRaises(ValueError):
                    self.check()
                self.state[field] = previous

    def test_unpublished_disk_evidence_is_not_reexecuted(self):
        launch.persist(self.root / "grading" / self.new["key"], {"intent":{"started":True}})
        with self.assertRaisesRegex(ValueError, "unpublished"):
            self.check()

    def test_changed_manifest_and_published_receipt_are_rejected(self):
        self.state["allowed"][self.new["key"]]["identity"] = "changed"
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.check()
        self.state["allowed"][self.new["key"]]["identity"] = fingerprint(self.new)
        self.state["batches"][self.manifest["key"]]["raw_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "published batch"):
            self.check()

    def test_changed_retained_evidence_and_duplicate_partition_are_rejected(self):
        self.snapshot["completed"]["retained"]["source_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "inherited raw"):
            self.check()
        self.snapshot["missing"].append(self.new)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check()


if __name__ == "__main__":
    unittest.main()
