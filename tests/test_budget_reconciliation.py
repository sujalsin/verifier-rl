from copy import deepcopy
from decimal import Decimal
import itertools
import unittest

from verifier_rl import compute_budget as budget, budget_reconciliation as audit
from verifier_rl.parallel_evaluation import fingerprint

RATES = {"cpu_hour_cost": ".04730", "mem_gib_hour_cost": ".00800",
         "gpu_hour_cost_l40s": "1.95000"}


def proof():
    return {"identity": "id", "call_id": "fc-test", "app_id": "ap-old", "retries": 0, "sandbox_starts": 0,
            "lifecycle": {"app_id": "ap-old", "state": "APP_STATE_STOPPED", "stopped_at": 1000},
            "call_info": {"function_call_id": "fc-test", "created_at": 100,
                          "total_inputs": 1, "pending_inputs": {"total": 0},
                          "failed_inputs": {"total": 1, "latest": [{"started_at": 120,
                              "finished_at": 500, "task_startup_time": 10}]}}}


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.ledger = budget.reserve(budget.initialize(), "old", "10", "id")
        self.saved = {"binding": "fixed", "ledger": self.ledger}

    def test_report_never_calls_reservations_provider_spend(self):
        ledger = budget.settle(budget.reserve(self.ledger, "done", "8", "done"), "done", "2", "done")
        report = audit.describe(ledger)
        self.assertEqual(report["estimated_settled_compute_usd"], "2")
        self.assertEqual(report["outstanding_maximum_reservations_usd"], "10")
        self.assertEqual(report["guard_balance_usd"], "32")
        self.assertFalse(report["is_provider_billing"])

    def test_provider_bound_includes_queue_startup_and_margin(self):
        receipt = audit.completed_call_bound(self.ledger["items"]["old"], proof(), RATES, "gpu")
        self.assertEqual(receipt["charged_seconds"], "530")
        self.assertEqual(Decimal(receipt["estimated_compute_usd"]), budget.cost(RATES, "gpu", 530))

    def test_unknown_active_mismatched_or_partial_calls_stay_reserved(self):
        variants = []
        p = proof(); p["lifecycle"]["state"] = "APP_STATE_DETACHED"; variants.append(p)
        p = proof(); p["call_info"]["pending_inputs"]["total"] = 1; variants.append(p)
        p = proof(); p["call_info"]["failed_inputs"]["latest"] = []; variants.append(p)
        p = proof(); p["call_info"]["total_inputs"] = 2; variants.append(p)
        p = proof(); p["identity"] = "other"; variants.append(p)
        p = proof(); p["retries"] = 1; variants.append(p)
        p = proof(); p["sandbox_starts"] = 4; variants.append(p)
        p = proof(); p["call_info"]["failed_inputs"]["latest"][0]["finished_at"] = 0; variants.append(p)
        for p in variants:
            with self.subTest(proof=p), self.assertRaises(ValueError):
                audit.completed_call_bound(self.ledger["items"]["old"], p, RATES, "gpu")

    def test_reconcile_preserves_ceiling_binding_and_unrelated_items(self):
        evidence = {"old": proof()}
        receipt = audit.completed_call_bound(self.ledger["items"]["old"], proof(), RATES, "gpu")
        proposal = {"before_hash": fingerprint(self.saved), "rates": RATES, "items": {"old": receipt}}
        with self.assertRaisesRegex(ValueError, "live accounting"):
            audit.reconcile_offline(self.saved, proposal, evidence, writers_stopped=False)
        result = audit.reconcile_offline(self.saved, proposal, evidence, writers_stopped=True)
        self.assertEqual(result["binding"], "fixed")
        self.assertEqual(result["ledger"]["ceiling"], "250")
        self.assertEqual(result["ledger"]["overhead"], "20")
        self.assertEqual(result["ledger"]["items"]["old"]["maximum"], "10")
        self.assertIsNone(self.saved["ledger"]["items"]["old"]["actual"])
        with self.assertRaisesRegex(ValueError, "ledger changed"):
            audit.reconcile_offline(result, proposal, evidence, writers_stopped=True)
        corrupt = deepcopy(proposal); corrupt["items"]["old"]["estimated_compute_usd"] = "0"
        with self.assertRaisesRegex(ValueError, "receipt"):
            audit.reconcile_offline(self.saved, corrupt, evidence, writers_stopped=True)

    def test_admission_escrow_drains_existing_work_without_cancellation(self):
        # Enumerate every interleaving of the four active settlement/reserve
        # pairs. At least one new reservation fails; already admitted work is
        # still allowed to settle. The escrow is a hold, not compute spend.
        for order in itertools.permutations(range(4)):
            state = budget.initialize()
            for i in range(4):
                state = budget.reserve(state, f"active-{i}", ".72", str(i))
            amount = Decimal("250") - budget.committed(state)
            state = budget.reserve(state, "maintenance-escrow", str(amount), "escrow")
            stopped = False
            for i in order:
                state = budget.settle(state, f"active-{i}", ".12", str(i))
                if not stopped:
                    with self.assertRaises(budget.BudgetReached):
                        budget.reserve(state, "next", ".72", "next")
                    stopped = True
            self.assertTrue(stopped)
            state = budget.settle(state, "maintenance-escrow", "0", "escrow")
            self.assertEqual(budget.committed(state), Decimal("20.48"))

    def test_all_valid_concurrent_settle_reserve_orders_eventually_close_admission(self):
        operations = [(i, action) for i in range(4) for action in ("settle", "reserve")]
        orders_checked = 0
        for order in itertools.permutations(operations):
            if any(order.index((i, "settle")) > order.index((i, "reserve")) for i in range(4)):
                continue
            state = budget.initialize()
            for i in range(4):
                state = budget.reserve(state, f"active-{i}", ".72", str(i))
            state = budget.reserve(state, "escrow", str(Decimal("250") - budget.committed(state)), "e")
            denied = False
            for i, action in order:
                if action == "settle":
                    state = budget.settle(state, f"active-{i}", ".12", str(i))
                elif not denied:
                    try:
                        state = budget.reserve(state, f"next-{i}", ".72", str(i))
                    except budget.BudgetReached:
                        denied = True
            self.assertTrue(denied)
            self.assertTrue(all(state["items"][f"active-{i}"]["actual"] == ".12" or
                                Decimal(state["items"][f"active-{i}"]["actual"]) == Decimal(".12") for i in range(4)))
            self.assertLessEqual(budget.committed(state), 250)
            orders_checked += 1
        self.assertEqual(orders_checked, 2520)


if __name__ == "__main__":
    unittest.main()
