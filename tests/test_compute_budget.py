from decimal import Decimal
import unittest
from verifier_rl import compute_budget as budget

RATES={"cpu_hour_cost":".04730","mem_gib_hour_cost":".00800",
       "cpu_hour_cost_sandbox":".141900","mem_gib_hour_cost_sandbox":".024000","gpu_hour_cost_l40s":"1.95000"}


class BudgetTests(unittest.TestCase):
    def test_reserves_before_spend_and_cannot_exceed_total(self):
        state=budget.reserve(budget.initialize(),"first","200","a")
        self.assertEqual(budget.committed(state),Decimal("220"))
        with self.assertRaises(budget.BudgetReached):
            budget.reserve(state,"second","31","b")
        self.assertEqual(budget.committed(state),Decimal("220"))
        state=budget.settle(state,"first","50","a")
        state=budget.reserve(state,"second","100","b")
        self.assertEqual(budget.committed(state),Decimal("170"))

    def test_idempotency_identity_and_no_double_refund(self):
        state=budget.reserve(budget.initialize(),"first","10","id")
        self.assertEqual(budget.reserve(state,"first","10","id"),state)
        with self.assertRaises(ValueError):
            budget.reserve(state,"first","10","other")
        state=budget.settle(state,"first","3","id")
        self.assertEqual(budget.settle(state,"first","3","id"),state)
        with self.assertRaises(ValueError):
            budget.settle(state,"first","2","id")
        with self.assertRaises(ValueError):
            budget.settle(state,"first","11","id")

    def test_invalid_values_rejected(self):
        for value in ("NaN","Infinity",-1):
            with self.assertRaises(ValueError):
                budget.reserve(budget.initialize(),"x",value,"id")

    def test_rate_math_includes_gpu_cpu_memory_and_nonpreemptible_premium(self):
        self.assertEqual(budget.hourly(RATES,"gpu"),Decimal("2.30060"))
        self.assertEqual(budget.hourly(RATES,"cpu"),Decimal(".18990"))
        self.assertEqual(budget.hourly(RATES,"sandbox"),Decimal(".147900"))

    def test_lost_or_unclean_attempt_retains_full_lifetime_reservation(self):
        raw={"entries":{"a":{"input":{"intent-1":{},"intent-2":{},
            "attempt-1":{"metadata":{"total_seconds":1,"cleanup":"unknown"}},
            "attempt-2":{"metadata":{"total_seconds":2.2,"cleanup":"terminated"}}}}}}
        self.assertEqual(budget.sandbox_seconds(raw),Decimal(123))
        raw["entries"]["a"]["input"].pop("attempt-1")
        self.assertEqual(budget.sandbox_seconds(raw),Decimal(123))
        raw["entries"]["a"]["input"]["attempt-2"]["metadata"]["total_seconds"]=150
        self.assertEqual(budget.sandbox_seconds(raw),Decimal(240))

    def test_fixed_250_ceiling_in_parallel_reservation_order(self):
        state=budget.initialize()
        # Serial service interleaves reservations but does not release live holds.
        for i in range(23):
            state=budget.reserve(state,str(i),"10",str(i))
        self.assertEqual(budget.committed(state),Decimal(250))
        with self.assertRaises(budget.BudgetReached):
            budget.reserve(state,"24",".01","24")
