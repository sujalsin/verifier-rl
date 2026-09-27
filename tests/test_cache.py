import unittest

from verifier_rl.cache import checked_answer, reference, reverse_scan_oracle, validate_input
from verifier_rl.suites import build_suites, exhaustive_cases, get, put, random_cases


class CacheTests(unittest.TestCase):
    def test_specification_examples(self):
        examples = [
            ([], []), ([get(0)], [None]),
            ([put(0), get(9), get(10), get(11)], [7, None, None]),
            ([put(0), put(5, value=8), get(10), get(15)], [8, None]),
            ([put(0, ttl=100), put(5, value=8, ttl=1), get(6)], [None]),
            ([put(0, ttl=5), get(5), put(5, value=8, ttl=2), get(5)], [None, 8]),
            ([put(0, value=0, ttl=3), get(0), put(0, "b", -5, 1), get(0, "b")], [0, -5]),
            ([put(0, "a", 1, 2), put(0, "b", 2, 5), get(2), get(2, "b"), get(5, "b")], [None, 2, None]),
        ]
        for ops, expected in examples:
            with self.subTest(ops=ops):
                self.assertEqual(reference(ops), expected)
                self.assertEqual(reverse_scan_oracle(ops), expected)

    def test_exhaustive_oracle_agreement(self):
        cases = tuple(exhaustive_cases())
        self.assertGreater(len(cases), 100)
        for case in cases:
            self.assertEqual(reference(case.operations), reverse_scan_oracle(case.operations))

    def test_seeded_oracle_agreement_and_all_generated_inputs(self):
        for c in random_cases(91027, 1000, "oracle-check", max_length=200):
            self.assertEqual(reference(c.operations), reverse_scan_oracle(c.operations))
        for s in build_suites():
            for c in s.cases:
                checked_answer(c.operations)

    def test_no_state_leak_or_old_value_resurrection(self):
        self.assertEqual(reference([put(0)]), [])
        self.assertEqual(reference([get(0)]), [None])
        ops = [put(0, ttl=100), put(1, value=8, ttl=1), get(3)]
        self.assertEqual(checked_answer(ops), [None])

    def test_invalid_input_rejected(self):
        for ops in (None, {}, [get(-1)], [get(1), get(0)], [get(True)],
                    [put(0, ttl=0)], [put(0, value=False)], [put(0, value=1001)],
                    [get(0, "UPPER")], [dict(get(0), unexpected=1)], [get(0)]*201):
            with self.subTest(ops=str(ops)[:60]), self.assertRaises(ValueError):
                validate_input(ops)
