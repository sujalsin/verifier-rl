from itertools import combinations_with_replacement, product
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from verifier_rl.suites import canonical_json
from verifier_rl.task_panel import (BOOKING, CACHE, LIMITER, TASK_IDS, Case, build_panel,
                                   build_suite, checked_answer, control_answer, prompt_for,
                                   suite_hash, task_for, valid_output, validate_call)

ROOT = Path(__file__).resolve().parents[1]


class TaskPanelTests(unittest.TestCase):
    def test_booking_hand_checked_examples(self):
        examples = [([], 0), ([[1, 5]], 1), ([[1, 3], [3, 5]], 1),
                    ([[0, 10], [2, 8], [3, 7]], 3), ([[1, 4]] * 3, 3),
                    ([[0, 3], [1, 3], [3, 6], [3, 5]], 2),
                    ([[8, 9], [0, 5], [2, 4], [3, 7]], 3),
                    ([[0, 1], [1, 2], [2, 3]], 1)]
        for bookings, answer in examples:
            self.assertEqual(checked_answer(BOOKING, {"bookings": bookings}), answer)

    def test_limiter_hand_checked_examples(self):
        examples = [([], 2, 10, []), ([(0, "a"), (1, "a"), (2, "a")], 2, 10, [True, True, False]),
                    ([(0, "a"), (9, "a"), (10, "a")], 1, 10, [True, False, True]),
                    ([(0, "a")] * 3, 2, 10, [True, True, False]),
                    ([(0, "a"), (0, "b"), (1, "a"), (1, "b")], 1, 10, [True, True, False, False]),
                    ([(0, "a"), (5, "a"), (9, "a"), (10, "a"), (15, "a")], 2, 10,
                     [True, True, False, True, True]),
                    ([(9, "a"), (10, "a")], 1, 10, [True, False]),
                    ([(0, "a"), (1, "a"), (2, "a")], 1, 1, [True, True, True])]
        for trace, limit, window, answer in examples:
            arguments = {"requests": [{"time": t, "user": u} for t, u in trace],
                         "limit": limit, "window": window}
            self.assertEqual(checked_answer(LIMITER, arguments), answer)

    def test_booking_exhaustive_multisets_and_permutation(self):
        alphabet = [[a, b] for a in range(3) for b in range(a + 1, 4)]
        for length in range(5):
            for bookings in combinations_with_replacement(alphabet, length):
                answer = checked_answer(BOOKING, {"bookings": list(bookings)})
                self.assertEqual(checked_answer(BOOKING, {"bookings": list(reversed(bookings))}), answer)

    def test_limiter_exhaustive_short_inputs(self):
        alphabet = [(t, u) for t in (0, 1) for u in ("a", "b")]
        for length in range(5):
            for trace in product(alphabet, repeat=length):
                if any(a[0] > b[0] for a, b in zip(trace, trace[1:])):
                    continue
                for limit, window in product((1, 2), repeat=2):
                    requests = [{"time": t, "user": u} for t, u in trace]
                    checked_answer(LIMITER, {"requests": requests, "limit": limit, "window": window})

    def test_random_references_and_user_independence(self):
        rng = random.Random(92028)
        for _ in range(200):
            bookings = []
            requests = []
            t = 0
            for _ in range(rng.randrange(30)):
                start = rng.randrange(50)
                bookings.append([start, start + rng.randrange(1, 20)])
                t += rng.randrange(4)
                requests.append({"time": t, "user": rng.choice(("a", "b", "longuser"))})
            checked_answer(BOOKING, {"bookings": bookings})
            limit, window = rng.randrange(1, 6), rng.randrange(1, 12)
            result = checked_answer(LIMITER, {"requests": requests, "limit": limit, "window": window})
            for user in ("a", "b", "longuser"):
                indices = [i for i, request in enumerate(requests) if request["user"] == user]
                separate = checked_answer(LIMITER, {"requests": [requests[i] for i in indices],
                                                    "limit": limit, "window": window})
                self.assertEqual(separate, [result[i] for i in indices])

    def test_domain_validation_and_no_bool_int_confusion(self):
        for bookings in ([[True, 2]], [[0, 1.0]], [[2, 2]], [[3, 1]], [[-1, 2]], [[0, 10001]],
                         [(0, 1)], [[0, 1, 2]], [[0, 1]] * 201, None):
            with self.assertRaises(ValueError):
                validate_call(BOOKING, {"bookings": bookings})
        valid = {"requests": [{"time": 1, "user": "a"}], "limit": 1, "window": 10}
        for change in ({"limit": True}, {"window": 0}, {"limit": 21}, {"extra": 1},
                       {"requests": [{"time": True, "user": "a"}]},
                       {"requests": [{"time": 1, "user": "UPPER"}]},
                       {"requests": [{"time": 1, "user": "a"}, {"time": 0, "user": "a"}]},
                       {"requests": [{"time": 1, "user": "a", "extra": 1}]}):
            with self.assertRaises(ValueError):
                validate_call(LIMITER, dict(valid, **change))
        with self.assertRaises(ValueError):
            validate_call(CACHE, {"bookings": []})
        with self.assertRaises(ValueError):
            validate_call("unknown", {})

    def test_exact_output_types(self):
        for task_id, good, bad in ((BOOKING, [0, 9], [True, 1.0, -1, [], None]),
                                    (LIMITER, [[], [True, False]], [[0, 1], [None], True]),
                                    (CACHE, [[], [0, None, -1]], [[True], [1.0], 0])):
            for output in good:
                self.assertTrue(valid_output(task_id, output))
            for output in bad:
                self.assertFalse(valid_output(task_id, output))

    def test_panel_reproducible_disjoint_and_boundary_control(self):
        first, second = build_panel(), build_panel()
        for task_id in TASK_IDS:
            train, development = first[task_id]["training"], first[task_id]["development"]
            self.assertEqual(train, second[task_id]["training"])
            self.assertEqual(len({c.input_hash for c in (*train, *development)}), 32)
            self.assertNotEqual(suite_hash(train), suite_hash(development))
            for case in (*train, *development):
                self.assertEqual(control_answer(task_id, case.arguments, "correct"), case.expected)
                self.assertEqual(control_answer(task_id, case.arguments, "inclusive_boundary") == case.expected,
                                 case.family != "boundary")
            altered = train[0].arguments
            altered.clear()
            self.assertTrue(train[0].arguments)

    def test_prompts_only_include_model_facing_contract(self):
        for task_id in TASK_IDS:
            prompt = prompt_for(task_id, ROOT)
            self.assertIn(task_for(task_id).entrypoint, prompt)
            for private in ("DIAGNOSTIC FAULTY", "REFERENCE SOLUTION PLAN", "GRADER DEFINITIONS",
                            "END CANONICAL", "DEVELOPMENT EXAMPLES"):
                self.assertNotIn(private, prompt)

    def test_reference_disagreement_stops(self):
        with patch("verifier_rl.task_panel.booking_oracle", return_value=999):
            with self.assertRaises(RuntimeError):
                build_panel()
        with patch("verifier_rl.task_panel.limiter_oracle", return_value=[False]):
            with self.assertRaises(RuntimeError):
                checked_answer(LIMITER, {"requests": [], "limit": 1, "window": 1})

    def test_unknown_split_or_noncanonical_case_rejected(self):
        with self.assertRaises(ValueError):
            build_suite(CACHE, "final")
        with self.assertRaises(ValueError):
            Case(BOOKING, "training", "ordinary", "x", '{"bookings": []}')
        with self.assertRaises(ValueError):
            Case(BOOKING, "training", "oops", "x", canonical_json({"bookings": []}))


if __name__ == "__main__":
    unittest.main()
