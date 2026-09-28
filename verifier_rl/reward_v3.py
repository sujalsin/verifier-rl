"""Opt-in domain-coverage proposal; no legacy scorer/trainer is changed.

Keep v2's family weights and complete-list comparison. Add longer traces and
wide-key metamorphic cases separately from reward-formula experiments.
"""

from fractions import Fraction
import random
import string

from .reward_v2 import FAMILIES, SEED, WEIGHTS, behavior_suite
from .suites import Suite, case, get, put

VERSION = "cache-balanced-probes-0.3"
COUNTS = {f: 8 for f in FAMILIES[:-1]} | {"edges": 4}


def coverage_suite(seed=SEED):
    original = behavior_suite(seed)
    rng = random.Random(seed + 300_000)
    keys = rng.sample(list(string.ascii_lowercase), 7)
    a, b, missing = keys[:3]
    t, value = rng.randint(10, 1000), rng.randint(1, 999)
    long_cases = {
        "retrieval": [put(t, a, value, 10_000)]
        + [get(t + i, missing if i % 13 == 0 else a) for i in range(199)],
        "expiry": [put(t, a, value, 40), put(t, b, -value, 80)]
        + [get(t + i, b) for i in range(1, 40)]
        + [get(t + 40, a), get(t + 40, b), get(t + 80, b)],
        "overwrite": [put(t, a, value, 200)]
        + [op for i in range(1, 21)
           for op in (put(t + i, a, -value if i % 2 else value, 2), get(t + i, a))]
        + [get(t + 22, a), get(t + 22, missing)],
        "multiple_keys": [put(t, k, i - 3, i + 1) for i, k in enumerate(keys[:6])]
        + [get(t + offset, k) for offset in range(8) for k in keys],
        "order": [op for i in range(30)
                  for op in (get(t, a), put(t, a, value if i % 2 else -value, 10),
                             get(t, a), get(t, missing))],
    }
    additions = []
    for family in FAMILIES[:-1]:
        additions.append(case(f"{family}/domain_long", long_cases[family], family))
        template = next(c for c in original.cases if c.tags == (family,))
        mapping = {}
        for key in sorted({op["key"] for op in template.operations}):
            while True:
                renamed = "".join(rng.choices(string.ascii_lowercase, k=8))
                if renamed not in mapping.values():
                    mapping[key] = renamed
                    break
        shift = rng.randint(100, 1000)
        operations = [dict(op, key=mapping[op["key"]], time=op["time"] + shift)
                      for op in template.operations]
        transformed = case(f"{family}/domain_wide_keys", operations, family)
        if transformed.expected != template.expected:
            raise AssertionError("key renaming/time translation changed oracle answers")
        additions.append(transformed)
    return Suite("balanced_v3", "training", original.cases + tuple(additions), seed, VERSION)


def coverage_scores(result, suite=None):
    """Version/identity checked weighted exact-case score, not per-element credit."""
    suite = suite or coverage_suite()
    if (suite.version != VERSION or suite.name != "balanced_v3" or suite.purpose != "training"
            or result["suite_hash"] != suite.fingerprint or result["suite"] != suite.name
            or result["total"] != len(suite.cases) or len(result["outcomes"]) != len(suite.cases)):
        raise ValueError("v3 suite identity/coverage mismatch")
    for c, o in zip(suite.cases, result["outcomes"]):
        if o["input_hash"] != c.input_hash or o["case"] != c.name or o["expected"] != c.expected:
            raise ValueError("v3 outcome identity mismatch")
        if o["passed"] is not None and type(o["passed"]) is not bool:
            raise ValueError("invalid correctness flag")
    if result["passed_count"] != sum(o["passed"] is True for o in result["outcomes"]):
        raise ValueError("v3 passed-count mismatch")
    unscored = any(o["passed"] is None for o in result["outcomes"]) or result["infrastructure_errors"]
    all_passed = None if unscored else all(o["passed"] for o in result["outcomes"])
    if result["all_passed"] is not all_passed or result["reward"] != (None if unscored else int(all_passed)):
        raise ValueError("v3 aggregate reward mismatch")
    if unscored:
        return {"binary": None, "partial": None, "family_pass_rates": None}
    rates = {}
    for family in FAMILIES:
        flags = [o["passed"] for c, o in zip(suite.cases, result["outcomes"]) if c.tags == (family,)]
        if len(flags) != COUNTS[family]:
            raise ValueError("unexpected v3 family sizes")
        rates[family] = Fraction(sum(flags), len(flags))
    return {"binary": int(all_passed),
            "partial": float(sum(WEIGHTS[f] * rates[f] for f in FAMILIES)),
            "family_pass_rates": {f: float(v) for f, v in rates.items()}}
