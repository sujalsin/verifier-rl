"""Versioned balanced cache probes; existing G1/G2/G3 and audit are untouched."""

from fractions import Fraction
import random

from .fixtures import FAULTS, fixture_impl
from .grading import ExecutionResult, Status, score_suite
from .suites import Suite, canonical_json, case, get, put

VERSION = "cache-balanced-probes-0.1"
SEED = 20260929
FAMILIES = ("retrieval", "expiry", "overwrite", "multiple_keys", "order", "edges")
WEIGHTS = {name: Fraction(19, 100) for name in FAMILIES[:-1]} | {"edges": Fraction(5, 100)}


def balanced_suite(seed=SEED):
    rng = random.Random(seed)
    cases = []
    for i in range(4):
        a, b, missing = rng.sample(list("abcdefghijklmno"), 3)
        v, w = rng.sample([x for x in range(-1000, 1001) if x != 0], 2)
        t, ttl = rng.randint(0, 1000), rng.randint(2, 15)
        probes = {
            "retrieval": [put(t, a, v, ttl), get(t, a), get(t + ttl - 1, a), get(t + ttl - 1, missing)],
            "expiry": [put(t, a, v, ttl), get(t + ttl - 1, a), get(t + ttl, a), get(t + ttl + 1, a)],
            "overwrite": ([put(t, a, v, 2), get(t + 1, a), put(t + 1, a, w, ttl + 2),
                           get(t + 2, a), get(t + ttl + 3, a)] if i % 2 == 0 else
                          [put(t, a, v, ttl + 8), get(t, a), put(t + 1, a, w, 2),
                           get(t + 2, a), get(t + 3, a), get(t + 4, a)]),
            "multiple_keys": [put(t, a, v, 2), put(t, b, w, ttl + 2), get(t + 1, a),
                              get(t + 1, b), get(t + 2, a), get(t + 2, b), get(t + ttl + 2, b)],
            "order": [get(t, a), put(t, a, v, 2), get(t, a), put(t, a, w, ttl),
                      get(t, a), get(t + ttl, a)],
        }
        for family, ops in probes.items():
            cases.append(case(f"{family}/{i}", ops, family))
    for name, ops in (
        ("empty", []), ("put_only", [put(0), put(1, "b", -2)]),
        ("missing", [get(0, "z"), get(1, "y")]),
        ("zero_negative", [put(0, "a", 0, 2), put(0, "b", -5, 3), get(0, "a"), get(1, "b"), get(3, "b")]),
    ):
        cases.append(case(f"edges/{name}", ops, "edges"))
    return Suite("balanced", "training", tuple(cases), seed, VERSION)


def balanced_scores(result, suite=None):
    """Derive two rewards from the SAME exact-output case outcomes, never rerun."""
    suite = suite or balanced_suite()
    if result["suite_hash"] != suite.fingerprint or len(result["outcomes"]) != len(suite.cases):
        raise ValueError("balanced suite identity/coverage mismatch")
    for c, o in zip(suite.cases, result["outcomes"]):
        if o["input_hash"] != c.input_hash or o["case"] != c.name or o["expected"] != c.expected:
            raise ValueError("balanced outcome identity mismatch")
        if o["passed"] is not None and type(o["passed"]) is not bool:
            raise ValueError("invalid case correctness flag")
    if any(o["passed"] is None for o in result["outcomes"]) or result["infrastructure_errors"]:
        return {"binary": None, "partial": None, "family_pass_rates": None}
    rates = {}
    for family in FAMILIES:
        flags = [o["passed"] for c, o in zip(suite.cases, result["outcomes"]) if c.tags == (family,)]
        if len(flags) != 4:
            raise ValueError("each behavior family must have four probes")
        rates[family] = Fraction(sum(flags), len(flags))
    return {"binary": int(all(o["passed"] for o in result["outcomes"])),
            "partial": float(sum(WEIGHTS[f] * rates[f] for f in FAMILIES)),
            "family_pass_rates": {f: float(v) for f, v in rates.items()}}


SHORTCUTS = ("always_none", "always_empty", "constant_seven", "ignore_expiration", "last_value", "random_answers")


def shortcut_output(operations, name, seed=0):
    """Fixed project-authored test doubles, not executable model submissions."""
    if name not in SHORTCUTS:
        raise ValueError("unknown shortcut")
    if name == "always_empty":
        return []
    rng, values, last, output = random.Random(seed), {}, None, []
    for op in operations:
        if op["op"] == "put":
            values[op["key"]] = last = op["value"]
        else:
            output.append(None if name == "always_none" else 7 if name == "constant_seven"
                          else values.get(op["key"]) if name == "ignore_expiration" else last if name == "last_value"
                          else rng.choice([None, rng.randint(-1000, 1000)]))
    return output


def validate_reward_design():
    rows = []
    for seed in (SEED, SEED + 1, SEED + 2):
        suite = balanced_suite(seed)
        for name in FAULTS + SHORTCUTS:
            outputs = [fixture_impl(c.operations, name) if name in FAULTS
                       else shortcut_output(c.operations, name, i + seed) for i, c in enumerate(suite.cases)]
            executions = {c.input_hash: (ExecutionResult(Status.COMPLETED, canonical_json(out).encode()),)
                          for c, out in zip(suite.cases, outputs)}
            scores = balanced_scores(score_suite(suite, executions), suite)
            if name == "correct" and scores["partial"] != 1:
                raise AssertionError("reference fixture must receive full credit")
            if name != "correct" and scores["binary"] != 0:
                raise AssertionError(f"full false acceptance: {name}")
            if name in ("always_none", "always_empty", "constant_seven", "random_answers") and scores["partial"] > .05:
                raise AssertionError(f"trivial-answer reward above ceiling: {name}")
            if name == "ignore_expiration" and not .05 < scores["partial"] < 1:
                raise AssertionError("basic retrieval must have distinguishable partial credit")
            rows.append({"seed": seed, "program": name, **scores})
    return {"kind": "authored_reward_validation", "version": VERSION, "passed": True,
            "model_learning_claim": False, "rows": rows,
            "limitations": "finite authored controls; not proof against all shortcuts or random strategies"}
