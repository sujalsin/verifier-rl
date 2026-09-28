"""Explicitly versioned behavior probes and trusted mutation controls.

V0.1 and its historical launchers are deliberately untouched. This module never
loads a model or executes candidate source. The controls are fixed authored code.
"""

import argparse
from fractions import Fraction
import random

from .fixtures import FAULTS, fixture_impl
from .grading import ExecutionResult, Status, score_suite
from .partial_reward import SHORTCUTS, balanced_scores, balanced_suite, shortcut_output
from .suites import Suite, canonical_json, case, get, put

VERSION = "cache-balanced-probes-0.2"
SEED = 20260929
FAMILIES = ("retrieval", "expiry", "overwrite", "multiple_keys", "order", "edges")
COUNTS = {f: 6 for f in FAMILIES[:-1]} | {"edges": 4}
WEIGHTS = {f: Fraction(19, 100) for f in FAMILIES[:-1]} | {"edges": Fraction(5, 100)}


def behavior_suite(seed=SEED):
    rng = random.Random(seed)
    a, b, missing = rng.sample(list("abcdefghijklmno"), 3)
    v, w = rng.randint(1, 900), -rng.randint(1, 900)
    t, d = rng.randint(0, 1000), rng.randint(4, 12)
    # Different structures, not merely more numeric variants of one structure.
    probes = {
        "retrieval": {
            "same_time_repeats": [put(t, a, v, d), get(t, a), get(t, a), get(t, missing)],
            "later_repeats": [put(t, a, v, d), get(t + 1, a), get(t + d - 1, a), get(t + d - 1, missing)],
            "interleaved": [put(t, a, v, d), put(t, b, w, d + 2), get(t, a), get(t, b), get(t, a), get(t, missing)],
            "zero": [put(t, a, 0, d), get(t, a), get(t + 1, a), get(t + 1, missing)],
            "negative": [put(t, a, w, d), get(t + 1, a), get(t + d - 1, a), get(t + d - 1, missing)],
            "replacement": [put(t, a, v, 1), get(t + 1, a), put(t + 1, a, w, d), get(t + 1, a), get(t + 2, a)],
        },
        # Every expiry probe includes a first read of an already-expired write.
        # A separate live value prevents always-None from passing the probe.
        "expiry": {
            "first_at_boundary": [put(t, a, v, d), put(t, b, w, d + 3), get(t + d, a), get(t + d, b)],
            "first_after_boundary": [put(t, a, v, d), put(t, b, w, d + 4), get(t + d + 2, a), get(t + d + 2, b)],
            "minimum_lifetime": [put(t, a, v, 1), put(t, b, w, 4), get(t + 1, a), get(t + 1, b), get(t + 2, a)],
            "reads_do_not_refresh": [put(t, a, v, d), put(t, b, w, d), get(t + 1, b), get(t + d, a), get(t + d, b)],
            "expired_before_other_write": [put(t, a, v, 1), put(t + 2, b, w, d), get(t + 2, a), get(t + 2, b)],
            "large_clock": [put(999980, a, -1000, 10), put(999980, b, 1000, 10000), get(1000000, a), get(1000000, b)],
        },
        "overwrite": {
            "extend": [put(t, a, v, 2), put(t + 1, a, w, d), get(t + 2, a), get(t + d + 1, a)],
            "shorten": [put(t, a, v, d + 5), put(t + 1, a, w, 2), get(t + 2, a), get(t + 3, a)],
            "expired_then_replaced": [put(t, a, v, 1), get(t + 1, a), put(t + 2, a, w, d), get(t + 2, a), get(t + 3, a), get(t + d + 2, a)],
            "same_time_repeat": [put(t, a, v, d), put(t, a, w, 2), get(t, a), get(t, a), get(t + 2, a)],
            "no_resurrection": [put(t, a, v, d + 10), put(t + 1, a, w, 1), put(t + 1, b, 0, d), get(t + 2, a), get(t + 2, b)],
            "three_writes": [put(t, a, v, d + 10), put(t, a, 0, 1), put(t + 1, a, w, d), get(t + 1, a), get(t + d + 1, a)],
        },
        "multiple_keys": {
            "independent_expiry": [put(t, a, v, 2), put(t, b, w, d), get(t + 2, a), get(t + 2, b), get(t + d, a), get(t + d, b)],
            "interleaved_repeat": [put(t, a, v, d), put(t, b, w, d), get(t, a), get(t, b), get(t + 1, a), get(t + 1, missing)],
            "overwrite_one": [put(t, a, v, d), put(t, b, w, d + 4), put(t + 1, a, 0, 1), get(t + 2, a), get(t + 2, b), get(t + 2, b)],
            "missing_read_no_effect": [put(t, a, v, d), get(t, b), get(t, a), get(t, missing), get(t, a)],
            "different_write_times": [put(t, a, v, 2), put(t + 1, b, w, 4), get(t + 2, a), get(t + 2, b), get(t + 5, b)],
            "zero_and_negative": [put(t, a, 0, 2), put(t, b, w, 4), get(t, a), get(t, b), get(t + 1, a), get(t + 2, a)],
        },
        "order": {
            "read_before_write": [get(t, a), put(t, a, v, d), get(t, a)],
            "last_same_time_write": [put(t, a, v, d), put(t, a, w, d), get(t, a), get(t, missing)],
            "read_write_read": [put(t, a, v, d), get(t, a), put(t, a, w, d), get(t, a), get(t, b)],
            "read_write_repeat": [get(t, a), put(t, a, v, d), get(t, a), put(t, a, w, d), get(t, a), get(t, a)],
            "expiry_then_replacement": [put(t, a, v, 1), get(t + 1, a), put(t + 1, a, w, d), get(t + 1, a)],
            "key_ties": [get(t, a), put(t, b, w, d), put(t, a, v, d), get(t, a), get(t, b), get(t, a)],
        },
        "edges": {
            "empty": [],
            "put_only": [put(0), put(1, "b", -2)],
            "missing": [get(0, "z"), get(1, "y")],
            "zero_negative": [put(0, "a", 0, 2), put(0, "b", -5, 3), get(0, "a"), get(1, "b"), get(3, "b")],
        },
    }
    cases = tuple(case(f"{family}/{name}", ops, family)
                  for family, rows in probes.items() for name, ops in rows.items())
    return Suite("balanced_v2", "training", cases, seed, VERSION)


def behavior_scores(result, suite=None):
    suite = suite or behavior_suite()
    if (suite.version != VERSION or suite.name != "balanced_v2"
            or result["suite_hash"] != suite.fingerprint
            or result["total"] != len(suite.cases) or len(result["outcomes"]) != len(suite.cases)):
        raise ValueError("v2 suite identity/coverage mismatch")
    for c, o in zip(suite.cases, result["outcomes"]):
        if o["input_hash"] != c.input_hash or o["case"] != c.name or o["expected"] != c.expected:
            raise ValueError("v2 outcome identity mismatch")
        if o["passed"] is not None and type(o["passed"]) is not bool:
            raise ValueError("invalid correctness flag")
    if result["passed_count"] != sum(o["passed"] is True for o in result["outcomes"]):
        raise ValueError("v2 passed-count mismatch")
    unscored = any(o["passed"] is None for o in result["outcomes"]) or result["infrastructure_errors"]
    all_passed = None if unscored else all(o["passed"] for o in result["outcomes"])
    if result["all_passed"] is not all_passed or result["reward"] != (None if unscored else int(all_passed)):
        raise ValueError("v2 aggregate reward mismatch")
    if unscored:
        return {"binary": None, "partial": None, "family_pass_rates": None}
    rates = {}
    for family in FAMILIES:
        flags = [o["passed"] for c, o in zip(suite.cases, result["outcomes"]) if c.tags == (family,)]
        if len(flags) != COUNTS[family]:
            raise ValueError("unexpected v2 family sizes")
        rates[family] = Fraction(sum(flags), len(flags))
    return {"binary": int(all(o["passed"] for o in result["outcomes"])),
            "partial": float(sum(WEIGHTS[f] * rates[f] for f in FAMILIES)),
            "family_pass_rates": {f: float(v) for f, v in rates.items()}}


COMBINED = ("consume_with_expiry", "consume_without_expiry", "consume_inclusive_expiry",
            "consume_shared_expiry", "ignore_expiry_zero_missing", "refresh_zero_missing")
CONTROLS = FAULTS + SHORTCUTS + COMBINED


def control_output(operations, name):
    """Fixed authored counterexamples, never execution of a submitted source string."""
    if name not in CONTROLS:
        raise ValueError("unknown v2 control")
    if name in FAULTS:
        return fixture_impl(operations, name)
    if name in SHORTCUTS:
        return shortcut_output(operations, name, seed=31415)
    if name in ("ignore_expiry_zero_missing", "refresh_zero_missing"):
        answers = (shortcut_output(operations, "ignore_expiration") if name == "ignore_expiry_zero_missing"
                   else fixture_impl(operations, "refresh_reads"))
        return [None if value == 0 else value for value in answers]
    values, answers, shared = {}, [], None
    for op in operations:
        if op["op"] == "put":
            shared = op["time"] + op["ttl"]
            values[op["key"]] = op["value"], shared
        else:
            entry = values.pop(op["key"], None)
            answer = None
            if entry is not None:
                value, expiry = entry
                if name == "consume_shared_expiry":
                    expiry = shared
                valid = op["time"] <= expiry if name == "consume_inclusive_expiry" else op["time"] < expiry
                if name == "consume_without_expiry" or valid:
                    answer = value
            answers.append(answer)
    return answers


def validate_behavior_reward(seeds=(SEED, SEED + 1, SEED + 2)):
    rows = []
    for seed in seeds:
        old, new = balanced_suite(seed), behavior_suite(seed)
        for name in CONTROLS:
            scores = []
            for suite, scoring in ((old, balanced_scores), (new, behavior_scores)):
                executions = {c.input_hash: (ExecutionResult(Status.COMPLETED,
                              canonical_json(control_output(c.operations, name)).encode()),) for c in suite.cases}
                scores.append(scoring(score_suite(suite, executions), suite))
            legacy, current = scores
            if name == "correct" and current["partial"] != 1:
                raise AssertionError("correct fixture must receive full credit")
            if name != "correct" and current["binary"] != 0:
                raise AssertionError(f"v2 full false acceptance: {name}")
            if name in ("always_none", "always_empty", "constant_seven", "random_answers") and current["partial"] > .05:
                raise AssertionError("trivial output exceeded edge-only allowance")
            if name == "consume_without_expiry" and (current["family_pass_rates"]["expiry"] != 0
                                                     or current["partial"] >= legacy["partial"]):
                raise AssertionError("expiry/delete-on-read confound remains")
            rows.append({"seed": seed, "control": name, "legacy": legacy, "v2": current})
        by_name = {r["control"]: r["v2"] for r in rows if r["seed"] == seed}
        if by_name["consume_with_expiry"]["partial"] <= by_name["consume_without_expiry"]["partial"]:
            raise AssertionError("correct expiry must improve this matched mutation control")
    return {"kind": "authored_v2_reward_validation", "version": VERSION, "passed": True,
            "model_results": False, "seeds": list(seeds), "rows": rows,
            "limitations": ["Finite authored controls, not proof against arbitrary shortcuts.",
                            "Family labels name test patterns, not isolated capabilities.",
                            "Partial credit is not full acceptance or a guarantee of useful RL.",
                            "These probes are development/training data, not an untouched final evaluation."]}


def main():
    from .cli import create_run_directory, save_suites, write_private
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new local directory; no cloud/model execution")
    args = parser.parse_args()
    report = validate_behavior_reward()
    directory = create_run_directory(args.out)
    save_suites(directory, (behavior_suite(),))
    write_private(directory / "controls.json", canonical_json(report))
    write_private(directory / "manifests.json", canonical_json([
        {"legacy": balanced_suite(seed).manifest(), "v2": behavior_suite(seed).manifest(),
         "legacy_hash": balanced_suite(seed).fingerprint, "v2_hash": behavior_suite(seed).fingerprint}
        for seed in report["seeds"]]))
    print(f"Validated {len(report['rows'])} authored control/seed combinations; no model evaluation.")
    for row in report["rows"]:
        if row["seed"] == SEED:
            print(row["control"], "v1", row["legacy"]["partial"], "v2", row["v2"]["partial"])


if __name__ == "__main__":
    main()
