"""Offline booking coverage/reward revision; no model, cloud, or candidate execution.

The completed booking experiments remain frozen. This module is not wired into
their launchers. Synthetic outcome maps below are grader controls, not model data.
"""

from collections import Counter
from functools import lru_cache
import json
import math
import random
from types import MappingProxyType

from . import booking_study as historical
from .grading import Status
from .model_trial import validate_submission
from .panel_execution import require_evidence, unpack_result
from .suites import canonical_json, digest
from .task_panel import BOOKING, FAMILIES, Case, booking_reference, build_suite
from .verifier_quality import compare_output

VERSION = "booking-coverage-reward-0.2"
COUNTS = MappingProxyType({"training": 96, "audit": 192})
SIZES = (1, 2, 3, 8, 25, 75, 150, 200)
SHAPES = ("binary", "linear", "logarithmic")
CONDITIONS = ("reference", "structured")
LOG_STRENGTH = 9.0


def multiset_key(bookings):
    """Preserve multiplicity; shuffling an old input does not make it new."""
    return canonical_json(sorted(bookings))


def historical_inputs():
    cases = [c for suite in historical.suites().values() for c in suite]
    cases.extend(build_suite(BOOKING, "development"))
    return {multiset_key(c.arguments["bookings"]) for c in cases}


def _random_intervals(rng, n, width):
    result = []
    for _ in range(n):
        start = rng.randrange(width)
        result.append([start, rng.randint(start + 1, width)])
    return result


def _place(rng, bookings):
    span = max(end for _, end in bookings)
    scale = rng.randint(1, min(8, 10000 // span))
    offset = rng.randint(0, 10000 - scale * span)
    result = [[offset + scale * a, offset + scale * b] for a, b in bookings]
    rng.shuffle(result)
    return result


def _input(rng, family, index):
    n = SIZES[index % len(SIZES)]
    if family == "ordinary":
        if index == 0:
            return [[0, rng.randint(1, 10000)]]
        if index == 1:
            return [[rng.randrange(10000), 10000]]
        mode = (index // 8) % 3
        values = (_random_intervals(rng, n, 1000) if mode == 0 else
                  [[3 * i, 3 * i + 1] for i in range(n)] if mode == 1 else
                  [[2 * i, 2 * i + 5] for i in range(n)])
    elif family == "state":
        mode = (index // 4) % 6
        if mode == 0:
            values = [[i, 2 * n + 1 - i] for i in range(n)]
        elif mode == 1:
            values = [[1, 5] for _ in range(n)]
        elif mode == 2:
            values = [[0, i + 1] for i in range(n)]
        elif mode == 3:
            values = [[i, n + 1] for i in range(n)]
        elif mode == 4:
            pool = _random_intervals(rng, max(1, n // 4), 30)
            values = [list(rng.choice(pool)) for _ in range(n)]
        else:
            values = [[i % 5, 8 + i % 7] for i in range(n)]
    elif family == "interaction":
        if index == 0:
            return []
        # Dense endpoint collisions, wider random mixtures, and two time clusters.
        mode = (index // 8) % 3
        values = _random_intervals(rng, n, (11, 250, 31)[mode])
        if mode == 2:
            values = [[a + (100 if i % 2 else 0), b + (100 if i % 2 else 0)]
                      for i, (a, b) in enumerate(values)]
    elif family == "boundary":
        n = max(2, n)
        mode = (index // 8) % 3
        if mode == 0:
            values = [[i, i + 1] for i in range(n)]
        elif mode == 1:
            values = [[0, 3] for _ in range(n // 2)] + [[3, 7] for _ in range(n - n // 2)]
        else:
            values = [[0, n + 1]] + [[i, i + 1] for i in range(1, n)]
    else:
        raise ValueError("unknown booking family")
    return _place(rng, values)


@lru_cache(maxsize=1)
def suites():
    """Fresh deterministic inputs; only the unique empty input may overlap."""
    seen = historical_inputs()
    result = {}
    for role, count in COUNTS.items():
        cases, local = [], set()
        for family in FAMILIES:
            rng = random.Random(f"{VERSION}/{role}/{family}")
            for index in range(count // len(FAMILIES)):
                for _ in range(1000):
                    bookings = _input(rng, family, index)
                    key = multiset_key(bookings)
                    if key not in local and (not bookings or key not in seen):
                        break
                else:
                    raise RuntimeError("could not construct a fresh distinct input")
                case = Case(BOOKING, "training" if role == "training" else "development",
                            family, f"{VERSION}/{role}/{family}/{index:02d}",
                            canonical_json({"bookings": bookings}))
                case.expected  # Both trusted mathematical implementations must agree.
                cases.append(case)
                local.add(key)
                seen.add(key)
        if len(cases) != count or len({c.input_hash for c in cases}) != count:
            raise RuntimeError("suite count or uniqueness mismatch")
        result[role] = tuple(cases)
    return MappingProxyType(result)


def cases_for(role):
    if role not in COUNTS:
        raise ValueError("role must be training or audit")
    return suites()[role]


@lru_cache(maxsize=2)
def suite_hash(role):
    return digest(canonical_json({"version": VERSION, "role": role,
                                 "cases": [c.manifest() for c in cases_for(role)]}))


def score_outcomes(role, outcomes):
    """Pure arithmetic for resolved, exact per-input booleans; not a trust boundary.

    Actual candidate records must enter via grade_submission. Missing/ambiguous
    cases are never silently failed, dropped from the denominator, or imputed.
    """
    cases = cases_for(role)
    if (set(outcomes) != {c.input_hash for c in cases}
            or any(type(value) is not bool for value in outcomes.values())):
        raise ValueError("complete resolved boolean outcomes required")
    passed = sum(outcomes.values())
    return {"version": VERSION, "role": role, "suite_hash": suite_hash(role),
            "outcomes": dict(outcomes), "passed": passed, "total": len(cases),
            "pass_fraction": passed / len(cases), "full_pass": passed == len(cases),
            "by_family": {family: {"passed": sum(outcomes[c.input_hash] for c in cases if c.family == family),
                                    "total": sum(c.family == family for c in cases)} for family in FAMILIES}}


def grade_submission(submission, records, role, image_id, *, execution_version=None):
    """Compare saved sandbox outputs; never execute, import, or repair a program."""
    check_execution = require_evidence
    if execution_version is not None:
        from . import supervised_execution
        if execution_version != supervised_execution.VERSION:
            raise ValueError("unknown execution protocol")
        check_execution = supervised_execution.require_evidence
    candidate = validate_submission(submission)
    cases = cases_for(role)
    rejected = candidate["extraction_status"].startswith("rejected_")
    if set(records) != (set() if rejected else {c.input_hash for c in cases}):
        raise ValueError("incomplete or extra execution evidence")
    outcomes, reasons, ids = {}, {}, []
    for case in cases:
        if rejected:
            passed, reason = False, "extraction_rejected"
        else:
            result = unpack_result(records[case.input_hash])
            ids.append(check_execution(result, BOOKING, image_id, source=candidate["source"], case=case))
            passed, reason = (compare_output(case, result.stdout) if result.status == Status.COMPLETED
                              else (False, result.status.value))
        outcomes[case.input_hash], reasons[case.input_hash] = passed, reason
    if len(ids) != len(set(ids)):
        raise ValueError("reused sandbox")
    return dict(score_outcomes(role, outcomes), reasons=reasons, sandbox_ids=ids,
                source_hash=digest(candidate["source"]), record_hash=digest(canonical_json(records)))


def shape_reward(fraction, shape):
    if (type(fraction) not in (int, float) or not math.isfinite(fraction)
            or not 0 <= fraction <= 1 or shape not in SHAPES):
        raise ValueError("finite pass fraction in [0, 1] and a known reward shape required")
    if shape == "binary":
        return float(fraction == 1)
    if shape == "linear" or fraction in (0, 1):
        return float(fraction)
    return math.log1p(LOG_STRENGTH * fraction) / math.log1p(LOG_STRENGTH)


def training_reward(report, condition="reference", shape="linear"):
    if report.get("version") != VERSION or report.get("role") != "training" or condition not in CONDITIONS:
        raise ValueError("only this version's training outcomes may supply rewards")
    checked = score_outcomes("training", report["outcomes"])
    if any(report.get(key) != value for key, value in checked.items()):
        raise ValueError("report differs from recomputed outcomes")
    cases = [c for c in cases_for("training")
             if condition == "reference" or c.arguments["bookings"]]
    fraction = sum(report["outcomes"][c.input_hash] for c in cases) / len(cases)
    return shape_reward(fraction, shape)


def coverage_summary():
    return {role: {"cases": len(cases), "families": dict(Counter(c.family for c in cases)),
                   "input_sizes": sorted({len(c.arguments["bookings"]) for c in cases}),
                   "maximum_bookings": max(len(c.arguments["bookings"]) for c in cases),
                   "suite_hash": suite_hash(role)} for role, cases in suites().items()}


def control_summary():
    """Trusted mathematical fault maps, NOT executions of model-written code."""
    result = {}
    for role, cases in suites().items():
        faults = {name: {} for name in ("correct", "empty_only", "inclusive_end", "deduplicate",
                                        "constant_zero", "constant_one", "return_length")}
        for case in cases:
            bookings, expected = case.arguments["bookings"], case.expected
            answers = {"correct": expected, "empty_only": expected if bookings else 1,
                       "inclusive_end": max((sum(a <= point <= b for a, b in bookings)
                                             for point, _ in bookings), default=0),
                       "deduplicate": booking_reference(sorted(set(tuple(b) for b in bookings))),
                       "constant_zero": 0, "constant_one": 1, "return_length": len(bookings)}
            for fault, answer in answers.items():
                faults[fault][case.input_hash] = answer == expected
        result[role] = {fault: {"passed": sum(outcomes.values()), "total": len(cases)}
                        for fault, outcomes in faults.items()}
    return result


def main():
    print(json.dumps({"version": VERSION, "status": "offline_only_not_connected_to_training",
                      "coverage": coverage_summary(), "authored_fault_maps": control_summary(),
                      "reward_examples": [{"pass_fraction": p, **{s: shape_reward(p, s) for s in SHAPES}}
                                          for p in (0, .25, .5, .75, 1)],
                      "baseline": "untouched Qwen2.5-Coder-1.5B-Instruct; no project SFT or RL"}, indent=2))


if __name__ == "__main__":
    main()
