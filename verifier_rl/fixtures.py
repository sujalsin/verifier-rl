"""Author-written diagnostic programs. Not model outputs or exploitation evidence.

The local demo calls these trusted functions directly. It NEVER executes a
user-supplied source file. source_for() exports standalone code for cloud checks.
"""

import inspect
import json

from .grading import ExecutionResult, Status, score_suite


def fixture_impl(operations, fault="correct"):
    cache = {}
    result = []
    shared_expiry = None
    if fault == "reorder_ties":
        operations = sorted(operations, key=lambda o: (o["time"], o["op"] == "get"))
    for op in operations:
        key, t = op["key"], op["time"]
        if op["op"] == "put":
            expiry = t + op["ttl"]
            if key in cache and fault == "stale_expiry":
                expiry = cache[key][1]
            if key in cache and fault == "max_expiry":
                expiry = max(expiry, cache[key][1])
            cache[key] = (op["value"], expiry, op["ttl"])
            shared_expiry = expiry
            if fault == "extra_put_output":
                result.append(None)
        else:
            stored = cache.get(key)
            answer = None
            if stored is not None:
                value, expiry, ttl = stored
                if fault == "shared_expiry":
                    expiry = shared_expiry
                valid = t <= expiry if fault == "inclusive_expiry" else t < expiry
                if valid:
                    answer = value
                    if fault == "refresh_reads":
                        cache[key] = (value, t + ttl, ttl)
                if fault == "zero_missing" and answer == 0:
                    answer = None
            if fault != "omit_missing" or answer is not None:
                result.append(answer)
    return result


FAULTS = ("correct", "inclusive_expiry", "stale_expiry", "max_expiry",
          "refresh_reads", "zero_missing", "shared_expiry", "reorder_ties",
          "extra_put_output", "omit_missing")


def source_for(fault):
    if fault not in FAULTS:
        raise ValueError("unknown fixture")
    return (inspect.getsource(fixture_impl)
            + f"\ndef simulate_cache(operations):\n    return fixture_impl(operations, {fault!r})\n")


def fixture_report(suites):
    rows = []
    for fault in FAULTS:
        executions = {}
        for suite in suites:
            for case in suite.cases:
                if case.input_hash not in executions:
                    output = fixture_impl(case.operations, fault)
                    executions[case.input_hash] = (ExecutionResult(
                        Status.COMPLETED, json.dumps(output).encode(),
                        metadata={"backend": "trusted_fixture_only"}),)
        rows.append({"fixture": fault, "suites": [score_suite(s, executions) for s in suites]})
    return {"kind": "hand_written_fixture_validation", "model_results": False, "fixtures": rows}
