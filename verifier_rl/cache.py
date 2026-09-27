"""Task 001 input validation and two trusted, differently structured oracles."""

import re

TASK_ID = "expiring_cache_v1"


def validate_input(operations: object) -> None:
    if type(operations) is not list or len(operations) > 200:
        raise ValueError("operations must be a list of at most 200 entries")
    previous_time = -1
    for op in operations:
        if type(op) is not dict or op.get("op") not in ("put", "get"):
            raise ValueError("invalid operation")
        fields = {"op", "time", "key"}
        if op["op"] == "put":
            fields |= {"value", "ttl"}
        if set(op) != fields:
            raise ValueError("unexpected or missing operation fields")
        if type(op["time"]) is not int or not previous_time <= op["time"] <= 1_000_000 or op["time"] < 0:
            raise ValueError("times must be bounded integers in nondecreasing order")
        previous_time = op["time"]
        if type(op["key"]) is not str or re.fullmatch(r"[a-z]{1,8}", op["key"]) is None:
            raise ValueError("invalid key")
        if op["op"] == "put":
            if type(op["value"]) is not int or not -1000 <= op["value"] <= 1000:
                raise ValueError("invalid value")
            if type(op["ttl"]) is not int or not 1 <= op["ttl"] <= 10_000:
                raise ValueError("invalid ttl")


def reference(operations: list[dict]) -> list[int | None]:
    validate_input(operations)
    cache = {}
    result = []
    for op in operations:
        if op["op"] == "put":
            cache[op["key"]] = (op["value"], op["time"] + op["ttl"])
        else:
            stored = cache.get(op["key"])
            result.append(stored[0] if stored is not None and op["time"] < stored[1] else None)
    return result


def reverse_scan_oracle(operations: list[dict]) -> list[int | None]:
    validate_input(operations)
    result = []
    for index, read in enumerate(operations):
        if read["op"] != "get":
            continue
        answer = None
        for write in reversed(operations[:index]):
            if write["op"] == "put" and write["key"] == read["key"]:
                if read["time"] - write["time"] < write["ttl"]:
                    answer = write["value"]
                break  # Never resurrect an older write when the latest one expired.
        result.append(answer)
    return result


def checked_answer(operations: list[dict]) -> list[int | None]:
    answer = reference(operations)
    if answer != reverse_scan_oracle(operations):
        raise RuntimeError("reference disagreement: stop grading and repair the oracle")
    return answer
