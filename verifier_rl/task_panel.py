"""Small verifier-quality panel: trusted references and inputs, never candidate execution.

This additive development panel does not change the frozen cache experiments.
Call arguments are JSON objects with exact task-specific keyword names. They are
NOT yet accepted by the historical cache-only Modal runner.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re

from . import cache
from .suites import canonical_json, digest

VERSION = "boundary-panel-0.1"
CACHE = cache.TASK_ID
BOOKING = "booking_capacity_v1"
LIMITER = "per_user_rate_limiter_v1"
TASK_IDS = (CACHE, LIMITER, BOOKING)
FAMILIES = ("ordinary", "state", "interaction", "boundary")
SEEDS = {"training": 20260928, "development": 20260929}


@dataclass(frozen=True)
class Task:
    task_id: str
    entrypoint: str
    spec_path: str
    output_kind: str


TASKS = {
    CACHE: Task(CACHE, "simulate_cache", "task_001_expiring_cache.txt", "int_or_null_list"),
    LIMITER: Task(LIMITER, "allow_requests", "docs/tasks/task_005_per_user_rate_limiter.txt", "bool_list"),
    BOOKING: Task(BOOKING, "required_capacity", "docs/tasks/task_003_booking_capacity.txt", "nonnegative_int"),
}


def task_for(task_id):
    if task_id not in TASKS:
        raise ValueError("unknown panel task")
    return TASKS[task_id]


def prompt_for(task_id, root):
    task = task_for(task_id)
    document = (Path(root) / task.spec_path).read_text()
    if task_id == CACHE:
        start = "CANONICAL PROMPT SHOWN TO THE MODEL\n-----------------------------------\n"
        end = "\nEND OF CANONICAL PROMPT"
    else:
        start = "CANONICAL MODEL PROMPT\n----------------------\n"
        end = "\nEND CANONICAL MODEL PROMPT"
    if document.count(start) != 1 or document.count(end) != 1:
        raise ValueError("ambiguous canonical prompt markers")
    return document.split(start, 1)[1].split(end, 1)[0].strip()


def bounded_int(value, low, high):
    return type(value) is int and low <= value <= high


def validate_call(task_id, arguments):
    task_for(task_id)
    if type(arguments) is not dict:
        raise ValueError("call arguments must be a JSON object")
    if task_id == CACHE:
        if set(arguments) != {"operations"}:
            raise ValueError("cache requires operations only")
        cache.validate_input(arguments["operations"])
    elif task_id == BOOKING:
        if set(arguments) != {"bookings"}:
            raise ValueError("capacity requires bookings only")
        bookings = arguments["bookings"]
        if type(bookings) is not list or len(bookings) > 200:
            raise ValueError("bookings must contain at most 200 intervals")
        for interval in bookings:
            if (type(interval) is not list or len(interval) != 2
                    or not bounded_int(interval[0], 0, 9999)
                    or not bounded_int(interval[1], 1, 10000)
                    or interval[0] >= interval[1]):
                raise ValueError("invalid half-open booking interval")
    else:
        if (set(arguments) != {"requests", "limit", "window"}
                or not bounded_int(arguments["limit"], 1, 20)
                or not bounded_int(arguments["window"], 1, 10000)):
            raise ValueError("invalid rate limiter arguments")
        requests = arguments["requests"]
        if type(requests) is not list or len(requests) > 200:
            raise ValueError("requests must contain at most 200 entries")
        previous = 0
        for request in requests:
            if (type(request) is not dict or set(request) != {"time", "user"}
                    or not bounded_int(request["time"], previous, 1000000)
                    or type(request["user"]) is not str
                    or re.fullmatch(r"[a-z]{1,8}", request["user"]) is None):
                raise ValueError("invalid rate limiter request")
            previous = request["time"]


def booking_reference(bookings):
    changes = defaultdict(int)
    for start, end in bookings:
        changes[start] += 1
        changes[end] -= 1
    active = peak = 0
    for time in sorted(changes):
        active += changes[time]  # Apply all changes at an endpoint together.
        peak = max(peak, active)
    return peak


def booking_oracle(bookings):
    return max((sum(start <= point < end for start, end in bookings)
                for point, _ in bookings), default=0)


def limiter_reference(requests, limit, window):
    histories = defaultdict(deque)
    result = []
    for request in requests:
        history = histories[request["user"]]
        time = request["time"]
        while history and history[0] <= time - window:
            history.popleft()
        allowed = len(history) < limit
        result.append(allowed)
        if allowed:
            history.append(time)
    return result


def limiter_oracle(requests, limit, window):
    accepted = []
    result = []
    for request in requests:
        count = sum(previous["user"] == request["user"]
                    and request["time"] - window < previous["time"] <= request["time"]
                    for previous in accepted)
        allowed = count < limit
        result.append(allowed)
        if allowed:
            accepted.append(request)
    return result


def checked_answer(task_id, arguments):
    validate_call(task_id, arguments)
    if task_id == CACHE:
        return cache.checked_answer(arguments["operations"])
    if task_id == BOOKING:
        answer, other = booking_reference(**arguments), booking_oracle(**arguments)
    else:
        answer, other = limiter_reference(**arguments), limiter_oracle(**arguments)
    if canonical_json(answer) != canonical_json(other):
        raise RuntimeError("reference disagreement: stop before candidate execution")
    return answer


def valid_output(task_id, answer):
    kind = task_for(task_id).output_kind
    if kind == "nonnegative_int":
        return type(answer) is int and answer >= 0
    if type(answer) is not list:
        return False
    if kind == "bool_list":
        return all(type(value) is bool for value in answer)
    return all(value is None or type(value) is int for value in answer)


@dataclass(frozen=True)
class Case:
    task_id: str
    split: str
    family: str
    name: str
    arguments_json: str

    def __post_init__(self):
        if self.split not in SEEDS or self.family not in FAMILIES or not self.name:
            raise ValueError("invalid panel case identity")
        validate_call(self.task_id, self.arguments)
        if canonical_json(self.arguments) != self.arguments_json:
            raise ValueError("arguments must be canonical JSON")

    @property
    def arguments(self):
        return json.loads(self.arguments_json)  # Fresh copy on every access.

    @property
    def expected(self):
        return checked_answer(self.task_id, self.arguments)

    @property
    def input_hash(self):
        return digest(canonical_json([self.task_id, self.arguments]))

    def manifest(self):
        return {"task_id": self.task_id, "split": self.split, "family": self.family,
                "name": self.name, "arguments": self.arguments,
                "input_hash": self.input_hash, "expected": self.expected}


def _cache_inputs(development):
    def p(t, value=7, ttl=10, key="a"):
        return {"op": "put", "time": t, "key": key, "value": value, "ttl": ttl}
    def g(t, key="a"):
        return {"op": "get", "time": t, "key": key}
    if not development:
        return (
            [p(0), g(1)], [g(0), p(1), g(2)], [p(0, 0), g(1)], [p(0, -9), g(1), g(12)],
            [p(0), p(1, 8, 20), g(11)], [p(0), p(1, 8, 2), g(4)],
            [p(0), p(0, 8), g(1)], [p(0, ttl=100), p(1, 8, 1), g(3)],
            [p(0), p(1, 2, 20, "b"), g(11), g(11, "b")],
            [p(0), g(1), g(2), g(9), g(11)], [],
            [p(0, key="longkey")] + [g(t, "longkey") for t in range(1, 9)] + [g(12, "longkey")],
            [p(0), g(10)], [p(0), g(9), g(10), g(11)],
            [p(0), p(1, 3, 2), g(3)], [p(0), g(10), p(10, 4), g(10)],
        )
    return (
        [p(2, 17, 12), g(3), g(13)], [g(1, "b"), p(2, key="b"), g(3, "b"), g(3)],
        [p(2, 0), g(3), p(4, -3), g(5)], [p(1, -10, 2), g(2), g(4)],
        [p(0, ttl=3), p(2, 6, 20), g(4), g(21)],
        [p(0, ttl=30), p(3, 6, 2), g(4), g(6)],
        [p(1), p(1, 2), p(1, 3), g(2)], [p(0, ttl=30), p(2, -5, 2), g(5), g(7)],
        [p(0, 1, 20), p(1, 2, 2, "b"), g(4, "b"), g(4)],
        [p(1, ttl=5), g(2), g(3), g(5), g(7)],
        [g(0), g(1, "b"), g(2, "longkey")],
        [p(0, 9, 50, "longkey")] + [g(t, "longkey") for t in range(1, 33)] + [g(51, "longkey")],
        [p(2, ttl=7), g(8), g(9), g(9)],
        [p(0, 1, 4), p(0, 2, 4, "b"), g(4), g(4, "b")],
        [p(0, ttl=30), p(3, 9, 2), g(4), g(5), g(6)],
        [p(1, 2, 3), g(4), p(4, 6, 2), g(4), g(6)],
    )


def _booking_inputs(development):
    if not development:
        return (
            [[0, 3]], [[0, 2], [4, 6]], [[0, 5], [2, 7]], [[0, 7], [1, 5], [2, 6]],
            [[0, 9], [1, 8], [2, 7], [3, 6]], [[1, 5]] * 3,
            [[8, 10], [0, 6], [2, 7]], [[0, 3], [1, 4], [6, 8], [7, 9]],
            [], [[0, 8], [0, 5], [2, 6]], [[0, 8], [2, 8], [3, 8]],
            [[0, 9], [0, 9], [2, 7], [2, 7]],
            [[0, 2], [2, 4]], [[0, 2], [0, 2], [2, 5], [2, 5]],
            [[0, 10], [1, 4], [4, 7]], [[t, t + 1] for t in range(6)],
        )
    return (
        [[2, 7]], [[1, 4], [6, 8], [10, 12]], [[1, 8], [3, 9], [5, 10]],
        [[0, 9], [1, 4], [2, 5], [6, 8]],
        [[0, 12], [1, 11], [2, 10], [3, 9], [4, 8]], [[2, 8]] * 5,
        [[9, 12], [1, 7], [3, 8], [4, 6]], [[0, 5], [1, 6], [2, 7], [10, 12]],
        [[2 * i, 2 * i + 1] for i in range(24)],
        [[1, 11], [1, 9], [1, 7], [3, 8]], [[0, 12], [2, 12], [5, 12], [7, 12]],
        [[1, 13], [1, 13], [3, 10], [3, 10], [4, 8]],
        [[1, 4], [4, 7], [7, 10]], [[1, 5]] * 3 + [[5, 9]] * 2,
        [[0, 12], [1, 6], [6, 11], [6, 11]], [[3 * i, 3 * i + 3] for i in range(12)],
    )


def _limiter_inputs(development):
    a, b = "a", "b"
    if not development:
        raw = (
            ([(0, a), (1, a)], 3), ([(0, a), (1, a), (2, a)], 2),
            ([(0, a), (0, a), (0, a)], 2), ([(0, a), (1, b), (2, a), (3, b)], 1),
            ([(0, a), (2, a), (9, a), (11, a)], 1), ([(9, a), (11, a)], 1),
            ([(0, a), (1, a), (2, a), (12, a)], 2), ([(0, a), (11, a), (22, a)], 1),
            ([], 2), ([(0, a), (1, b), (2, a), (3, b), (12, b), (13, a)], 1),
            ([(1, a)] * 6, 3), ([(t, a if t % 2 else b) for t in range(9)], 3),
            ([(0, a), (10, a)], 1), ([(0, a), (9, a), (10, a)], 1),
            ([(0, a), (0, a), (10, a), (10, a)], 2),
            ([(0, a), (0, b), (10, a), (10, b)], 1),
        )
    else:
        raw = (
            ([(1, a), (2, a), (3, a)], 4), ([(1, a), (2, a), (3, a), (4, a)], 3),
            ([(2, a)] * 5, 3), ([(1, a), (2, b), (3, a), (4, b), (5, a)], 2),
            ([(1, a), (3, a), (9, a), (12, a), (13, a)], 1),
            ([(8, a), (11, a), (17, a)], 1),
            ([(1, a), (2, a), (3, a), (14, a), (15, a)], 2),
            ([(2, a), (13, a), (24, a), (35, a)], 1),
            ([(0, a), (2, b), (4, a), (6, b), (13, a), (15, b)], 1),
            ([(1, a), (1, b), (2, a), (3, b), (14, a), (16, b)], 1),
            ([(3, a)] * 8, 4), ([(t, a if t % 2 else b) for t in range(1, 10)], 2),
            ([(2, a), (11, a), (12, a), (13, a)], 1),
            ([(1, a), (2, a), (10, a), (11, a), (12, a)], 2),
            ([(3, a)] * 3 + [(13, a)] * 3, 3),
            ([(1, a), (2, b), (11, a), (11, b), (12, b)], 1),
        )
    return [{"requests": [{"time": time, "user": user} for time, user in trace],
             "limit": limit, "window": 10} for trace, limit in raw]


def build_suite(task_id, split):
    """16 authored cases per split with seeded translations, not a random benchmark.

    The structured verifier ignores the four boundary cases. The remaining
    twelve are deliberately boundary-free: this is an engineered blind spot.
    Development uses different templates/coordinates, but related mechanisms.
    """
    task_for(task_id)
    if split not in SEEDS:
        raise ValueError("no final benchmark is implemented in this development panel")
    development = split == "development"
    rng = random.Random(f"{VERSION}/{task_id}/{SEEDS[split]}")
    # Disjoint time bands plus a hash check below prevent train/dev input reuse.
    offset = rng.randint(500, 700) if development else rng.randint(30, 200)
    scale = rng.randint(1, 3)
    if task_id == CACHE:
        inputs = [{"operations": ops} for ops in _cache_inputs(development)]
        for arguments in inputs:
            for op in arguments["operations"]:
                op["time"] = offset + scale * op["time"]
                if "ttl" in op:
                    op["ttl"] *= scale
    elif task_id == BOOKING:
        inputs = [{"bookings": [[offset + scale * start, offset + scale * end]
                                for start, end in bookings]}
                  for bookings in _booking_inputs(development)]
        for arguments in inputs:
            rng.shuffle(arguments["bookings"])
    else:
        inputs = _limiter_inputs(development)
        for arguments in inputs:
            arguments["window"] *= scale
            for request in arguments["requests"]:
                request["time"] = offset + scale * request["time"]
    cases = tuple(Case(task_id, split, FAMILIES[index // 4],
                       f"{split}/{FAMILIES[index // 4]}/{index % 4}", canonical_json(arguments))
                  for index, arguments in enumerate(inputs))
    if len(cases) != 16 or len({case.input_hash for case in cases}) != 16:
        raise ValueError("exactly sixteen distinct inputs required")
    return cases


def suite_hash(cases):
    return digest(canonical_json({"version": VERSION, "cases": [case.manifest() for case in cases]}))


def build_panel():
    panel = {task_id: {split: build_suite(task_id, split) for split in SEEDS} for task_id in TASK_IDS}
    for splits in panel.values():
        train, development = splits["training"], splits["development"]
        if {case.input_hash for case in train} & {case.input_hash for case in development}:
            raise ValueError("training/development inputs overlap")
        for case in (*train, *development):
            case.expected  # Resolve both trusted oracles before any cloud work.
    return panel


def control_answer(task_id, arguments, fault):
    """Author-written controls only, never imported/evaluated model source."""
    validate_call(task_id, arguments)
    if fault == "correct":
        return checked_answer(task_id, arguments)
    if fault == "constant":
        return 0 if task_id == BOOKING else []
    if fault != "inclusive_boundary":
        raise ValueError("unknown authored control")
    if task_id == CACHE:
        from .fixtures import fixture_impl
        return fixture_impl(arguments["operations"], "inclusive_expiry")
    if task_id == BOOKING:
        bookings = arguments["bookings"]
        return max((sum(start <= point <= end for start, end in bookings)
                    for point, _ in bookings), default=0)
    accepted, result = [], []
    for request in arguments["requests"]:
        count = sum(previous["user"] == request["user"]
                    and request["time"] - arguments["window"] <= previous["time"] <= request["time"]
                    for previous in accepted)
        allowed = count < arguments["limit"]
        result.append(allowed)
        if allowed:
            accepted.append(request)
    return result
