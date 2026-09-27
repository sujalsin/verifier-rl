"""Versioned, seeded development suites. None is an untouched final evaluation."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import random

from .cache import TASK_ID, checked_answer, validate_input

SUITE_VERSION = "cache-coverage-0.1"
DEFAULT_SEED = 20260924


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Case:
    name: str
    input_json: str
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        operations = json.loads(self.input_json)
        validate_input(operations)
        if canonical_json(operations) != self.input_json:
            raise ValueError("case input must be canonical JSON")

    @property
    def operations(self):
        return json.loads(self.input_json)  # Fresh copy; fixtures cannot mutate a suite.

    @property
    def expected(self):
        return checked_answer(self.operations)

    @property
    def input_hash(self):
        return digest(self.input_json)


@dataclass(frozen=True)
class Suite:
    name: str
    purpose: str
    cases: tuple[Case, ...]
    seed: int
    version: str = SUITE_VERSION

    def __post_init__(self):
        if self.purpose not in ("training", "audit") or not self.cases:
            raise ValueError("nonempty suite with explicit purpose required")
        if len({c.name for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate case names")

    def manifest(self):
        return {
            "task_id": TASK_ID, "name": self.name, "purpose": self.purpose,
            "version": self.version, "seed": self.seed,
            "cases": [{"name": c.name, "input": c.operations,
                       "expected": c.expected, "tags": list(c.tags)} for c in self.cases],
        }

    @property
    def fingerprint(self):
        return digest(canonical_json(self.manifest()))


def put(t, key="a", value=7, ttl=10):
    return {"op": "put", "time": t, "key": key, "value": value, "ttl": ttl}


def get(t, key="a"):
    return {"op": "get", "time": t, "key": key}


def case(name, operations, *tags):
    return Case(name, canonical_json(operations), tuple(tags))


def ordinary_cases():
    return (
        case("ordinary/read", [put(0), get(4)], "ordinary"),
        case("ordinary/missing", [get(0, "z"), put(1), get(2)], "missing"),
        case("ordinary/keys", [put(0, "a", 3), put(1, "b", 9), get(2, "a"), get(3, "b")], "independent_keys"),
        case("ordinary/expired", [put(0, ttl=3), get(5)], "after_expiry"),
        case("ordinary/repeated", [put(0, ttl=100), get(1), get(2), get(3)], "repeated_reads"),
    )


def targeted_cases():
    return (
        case("target/boundary", [put(0), get(9), get(10), get(11)], "exact_expiry"),
        case("target/extend", [put(0), put(5, value=8), get(10), get(15)], "overwrite_extend"),
        case("target/shorten", [put(0, ttl=100), put(5, value=8, ttl=1), get(6)], "overwrite_shorten"),
        case("target/tie_order", [put(0, ttl=5), get(5), put(5, value=8, ttl=2), get(5)], "tie_order"),
        case("target/zero", [put(0, value=0, ttl=3), get(0)], "zero"),
        case("target/negative", [put(0, value=-5, ttl=1), get(0)], "negative"),
        case("target/empty", [], "empty"),
        case("target/put_only", [put(0), put(1, "b"), put(2, value=0)], "put_only"),
        case("target/reads_no_refresh", [put(0, ttl=5), get(4), get(5), get(8)], "reads_no_refresh"),
        case("target/independent", [put(0, "a", 1, 2), put(0, "b", 2, 5), get(2), get(2, "b"), get(5, "b")], "independent_keys"),
        case("target/read_before_put", [get(0), put(0), get(0)], "tie_order"),
        case("target/last_write", [put(0), put(0, value=9, ttl=1), get(0), get(1)], "last_write"),
        case("target/expired_overwrite", [put(0, ttl=1), put(3, value=9, ttl=2), get(3), get(5)], "expired_overwrite"),
        case("target/no_resurrection", [put(0, ttl=100), put(1, value=8, ttl=1), get(3)], "no_resurrection"),
        case("target/missing_outputs", [get(0), get(1, "b"), get(2)], "missing"),
        case("target/max_bounds", [put(999_999, "abcdefgh", -1000, 10_000), get(1_000_000, "abcdefgh")], "bounds"),
    )


def random_cases(seed, count, prefix, max_length=12):
    """Uniform length 0..max_length, put p=.55, 3 keys, times advance 0..4.

    Initial time 0..20; values uniform -10..10; TTL uniform 1..12.
    Narrow distributions deliberately make collisions possible, not mandatory.
    These distributions do not cover the whole input domain; audit adds bounds.
    """
    rng = random.Random(seed)
    result = []
    for index in range(count):
        t = rng.randint(0, 20)
        operations = []
        for _ in range(rng.randint(0, max_length)):
            t += rng.randint(0, 4)
            key = rng.choice(("a", "b", "c"))
            if rng.random() < .55:
                operations.append(put(t, key, rng.randint(-10, 10), rng.randint(1, 12)))
            else:
                operations.append(get(t, key))
        result.append(case(f"{prefix}/{index:03d}", operations, "random"))
    return tuple(result)


def exhaustive_cases():
    """All valid length <=3 sequences from this explicit 8-operation alphabet."""
    alphabet = tuple(op for t in (0, 1) for op in
                     (put(t, value=0, ttl=1), put(t, value=1, ttl=2), get(t), get(t, "b")))
    index = 0
    for length in range(4):
        for ops in product(alphabet, repeat=length):
            if any(a["time"] > b["time"] for a, b in zip(ops, ops[1:])):
                continue
            yield case(f"exhaustive/{index:04d}", list(ops), "exhaustive")
            index += 1


def build_suites(seed=DEFAULT_SEED):
    core = random_cases(seed, 16, "core")
    g1 = Suite("g1", "training", ordinary_cases(), seed)
    g2 = Suite("g2", "training", core + random_cases(seed + 1, 16, "extra"), seed)
    g3 = Suite("g3", "training", core + targeted_cases(), seed)
    # Audit is separately generated, not merely the targeted training suite.
    audit = list(random_cases(seed + 100_000, 64, "audit-random", max_length=80))
    for ttl in (1, 4, 13, 10_000):
        for offset in (-1, 0, 1):
            audit.append(case(f"audit-boundary/{ttl}/{offset}",
                              [put(31, "audit", -17, ttl), get(31 + ttl + offset, "audit")], "audit_boundary"))
    audit.extend(exhaustive_cases())
    seen = {c.input_hash for s in (g1, g2, g3) for c in s.cases}
    unique = []
    for c in audit:
        if c.input_hash not in seen:
            unique.append(c)
            seen.add(c.input_hash)
    return (g1, g2, g3, Suite("audit", "audit", tuple(unique), seed + 100_000))


def coverage(suite):
    """Observed input features, counted once per case; not claims of bug detection."""
    counts = Counter()
    lengths = []
    for c in suite.cases:
        ops = c.operations
        lengths.append(len(ops))
        features = set()
        if not ops: features.add("empty")
        if ops and all(o["op"] == "put" for o in ops): features.add("put_only")
        if len({o["key"] for o in ops}) > 1: features.add("multiple_keys")
        last = {}
        previous = None
        for o in ops:
            if o["time"] == previous: features.add("tied_times")
            previous = o["time"]
            if o["op"] == "put":
                if o["key"] in last: features.add("overwrite")
                if o["value"] == 0: features.add("zero")
                if o["value"] < 0: features.add("negative")
                last[o["key"]] = o["time"] + o["ttl"]
            elif o["key"] in last and o["time"] == last[o["key"]]:
                features.add("exact_expiry")
        counts.update(features)
    return {"cases": len(lengths), "operations": sum(lengths),
            "min_length": min(lengths), "max_length": max(lengths),
            "mean_length": sum(lengths) / len(lengths), "features": dict(sorted(counts.items()))}
