"""Conservative, serialized reservations for a finite cloud study.

This is an application guard, not a provider billing cap. Retain reservations
for lost calls; only a complete trusted receipt can release unused resources.
The cloud caller must serialize transitions (one non-concurrent worker).
"""

from copy import deepcopy
from decimal import Decimal, ROUND_CEILING


class BudgetReached(ValueError):
    pass


def number(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("nonnegative finite accounting value required")
    return result


def initialize(ceiling="250", overhead="20"):
    if number(overhead) > number(ceiling):
        raise ValueError("overhead exceeds ceiling")
    return {"ceiling": str(number(ceiling)), "overhead": str(number(overhead)), "items": {}}


def committed(state):
    return number(state["overhead"]) + sum((number(v["actual"] if v["actual"] is not None else v["maximum"])
                                           for v in state["items"].values()), Decimal(0))


def reserve(state, key, maximum, identity):
    result = deepcopy(state)
    item = {"maximum": str(number(maximum)), "identity": identity, "actual": None}
    if key in result["items"]:
        old = result["items"][key]
        if any(old[k] != item[k] for k in ("maximum", "identity")):
            raise ValueError("reservation identity changed")
        return result
    if committed(result) + number(maximum) > number(result["ceiling"]):
        raise BudgetReached(f"reservation {key} would exceed ${result['ceiling']}; preserve progress")
    result["items"][key] = item
    return result


def settle(state, key, actual, identity):
    result = deepcopy(state)
    old = result["items"][key]
    value = number(actual)
    if identity != old["identity"] or value > number(old["maximum"]):
        raise ValueError("receipt exceeds or differs from reservation")
    if old["actual"] is not None and number(old["actual"]) != value:
        raise ValueError("settlement changed")
    old["actual"] = str(value)
    return result


def hourly(rates, kind):
    cpu, mem = number(rates["cpu_hour_cost"]), number(rates["mem_gib_hour_cost"])
    if kind == "gpu":
        return number(rates["gpu_hour_cost_l40s"]) + 2*cpu + 32*mem
    if kind == "cpu":
        return 3*(cpu + 2*mem)  # nonpreemptible CPU and memory premium
    if kind == "sandbox":
        return number(rates["cpu_hour_cost_sandbox"]) + number(rates["mem_gib_hour_cost_sandbox"])/4
    raise ValueError("unknown resource kind")


def cost(rates, kind, seconds):
    return hourly(rates, kind) * number(seconds)/3600


def sandbox_seconds(raw):
    seconds = Decimal(0)
    for entries in raw["entries"].values():
        for entry in entries.values():
            for n in (1,2):
                if f"intent-{n}" not in entry:
                    continue
                metadata = entry.get(f"attempt-{n}", {}).get("metadata", {})
                value = metadata.get("total_seconds")
                if metadata.get("cleanup") != "terminated" or value is None:
                    seconds += 120
                else:
                    # This timer starts before create and ends after cleanup;
                    # ceil it rather than adding overhead once per tiny input.
                    elapsed = number(value).to_integral_value(rounding=ROUND_CEILING)
                    # Lifecycle timing also includes API/startup/cleanup waits;
                    # the sandbox itself has an unchanged 120-second hard TTL.
                    seconds += min(Decimal(120), elapsed)
    return seconds
