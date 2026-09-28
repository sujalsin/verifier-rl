"""Bounded recovery for provider timeouts before candidate execution begins."""

from copy import deepcopy
from dataclasses import dataclass, replace

from .grading import Status
from .modal_backend import RUNNER
from .suites import digest


def is_preflight_timeout(attempt):
    metadata = attempt.get("metadata", {})
    return (attempt["status"] == Status.INFRASTRUCTURE_ERROR.value
            and attempt.get("detail", "").startswith("preflight:")
            and metadata.get("preflight_returncode") == -1
            and "returncode" not in metadata  # Candidate process was never started.
            and metadata.get("cleanup") == "terminated")


@dataclass
class RetryBudget:
    limit: int = 3
    used: int = 0


class PreflightOnlyRetries:
    """At most one replacement per input, with one shared pilot-wide budget."""
    def __init__(self, backend, budget):
        self.backend, self.budget = backend, budget
        self.replaced = set()

    def reserve(self, request):
        key = (request.source, request.input_json)
        if key in self.replaced or self.budget.used >= self.budget.limit:
            return False
        self.replaced.add(key)
        self.budget.used += 1
        return True

    async def execute(self, request):
        result = await self.backend.execute(request)
        if result.status != Status.INFRASTRUCTURE_ERROR:
            return result
        eligible = is_preflight_timeout({"status": result.status.value, "detail": result.detail,
                                         "metadata": result.metadata})
        # No other service failure or candidate failure becomes retryable here.
        return replace(result, retryable=eligible and self.reserve(request))


def validate_prior(sample, report, suites, image_id):
    if ((report.get("arm"), report.get("seed")) != (sample["arm"], sample["seed"])
            or report["candidate_hash"] != digest(sample["source"])):
        raise ValueError("recovery source/sample mismatch")
    if {s["suite"]: s["suite_hash"] for s in report["suites"]} != {s.name: s.fingerprint for s in suites}:
        raise ValueError("recovery suite mismatch")
    pending = set()
    for suite, saved in zip(suites, report["suites"]):
        if saved["suite"] != suite.name or len(saved["outcomes"]) != len(suite.cases):
            raise ValueError("recovery case coverage mismatch")
        for case, outcome in zip(suite.cases, saved["outcomes"]):
            if (outcome["input_hash"] != case.input_hash or outcome["case"] != case.name
                    or outcome["expected"] != case.expected or not outcome["attempts"]):
                raise ValueError("recovery case identity mismatch")
            for attempt in outcome["attempts"]:
                if attempt["status"] == Status.EXTRACTION_REJECTED.value:
                    if not sample["extraction_status"].startswith("rejected_"):
                        raise ValueError("unexpected extraction rejection")
                    continue
                metadata = attempt["metadata"]
                if (metadata.get("image_id") != image_id or metadata.get("runner_hash") != digest(RUNNER)
                        or metadata.get("cleanup") != "terminated"):
                    raise ValueError("recovery environment/cleanup mismatch")
            final = outcome["attempts"][-1]
            if final["status"] == Status.INFRASTRUCTURE_ERROR.value:
                if not is_preflight_timeout(final):
                    raise ValueError("failure is outside this preflight-only recovery policy")
                pending.add(case.input_hash)
    return pending


def merge_replacements(prior, replacement):
    """Keep all old attempts and replace only unresolved preflight outcomes."""
    if prior["candidate_hash"] != replacement["candidate_hash"]:
        raise ValueError("replacement source mismatch")
    updates = {o["input_hash"]: o for s in replacement["suites"] for o in s["outcomes"]}
    pending = {o["input_hash"] for s in prior["suites"] for o in s["outcomes"]
               if o["attempts"] and is_preflight_timeout(o["attempts"][-1])}
    if set(updates) != pending:
        raise ValueError("replacement must cover exactly the unresolved preflight inputs")
    result = deepcopy(prior)
    for suite in result["suites"]:
        for index, old in enumerate(suite["outcomes"]):
            if old["input_hash"] not in updates:
                continue
            new = deepcopy(updates[old["input_hash"]])
            if old["expected"] != new["expected"]:
                raise ValueError("replacement expected output mismatch")
            new["case"] = old["case"]
            new["attempts"] = old["attempts"] + new["attempts"]
            suite["outcomes"][index] = new
        suite["passed_count"] = sum(o["passed"] is True for o in suite["outcomes"])
        suite["infrastructure_errors"] = sum(o["passed"] is None for o in suite["outcomes"])
        suite["all_passed"] = None if suite["infrastructure_errors"] else suite["passed_count"] == suite["total"]
        suite["reward"] = (int(suite["all_passed"]) if suite["purpose"] == "training"
                           and suite["all_passed"] is not None else None)
    unique = {o["input_hash"]: o for s in result["suites"] for o in s["outcomes"]}
    result["execution_config"]["attempts"] = sum(
        len(o["attempts"]) for o in unique.values()
        if o["attempts"][-1]["status"] != Status.EXTRACTION_REJECTED.value)
    result["recovery_policy"] = "one-confirmed-preflight-replacement-per-input; three-per-pilot"
    return result
