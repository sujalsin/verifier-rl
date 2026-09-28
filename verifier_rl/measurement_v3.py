"""Frozen, CPU-only paired grader measurement. Never execute source locally."""

from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import inspect
from statistics import mean

from .diagnostics import diagnose_sample, identity
from .fixtures import fixture_impl
from .modal_backend import Limits, RUNNER
from .model_trial import submission_from_completion, validate_submission
from .reward_review import authored_output, group_advantages, review_pilot
from .reward_v2 import behavior_scores, behavior_suite
from .reward_v3 import coverage_scores, coverage_suite
from .suites import canonical_json, digest

VERSION = "saved-cache-v2-v3-measurement-0.1"
SOURCE_RUN = "qwen-grpo-partial-20260928T010255-4a0e6d52"
SOURCE_HASHES = {
    "generation": "b1ea25a862c7ba86abb60fe2cab497621f5463ece3f3b5ead69ec102f87c5be7",
    "reports": "5b40b23ea6a6539fb1b55d81b0fc41416c92063ffc763a4c1aa7d103a2179d89",
}
CONTROLS = ("correct", "short_inputs_only", "single_character_keys_only")
CONCURRENCY = 8
CANDIDATE_SECONDS = 300
CONTROLLER_SECONDS = 2400


def suites():
    return behavior_suite(), coverage_suite()


def entry_id(sample):
    arm, seed = identity(sample)
    if arm not in ("control", "training", "before", "after") or seed < 0:
        raise ValueError("unexpected measurement identity")
    return f"{arm}-{seed}"


def control_samples():
    # Only project-authored code is exported. Candidate source is never exec'd
    # on the controller. Cloud controls use the same runner as model programs.
    result = []
    for i, name in enumerate(CONTROLS):
        predicate = {"correct": "False", "short_inputs_only": "len(operations) > 6",
                     "single_character_keys_only": "any(len(op['key']) > 1 for op in operations)"}[name]
        source = (inspect.getsource(fixture_impl) + "\ndef simulate_cache(operations):\n"
                  + f"    if {predicate}:\n"
                  + "        return [None for op in operations if op['op'] == 'get']\n"
                  + "    return fixture_impl(operations)\n")
        result.append({**submission_from_completion(source), "arm": "control", "seed": i,
                       "control": name})
    return result


def saved_samples(generation):
    return [s for r in generation["rollout_records"] for s in r["samples"]] + generation["samples"]


def previous_reports(generation, reports):
    return {entry_id(r): r for record in generation["rollout_records"] for r in record["reports"]} | {
        entry_id(r): r for r in reports}


def measurement_plan(generation, reports):
    if (generation["run_id"] != SOURCE_RUN
            or digest(canonical_json(generation)) != SOURCE_HASHES["generation"]
            or digest(canonical_json(reports)) != SOURCE_HASHES["reports"]):
        raise ValueError("measurement requires unchanged frozen pilot artifacts")
    review_pilot(generation, reports)  # Recompute old scores and saved-output evidence, not source.
    samples = saved_samples(generation)
    expected = ([('training', i) for i in range(9000, 9016)]
                + [(arm, i) for arm in ("before", "after") for i in range(8000, 8008)])
    if [identity(s) for s in samples] != expected:
        raise ValueError("expected the 32 saved pilot records in original order")
    entries = control_samples() + samples
    for s in entries:
        validate_submission(s)
    old, new = suites()
    if old.cases != new.cases[:len(old.cases)]:
        raise ValueError("v3 must preserve every v2 input in order")
    count = len({c.input_hash for suite in (old, new) for c in suite.cases})
    return {"version": VERSION, "source_run_id": SOURCE_RUN, "source_hashes": dict(SOURCE_HASHES),
            "entry_hashes": {entry_id(s): digest(canonical_json(s)) for s in entries},
            "source_code_hashes": {entry_id(s): digest(s["source"]) for s in entries},
            "entry_order": [entry_id(s) for s in entries],
            "model_records": len(samples), "unique_model_sources": len({s['source'] for s in samples}),
            "controls": list(CONTROLS), "suite_hashes": [s.fingerprint for s in (old, new)],
            "suite_cases": [len(s.cases) for s in (old, new)], "unique_inputs_per_entry": count,
            "max_sandbox_executions": sum(not s["extraction_status"].startswith("rejected_")
                                          for s in entries) * count,
            "concurrency": CONCURRENCY, "creation_interval_seconds": .26, "max_retries": 0,
            "candidate_controller_seconds": CANDIDATE_SECONDS, "controller_seconds": CONTROLLER_SECONDS,
            "runner_hash": digest(RUNNER), "limits": asdict(Limits()),
            "training": False, "new_model_samples": 0, "no_automatic_training": True,
            "stop_on_v2_repeatability_change": True, "stop_on_unattributed_termination": True}


def budget_check(plan, billing, rates):
    """Conservative resource envelope, not an account/provider spending cap."""
    def amount(value):
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError("invalid billing value")
        return result
    sandbox_hour = amount(rates["cpu_hour_cost_sandbox"]) + amount(rates["mem_gib_hour_cost_sandbox"]) / 4
    controller_hour = amount(rates["cpu_hour_cost"]) + amount(rates["mem_gib_hour_cost"])
    sandbox_cost = sandbox_hour * plan["max_sandbox_executions"] * plan["limits"]["sandbox_lifetime_seconds"] / 3600
    controller_cost = controller_hour * (plan["controller_seconds"]
                       + len(plan["entry_order"]) * plan["candidate_controller_seconds"]) / 3600
    reserve = Decimal("1")  # Images/startup/storage/lag allowance; not measured usage.
    envelope = amount(billing["metered_cost"]) + sandbox_cost + controller_cost + reserve
    if envelope > Decimal("10"):
        raise ValueError("conservative envelope exceeds the user's $10 total trial budget")
    return {"billing_before": billing, "rates": rates, "sandbox_resource_envelope_usd": str(sandbox_cost),
            "controller_resource_envelope_usd": str(controller_cost), "overhead_reserve_usd": str(reserve),
            "total_metered_envelope_usd": str(envelope), "total_trial_budget_usd": "10",
            "provider_hard_spending_cap": False}


def unique_outcomes(report):
    return {o["input_hash"]: o for suite in report["suites"] for o in suite["outcomes"]}


def repeatability_changes(previous, current):
    old, fresh = previous["suites"][0], current["suites"][0]
    if old["suite_hash"] != behavior_suite().fingerprint or fresh["suite_hash"] != old["suite_hash"]:
        raise ValueError("repeatability comparison requires matching v2 suites")
    def signature(o):
        return (o["passed"], o["reason"], o["actual"],
                [a["metadata"].get("returncode") for a in o["attempts"]],
                [a["metadata"].get("stdout_sha256") for a in o["attempts"]])
    return [{"case": a["case"], "previous_reason": a["reason"], "fresh_reason": b["reason"]}
            for a, b in zip(old["outcomes"], fresh["outcomes"]) if signature(a) != signature(b)]


def require_measurement_report(sample, report, image_id):
    selected = suites()
    if [s["suite_hash"] for s in report["suites"]] != [s.fingerprint for s in selected]:
        raise ValueError("exact paired v2/v3 suites required")
    row = diagnose_sample(sample, report, selected)
    if any(s["all_passed"] is None for s in report["suites"]):
        raise ValueError("unresolved measurement infrastructure failure")
    for suite, saved in zip(selected, report["suites"]):
        if (saved.get("purpose") != suite.purpose or saved.get("suite_version") != suite.version
                or saved.get("seed") != suite.seed):
            raise ValueError("suite metadata mismatch")
    if report["execution_config"].get("max_retries") != 0:
        raise ValueError("measurement must not retry")
    if set(row["stdout_evidence"]) - {"verified_complete_stdout", "not_completed"}:
        raise ValueError("incomplete stdout evidence requires review")
    ids = set()
    for outcome in unique_outcomes(report).values():
        if len(outcome["attempts"]) != 1:
            raise ValueError("zero retries and exactly one recorded attempt required")
        attempt = outcome["attempts"][0]
        if attempt["status"] == "extraction_rejected":
            continue
        m = attempt["metadata"]
        if (m.get("backend") != "modal" or m.get("image_id") != image_id
                or m.get("runner_hash") != digest(RUNNER) or m.get("cleanup") != "terminated"
                or m.get("block_network") is not True or m.get("limits") != asdict(Limits())
                or m.get("reset") != "fresh_sandbox_per_input" or m.get("sdk_version") != "1.5.5"
                or m.get("creation_interval_seconds") != .26 or m.get("preflight_returncode") != 0):
            raise ValueError("measurement environment, preflight or cleanup mismatch")
        if not m.get("sandbox_id") or m["sandbox_id"] in ids:
            raise ValueError("missing or reused sandbox identity")
        ids.add(m["sandbox_id"])
        code = m.get("returncode")
        if attempt["status"] == "timeout" or (type(code) is int and (code < 0 or code >= 128)):
            raise ValueError("unattributed termination: preserve evidence and stop for review")
    if len(ids) != report["execution_config"]["attempts"]:
        raise ValueError("sandbox/attempt coverage mismatch")
    if sample["arm"] == "control":
        name = sample["control"]
        if name not in CONTROLS:
            raise ValueError("unexpected authored control")
        for suite, result in zip(selected, report["suites"]):
            for case, outcome in zip(suite.cases, result["outcomes"]):
                expected_actual = authored_output(case.operations, name)
                if (outcome["actual"] != expected_actual
                        or outcome["passed"] is not (expected_actual == case.expected)):
                    raise ValueError("cloud control differs from authored behavior")
    return row


def measurement_summary(generation, previous, reports, image_id):
    plan = measurement_plan(generation, previous)
    entries = control_samples() + saved_samples(generation)
    if [entry_id(r) for r in reports] != plan["entry_order"]:
        raise ValueError("all controls and 32 program records required exactly once, in order")
    prior = previous_reports(generation, previous)
    rows, all_ids = [], set()
    for sample, report in zip(entries, reports):
        diagnostic = require_measurement_report(sample, report, image_id)
        ids = {a["metadata"]["sandbox_id"] for o in unique_outcomes(report).values()
               for a in o["attempts"] if a["status"] != "extraction_rejected"}
        if ids & all_ids:
            raise ValueError("sandbox was reused across programs")
        all_ids.update(ids)
        entry = entry_id(sample)
        old = prior.get(entry)
        changes = repeatability_changes(old, report) if old else []
        fresh_v2, v3 = behavior_scores(report["suites"][0]), coverage_scores(report["suites"][1])
        if v3["binary"] > fresh_v2["binary"]:
            raise ValueError("adding cases cannot increase full-suite acceptance")
        added = report["suites"][1]["outcomes"][len(behavior_suite().cases):]
        row = {"entry": entry, "arm": sample["arm"], "seed": sample["seed"],
               "source_hash": digest(sample["source"]), "control": sample.get("control"),
               "historical_v2": behavior_scores(old["suites"][0]) if old else None,
               "fresh_v2": fresh_v2, "v3": v3, "v2_repeatability_changes": changes,
               "added_case_outcomes": dict(Counter(o["reason"] for o in added)),
               "added_failures": [{"case": o["case"], "reason": o["reason"]} for o in added if not o["passed"]],
               "partial_delta": v3["partial"] - fresh_v2["partial"], "diagnostic": diagnostic}
        rows.append(row)
    if len(all_ids) != plan["max_sandbox_executions"]:
        raise ValueError("completed measurement does not match execution budget")
    groups = []
    training = [r for r in rows if r["arm"] == "training"]
    for i in range(4):
        group = training[i * 4:i * 4 + 4]
        comparisons = {}
        for kind in ("historical_v2", "fresh_v2", "v3"):
            rewards = [r[kind]["partial"] for r in group]
            advantages = group_advantages(rewards)
            comparisons[kind] = {"rewards": rewards, "advantages": advantages,
                "uniform": len(set(rewards)) == 1,
                "positive_nonfull_ids": [r["seed"] for r, a in zip(group, advantages)
                                         if a > 0 and not r[kind]["binary"]]}
        groups.append({"step": i + 1, "comparisons": comparisons})
    arms = {}
    for arm in ("training", "before", "after"):
        selected_rows = [r for r in rows if r["arm"] == arm]
        arms[arm] = {"program_records": len(selected_rows),
            "unique_sources": len({r["source_hash"] for r in selected_rows}),
            **{kind: {"mean_partial": mean(r[kind]["partial"] for r in selected_rows),
                      "full_passes": sum(r[kind]["binary"] for r in selected_rows)}
               for kind in ("historical_v2", "fresh_v2", "v3")}}
    return {"version": VERSION, "plan": plan, "rows": rows, "arms": arms, "groups": groups,
            "controls_passed": True, "v2_repeatability_passed": not any(r["v2_repeatability_changes"] for r in rows),
            "recorded_sandbox_executions": len(all_ids), "training": False, "new_model_samples": 0,
            "limitations": ["32 existing program records, 29 unique sources; executions are not independent model samples.",
                "Same programs on different test distributions; changed scores are not learning.",
                "Higher partial scores can result from diluting old failures with added passing cases.",
                "Group advantages are counterfactual diagnostics, not weight updates or an alternative RL trajectory.",
                "Authored control failures do not establish model-discovered exploitation.",
                "No strict audit rerun, no new final evaluation, no change to the historical exit-137 outcome.",
                "Known development suites do not prove general correctness or reward-hacking resistance."]}
