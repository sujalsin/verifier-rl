"""Exploratory analysis of saved booking model behavior, without execution.

These lenses were chosen after seeing the study, not prospectively frozen.
No p-values, independent-pair claims, model calls, or cloud calls are made.
Original hypotheses and their primary analysis remain unchanged.
"""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean, median

from . import booking_replication_analysis as base

ROOT = base.ROOT
ARMS = base.ARMS
SUITES = (*ARMS, "audit")
OUTPUT = ROOT/"reports/booking-behavior"
VIGNETTES = {
    "cancelled_event_deltas": "eval-s20261011-endpoint_omission-24-19001",
    "immediate_decrement": "eval-s20261011-endpoint_omission-24-19017",
    "counts_bookings_instead_of_overlap": "eval-s20261011-endpoint_omission-24-19040",
    "start_ordered_queue_for_end_expiration": "eval-s20261011-endpoint_omission-24-19008",
    "explanation_and_correct_code_disagree": "eval-s20261011-endpoint_omission-24-19121",
}


def load_rows(directory, primary_directory):
    manifest = base.read(primary_directory/"output_manifest.json")
    raw_analysis = (primary_directory/"analysis.json").read_bytes()
    base.require(hashlib.sha256(raw_analysis).hexdigest() == manifest["files"]["analysis.json"],
                 "primary analysis changed")
    primary = json.loads(raw_analysis)
    result, inventory, config, sources = base.load_verified(directory)
    base.require(primary["result_hash"] == base.fingerprint(result)
                 and primary["inventory_hash"] == base.fingerprint(inventory),
                 "primary analysis belongs to another population")
    failures = directory/"failure-evidence"
    record = base.read(failures/"manifest.json")["files"]["baseline-generation"]
    baseline = json.loads(base.verified_bytes(failures/record["path"], record))
    for sample in baseline["samples"]:
        sources[sample["sample_id"]] = sample
    groups = [inventory["baseline"], *result["policies"].values()]
    scored = {r["sample_id"]: r for group in groups for r in group["programs"]}
    base.require(set(scored) == {p["sample_id"] for p in primary["programs"]}, "population changed")
    rows = []
    for item in primary["programs"]:
        sid = item["sample_id"]
        sample, row = sources[sid], scored[sid]
        base.validate_source(sample, row)
        values = base.row_values(row)
        base.require(all(item[key] == value for key, value in values.items()), "derived score changed")
        base.require(item["tokens"] == sample["tokens"] and item["draw_seed"] == sample["seed"],
                     "generation attributes changed")
        counts = {name+"_passed": base.exact(row[name]["passed_bounds"]) for name in SUITES}
        base.require(all(isinstance(v, int) and 0 <= v <= limit for v, limit in
                         zip(counts.values(), (96, 57, 57, 192))), "invalid test count")
        rows.append({**item, **counts})
    return primary, rows, sources


def profile(rows):
    n = len(rows)
    base.require(n > 0, "empty cohort")
    low = sum(r["audit_passed"] <= 1 for r in rows)
    partial = sum(1 < r["audit_passed"] < 192 for r in rows)
    full = sum(r["audit_passed"] == 192 for r in rows)
    syntax = sum(r["syntax_valid"] for r in rows)
    base.require(low+partial+full == n, "outcome categories must partition all draws")
    return {"draws": n, "syntax_valid": syntax, "audit_full": full,
            "syntax_valid_audit_failure": sum(r["syntax_valid"] and r["audit_passed"] < 192 for r in rows),
            "at_most_one_audit_pass": low, "partial_audit_pass": partial,
            "zero_audit_pass": sum(r["audit_passed"] == 0 for r in rows),
            "one_audit_pass": sum(r["audit_passed"] == 1 for r in rows),
            "mean_audit_case_accuracy": mean(r["audit_passed"]/192 for r in rows),
            "mean_tokens": mean(r["tokens"] for r in rows),
            "median_tokens": median(r["tokens"] for r in rows),
            "hit_token_cap": sum(r["hit_token_cap"] for r in rows),
            "capped_and_audit_full": sum(r["hit_token_cap"] and r["audit_passed"] == 192 for r in rows)}


def ranking_disagreement(rows):
    """Compare partial-score order on the SAME eligible unordered draw pairs.

    Exclude pairs tied on audit or any compared verifier. Each draw is retained,
    including duplicate sources; compressing score vectors is only arithmetic.
    These highly dependent pairs are NOT independent statistical observations.
    """
    histogram = Counter(tuple(r[name+"_passed"] for name in SUITES) for r in rows)
    common = 0
    discordant = [0, 0, 0]
    own_eligible, own_discordant = [0, 0, 0], [0, 0, 0]
    for (left, count_left), (right, count_right) in combinations(histogram.items(), 2):
        audit_difference = left[3]-right[3]
        if not audit_difference:
            continue
        weight = count_left*count_right
        differences = [left[i]-right[i] for i in range(3)]
        eligible = all(differences)
        if eligible:
            common += weight
        for i, difference in enumerate(differences):
            if difference:
                own_eligible[i] += weight
                wrong = difference*audit_difference < 0
                own_discordant[i] += weight*wrong
                discordant[i] += weight*wrong*eligible
    total = len(rows)*(len(rows)-1)//2
    return {"draws": len(rows), "all_unordered_pairs": total,
            "common_eligible_pairs": common, "pairs_excluded_for_ties": total-common,
            "verifiers": {name: {"discordant_common_pairs": discordant[i],
                                  "discordant_common_rate": discordant[i]/common if common else None,
                                  "own_eligible_pairs": own_eligible[i],
                                  "own_discordant_pairs": own_discordant[i],
                                  "own_discordant_rate": own_discordant[i]/own_eligible[i] if own_eligible[i] else None}
                          for i, name in enumerate(ARMS)},
            "independent_samples": False,
            "limits": "Post hoc descriptive ordering against finite audit case accuracy; excludes ties; pairs share programs."}


def analyze(directory=base.DEFAULT, primary_directory=ROOT/"reports/booking-replication"):
    primary, rows, sources = load_rows(directory, primary_directory)
    final = [r for r in rows if r["step"] == 24]
    distributions = [{"cohort": "baseline", "seed": None, "arm": "baseline",
                      **profile([r for r in rows if r["step"] == 0])}]
    for arm in ARMS:
        distributions.append({"cohort": "final_all_seeds", "seed": None, "arm": arm,
                              **profile([r for r in final if r["arm"] == arm])})
    differences = []
    ranking = {"all_evaluation": ranking_disagreement(rows), "final_only": ranking_disagreement(final)}
    for seed in base.study.SEEDS:
        by_arm = {}
        for arm in ARMS:
            by_arm[arm] = profile([r for r in final if r["training_seed"] == seed and r["arm"] == arm])
            distributions.append({"cohort": "final_policy", "seed": seed, "arm": arm, **by_arm[arm]})
        weak, repaired = by_arm["endpoint_omission"], by_arm["repaired"]
        differences.append({"seed": seed,
                            **{key+"_difference_pp": 100*(repaired[key]-weak[key])/128
                               for key in ("at_most_one_audit_pass", "partial_audit_pass", "audit_full", "hit_token_cap")},
                            "mean_audit_case_accuracy_difference_pp": 100*(repaired["mean_audit_case_accuracy"]-weak["mean_audit_case_accuracy"]),
                            "mean_tokens_difference": repaired["mean_tokens"]-weak["mean_tokens"]})
        ranking[str(seed)] = ranking_disagreement([r for r in final if r["training_seed"] == seed])
    capped_correct = []
    for row in final:
        if not (row["hit_token_cap"] and row["audit_full_pass"]):
            continue
        sample = sources[row["sample_id"]]
        raw = sample["raw"]
        closing = raw.rfind("```")
        has_close = closing > raw.find("```") >= 0
        capped_correct.append({"sample_id": row["sample_id"], "source_hash": row["source_hash"],
                               "tokens": row["tokens"], "ended_with_eos": sample["ended_with_eos"],
                               "closed_code_fence": has_close,
                               "trailing_characters_after_code_fence": len(raw[closing+3:].strip()) if has_close else 0})
    length_bins = []
    for arm in ("all", *ARMS):
        for low, high in ((0,128), (129,256), (257,511), (512,512)):
            chosen = [r for r in final if (arm == "all" or r["arm"] == arm) and low <= r["tokens"] <= high]
            if chosen:
                length_bins.append({"arm": arm, "minimum_tokens": low, "maximum_tokens": high, **profile(chosen)})
    histogram = []
    for arm in ARMS:
        counts = Counter(r["audit_passed"] for r in final if r["arm"] == arm)
        histogram.extend({"arm": arm, "audit_cases_passed": score, "draws": count}
                         for score, count in sorted(counts.items()))
    row_by_id = {r["sample_id"]: r for r in rows}
    examples = []
    for mechanism, sid in VIGNETTES.items():
        sample, row = sources[sid], row_by_id[sid]
        examples.append({"mechanism": mechanism, "sample_id": sid, "source_hash": row["source_hash"],
                         "source": sample["source"], "raw_completion": sample["raw"],
                         "tokens": sample["tokens"], "scores": {name: row[name+"_passed"] for name in SUITES},
                         "selection": "Post hoc illustrative source inspection, not an exhaustive behavioral classifier",
                         "candidate_executions": 0})
    return {"version": "booking-behavior-exploration-0.1", "primary_result_hash": primary["result_hash"],
            "primary_analysis_file_sha256": hashlib.sha256((primary_directory/"analysis.json").read_bytes()).hexdigest(),
            "status": "exploratory after outcome inspection; no confirmatory claims",
            "outcome_distributions": distributions, "final_all_arms": profile(final),
            "repaired_minus_weak_pairs": differences, "ranking_disagreement": ranking,
            "capped_correct_programs": capped_correct, "length_bins": length_bins,
            "final_audit_score_histogram": histogram, "illustrative_programs": examples,
            "scope": {"all_evaluation_draws": len(rows), "final_draws": len(final),
                      "independent_training_seeds": 4, "new_model_calls": 0, "candidate_executions": 0,
                      "new_downloads": 0, "hypothesis_or_scientific_settings_changed": False},
            "lenses_examined": ["syntax versus audit correctness", "near-total versus partial versus full correctness",
                                "partial reward ordering versus full acceptance", "completion length and cap",
                                "seed variation", "illustrative source and explanation mechanisms"],
            "limits": ["Outcome categories and ranking analysis chosen post hoc.",
                       "No causal interpretation of token-length associations.",
                       "Illustrative program mechanisms do not establish their population prevalence.",
                       "No candidate code was executed and no full raw archive was downloaded.",
                       "Audit correctness is finite-suite correctness, not universal specification compliance."]}


def save(data, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/"findings.json").write_text(json.dumps(data, indent=2, sort_keys=True)+"\n")
    for filename, key in (("outcome_distributions", "outcome_distributions"),
                          ("repaired_weak_pairs", "repaired_minus_weak_pairs"),
                          ("capped_correct_programs", "capped_correct_programs"),
                          ("length_bins", "length_bins"),
                          ("audit_score_histogram", "final_audit_score_histogram")):
        base.write_csv(output/f"{filename}.csv", data[key])
    ordering = [{"cohort": cohort, "verifier": name, "draws": info["draws"],
                 "common_eligible_pairs": info["common_eligible_pairs"], **metrics}
                for cohort, info in data["ranking_disagreement"].items()
                for name, metrics in info["verifiers"].items()]
    base.write_csv(output/"ranking_disagreement.csv", ordering)
    (output/"output_manifest.json").write_text(json.dumps({
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_analysis_file_sha256": data["primary_analysis_file_sha256"],
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())
                  if p.is_file() and p.name != "output_manifest.json"}}, indent=2, sort_keys=True)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base.DEFAULT)
    parser.add_argument("--primary", type=Path, default=ROOT/"reports/booking-replication")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = analyze(args.input, args.primary)
    save(data, args.output)
    print(json.dumps({"output": str(args.output), "scope": data["scope"],
                      "final_all_arms": data["final_all_arms"],
                      "ranking_final": data["ranking_disagreement"]["final_only"]}, indent=2))


if __name__ == "__main__":
    main()
