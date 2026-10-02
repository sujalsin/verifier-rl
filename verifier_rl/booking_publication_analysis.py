"""Offline publication checks computed directly from the complete study record.

Uses no candidate execution, model calls, cloud access, or prior analysis kernels.
New robustness checks are descriptive and post hoc, not new primary endpoints.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/booking_complete_study_record.md"
MANIFEST = ROOT / "reports/booking-complete/manifest.json"
OUTPUT = ROOT / "reports/booking-publication"
ARMS = ("reference", "endpoint_omission", "repaired")
LABELS = dict(zip(ARMS, ("Reference", "Weak", "Repaired")))
COLORS = dict(zip(ARMS, ("#2563eb", "#b45309", "#15803d")))
SEEDS = tuple(range(20261011, 20261015))
T_DF3 = 3.182446305284263


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def literal_blocks(text):
    """Variable-length fences keep embedded model Markdown inert."""
    blocks, delimiter, language, body = [], None, None, []
    for line in text.splitlines(keepends=True):
        if delimiter is None:
            opening = re.fullmatch(r"(`{3,})([a-z]*)\n", line)
            if opening:
                delimiter, language = opening.groups()
                body = []
        elif line.rstrip("\n") == delimiter:
            value = "".join(body)
            blocks.append((language, value[:-1] if value.endswith("\n") else value))
            delimiter = None
        else:
            body.append(line)
    require(delimiter is None, "unclosed literal evidence block")
    return blocks


def exact(bounds):
    require(len(bounds) == 2 and bounds[0] == bounds[1], "unknown outcome is not a point estimate")
    return bounds[0]


def flatten(record, source):
    sid, scores, generation = record["sample_id"], record["scores"], record["generation"]
    require(sid == scores["sample_id"] == generation["sample_id"], "sample identity changed")
    require(sha(source.encode()) == scores["source_hash"], "source binding changed")
    match = re.fullmatch(r"eval-(baseline|s(\d{8})-(reference|endpoint_omission|repaired))-(00|12|24)-(\d+)", sid)
    require(match is not None, "unexpected sample identifier")
    _, seed, arm, step, draw = match.groups()
    require(int(draw) == generation["seed"], "draw seed changed")
    require(scores["unknown_inputs"] == 0, "replication contains unknown outcomes")
    row = {"sample_id": sid, "training_seed": int(seed) if seed else None,
           "arm": arm or "baseline", "step": int(step), "draw_seed": int(draw),
           "source_hash": scores["source_hash"], "tokens": generation["tokens"],
           "syntax_valid": int(scores["syntax_valid"]), "hit_token_cap": int(scores["hit_token_cap"]),
           "inclusive_signature": exact(scores["inclusive_signature_bounds"]),
           "weak_only_acceptance": exact(scores["weak_only_acceptance_bounds"])}
    for name, total in zip((*ARMS, "audit"), (96, 57, 57, 192)):
        value = scores[name]
        passed = exact(value["passed_bounds"])
        require(value["total"] == total and value["unknown"] == 0, "invalid scorer scope")
        require(type(passed) is int and 0 <= passed <= total, "invalid pass count")
        require(exact(value["full_pass_bounds"]) == int(passed == total), "full-pass inconsistency")
        require(math.isclose(exact(value["reward_bounds"]), passed / total), "reward inconsistency")
        row[name + "_passed"] = passed
        row[name + "_full_pass"] = int(passed == total)
    row["audit_case_accuracy"] = row["audit_passed"] / 192
    return row


def load(archive=ARCHIVE, manifest_path=MANIFEST):
    raw = archive.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    require(sha(raw) == manifest["report_sha256"] and len(raw) == manifest["report_bytes"],
            "archive differs from its recorded hash")
    blocks = literal_blocks(raw.decode())
    rows, cases, checkpoints, training_logs = [], None, [], []
    for index, (kind, value) in enumerate(blocks):
        if kind != "json":
            continue
        item = json.loads(value)  # Read the complete embedded structured evidence, not just summaries.
        if isinstance(item, dict) and item.get("record_type") == "evaluation_response":
            require(blocks[index + 1][0] == "python" and blocks[index + 2][0] == "text",
                    "missing literal source or response")
            rows.append(flatten(item, blocks[index + 1][1]))
        elif isinstance(item, dict) and item.get("record_type") == "checkpoint_receipt":
            checkpoints.append(item)
        elif isinstance(item, dict) and "log_history" in item and "global_step" in item:
            training_logs.append(item)
        elif isinstance(item, list) and len(item) == 288 and isinstance(item[0], dict) and "case_id" in item[0]:
            require(cases is None, "duplicate case inventory")
            cases = item
    require(len(rows) == len({r["sample_id"] for r in rows}) == 2048, "incomplete evaluation population")
    cohorts = defaultdict(list)
    for row in rows:
        cohorts[(row["training_seed"], row["arm"], row["step"])].append(row)
    expected = {(None, "baseline", 0)} | {(s, a, t) for s in SEEDS for a in ARMS for t in (12, 24)}
    require(set(cohorts) == expected, "cohort population changed")
    for (_, _, step), group in cohorts.items():
        n = 32 if step == 12 else 128
        require(len(group) == n and {r["draw_seed"] for r in group} == set(range(19000, 19000 + n)),
                "fixed evaluation panel changed")
    require(len(checkpoints) == 288 and len(training_logs) == 12, "training records incomplete")
    require(cases is not None, "case inventory absent")
    return rows, cases, {"archive_sha256": sha(raw), "archive_bytes": len(raw),
                         "primary_result_hash": manifest["primary_result_hash"],
                         "literal_blocks_read": len(blocks), "checkpoint_receipts_read": len(checkpoints),
                         "training_logs_read": len(training_logs)}


def profile(rows):
    n = len(rows)
    require(n > 0, "empty population")
    buckets = {"near_zero": [r for r in rows if r["audit_passed"] <= 1],
               "partial": [r for r in rows if 1 < r["audit_passed"] < 192],
               "full": [r for r in rows if r["audit_passed"] == 192]}
    return {"n": n, "audit_full_count": len(buckets["full"]),
            "audit_full_pass": len(buckets["full"]) / n,
            "inclusive_count": sum(r["inclusive_signature"] for r in rows),
            "inclusive_signature": mean(r["inclusive_signature"] for r in rows),
            "weak_only_count": sum(r["weak_only_acceptance"] for r in rows),
            "audit_case_accuracy": mean(r["audit_case_accuracy"] for r in rows),
            "hit_token_cap": mean(r["hit_token_cap"] for r in rows),
            "syntax_valid": mean(r["syntax_valid"] for r in rows),
            "buckets": {key: {"count": len(group), "audit_passed_slots": sum(r["audit_passed"] for r in group),
                               "contribution_to_mean_pp": 100 * sum(r["audit_passed"] for r in group) / (n * 192)}
                        for key, group in buckets.items()}}


def paired(values):
    require(len(values) == 4 and all(math.isfinite(v) for v in values), "need four finite seed differences")
    center = mean(values)
    half = T_DF3 * stdev(values) / 2
    return {"mean_pp": 100 * center, "ci95_pp": [100 * (center - half), 100 * (center + half)],
            "seed_differences_pp": [100 * v for v in values],
            "leave_one_seed_out_mean_pp": {str(seed): 100 * mean(values[:i] + values[i+1:])
                                           for i, seed in enumerate(SEEDS)},
            "method": "exploratory paired t df3; leave-one-seed-out values are influence checks, not intervals"}


def analyze(rows, cases, provenance):
    final = [r for r in rows if r["step"] == 24]
    by_arm = {a: profile([r for r in final if r["arm"] == a]) for a in ARMS}
    policies = {(s, a): profile([r for r in final if r["training_seed"] == s and r["arm"] == a])
                for s in SEEDS for a in ARMS}
    metrics = ("audit_full_pass", "inclusive_signature", "audit_case_accuracy", "hit_token_cap")
    contrasts = {name: {metric: paired([policies[s, left][metric] - policies[s, right][metric] for s in SEEDS])
                        for metric in metrics}
                 for name, left, right in (("weak_minus_reference", "endpoint_omission", "reference"),
                                            ("repaired_minus_weak", "repaired", "endpoint_omission"))}
    confusion = {}
    for arm in ARMS:
        counts = Counter((r[arm + "_full_pass"], r["audit_full_pass"]) for r in rows)
        confusion[arm] = {"true_accept": counts[1, 1], "false_accept": counts[1, 0],
                          "true_reject": counts[0, 0], "false_reject": counts[0, 1]}
    failures = [r for r in rows if r["endpoint_omission_full_pass"] and not r["audit_full_pass"]]
    require({r["sample_id"] for r in failures} == {r["sample_id"] for r in rows if r["weak_only_acceptance"]},
            "false-acceptance definitions no longer agree on this cohort")
    windows = []
    for block in range(4):
        for arm in ARMS:
            group = [r for r in final if r["arm"] == arm and (r["draw_seed"] - 19000) // 32 == block]
            windows.append({"block": block, "draw_seed_min": 19000 + block * 32,
                            "draw_seed_max": 19031 + block * 32, "arm": arm, **profile(group)})
    decomposition = {key: by_arm["repaired"]["buckets"][key]["contribution_to_mean_pp"] -
                          by_arm["endpoint_omission"]["buckets"][key]["contribution_to_mean_pp"]
                     for key in ("near_zero", "partial", "full")}
    survival = []
    for threshold in range(193):
        counts = {a: sum(r["audit_passed"] >= threshold for r in final if r["arm"] == a) for a in ARMS}
        survival.append({"minimum_audit_cases": threshold, "minimum_accuracy_pct": 100 * threshold / 192,
                         **{a + "_count": counts[a] for a in ARMS},
                         "repaired_minus_weak_pp": 100 * (counts["repaired"] - counts["endpoint_omission"]) / 512})
    training = {c["input_hash"]: c for c in cases if c["suite"] == "training"}
    audit = {c["input_hash"]: c for c in cases if c["suite"] == "audit"}
    common = sorted(training.keys() & audit.keys())
    require(len(training) == 96 and len(audit) == 192 and len(common) == 1, "suite overlap changed")
    require(training[common[0]]["arguments"] == {"bookings": []}, "shared case is not the empty input")
    return {"status": "offline publication reanalysis; sensitivity checks are post hoc",
            "provenance": provenance, "scope": {"draws": len(rows), "final_draws": len(final),
                                                "training_seed_triplets": 4, "new_executions": 0},
            "baseline": profile([r for r in rows if r["step"] == 0]),
            "final_arms": by_arm,
            "final_policies": [{"seed": s, "arm": a, **policies[s, a]} for s in SEEDS for a in ARMS],
            "contrasts": contrasts, "confusion": confusion,
            "false_acceptance": {"draws": len(failures), "distinct_sources": len({r["source_hash"] for r in failures}),
                                 "exact_signature": sum(r["inclusive_signature"] for r in failures),
                                 "non_signature": sum(not r["inclusive_signature"] for r in failures)},
            "disjoint_sampling_blocks": windows, "survival_curve": survival,
            "repaired_minus_weak_accuracy_decomposition_pp": decomposition,
            "suite_overlap": [{"input_hash": h, "arguments": training[h]["arguments"],
                               "training_case_id": training[h]["case_id"], "audit_case_id": audit[h]["case_id"]}
                              for h in common],
            "limits": ["Consistency check of saved evidence, not independent training replication or a full implementation audit.",
                       "Four training seeds; correlated draws and tests are not additional training replications.",
                       "Paired t intervals are model-based, fragile with four seeds, and not multiplicity adjusted.",
                       "Leave-one-out, decomposition, threshold curves and disjoint blocks chosen after inspecting results.",
                       "Buckets describe sampled output populations, not causal transitions of individual programs.",
                       "Disjoint draw blocks share the same trained models and are not independent replications.",
                       "Finite development audit and one deliberately diagnostic task limit generalization."]}


def svg_start(title, description, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            f'<rect width="860" height="{height}" fill="#ffffff"/>',
            '<g font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#17202a">',
            f'<text x="32" y="34" font-size="22" font-weight="700">{escape(title)}</text>']


def text(parts, x, y, value, **attrs):
    options = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    parts.append(f'<text x="{x}" y="{y}" {options}>{escape(str(value))}</text>')


def effects_figure(data):
    parts = svg_start("Grading improved but training effects remain uncertain",
                      "Four paired training-seed differences plus exploratory mean and 95 percent paired t interval.", 580)
    text(parts, 32, 62, "Dots are seed differences; black diamonds and bars show the mean and exploratory 95% interval.")
    x = lambda pp: 330 + (pp + 14) * 470 / 28
    for tick in range(-10, 11, 5):
        parts.append(f'<path d="M {x(tick)} 96 V 471" stroke="#e3e7ed"/>')
        text(parts, x(tick), 493, f"{tick:+d}" if tick else "0", text_anchor="middle")
    parts.append(f'<path d="M {x(0)} 96 V 471" stroke="#5b6471" stroke-dasharray="4 4"/>')
    for i, (contrast, metric, label, subtitle) in enumerate((
        ("weak_minus_reference", "audit_full_pass", "Weak minus reference", "Full audit pass rate"),
        ("weak_minus_reference", "inclusive_signature", "Weak minus reference", "Exact target bug rate"),
        ("repaired_minus_weak", "audit_full_pass", "Repaired minus weak", "Full audit pass rate"),
        ("repaired_minus_weak", "inclusive_signature", "Repaired minus weak", "Exact target bug rate"))):
        value, y = data["contrasts"][contrast][metric], 135 + i * 99
        text(parts, 32, y - 5, label, font_weight="700")
        text(parts, 32, y + 17, subtitle)
        lo, hi = value["ci95_pp"]
        parts.append(f'<path d="M {x(lo)} {y} H {x(hi)}" stroke="#17202a" stroke-width="2"/>')
        for pp, offset in zip(value["seed_differences_pp"], (-15, -5, 5, 15)):
            parts.append(f'<circle cx="{x(pp)}" cy="{y + offset}" r="4" fill="#64748b"/>')
        center = x(value["mean_pp"])
        parts.append(f'<path d="M {center} {y-7} l 7 7 l -7 7 l -7 -7 Z" fill="#17202a"/>')
        text(parts, x(0), y + 39, f'{value["mean_pp"]:+.2f} pp [{lo:+.2f}, {hi:+.2f}]', text_anchor="middle", font_size="13")
    text(parts, 565, 525, "Difference in percentage points", text_anchor="middle")
    text(parts, 32, 557, "4 seed pairs · 128 final draws per policy · one task · intervals chosen during analysis", font_size="13")
    return "\n".join(parts + ["</g></svg>\n"])


def distribution_figure(data):
    parts = svg_start("Partial accuracy and full correctness give different rankings",
                      "512 final draws per condition, split into zero or one audit pass, partial success, and full success.", 430)
    text(parts, 32, 64, "All 512 final draws per condition are retained. Category counts appear inside each bar.")
    shades = {"near_zero": "#cbd5e1", "partial": "#f2c57c", "full": "#93c5aa"}
    for i, arm in enumerate(ARMS):
        y, x0 = 125 + i * 88, 160
        row = data["final_arms"][arm]
        text(parts, 32, y + 28, LABELS[arm], font_weight="700")
        for bucket in shades:
            n = row["buckets"][bucket]["count"]
            width = n / 512 * 420
            parts.append(f'<rect x="{x0}" y="{y}" width="{width}" height="42" fill="{shades[bucket]}"/>')
            text(parts, x0 + width / 2, y + 27, n, text_anchor="middle")
            x0 += width
        text(parts, 610, y + 17, f'Mean case accuracy {100*row["audit_case_accuracy"]:.2f}%')
        text(parts, 610, y + 40, f'Full audit pass {100*row["audit_full_pass"]:.2f}%')
    for x0, bucket, label in ((32, "near_zero", "0 or 1 of 192 cases"), (297, "partial", "2 to 191 cases"), (554, "full", "All 192 cases")):
        parts.append(f'<rect x="{x0}" y="362" width="16" height="16" fill="{shades[bucket]}"/>')
        text(parts, x0 + 25, 376, label)
    text(parts, 32, 409, "Post hoc categories · sampled populations, not transitions of individual programs", font_size="13")
    return "\n".join(parts + ["</g></svg>\n"])


def threshold_figure(data):
    parts = svg_start("Outcome comparison across every audit threshold",
                      "Fraction of final draws reaching at least each audit score; descriptive empirical survival curves.", 445)
    text(parts, 32, 64, "Every integer threshold from 0 to 192 is shown; no threshold is selected as a new primary metric.")
    x = lambda value: 82 + value / 192 * 700
    y = lambda value: 335 - value / 512 * 230
    for tick in (0, 25, 50, 75, 100):
        parts.append(f'<path d="M 82 {y(tick/100*512)} H 782" stroke="#e3e7ed"/>')
        text(parts, 70, y(tick/100*512) + 5, f"{tick}%", text_anchor="end")
        text(parts, x(tick/100*192), 361, f"{tick}%", text_anchor="middle")
    for i, arm in enumerate(ARMS):
        points = " ".join(f'{x(v["minimum_audit_cases"]):.3f},{y(v[arm+"_count"]):.3f}' for v in data["survival_curve"])
        parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[arm]}" stroke-width="2.7"/>')
        text(parts, 110 + i*240, 88, LABELS[arm], fill=COLORS[arm], font_weight="700")
    text(parts, 430, 390, "Minimum fraction of audit cases passed", text_anchor="middle")
    text(parts, 32, 425, "Vertical axis: share of 512 draws meeting threshold · post hoc · no confidence bands or causal claim", font_size="13")
    return "\n".join(parts + ["</g></svg>\n"])


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(data, rows, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    write_csv(output / "program_rows.csv", rows)
    write_csv(output / "audit_thresholds.csv", data["survival_curve"])
    write_csv(output / "sampling_blocks.csv", [{k: v for k, v in r.items() if k != "buckets"}
                                             for r in data["disjoint_sampling_blocks"]])
    write_csv(output / "seed_sensitivity.csv", [
        {"contrast": contrast, "metric": metric, "omitted_seed": seed, "mean_all_four_pp": info["mean_pp"],
         "mean_remaining_three_pp": value}
        for contrast, metrics in data["contrasts"].items() for metric, info in metrics.items()
        for seed, value in info["leave_one_seed_out_mean_pp"].items()])
    for name, function in (("paired_effects", effects_figure), ("outcome_distribution", distribution_figure),
                           ("audit_thresholds", threshold_figure)):
        (output / (name + ".svg")).write_text(function(data))
    names = ("analysis.json", "program_rows.csv", "audit_thresholds.csv", "sampling_blocks.csv", "seed_sensitivity.csv",
             "paired_effects.svg", "outcome_distribution.svg", "audit_thresholds.svg")
    (output / "manifest.json").write_text(json.dumps({"archive_sha256": data["provenance"]["archive_sha256"],
        "analyzer_sha256": sha(Path(__file__).read_bytes()),
        "files": {name: sha((output / name).read_bytes()) for name in names}}, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows, cases, provenance = load()
    data = analyze(rows, cases, provenance)
    save(data, rows, args.output)
    print(json.dumps({"scope": data["scope"], "false_acceptance": data["false_acceptance"],
                      "contrasts": data["contrasts"], "decomposition_pp": data["repaired_minus_weak_accuracy_decomposition_pp"],
                      "suite_overlap": data["suite_overlap"]}, indent=2))


if __name__ == "__main__":
    main()
