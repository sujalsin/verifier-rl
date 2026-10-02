"""Recompute the blog's arithmetic from a portable table of saved scores.

Default: standard library, offline, no model or candidate execution.
--export-evidence additionally validates the existing local evidence bundle.
--figures uses Matplotlib; --preview uses markdown-it-py. Neither is needed
to reproduce the numbers. Run from a checkout containing the score CSV.
"""

import argparse
from collections import Counter
import csv
import hashlib
from html import escape
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/booking-blog"
ARMS = ("reference", "endpoint_omission", "repaired")
LABELS = ("Reference", "Weak", "Repaired")
SEEDS = tuple(range(20261011, 20261015))
SCORES = ("reference_passed", "endpoint_omission_passed", "repaired_passed", "audit_passed")
TOTALS = (96, 57, 57, 192)
T975 = 3.182446305284263


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def export_evidence(output):
    sys.path.insert(0, str(ROOT))
    from verifier_rl import booking_behavior_analysis as behavior
    from verifier_rl import booking_replication as study

    primary, rows, _ = behavior.load_rows(behavior.base.DEFAULT, ROOT / "reports/booking-replication")
    with (output / "program_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train = {c.input_hash: c for c in study.pilot.cases_for("training")}
    audit = {c.input_hash: c for c in study.pilot.coverage.cases_for("audit")}
    overlap = [{"input_hash": h, "arguments": train[h].arguments,
                "training_case": train[h].name, "audit_case": audit[h].name}
               for h in sorted(train.keys() & audit.keys())]
    require(len(overlap) == 1 and overlap[0]["arguments"] == {"bookings": []}, "suite overlap changed")
    inputs = ("docs/booking_complete_study_record.md", "docs/booking_replication_repair_protocol.txt",
              "reports/booking-replication/analysis.json", "reports/booking-replication/output_manifest.json",
              "reports/booking-replication/program_metrics.csv", "verifier_rl/booking_replication.py",
              "verifier_rl/booking_behavior_analysis.py")
    write_json(output / "source_manifest.json", {
        "score_csv_sha256": sha(output / "program_scores.csv"),
        "source_result_hash": primary["result_hash"], "source_inventory_hash": primary["inventory_hash"],
        "source_files": {p: sha(ROOT / p) for p in inputs},
        "suite_overlap": overlap, "training_inputs_executed_per_extracted_program": 96,
        "training_inputs_scored": dict(zip(ARMS, TOTALS[:3])),
        "scope": "Validated export of saved score summaries and generation bindings; no new executions. "
                 "The CSV supports arithmetic reproduction, not an independent replay of all raw evidence."
    })


def load_rows(path, manifest):
    require(sha(path) == manifest["score_csv_sha256"], "score CSV differs from its export manifest")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    integer_fields = ("step", "draw_seed", "tokens", "audit_full_pass", "inclusive_signature",
                      "inclusive_signature_all_inputs", "weak_only_acceptance", "hit_token_cap",
                      "syntax_valid", "reference_full_pass", "endpoint_omission_full_pass",
                      "repaired_full_pass", *SCORES)
    for row in rows:
        row["training_seed"] = int(row["training_seed"]) if row["training_seed"] else None
        for key in integer_fields:
            row[key] = int(row[key])
        row["audit_case_accuracy"] = float(row["audit_case_accuracy"])
        for key, total in zip(SCORES, TOTALS):
            require(0 <= row[key] <= total, "invalid score numerator")
        for arm, total in zip((*ARMS, "audit"), TOTALS):
            require(row[arm + "_full_pass"] == int(row[arm + "_passed"] == total), "full-pass binding differs")
        require(math.isclose(row["audit_case_accuracy"], row["audit_passed"] / 192, abs_tol=1e-15),
                "audit case accuracy differs")
    expected = {f"eval-baseline-00-{d}" for d in range(19000, 19128)}
    for seed in SEEDS:
        for arm in ARMS:
            for step, n in ((12, 32), (24, 128)):
                expected.update(f"eval-s{seed}-{arm}-{step:02d}-{d}" for d in range(19000, 19000+n))
    require(len(rows) == len(expected) == 2048 and {r["sample_id"] for r in rows} == expected,
            "fixed population contains missing, extra or duplicate draws")
    for row in rows:
        label = "baseline" if row["training_seed"] is None else f"s{row['training_seed']}-{row['arm']}"
        require(row["sample_id"] == f"eval-{label}-{row['step']:02d}-{row['draw_seed']}", "identity mismatch")
    return rows


def profile(rows):
    return {"draws": len(rows), "audit_full": sum(r["audit_full_pass"] for r in rows),
            "target_bug": sum(r["inclusive_signature"] for r in rows),
            "weak_only": sum(r["weak_only_acceptance"] for r in rows),
            "mean_case_accuracy": sum(r["audit_passed"] for r in rows) / (len(rows)*192),
            "at_most_one_pass": sum(r["audit_passed"] <= 1 for r in rows),
            "partial": sum(1 < r["audit_passed"] < 192 for r in rows),
            "token_cap": sum(r["hit_token_cap"] for r in rows)}


def paired_effect(differences):
    require(len(differences) == 4, "four paired seeds required")
    average = statistics.mean(differences)
    half_width = T975 * statistics.stdev(differences) / math.sqrt(4)
    return {"differences_pp": differences, "mean_pp": average,
            "exploratory_t95_pp": [average-half_width, average+half_width],
            "leave_one_seed_out_means_pp": [statistics.mean(differences[:i]+differences[i+1:]) for i in range(4)]}


def rank_disagreement(rows):
    histogram = Counter(tuple(r[k] for k in SCORES) for r in rows)
    eligible = 0
    wrong = [0, 0, 0]
    for (a, na), (b, nb) in combinations(histogram.items(), 2):
        deltas = [x-y for x, y in zip(a, b)]
        if all(deltas):
            eligible += na*nb
            for i in range(3):
                wrong[i] += na*nb * (deltas[i]*deltas[3] < 0)
    return {"eligible_dependent_pairs": eligible,
            "discordant_pairs": dict(zip(ARMS, wrong)),
            "rates": {arm: n/eligible for arm, n in zip(ARMS, wrong)}}


def normalized_example(rewards):
    average, sd = statistics.mean(rewards), statistics.stdev(rewards)
    return {"rewards": rewards, "idealized_advantages": [(r-average)/sd if sd else 0 for r in rewards]}


def compute(rows):
    final = [r for r in rows if r["step"] == 24]
    cohorts = {(s, a): [r for r in final if r["training_seed"] == s and r["arm"] == a]
               for s in SEEDS for a in ARMS}
    require(all(len(v) == 128 for v in cohorts.values()), "incorrect final cohort")
    effects = {}
    for first, second in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[1])):
        effects[first + "_minus_" + second] = {}
        for metric in ("audit_full_pass", "inclusive_signature", "audit_case_accuracy"):
            ds = [100*(statistics.mean(r[metric] for r in cohorts[s, first]) -
                       statistics.mean(r[metric] for r in cohorts[s, second])) for s in SEEDS]
            effects[first + "_minus_" + second][metric] = paired_effect(ds)
    confusion = {}
    for arm in ARMS:
        accepted = [r for r in rows if r[arm + "_full_pass"]]
        false = [r for r in accepted if not r["audit_full_pass"]]
        confusion[arm] = {"accepted_audit_pass": len(accepted)-len(false), "accepted_audit_fail": len(false),
                          "rejected_audit_pass": sum(r["audit_full_pass"] and not r[arm + "_full_pass"] for r in rows),
                          "rejected_audit_fail": sum(not r["audit_full_pass"] and not r[arm + "_full_pass"] for r in rows),
                          "false_accept_distinct_source_hashes": len({r["source_hash"] for r in false})}
    valid = [r for r in rows if not r["extraction_status"].startswith("rejected_")]
    by_source = {}
    for row in valid:
        prior = by_source.setdefault(row["source_hash"], row)
        require(all(prior[k] == row[k] for k in SCORES), "duplicate source has inconsistent saved scores")
    unique = list(by_source.values())
    return {
        "scope": "Offline arithmetic reanalysis of saved scores; no new training or candidate execution.",
        "evaluation_draws": len(rows), "training_seed_order": SEEDS,
        "confusion_all_2048_draws": confusion,
        "baseline": profile([r for r in rows if r["step"] == 0]),
        "final": {a: profile([r for r in final if r["arm"] == a]) for a in ARMS},
        "final_by_seed": {str(s): {a: profile(cohorts[s, a]) for a in ARMS} for s in SEEDS},
        "paired_effects": effects,
        "fixed_prefix_sensitivity": {str(n): {a: profile([r for r in final if r["arm"] == a and r["draw_seed"] < 19000+n])
                                             for a in ARMS} for n in (32, 128)},
        "source_deduplication_descriptive": {
            "extracted_draws": len(valid), "distinct_extracted_sources": len(unique),
            "distinct_audit_pass_sources": sum(r["audit_full_pass"] for r in unique),
            "distinct_weak_false_accept_sources": sum(r["endpoint_omission_full_pass"] and not r["audit_full_pass"] for r in unique),
            "distinct_repaired_false_accept_sources": sum(r["repaired_full_pass"] and not r["audit_full_pass"] for r in unique),
            "note": "Different descriptive unit; source deduplication does not estimate policy draw frequency or add independent training runs."},
        "partial_score_ordering_final": rank_disagreement(final),
        "synthetic_group_examples_not_observed_rollouts": {
            "definition": "(reward - group mean) / sample standard deviation; zero for a tied group. Numerical stability epsilon omitted.",
            "mixed_weak": normalized_example([1, 1, 1/57, 0]),
            "mixed_repaired": normalized_example([1, 49/57, 1/57, 0]),
            "three_correct_weak": normalized_example([1, 1, 1, 1]),
            "three_correct_repaired": normalized_example([1, 1, 1, 49/57])},
        "limits": ["Four paired training seeds on one task; intervals are exploratory, conditional on the fixed draw panel.",
                   "Leave-one-seed-out, deduplication and synthetic examples are publication-stage follow-up checks.",
                   "No equivalence test, causal mechanism identification or full raw archive audit."]}


def figures(result, output):
    # Keep plotting dependencies optional and the cache in a writable temp folder.
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "booking-blog-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ("#2854A1", "#B65C17", "#00796B")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False, "axes.titleweight": "bold",
                         "svg.fonttype": "none", "svg.hashsalt": "booking-publication"})

    def save(fig, name):
        fig.savefig(output / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
        fig.savefig(output / (name + ".png"), bbox_inches="tight", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0))
    contrasts = ("endpoint_omission_minus_reference", "repaired_minus_endpoint_omission")
    for ax, metric, title in zip(axes, ("audit_full_pass", "inclusive_signature"),
                                ("Full audit pass rate", "Inclusive endpoint signature frequency")):
        ax.axvline(0, color="#7B8494", linestyle="--", linewidth=1)
        for y, contrast in enumerate(contrasts):
            item = result["paired_effects"][contrast][metric]
            for offset, value in zip((-.12, -.04, .04, .12), item["differences_pp"]):
                ax.scatter(value, y+offset, color=colors[y+1], s=32, alpha=.8, zorder=4)
            lo, hi = item["exploratory_t95_pp"]
            ax.plot([lo, hi], [y+.22, y+.22], color=colors[y+1], linewidth=2)
            ax.scatter(item["mean_pp"], y+.22, color=colors[y+1], marker="D", s=55, zorder=5)
        ax.set_yticks([.06, 1.06], ["Weak − Reference", "Repaired − Weak"])
        ax.set_ylim(1.65, -.45)
        ax.set_title(title, loc="left", pad=12)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.set_xlabel("Difference in percentage points")
    axes[0].set_xlim(-15, 15)
    axes[1].set_xlim(-3.6, 3.6)
    fig.suptitle("Four paired seeds leave the training effects uncertain", x=.03, ha="left", fontsize=16, weight="bold")
    legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#475569", label="Individual seed difference"),
              Line2D([0], [0], marker="D", color="#475569", label="Mean and exploratory 95% paired t interval")]
    fig.legend(handles=legend, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(.55, .02))
    fig.subplots_adjust(left=.23, right=.97, top=.88, bottom=.21, hspace=.8)
    save(fig, "paired_effects")

    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    left = [0, 0, 0]
    category_colors = ("#CBD5E1", "#6D8FBC", "#087F6C")
    for key, label, color in zip(("at_most_one_pass", "partial", "audit_full"),
                                 ("0 or 1 audit case", "2 to 191 audit cases", "All 192 audit cases"), category_colors):
        values = [result["final"][a][key]/512*100 for a in ARMS]
        ax.barh(range(3), values, left=left, height=.57, color=color, label=label)
        for i, (value, start) in enumerate(zip(values, left)):
            count = result["final"][ARMS[i]][key]
            ax.text(start+value/2, i, str(count), va="center", ha="center", color="#14273B" if key == "at_most_one_pass" else "white", weight="bold")
        left = [a+b for a, b in zip(left, values)]
    ax.set_yticks(range(3), LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25), ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of final draws per training condition (512 each)")
    ax.legend(loc="upper center", bbox_to_anchor=(.48, -.25), frameon=False, ncol=3, fontsize=10)
    fig.suptitle("Partial accuracy and complete success tell different stories", x=.03, ha="left", fontsize=15, weight="bold")
    fig.text(.03, .87, "Labels show program draws. Outcome categories are exploratory.", color="#475569")
    fig.subplots_adjust(left=.14, right=.97, top=.76, bottom=.28)
    save(fig, "outcome_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7))
    for ax, key, title in zip(axes, ("audit_full", "target_bug"), ("Full audit pass rate", "Target bug frequency")):
        for arm_index, (arm, label, color) in enumerate(zip(ARMS, LABELS, colors)):
            panels = [result["fixed_prefix_sensitivity"][str(n)][arm] for n in (32, 128)]
            values = [p[key]/p["draws"]*100 for p in panels]
            positions = [x + (arm_index-1)*.24 for x in (0, 1)]
            ax.bar(positions, values, width=.21, color=color, label=label)
            for x, y, panel in zip(positions, values, panels):
                ax.annotate(str(panel[key]), (x, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, color=color)
        ax.set_xticks((0, 1), ("First 32\nper policy", "All 128\nper policy"))
        ax.set_xlim(-.5, 1.5)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Percent of draws")
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylim(0, 30)
    axes[1].set_ylim(-.15, 2.65)
    fig.suptitle("The fixed prefix gave a different impression", x=.03, ha="left", fontsize=16, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .03))
    fig.text(.03, .01, "Bar labels count draws. Denominators: 128 per condition in the prefix; 512 in the full panel.", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.09, right=.97, top=.8, bottom=.24, wspace=.4)
    save(fig, "prefix_sensitivity")


def preview(output):
    from markdown_it import MarkdownIt
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    for filename, name in (("booking_verifier_article.md", "index.html"), ("booking_publication_review.md", "methods.html")):
        doc = ROOT / "docs" / filename
        tokens = renderer.parse(doc.read_text())
        for token in tokens:
            for child in token.children or []:
                for attr in ("href", "src"):
                    target = child.attrGet(attr)
                    if not target or "://" in target or target.startswith("#"):
                        continue
                    filepart, separator, fragment = target.partition("#")
                    resolved = (doc.parent / filepart).resolve()
                    if resolved.name == "booking_verifier_article.md":
                        relative = "index.html"
                    elif resolved.name == "booking_publication_review.md":
                        relative = "methods.html"
                    else:
                        import os
                        relative = os.path.relpath(resolved, output)
                    child.attrSet(attr, relative + (separator+fragment if separator else ""))
        body = renderer.renderer.render(tokens, renderer.options, {})
        title = doc.read_text().splitlines()[0].lstrip("# ")
        (output / name).write_text("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{escape(title)}</title><style>"
            "*{box-sizing:border-box}body{margin:0;background:#fafbfc;color:#172536;font:18px/1.75 Georgia,serif}"
            "main{max-width:850px;margin:64px auto;padding:0 28px 70px}h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.2;letter-spacing:-.025em}"
            "h1{font-size:clamp(2rem,5vw,3.2rem);margin-bottom:28px}h2{font-size:1.55rem;margin-top:48px}"
            "a{color:#185b91;text-underline-offset:3px}img{width:100%;height:auto;margin:12px 0}"
            "table{border-collapse:collapse;width:100%;font:14px/1.5 system-ui,sans-serif;display:block;overflow-x:auto;margin:26px 0}"
            "th,td{padding:11px 12px;border-bottom:1px solid #d9e1e8;text-align:left}th{background:#edf2f5}"
            "pre{padding:20px;background:#eaf0f4;border-radius:6px;overflow:auto;font:14px/1.6 monospace}"
            "code{font-size:.85em}blockquote{margin:24px 0;padding:4px 24px;border-left:3px solid #00796b;color:#34485a}"
            "li{margin:.6em 0}p{margin:1.1em 0}strong{font-weight:700}"
            "@media(max-width:600px){main{margin:30px auto;padding:0 18px 40px}body{font-size:17px}}"
            "</style></head><body><main>" + body + "</main></body></html>\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--export-evidence", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.export_evidence:
        export_evidence(args.output)
    source = json.loads((args.output / "source_manifest.json").read_text())
    result = compute(load_rows(args.output / "program_scores.csv", source))
    write_json(args.output / "analysis.json", result)
    if args.figures:
        figures(result, args.output)
    if args.preview:
        preview(args.output)
    names = ("analysis.json", "source_manifest.json", "program_scores.csv", "index.html", "methods.html",
             "paired_effects.svg", "paired_effects.png", "outcome_distribution.svg", "outcome_distribution.png",
             "prefix_sensitivity.svg", "prefix_sensitivity.png")
    outputs = {name: sha(args.output / name) for name in names if (args.output / name).exists()}
    write_json(args.output / "output_manifest.json", {"script_sha256": sha(Path(__file__)), "files": outputs})
    print(json.dumps({"draws": result["evaluation_draws"], "final": result["final"],
                      "deduplication": result["source_deduplication_descriptive"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
