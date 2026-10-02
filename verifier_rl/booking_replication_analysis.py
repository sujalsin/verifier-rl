"""Evidence-only analysis of the frozen four-seed booking study.

Standard library only. No cloud calls, model loading, exec/eval, candidate
imports, or generated-code execution. Outputs are derived, not raw evidence.
Uncertainty is at the training-seed level, not the test-input level.
"""

import argparse
import ast
from collections import Counter
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
from statistics import mean, stdev

from . import booking_replication as study
from . import booking_failure_analysis as pilot_analysis
from .parallel_evaluation import checked_program, fingerprint
from .suites import digest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/"runs"/study.RUN_ID/"local-analysis"
VERSION = "booking-replication-analysis-0.1"
ARMS = study.ARMS
LABELS = {"reference": "Reference", "endpoint_omission": "Weak", "repaired": "Repaired"}
COLORS = ["#2563eb", "#d97706", "#15803d", "#9333ea"]
T975_DF3 = 3.182446305284263
METRICS = ("audit_full_pass", "inclusive_signature", "weak_only_acceptance",
           "audit_case_accuracy", "hit_token_cap", "syntax_valid")


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def verified_bytes(path, record):
    data = path.read_bytes()
    require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"],
            "downloaded evidence changed: "+str(path))
    return data


def validate_source(sample, row):
    require(sample["sample_id"] == row["sample_id"] and digest(sample["source"]) == row["source_hash"],
            "generated source differs from evaluated source: "+row["sample_id"])


def exact(bounds):
    require(len(bounds) == 2 and bounds[0] == bounds[1], "unknown outcome must not become a point estimate")
    return bounds[0]


def seed_interval(values):
    """Model-based 95% paired t interval for exactly four seed differences.

    Assumes independent approximately normal seed differences, conditional on
    this task and evaluation seed panel. With n=4 this is fragile, especially
    for rare errors. It is not a missing-data bound or an equivalence test.
    """
    require(len(values) == 4, "four independent training seed pairs required")
    require(all(math.isfinite(v) for v in values), "nonfinite seed difference")
    average = mean(values)
    sd = stdev(values)
    half = T975_DF3*sd/math.sqrt(4)
    return {"mean": average, "sd": sd, "ci95_model_based": [average-half, average+half],
            "seed_differences": list(values), "n_training_seed_pairs": 4,
            "degenerate_observed_variance": sd == 0,
            "method": "paired t, df=3; exploratory model-based uncertainty, not multiplicity-adjusted"}


def row_values(row):
    return {"audit_full_pass": exact(row["audit"]["full_pass_bounds"]),
            "inclusive_signature": exact(row["inclusive_signature_bounds"]),
            "inclusive_signature_all_inputs": exact(row["inclusive_all_inputs_signature_bounds"]),
            "weak_only_acceptance": exact(row["weak_only_acceptance_bounds"]),
            "audit_case_accuracy": exact(row["audit"]["reward_bounds"]),
            "hit_token_cap": int(row["hit_token_cap"]), "syntax_valid": int(row["syntax_valid"]),
            **{name+"_full_pass": exact(row[name]["full_pass_bounds"])
               for name in ("reference", "endpoint_omission", "repaired")}}


def confusion(rows, suite):
    counts = Counter((exact(r[suite]["full_pass_bounds"]), exact(r["audit"]["full_pass_bounds"]))
                     for r in rows)
    tp, fp, tn, fn = (counts[key] for key in ((1,1), (1,0), (0,0), (0,1)))
    return {"suite": suite, "programs": len(rows), "true_accept": tp,
            "false_accept": fp, "true_reject": tn, "false_reject": fn,
            "acceptance_precision": tp/(tp+fp) if tp+fp else None,
            "false_accept_rate_among_audit_failures": fp/(fp+tn) if fp+tn else None,
            "interpretation": "descriptive fixed mixed-policy cohort; audit is not universal correctness"}


def load_verified(directory):
    bundle = directory/"modal-results"
    manifest = read(directory/"download-manifest.json")
    for name, record in manifest["files"].items():
        verified_bytes(bundle/name, record)
    result, inventory, config, recovery = [read(bundle/f"{name}.json")
                                           for name in ("result", "inventory", "config", "recovery")]
    require(result["status"] == "completed", "study incomplete")
    require(fingerprint(config) == inventory["config_hash"] == result["provenance"]["config_hash"],
            "configuration binding differs")
    require(fingerprint(recovery) == result["provenance"]["recovery_hash"], "recovery binding differs")
    require(fingerprint(read(bundle/"predecessor-state.json")) == recovery["old_state_hash"],
            "predecessor evidence differs")
    study.validate_plan(config["plan"])
    for name, expected in config["scientific_sources"].items():
        require(digest((ROOT/name).read_text()) == expected, "frozen scientific source changed: "+name)
    require([c.input_hash for c in study.pilot.cases_for("evaluation")] == config["case_hashes"],
            "fixed test inputs changed")
    require(len(inventory["completed"])+len(inventory["missing"]) == 480, "batch population changed")
    require(study.analyze(inventory["baseline"], result["policies"]) == result["analysis"],
            "paired analysis differs from recorded final report")
    sources = {}
    for seed in study.SEEDS:
        arms = [inventory["saved_arms"][study.label(seed, a)] for a in ARMS]
        require(all(a["first_tokens"] == arms[0]["first_tokens"] for a in arms), "first rollouts not matched")
        for arm in arms:
            study.validate_arm(arm, config["plan"])
            key = study.label(seed, arm["arm"])
            require(arm["metrics"] == result["training_metrics"][key], "training metrics differ")
            for evaluation in arm["evaluations"].values():
                for sample in evaluation["samples"]:
                    require(sample["sample_id"] not in sources, "duplicate sample identity")
                    sources[sample["sample_id"]] = sample
    baseline = inventory["baseline"]
    require(study.policy_summary(baseline["programs"], "baseline", 0) == baseline["summary"],
            "baseline denominator or summary changed")
    for key, group in result["policies"].items():
        label, step = key.rsplit("-", 1)
        require(study.policy_summary(group["programs"], label, int(step)) == group["summary"],
                "policy denominator or summary changed")
        for row in group["programs"]:
            validate_source(sources[row["sample_id"]], row)
    require(all(g["summary"]["unknown_inputs"] == 0 for g in [baseline, *result["policies"].values()]),
            "unresolved outcomes require bounds, not this exact-outcome analysis")
    return result, inventory, config, sources


def failure_catalog(directory, result, inventory, config, sources):
    evidence = directory/"failure-evidence"
    manifest = read(evidence/"manifest.json")
    require(manifest["result_hash"] == fingerprint(result) and manifest["inventory_hash"] == fingerprint(inventory),
            "failure evidence belongs to another population")
    documents = {}
    for key, record in manifest["files"].items():
        data = verified_bytes(evidence/record["path"], record)
        documents[key] = json.loads(data)
    generation = documents.pop("baseline-generation")
    require(generation["parameter_hash"] == config["plan"]["experiment"]["initial_parameter_hash"],
            "wrong baseline weights")
    for index in range(32):
        study.validate_batch(f"eval-baseline-00-{index:02d}", generation["samples"][4*index:4*index+4],
                             "evaluation", config["plan"])
    sources.update((s["sample_id"], s) for s in generation["samples"])
    groups = {"baseline-0": inventory["baseline"], **result["policies"]}
    all_rows = [r for g in groups.values() for r in g["programs"]]
    for row in all_rows:
        validate_source(sources[row["sample_id"]], row)
    selected = {r["sample_id"]: r for r in all_rows if r["weak_only_acceptance_bounds"][1]}
    require(set(selected) == set(documents), "must inspect every weak-only acceptance")
    cases = study.pilot.cases_for("evaluation")
    catalog = []
    for sid, row in selected.items():
        sample = sources[sid]
        outcomes = checked_program(sample, documents[sid], cases, config["runtime"]["image_id"])
        require(study.program_summary(sample, outcomes, "evaluation") == row,
                "replayed protected outcomes differ from saved scores: "+sid)
        failed = [c for c in cases if outcomes[c.input_hash]["passed"] is False]
        witness = min(failed, key=lambda c: (len(c.arguments["bookings"]), c.input_hash))
        catalog.append({"sample_id": sid, "source_hash": row["source_hash"], "source": sample["source"],
                        "ast_hash": digest(ast.dump(ast.parse(sample["source"]), include_attributes=False)),
                        "inclusive_signature": exact(row["inclusive_signature_bounds"]),
                        "inclusive_all_inputs_signature": exact(row["inclusive_all_inputs_signature_bounds"]),
                        "matched_inclusive_inputs": sum(v["inclusive_match"] is True for v in outcomes.values()),
                        "failed_unique_inputs": len(failed), "unique_inputs": len(cases),
                        "failure_families": dict(Counter(c.family for c in failed)),
                        "all_failures_have_shared_endpoint": all(study.pilot.contrast.has_shared_endpoint(c) for c in failed),
                        "scores": {name: exact(row[name]["passed_bounds"])
                                   for name in ("reference", "endpoint_omission", "repaired", "audit")},
                        "witness": {"case_id": witness.name, "input_hash": witness.input_hash,
                                    "arguments": witness.arguments, "expected": witness.expected,
                                    "observed": outcomes[witness.input_hash]["actual"]}})
    return catalog


def training_analysis(result, inventory):
    summaries, updates = [], []
    for policy, metrics in result["training_metrics"].items():
        logs = [x for x in metrics["log_history"] if "grad_norm" in x]
        require([x["step"] for x in logs] == list(range(1, 25)), "missing or duplicate optimizer step log")
        bounds = inventory["saved_arms"][policy]["boundaries"]
        zero = []
        for log in logs:
            step = log["step"]
            before = metrics["before_parameter_hash"] if step == 1 else bounds[str(step-1)]["parameter_hash"]
            changed = before != bounds[str(step)]["parameter_hash"]
            item = {"policy": policy, "seed": int(policy[1:9]), "arm": policy.split("-",1)[1],
                    "step": step, "reward": log["reward"], "reward_std": log["reward_std"],
                    "grad_norm": log["grad_norm"], "logged_loss": log["loss"],
                    "zero_std_fraction": log["frac_reward_zero_std"],
                    "mean_completion_tokens": log["completions/mean_length"],
                    "clipped_fraction": log["completions/clipped_ratio"],
                    "entropy": log["entropy"], "parameter_hash_changed": changed}
            require(all(math.isfinite(item[k]) for k in ("reward", "reward_std", "grad_norm", "logged_loss")),
                    "nonfinite training metric")
            updates.append(item)
            if log["grad_norm"] == 0:
                zero.append(item)
        require(sum(x["grad_norm"] > 0 for x in logs) == metrics["nonzero_gradient_steps"],
                "nonzero-gradient count differs")
        summaries.append({"policy": policy, "seed": int(policy[1:9]), "arm": policy.split("-",1)[1],
                          "updates": 24, "nonzero_gradient_steps": metrics["nonzero_gradient_steps"],
                          "zero_gradient_steps": len(zero), "generated_tokens": metrics["generated_training_tokens"],
                          "mean_reward": mean(x["reward"] for x in logs),
                          "first_six_mean_reward": mean(x["reward"] for x in logs[:6]),
                          "last_six_mean_reward": mean(x["reward"] for x in logs[-6:]),
                          "mean_gradient_norm": mean(x["grad_norm"] for x in logs),
                          "mean_clipped_fraction": mean(x["completions/clipped_ratio"] for x in logs),
                          "parameters_changed": metrics["parameters_changed"],
                          "final_reload_verified": metrics["final_reload_verified"]})
    return summaries, updates


def analyze(directory, *, with_pilot=True):
    result, inventory, config, sources = load_verified(directory)
    failures = failure_catalog(directory, result, inventory, config, sources)
    groups = {"baseline-0": inventory["baseline"], **result["policies"]}
    programs, policies, common_panel = [], [], []
    for key, group in groups.items():
        label, step = key.rsplit("-",1)
        seed, arm = (None, "baseline") if label == "baseline" else (int(label[1:9]), label.split("-",1)[1])
        rows = group["programs"]
        for row in rows:
            sample = sources[row["sample_id"]]
            programs.append({"sample_id": row["sample_id"], "training_seed": seed, "arm": arm,
                             "step": int(step), "draw_seed": sample["seed"], "source_hash": row["source_hash"],
                             "tokens": sample["tokens"], "extraction_status": row["extraction_status"],
                             **row_values(row)})
        item = {"policy": key, "seed": seed, "arm": arm, "step": int(step), "programs": len(rows),
                **{m: mean(row_values(r)[m] for r in rows) for m in METRICS},
                "unique_sources": len({r["source_hash"] for r in rows}),
                "mean_tokens": mean(sources[r["sample_id"]]["tokens"] for r in rows)}
        policies.append(item)
        fixed = [r for r in rows if int(r["sample_id"].rsplit("-",1)[1]) < 19032]
        require(len(fixed) == 32, "common draw panel changed")
        common_panel.append({"policy": key, "seed": seed, "arm": arm, "step": int(step), "programs": 32,
                             **{m: mean(row_values(r)[m] for r in fixed) for m in METRICS}})
    by_key = {p["policy"]: p for p in policies}
    effects, pairs = {}, []
    for first, second in (("endpoint_omission", "reference"), ("repaired", "endpoint_omission")):
        contrast = first+"_minus_"+second
        effects[contrast] = {}
        for metric in METRICS:
            differences = []
            for seed in study.SEEDS:
                difference = by_key[f"{study.label(seed, first)}-24"][metric]-by_key[f"{study.label(seed, second)}-24"][metric]
                differences.append(difference)
                pairs.append({"contrast": contrast, "metric": metric, "seed": seed,
                              "difference": difference, "percentage_points": 100*difference})
            effects[contrast][metric] = seed_interval(differences)
    final_means = {arm: {m: mean(by_key[f"{study.label(s,arm)}-24"][m] for s in study.SEEDS)
                        for m in METRICS} for arm in ARMS}
    all_rows = [r for group in groups.values() for r in group["programs"]]
    training, updates = training_analysis(result, inventory)
    pilot = None
    if with_pilot:
        path = ROOT/"runs/qwen-booking-matched-training-20260929-v1/analysis/weak-only-001"
        pilot = pilot_analysis.analyze(path)
        # Keep evidence/units separate. Never add the pilot to the four-seed estimates.
        pilot = {k: pilot[k] for k in ("result_sha256", "policy_summary", "limitations", "candidate_executions")}
    benchmark_path = ROOT/"runs/booking-program-sandbox-benchmark-20260930-v1/result.json"
    benchmark = read(benchmark_path)
    require(benchmark["status"] == "passed" and benchmark["controls"] == 13, "benchmark did not pass")
    engineering = {"benchmark_file_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
                   "authored_controls": benchmark["controls"],
                   "parity_pairs": sum(p["matched_inputs"] for p in benchmark["parity"].values()),
                   "benchmark_inputs_per_condition": benchmark["timings"]["parallel"]["inputs"],
                   "sequential_seconds": benchmark["timings"]["sequential"]["seconds"],
                   "parallel_seconds": benchmark["timings"]["parallel"]["seconds"],
                   "observed_benchmark_speedup": benchmark["timings"]["sequential"]["seconds"]/benchmark["timings"]["parallel"]["seconds"],
                   "full_study_speedup_measured": False,
                   "unique_evaluation_sandboxes": result["unique_evaluation_sandboxes"],
                   "finalization_accounting_not_provider_bill": result["accounting_not_provider_billing"]}
    return {"version": VERSION, "result_hash": fingerprint(result), "inventory_hash": fingerprint(inventory),
            "workload": study.workload(), "baseline": by_key["baseline-0"], "final_arm_means": final_means,
            "policies": policies, "programs": programs, "paired_effects": effects, "seed_pairs": pairs,
            "common_32_draw_trajectories": common_panel,
            "verifier_confusion": [confusion(all_rows, a) for a in ARMS],
            "failure_catalog": failures, "training": training, "training_updates": updates,
            "pilot_separate": pilot, "engineering": engineering,
            "evidence_scope": {"program_summaries_verified": len(all_rows),
                               "generated_sources_bound_to_scores": len(sources),
                               "weak_only_programs_replayed_from_original_records": len(failures),
                               "protected_input_outcomes_replayed": len(failures)*287,
                               "unknown_inputs": 0, "candidate_executions": 0,
                               "full_raw_archive_reaudited_locally": False},
            "statistical_limits": ["Four independent training seed triplets on one task.",
                                   "Evaluation draws share seeds across policies; tests are not independent replications.",
                                   "Shared baseline measured once, never replicated as four independent baselines.",
                                   "Intervals are exploratory paired t approximations with df=3; normality is untestable here.",
                                   "No multiplicity adjustment or confirmatory significance/equivalence claim.",
                                   "Audit is an existing development suite, not an untouched final benchmark.",
                                   "The verifier confusion cohort mixes baseline, intermediate and final policies.",
                                   "Pilot retained separately; no favorable-checkpoint selection."]}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_header(width, height, title, description):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            f'<title>{escape(title)}</title><desc>{escape(description)}</desc>',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#172033} .small{font-size:12px} .label{font-size:14px} .title{font-size:19px;font-weight:bold}</style>']


def text_svg(parts, x, y, text, cls="label", anchor="start"):
    parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(str(text))}</text>')


def policy_plot(analysis):
    parts = svg_header(920, 470, "Final policy outcomes across four training seeds",
                       "All seed triplets; full audit passes and target error rates. Baseline is one shared 128 draw sample.")
    text_svg(parts, 25, 30, "Final policies across all four training seeds", "title")
    panels = [("audit_full_pass", "Audit full pass", 40),
              ("inclusive_signature", "Inclusive endpoint signature", 5),
              ("weak_only_acceptance", "Weak only acceptance", 5)]
    for j, (metric, title, maximum) in enumerate(panels):
        left, top, width, height = 65+j*300, 95, 210, 245
        text_svg(parts, left+105, 65, title, anchor="middle")
        y = lambda v: top+height-height*v/maximum
        for index in range(5):
            tick = maximum*index/4
            parts.append(f'<line x1="{left}" y1="{y(tick)}" x2="{left+width}" y2="{y(tick)}" stroke="#e2e8f0"/>')
            text_svg(parts, left-8, y(tick)+4, f"{tick:g}%", "small", "end")
        base_y = y(100*analysis["baseline"][metric])
        parts.append(f'<line x1="{left}" y1="{base_y}" x2="{left+width}" y2="{base_y}" stroke="#64748b" stroke-dasharray="5 4"/>')
        for n, seed in enumerate(study.SEEDS):
            values = [next(p[metric] for p in analysis["policies"] if p["seed"] == seed and p["arm"] == arm and p["step"] == 24)
                      for arm in ARMS]
            points = " ".join(f"{left+20+85*k},{y(100*v)}" for k,v in enumerate(values))
            parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[n]}" stroke-width="1.5" opacity="0.75"/>')
            for k,v in enumerate(values):
                parts.append(f'<circle cx="{left+20+85*k}" cy="{y(100*v)}" r="4" fill="{COLORS[n]}"/>')
        for k, arm in enumerate(ARMS):
            x = left+20+85*k
            average = 100*analysis["final_arm_means"][arm][metric]
            parts.append(f'<rect x="{x-6}" y="{y(average)-3}" width="12" height="6" fill="#111827"/>')
            text_svg(parts, x, top+height+25, LABELS[arm], "small", "middle")
            text_svg(parts, x, top+height+43, f"{average:.2f}%", "small", "middle")
    for n, seed in enumerate(study.SEEDS):
        x = 70+n*180
        parts.append(f'<circle cx="{x}" cy="413" r="4" fill="{COLORS[n]}"/>')
        text_svg(parts, x+10, 417, f"Seed {seed}", "small")
    text_svg(parts, 25, 448, "128 draws per final policy. Black marks = seed means. Dashed lines = shared baseline, not four baseline replicas.", "small")
    return "\n".join(parts+["</svg>"])


def paired_plot(analysis):
    parts = svg_header(950, 455, "Paired final checkpoint effects and exploratory uncertainty",
                       "Four seed differences per contrast and a model-based paired t interval with three degrees of freedom.")
    text_svg(parts, 25, 30, "Paired final checkpoint effects", "title")
    text_svg(parts, 25, 55, "Dots are the four seed differences. Diamond and bar show mean and exploratory 95% paired t interval.", "small")
    rows = []
    for contrast, values in analysis["paired_effects"].items():
        for metric in ("audit_full_pass", "inclusive_signature"):
            rows.append((contrast, metric, values[metric]))
    low, high, left, width = -16, 16, 330, 440
    scale = lambda v: left+(v-low)*width/(high-low)
    for tick in (-15, -10, -5, 0, 5, 10, 15):
        parts.append(f'<line x1="{scale(tick)}" y1="85" x2="{scale(tick)}" y2="355" stroke="{"#94a3b8" if tick == 0 else "#e2e8f0"}"/>')
        text_svg(parts, scale(tick), 380, f"{tick:+d}", "small", "middle")
    for index, (contrast, metric, effect) in enumerate(rows):
        y = 120+index*68
        title = "Weak minus reference" if contrast.startswith("endpoint") else "Repaired minus weak"
        text_svg(parts, 25, y-5, title)
        text_svg(parts, 25, y+14, "Audit full pass" if metric == "audit_full_pass" else "Inclusive endpoint signature", "small")
        lo, hi = [100*v for v in effect["ci95_model_based"]]
        average = 100*effect["mean"]
        parts.append(f'<line x1="{scale(lo)}" y1="{y}" x2="{scale(hi)}" y2="{y}" stroke="#111827" stroke-width="2"/>')
        for k,v in enumerate(effect["seed_differences"]):
            parts.append(f'<circle cx="{scale(100*v)}" cy="{y+(-10 if k%2 == 0 else 10)}" r="4" fill="{COLORS[k]}"/>')
        x = scale(average)
        parts.append(f'<polygon points="{x},{y-6} {x+6},{y} {x},{y+6} {x-6},{y}" fill="#111827"/>')
        text_svg(parts, 792, y-3, f"{average:+.2f} pp", "label")
        text_svg(parts, 792, y+16, f"[{lo:+.2f}, {hi:+.2f}]", "small")
    text_svg(parts, 550, 403, "Difference in percentage points", "small", "middle")
    text_svg(parts, 25, 431, "n = 4 paired training seeds; intervals assume approximately normal seed differences. No equivalence or significance claim.", "small")
    return "\n".join(parts+["</svg>"])


def training_plot(analysis):
    parts = svg_header(960, 790, "All training reward traces", "Raw group mean reward for all 288 updates; objectives differ across columns.")
    text_svg(parts, 25, 30, "All 12 training runs", "title")
    text_svg(parts, 25, 55, "Group mean rewards under each arm's own verifier; cross-column reward magnitudes are not common correctness scores.", "small")
    for row, seed in enumerate(study.SEEDS):
        for col, arm in enumerate(ARMS):
            left, top, width, height = 50+col*310, 110+row*160, 265, 100
            text_svg(parts, left, top-14, f"{seed}   {LABELS[arm]}")
            for value in (0, .5, 1):
                y = top+height-height*value
                parts.append(f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" stroke="#e2e8f0"/>')
                text_svg(parts, left-6, y+4, str(value), "small", "end")
            updates = [u for u in analysis["training_updates"] if u["seed"] == seed and u["arm"] == arm]
            points = " ".join(f"{left+(u['step']-1)*width/23},{top+height-height*u['reward']}" for u in updates)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[row]}" stroke-width="1.5"/>')
            for u in updates:
                if u["grad_norm"] == 0:
                    parts.append(f'<circle cx="{left+(u["step"]-1)*width/23}" cy="{top+height-height*u["reward"]}" r="5" fill="white" stroke="#dc2626" stroke-width="2"/>')
            for tick in (1,12,24):
                text_svg(parts, left+(tick-1)*width/23, top+height+19, tick, "small", "middle")
    text_svg(parts, 25, 772, "x = optimizer update. Red rings = zero-gradient groups. No smoothing, discarded seeds, or selected best checkpoint.", "small")
    return "\n".join(parts+["</svg>"])


def save_outputs(analysis, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/"analysis.json").write_text(json.dumps(analysis, sort_keys=True, indent=2)+"\n")
    for name, key in (("policy_metrics", "policies"), ("program_metrics", "programs"),
                      ("seed_pairs", "seed_pairs"), ("training_summary", "training"),
                      ("training_updates", "training_updates"), ("common_draw_trajectories", "common_32_draw_trajectories"),
                      ("verifier_confusion", "verifier_confusion")):
        write_csv(output/f"{name}.csv", analysis[key])
    for name, render in (("policy_outcomes", policy_plot), ("paired_effects", paired_plot), ("training_rewards", training_plot)):
        (output/f"{name}.svg").write_text(render(analysis)+"\n")
    (output/"output_manifest.json").write_text(json.dumps({
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_result_hash": analysis["result_hash"],
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())
                  if p.is_file() and p.name != "output_manifest.json"}}, sort_keys=True, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/booking-replication")
    args = parser.parse_args()
    analysis = analyze(args.input)
    save_outputs(analysis, args.output)
    print(json.dumps({"output": str(args.output), "evidence": analysis["evidence_scope"],
                      "final_means": analysis["final_arm_means"], "paired_effects": analysis["paired_effects"]}, indent=2))


if __name__ == "__main__":
    main()
