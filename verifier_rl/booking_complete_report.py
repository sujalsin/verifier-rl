"""Build one exhaustive local research record from existing booking evidence.

No model loading, cloud calls, or candidate execution. Markdown is a report,
not a byte-for-byte replacement for the original checkpoint/log archive.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

from . import booking_replication_analysis as primary
from . import booking_behavior_analysis as behavior

ROOT = primary.ROOT
RUN = ROOT/"runs"/primary.study.RUN_ID
LOCAL = RUN/"local-analysis"
PILOT = ROOT/"runs/qwen-booking-matched-training-20260929-v1/analysis/weak-only-001"
OUTPUT = ROOT/"docs/booking_complete_study_record.md"
MANIFEST = ROOT/"reports/booking-complete/manifest.json"
SUITES = ("reference", "endpoint_omission", "repaired", "audit")
LABELS = primary.LABELS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fence(value, language="text"):
    """Keep arbitrary generated text literal, including embedded Markdown fences."""
    value = str(value)
    longest = max((len(x) for x in re.findall(r"`+", value)), default=0)
    delimiter = "`"*max(3, longest+1)
    return f"{delimiter}{language}\n{value}\n{delimiter}\n\n"


def cell(value):
    if isinstance(value, float):
        value = f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    rows = list(rows)
    primary.require(all(len(r) == len(headers) for r in rows), "table width mismatch")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(cell, r)) + " |" for r in rows)])+"\n\n"


def pct(value):
    return f"{100*value:.2f}%"


def bounds(values, *, percentage=False):
    fn = pct if percentage else str
    return fn(values[0]) if values[0] == values[1] else f"{fn(values[0])} to {fn(values[1])}"


class Sources:
    def __init__(self):
        self.entries = {}

    def read_bytes(self, path):
        path = Path(path)
        data = path.read_bytes()
        self.entries[str(path.relative_to(ROOT))] = {"bytes": len(data), "sha256": sha(data)}
        return data

    def read(self, path):
        return json.loads(self.read_bytes(path))

    def text(self, path):
        return self.read_bytes(path).decode()

    def checked_outputs(self, directory, analyzer):
        manifest = self.read(directory/"output_manifest.json")
        primary.require(sha(self.read_bytes(analyzer)) == manifest["analyzer_sha256"], "analyzer changed")
        for name, expected in manifest["files"].items():
            primary.require(sha(self.read_bytes(directory/name)) == expected, "derived file changed: "+name)


def load():
    sources = Sources()
    sources.checked_outputs(ROOT/"reports/booking-replication", Path(primary.__file__))
    sources.checked_outputs(ROOT/"reports/booking-behavior", Path(behavior.__file__))
    analysis = sources.read(ROOT/"reports/booking-replication/analysis.json")
    exploratory = sources.read(ROOT/"reports/booking-behavior/findings.json")
    primary.require(exploratory["primary_analysis_file_sha256"] == sha(
        sources.read_bytes(ROOT/"reports/booking-replication/analysis.json")), "exploration binding changed")
    result, inventory, config, generated = primary.load_verified(LOCAL)
    for path in sorted((LOCAL/"modal-results").glob("*.json")):
        sources.read_bytes(path)
    sources.read_bytes(LOCAL/"download-manifest.json")
    primary.require(analysis["result_hash"] == primary.fingerprint(result)
                    and analysis["inventory_hash"] == primary.fingerprint(inventory), "primary binding changed")
    failure_manifest = sources.read(LOCAL/"failure-evidence/manifest.json")
    raw_failures = {}
    for key, record in failure_manifest["files"].items():
        path = LOCAL/"failure-evidence"/record["path"]
        raw = sources.read_bytes(path)
        primary.require(sha(raw) == record["sha256"] and len(raw) == record["bytes"], "raw evidence changed")
        if key == "baseline-generation":
            baseline_generation = json.loads(raw)
            generated.update((s["sample_id"], s) for s in baseline_generation["samples"])
        else:
            raw_failures[key] = json.loads(raw)
    groups = {"baseline-0": inventory["baseline"], **result["policies"]}
    rows = {r["sample_id"]: r for group in groups.values() for r in group["programs"]}
    primary.require(len(rows) == len(generated) == 2048 and set(rows) == set(generated), "incomplete evaluation population")
    for sid, row in rows.items():
        primary.validate_source(generated[sid], row)
    primary.require(set(raw_failures) == {r["sample_id"] for r in analysis["failure_catalog"]}, "failure population changed")
    primary.require(sum(len(a["boundaries"]) for a in inventory["saved_arms"].values()) == 288,
                    "checkpoint population changed")
    # Reuse the validators, never the candidate programs, for every exported failure.
    rebuilt_failures = primary.failure_catalog(LOCAL, result, inventory, config, generated)
    primary.require(rebuilt_failures == analysis["failure_catalog"], "failure interpretation changed")
    pilot_analysis = primary.pilot_analysis.analyze(PILOT)
    primary.require(pilot_analysis["policy_summary"] == analysis["pilot_separate"]["policy_summary"],
                    "pilot summary changed")
    pilot_result = sources.read(PILOT/"verified_final_result.json")
    sources.read_bytes(PILOT/"verification_receipt.json")
    for record in sources.read(PILOT/"download_manifest.json"):
        data = sources.read_bytes(PILOT/"evidence"/record["path"])
        primary.require(sha(data) == record["sha256"] and len(data) == record["bytes"], "pilot evidence changed")
    return {"sources": sources, "primary": analysis, "behavior": exploratory,
            "result": result, "inventory": inventory, "config": config, "groups": groups,
            "generations": generated, "rows": rows, "failure_documents": raw_failures,
            "baseline_generation": baseline_generation, "pilot_analysis": pilot_analysis,
            "pilot_result": pilot_result}


class Report:
    def __init__(self):
        self.parts = []
        self.sections = []

    def add(self, text):
        self.parts.append(text.rstrip()+"\n\n")

    def heading(self, title, level=2):
        self.add("#"*level+" "+title)
        if level == 2:
            self.sections.append(title)

    def data(self, value, label=None, *, compact=False):
        if label:
            self.add(label)
        self.parts.append(fence(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                          indent=None if compact else 2,
                                          separators=(",", ":") if compact else None), "json"))

    def prose_document(self, sources, path):
        content = sources.text(path).splitlines()[1:]
        in_fence = False
        out = []
        for line in content:
            if line.startswith("```"):
                in_fence = not in_fence
            if not in_fence and line.startswith("#"):
                line = "#"+line
            # Numerical records below make the report usable without adjacent images.
            if not in_fence and line.startswith("!["):
                continue
            out.append(line)
        self.add("\n".join(out))


def cohort_label(key):
    if key == "baseline-0":
        return "Original policy baseline"
    policy, step = key.rsplit("-", 1)
    seed, arm = policy[1:].split("-", 1)
    return f"Seed {seed} {LABELS[arm]} update {step}"


def counts_for(rows):
    return [sum(primary.exact(r[s]["full_pass_bounds"]) for r in rows) for s in SUITES]


def record_cases():
    suites = primary.study.pilot.coverage.suites()
    weak, omitted = primary.study.pilot.contrast.partition(suites["training"])
    removed, added, repaired = primary.study.repair()
    members = {name: {c.input_hash for c in seq} for name, seq in
               (("reference", suites["training"]), ("weak", weak), ("omitted", omitted),
                ("repaired", repaired), ("audit", suites["audit"]), ("removed", removed), ("added", added))}
    return [{"case_id": case.name, "suite": role, "family": case.family,
             "input_hash": case.input_hash, "arguments": case.arguments, "expected": case.expected,
             "inclusive_oracle_expected": primary.study.pilot.contrast.inclusive_answer(case.arguments_json),
             "shared_endpoint": primary.study.pilot.contrast.has_shared_endpoint(case),
             "memberships_by_input": {name: case.input_hash in hashes for name, hashes in members.items()}}
            for role, seq in suites.items() for case in seq]


def render(data):
    report = Report()
    sources, a, b = data["sources"], data["primary"], data["behavior"]
    inv, config, result = data["inventory"], data["config"], data["result"]
    groups, generations, rows = data["groups"], data["generations"], data["rows"]
    report.heading("Complete booking verifier RL study record", 1)
    report.add("This is the detailed analytical record of the completed four-seed, three-condition booking-capacity study, "
               "with its separate earlier pilot. It consolidates the research questions, methods, every saved evaluation "
               "cohort, every training update and checkpoint receipt, every evaluated response and score, exploratory "
               "observations, inspected failure evidence, and operational recovery history into one file. It is written "
               "for the project owner to understand, audit, and explain the work—not as a claim that every hypothesis succeeded.")
    report.add("The central results are an observed verifier blind spot and a repair that rejects all 38 observed weak false "
               "acceptances without rejecting any of 440 audit-passing draws. Four paired training seeds did not establish "
               "amplification of the target bug or a training benefit from the repair. Broader observations show disagreement "
               "between partial credit and full correctness, between acceptance decisions and partial-score ordering, and "
               "between response appearance and executable behavior. Both positive and inconclusive results are retained.")
    report.add("Replication run: `"+primary.study.RUN_ID+"`. This is an offline snapshot assembled from the local evidence. "
               "No new cloud jobs, model calls, candidate executions, or downloads are part of producing this document.")
    report.add("CONTENTS_PLACEHOLDER")
    report.heading("Coverage and important exclusions")
    report.parts.append(table(["Record", "Coverage"], [
        ["Training conditions", "4 seeds × 3 arms = 12 runs"],
        ["Training updates and checkpoint receipts", "24 per run = 288; every available metric field and receipt included"],
        ["Evaluation cohorts", "1 shared baseline + 12 at update 12 + 12 at update 24 = 25"],
        ["Evaluation draws", "128 baseline + 384 intermediate + 1536 final = 2048; all responses, extracted sources and score records included"],
        ["Post-training evaluation batches", "480 derived batch summaries; 330 retained plus 150 newly completed in the final recovery"],
        ["False-acceptance investigation", "All 38 draws, their witnesses and all 10906 original protected input records available locally"],
        ["Test definitions", "96 training plus 192 audit entries; 287 unique inputs; all arguments and expected results included"],
        ["Exploratory analysis", "All reported distributions, 48 original paired metric differences, rank comparisons, length bins and selected examples"],
        ["Earlier pilot", "Separate 5-cohort summary and complete locally exported final report, with 3 unresolved outcomes retained"],
        ["Operational evidence", "Benchmark and available control receipts, incident records, accounting snapshots, hashes and provenance"],
    ]))
    report.add("This file is exhaustive for the defined analytical records, not a full cloud backup. It does not contain "
               "model or optimizer tensors, every cloud log line, the entire per-input archive for all candidates, or raw "
               "training rollout/reward journals that were not part of the compact local export. The 1,152 training "
               "completions are represented by saved update aggregates and the initial matched token groups; do not "
               "mistake those for all 1,152 original response texts. No missing records are fabricated.")
    report.add("Checkpoint state was saved after every update, but retention kept the last two full trainer states and "
               "separate model snapshots at updates 12 and 24. All 288 boundary receipts are available; that does not mean "
               "all 288 tensor directories are still retained. Only baseline and updates 12 and 24 received the declared "
               "fixed evaluations. Other updates have training metrics and checkpoint evidence, not invented audit results.")
    report.add("Large raw protected records are placed at the end. The opening narrative and tables are the readable "
               "research report; the appendices preserve exact fields. Source hashes and file paths support traceability. "
               "Historical launch commands and generated code are quoted as evidence, not instructions to execute them.")

    report.heading("Definitions and how to interpret the numbers")
    report.parts.append(table(["Term", "Meaning and limitation"], [
        ["Reference", "Training reward from all 96 training cases"],
        ["Weak or endpoint_omission", "Training reward from 57 cases after omitting 39 shared-endpoint cases"],
        ["Repaired", "57-case reward: replace 8 ordinary weak cases with 8 distinguishing endpoint cases"],
        ["Audit", "192 fixed development cases; never used as the scalar training reward"],
        ["Full audit pass", "Pass every audit case; finite-suite success, not universal correctness"],
        ["Mean case accuracy", "Average fraction of audit tests passed across all program draws, including failures"],
        ["Inclusive signature", "Match the declared inclusive-end behavioral signature on its prescribed cases"],
        ["Weak-only acceptance", "Pass every weak case while failing the reference criterion; distinct from an exact bug signature"],
        ["Seed", "Training seed identifies a policy replication; draw seed identifies a generated evaluation response"],
        ["Bounds", "Unknown-outcome lower and upper bounds; equal endpoints mean an exact saved value, not certainty about a population"],
        ["Confidence interval", "Exploratory paired t interval over 4 training-seed differences; not a missing-data bound"],
        ["Token cap", "Response reached the generation length limit; extracted code may nevertheless be complete"],
        ["grad_norm", "Logged gradient norm; not the same as parameter displacement after an optimizer step"],
        ["loss", "Logged scalar objective; zero value does not imply zero derivative"],
        ["Guard balance", "Settled compute estimates + outstanding maximum reservations + overhead allowance; not provider billing"],
    ]))
    report.add("Tests are nested within programs, which are nested within trained policies. Neither hundreds of tests per "
               "program nor hundreds of thousands of derived program pairs create additional independent training seeds. "
               "Shared baseline observations are counted once, not replicated four times.")

    report.heading("Exact task prompt and experimental configuration")
    report.add("The prompt explicitly specifies half-open intervals and requests code without prose. Output extraction and "
               "functional scoring nevertheless evaluate extracted code; a response may pass that functional metric while "
               "violating the no-prose instruction. No separate comprehensive instruction-following metric was prespecified.")
    report.parts.append(fence(config["plan"]["experiment"]["prompt"]))
    report.add("Pinned GPU environment in the recorded launcher: Python 3.12, Modal 1.5.5, PyTorch 2.8.0, Transformers "
               "4.57.1, TRL 0.28.0, datasets 3.5.1, accelerate 1.12.0; L40S GPU with two CPU cores and 32 GiB memory. "
               "These are recorded study settings, not recommendations about current software. The exact frozen plan "
               "and runtime metadata in the final provenance appendix distinguish original protocol settings from "
               "later execution-only amendments.")
    sources.read_bytes(ROOT/"modal_booking_replication.py")
    report.add("The full configuration is embedded in [Final recovery provenance and reproduction]"
               "(#final-recovery-provenance-and-reproduction). The following findings explain the experimental "
               "settings and results before the exhaustive raw records.")

    report.heading("Primary findings and full scientific interpretation")
    report.prose_document(sources, ROOT/"docs/booking_replication_analysis.md")
    report.heading("Exploratory observations and full behavioral interpretation")
    report.prose_document(sources, ROOT/"docs/booking_behavior_findings.md")

    report.heading("Every evaluation checkpoint")
    report.add("All 25 cohorts are shown. Counts for the four scorers apply to the same generated programs. The baseline "
               "is shared. Update 12 has 32 draws per policy; update 24 has 128, so raw counts across those updates "
               "should not be compared without their denominators.")
    metrics = {p["policy"]: p for p in a["policies"]}
    report.parts.append(table(["Cohort", "N", "Reference full", "Weak full", "Repaired full", "Audit full", "Audit mean"],
        [[key, len(g["programs"]), *counts_for(g["programs"]), pct(metrics[key]["audit_case_accuracy"])]
         for key, g in sorted(groups.items())]))
    report.parts.append(table(["Cohort", "Audit full rate", "Inclusive count", "Weak-only count", "Syntax valid", "Capped", "Unique sources", "Mean tokens"],
        [[key, pct(metrics[key]["audit_full_pass"]), bounds(g["summary"]["inclusive_signature_bounds"]),
          bounds(g["summary"]["weak_only_acceptance_bounds"]), g["summary"]["syntax_valid"],
          g["summary"]["hit_token_cap"], g["summary"]["unique_sources"], metrics[key]["mean_tokens"]]
         for key, g in sorted(groups.items())]))
    for key, group in sorted(groups.items()):
        report.heading(cohort_label(key), 3)
        report.add("Saved cohort identifier: `"+key+"`.")
        if key == "baseline-0":
            parameter_hash = data["baseline_generation"]["parameter_hash"]
        else:
            policy, step = key.rsplit("-", 1)
            parameter_hash = inv["saved_arms"][policy]["evaluations"][step]["parameter_hash"]
        report.add("Evaluation model parameter hash: `"+parameter_hash+"`.")
        report.parts.append(table(["Scorer", "Case count per draw", "Total passed case slots", "Mean case accuracy", "Full passes"],
            [[LABELS.get(s, s.title()), group["programs"][0][s]["total"],
              sum(primary.exact(r[s]["passed_bounds"]) for r in group["programs"]),
              pct(mean(primary.exact(r[s]["reward_bounds"]) for r in group["programs"])),
              sum(primary.exact(r[s]["full_pass_bounds"]) for r in group["programs"])] for s in SUITES]))
        report.data(group["summary"], "Every field of the saved cohort summary.")

    report.heading("All paired contrasts and checkpoint trajectories")
    report.add("The first two metrics in each declared contrast are primary; the others are secondary. All six metrics "
               "for each of the two contrasts, with all four seed differences, are retained below. The interval calculation is exploratory "
               "and uses four seed pairs, with no multiplicity adjustment or equivalence claim.")
    report.parts.append(table(["Contrast", "Metric", "Mean difference pp", "Lower 95 pp", "Upper 95 pp", "Four seed differences pp"],
        [[c, m, 100*v["mean"], 100*v["ci95_model_based"][0], 100*v["ci95_model_based"][1],
          ", ".join(f"{100*x:+.6f}" for x in v["seed_differences"])]
         for c, ms in a["paired_effects"].items() for m, v in ms.items()]))
    report.data(a["paired_effects"], "Exact paired calculations, including observed variances and methodological flags.")
    report.parts.append(table(["Cohort", "Common N", "Audit full", "Inclusive", "Weak only", "Audit mean", "Capped", "Syntax valid"],
        [[p["policy"], p["programs"], pct(p["audit_full_pass"]), pct(p["inclusive_signature"]),
          pct(p["weak_only_acceptance"]), pct(p["audit_case_accuracy"]), pct(p["hit_token_cap"]), pct(p["syntax_valid"])]
         for p in a["common_32_draw_trajectories"]]))
    report.add("The common panel uses draw seeds 19000 through 19031 at every checkpoint. It is not a substitute for the "
               "complete final cohort. In particular, the repaired final target-error count is zero on this prefix "
               "but eleven across all final draws, and the weak-versus-repaired full-pass ordering changes.")
    report.data(a["seed_pairs"], "All 48 seed-level metric differences, unrounded.")

    report.heading("Every training update and checkpoint receipt")
    report.add("Each run has 24 optimizer updates. The table gives key logged diagnostics; the adjacent JSON retains "
               "every saved log field, including learning rate, lengths, clipping statistics, token counters and "
               "step times. Logged step time is not total study wall time: checkpointing, evaluation, queueing and "
               "resumed intervals complicate that interpretation. A receipt proves recorded state validation, not "
               "that its model tensors were downloaded or are all still retained.")
    report.parts.append(table(["Policy", "Updates", "Positive gradient", "Zero gradient", "Generated tokens", "Mean reward", "First six", "Last six", "Final reload"],
        [[p["policy"], p["updates"], p["nonzero_gradient_steps"], p["zero_gradient_steps"], p["generated_tokens"],
          p["mean_reward"], p["first_six_mean_reward"], p["last_six_mean_reward"], p["final_reload_verified"]]
         for p in a["training"]]))
    for policy, arm in sorted(inv["saved_arms"].items()):
        report.heading(cohort_label(policy+"-24").replace(" update 24", " training record"), 3)
        report.add("Policy identifier: `"+policy+"`. Rewards use this arm's own verifier; cross-arm reward magnitudes "
                   "are not a common correctness scale. Audit outcomes are listed above, not inferred from reward.")
        updates = [u for u in a["training_updates"] if u["policy"] == policy]
        report.parts.append(table(["Update", "Reward", "Reward std", "Grad norm", "Loss", "Mean tokens", "Capped fraction", "Entropy", "Weights changed"],
            [[u[k] for k in ("step", "reward", "reward_std", "grad_norm", "logged_loss", "mean_completion_tokens",
                            "clipped_fraction", "entropy", "parameter_hash_changed")] for u in updates]))
        report.data(arm["metrics"], "Complete saved training metrics and log history.")
        report.data(arm["first_tokens"], "Exact initial completion token groups used to verify matched starts across arms.")
        for step in range(1,25):
            report.data({"record_type": "checkpoint_receipt", "policy": policy, "update": step,
                         "receipt": arm["boundaries"][str(step)]})
    report.add("All 288 scalar losses were logged as zero or negative zero. Of these updates, 285 had nonzero gradient "
               "norms. The three zero-gradient groups had identical rewards within each group and still changed "
               "parameter hashes. That is consistent with retained AdamW moments; no new tensor-level decomposition "
               "of those parameter changes was performed for this report.")

    report.heading("Every evaluation batch")
    report.add("The following 480 post-training batch rows are derived from final verified program summaries. They "
               "are not fabricated copies of individual cloud batch receipts. The recovery inventory is an earlier "
               "330-complete and 150-missing snapshot; its word missing does not mean those batches are still unfinished. "
               "The final result confirms the remaining 150 completed. Baseline contributes another 32 batches.")
    batch_rows = []
    for key, group in sorted(groups.items()):
        if key == "baseline-0":
            continue
        ordered = sorted(group["programs"], key=lambda row: generations[row["sample_id"]]["seed"])
        for index in range(0,len(ordered),4):
            key_batch = "eval-"+key+f"-{index//4:02d}"
            chunk = ordered[index:index+4]
            batch_rows.append([key_batch, "retained" if key_batch in inv["completed"] else "final recovery",
                               len(chunk), sum(primary.exact(r["audit"]["full_pass_bounds"]) for r in chunk),
                               sum(r["unknown_inputs"] for r in chunk)])
    primary.require(len(batch_rows) == 480 and sum(r[1] == "retained" for r in batch_rows) == 330,
                    "batch reconstruction differs")
    report.parts.append(table(["Batch", "Final recovery role", "Draws", "Audit full passes", "Unknown inputs"], batch_rows))

    report.heading("Every evaluated program in compact tables")
    report.add("Every draw remains in the denominator, including extraction or syntax failures and duplicate sources. "
               "The draw seed together with the cohort heading identifies the full sample ID. R, W, P and A denote "
               "passed reference, weak, repaired and audit cases, with denominators 96, 57, 57 and 192. Full response "
               "text, source, generation metadata and all exact score fields are in the response appendix.")
    for key, group in sorted(groups.items()):
        report.heading(cohort_label(key), 3)
        report.parts.append(table(["Draw seed", "R", "W", "P", "A", "Inclusive", "Weak only", "Tokens", "Capped", "Syntax"],
            [[generations[r["sample_id"]]["seed"], *(primary.exact(r[s]["passed_bounds"]) for s in SUITES),
              primary.exact(r["inclusive_signature_bounds"]), primary.exact(r["weak_only_acceptance_bounds"]),
              generations[r["sample_id"]]["tokens"], int(r["hit_token_cap"]), int(r["syntax_valid"])]
             for r in group["programs"]]))

    report.heading("All verifier errors and concrete failure investigations")
    report.data(a["verifier_confusion"], "Complete confusion counts on the mixed 2048-draw cohort.")
    report.add("The following catalog is exhaustive for the 38 weak-only acceptance draws. It is not an exhaustive "
               "taxonomy of every failing program. Each entry includes scores, family counts, an original witness, "
               "source identity and whether it matched the exact inclusive oracle. Four failures are order-sensitive "
               "endpoint variants outside the exact target signature. The full 287 original records for each catalog "
               "entry are embedded at the end of this document.")
    for index, item in enumerate(a["failure_catalog"],1):
        report.heading(f"False acceptance {index:02d}",3)
        report.add("Sample: `"+item["sample_id"]+"`.")
        report.data({k:v for k,v in item.items() if k != "source"})
        report.parts.append(fence(item["source"],"python"))
    report.add("Five additional source-inspection examples follow. They were selected after outcome inspection and "
               "do not provide mechanism prevalence estimates. The final example is fully audit-passing despite "
               "an explanation that misdescribes its tie-breaking rule.")
    for index, item in enumerate(b["illustrative_programs"],1):
        report.heading(f"Illustrative behavior {index}",3)
        report.data({k:v for k,v in item.items() if k not in {"source","raw_completion"}})
        report.parts.append(fence(item["source"],"python"))
        report.parts.append(fence(item["raw_completion"]))

    report.heading("All exploratory distributions and ranking calculations")
    for title,key in (("Outcome distributions","outcome_distributions"),
                      ("All repaired minus weak seed differences","repaired_minus_weak_pairs"),
                      ("Rank disagreement counts and denominators","ranking_disagreement"),
                      ("All token capped passing responses","capped_correct_programs"),
                      ("All response length bins","length_bins"),
                      ("Complete audit score histogram","final_audit_score_histogram")):
        report.heading(title,3)
        report.data(b[key])
    report.data({k:b[k] for k in ("scope","status","lenses_examined","limits")})

    report.heading("Earlier matched pilot kept separate")
    report.add("The following narrative is the historical September 30 pilot report. Its next research step was "
               "subsequently addressed by the four-seed replication described above; it is not a new launch request. "
               "Its 32-draw cohorts use sampling seeds 18000 through 18031, and its training seed was 20261003. "
               "The three unknown outcomes are still unknown in that evidence; they are not made exact by the later study.")
    report.prose_document(sources,ROOT/"docs/booking_failure_analysis.md")
    report.data(data["pilot_result"],"Complete locally exported pilot final result, including all post-training score rows and all cohort summaries.")
    pilot_root = PILOT/"evidence/qwen-booking-matched-training-20260929-v1"
    report.data(sources.read(pilot_root/"baseline.json"),"Pilot baseline including all 32 original score rows.")
    report.data(data["pilot_analysis"],"Complete pilot failure analysis, all inspected per-input observations and same-seed source comparisons.")
    report.add("The targeted pilot export has raw generation files for baseline, weak update 12, weak update 24, and "
               "reference update 24. It does not contain the reference update-12 response texts or all pilot trainer "
               "state receipts. Their final evaluation scores remain included above. This availability gap is not "
               "filled by assuming identical responses across checkpoints.")
    manifest_pilot = sources.read(PILOT/"download_manifest.json")
    for item in manifest_pilot:
        if "/evaluation-" in item["path"] or "/generation/" in item["path"]:
            report.data(sources.read(PILOT/"evidence"/item["path"]),"Saved pilot generations: `"+item["path"]+"`.")

    report.heading("Execution history controls errors and recovery")
    report.add("This chronology distinguishes systems incidents from model errors. A storage or coordination failure "
               "is not a wrong answer, and an unknown candidate outcome is not permission to rerun it. The final "
               "replication has zero unresolved evaluation outcomes, while historical failed attempts and pilot "
               "unknowns remain preserved. Historical documents below contain then-current statuses and commands; "
               "they are not current operational instructions.")
    report.parts.append(table(["Stage", "Observed issue or test", "Resolution or evidence"],[
        ["Per-test execution", "Cleanup uncertainty and workspace start-rate constraints", "Stop on unknowns; qualify program-level isolation before replacement"],
        ["Program benchmark", "One sandbox per program with fresh process per input", "13 controls; 84 parity pairs; 2009-input paired benchmark"],
        ["Training recovery", "Workspace spending limit prevented admissions", "User resolved provider limit; reviewed checkpoint continuation within original project ceiling"],
        ["Storage overload", "Per-test Dict publication error interrupted result collection", "Durable evidence before publication; bounded retry; retained 25 outcomes and executed only 71 proven unsubmitted inputs"],
        ["Resume guard", "Whole-source equality rejected approved storage amendment", "Source-aware guard review at checkpoints 20 and 11; no scientific changes"],
        ["Parallel evaluation", "Four workers and up to 16 program sandboxes", "Cloud canary checked outcome parity and storage recovery before handoff"],
        ["Accounting handoff", "Outstanding maximum reservations mistaken for spent money", "Exclusive-writer handoff; verified reduction of obsolete holds; ceiling remained 250 USD"],
        ["App identity", "FunctionCallFromId returned NotFoundError", "Use hydrated app identity; preserve 198 completed batches; cloud failure-injection controls"],
        ["Permit admission", "Cross-container absolute clocks used for expiry", "Own-process monotonic intervals; clock-skew and delayed-grant controls"],
        ["Final recovery", "328 published batches plus 2 complete unpublished batches", "Reconstruct 2 receipts without execution; retain 330 and finish 150 fixed remaining batches"],
    ]))
    admin = RUN/"amendments/program-sandbox-001"
    evidence_paths = [ROOT/"runs/booking-program-sandbox-benchmark-20260930-v1/result.json",
                     ROOT/"runs/qwen-booking-full-state-control-20260929-v3/result.json",
                     admin/"recoveries/spend-limit-20261001-001/result.json",
                     admin/"storage-amendments/evidence-storage-001/result.json",
                     admin/"storage-amendments/evidence-storage-001/control-source-001/result.json",
                     admin/"parallel-evaluation-001/canary-001/result.json",
                     admin/"parallel-evaluation-001/research-003/identity-control.json",
                     admin/"parallel-evaluation-001/research-004/control.json",
                     admin/"parallel-evaluation-001/research-004/prepare.json"]
    for path in evidence_paths:
        report.data(sources.read(path),"Saved control or recovery result: `"+str(path.relative_to(ROOT))+"`.")
    report.add("The three-step full-state control is an engineering equivalence check, not model learning evidence. "
               "The 2.75× figure comes from equal 2009-input sequential and parallel workloads. The later canary "
               "compares one serial batch with four overlapping batches and is not an equal-work end-to-end study speedup.")
    historical_docs = ("booking_replication_repair_protocol.txt", "program_sandbox_benchmark.txt",
                       "program_sandbox_integration.txt", "program_spend_recovery.txt",
                       "program_storage_recovery.txt", "parallel_evaluation_handoff.txt",
                       "budget_reconciliation.txt", "permit_clock_recovery.txt")
    for index,name in enumerate(historical_docs,1):
        report.heading(f"Historical protocol and incident record {index}",3)
        report.add("Verbatim repository record: `docs/"+name+"`. Historical statuses are superseded by the final result.")
        report.parts.append(fence(sources.text(ROOT/"docs"/name)))

    report.heading("Accounting observations without conflating them with billing")
    report.add("The completion result recorded 108.3014665376 USD estimated settled compute, 52.5936805 USD in "
               "outstanding maximum reservations and a 20 USD allowance, for a guard balance of 180.8951470376 USD. "
               "Later read-only snapshots may differ after settlement. Neither number is a provider invoice. The "
               "250 USD application ceiling was unchanged; storage and workspace-wide billing have different scopes.")
    report.data(result["accounting_not_provider_billing"],"Completion-time accounting snapshot.")
    accounting_paths = sorted((RUN/"budget-reconciliation-001").glob("*/report.json"))
    reports = [(p,sources.read(p)) for p in accounting_paths]
    report.parts.append(table(["Checked UTC", "Completed batches", "Estimated settled USD", "Maximum holds USD", "Guard USD", "Headroom USD"],
        [[r.get("checked_utc",p.parent.name), r.get("evaluation",{}).get("completed_batches","not recorded"),
          *(r.get("accounting",{}).get(k,"not recorded") for k in ("estimated_settled_compute_usd",
             "outstanding_maximum_reservations_usd","guard_balance_usd","headroom_usd"))] for p,r in reports]))
    for path,record in reports:
        report.data(record,"Exact snapshot: `"+str(path.relative_to(ROOT))+"`.")
    report.add("Provider monthly figures in these snapshots cover the workspace and its recorded billing period, "
               "including credits or other resources; they must not be relabeled as this study's final compute cost. "
               "Proposed hold reductions marked applied_by_this_command=false are proposals, not completed releases.")

    report.heading("What the study establishes and what remains unresolved")
    report.add("Established on the observed data: an endpoint-related verifier blind spot; rejection of its 38 observed "
               "false acceptances by a count-matched repair; full population and seed-level results; concrete metric "
               "disagreements; documented algorithmic mistakes; passing extracted code in 29 capped responses; an "
               "explanation/code contradiction; and tested recovery and throughput behavior. These do not require "
               "novelty claims to be useful empirical observations.")
    report.add("Not established: a reproducible amplification effect, a reliable training benefit or harm from the "
               "repair, universal zero verifier error, intentional reward hacking, hidden reasoning mechanisms, a "
               "causal benefit from shorter responses, multi-task generalization, frontier-model capabilities, "
               "security certification, or the provider-billed cost of this study in isolation. The finite audit, "
               "four training seeds and post hoc exploratory choices limit inference.")
    report.data(a["statistical_limits"],"All primary analysis limitations.")
    report.add("A separate follow-up would need its own question, endpoints, uncertainty target and data boundaries. "
               "No further experiment is launched by this record. The completed study should not be extended merely "
               "until a preferred result appears.")
    report.heading("Resume blog and interview interpretation")
    report.prose_document(sources,ROOT/"docs/frontier_evals_project_brief.md")

    report.heading("Every test case and verifier membership")
    cases = record_cases()
    primary.require(len(cases) == 288 and len({c["input_hash"] for c in cases}) == 287, "case count changed")
    report.add("All logical training and audit entries follow. The mandatory empty input occurs in both suites, "
               "so 288 suite entries represent 287 unique inputs. Membership flags are keyed by input identity; "
               "the shared empty input therefore belongs to both suites. Expected values come from the frozen "
               "trusted task implementation, not from executing a generated candidate.")
    family_counts = Counter((c["suite"],c["family"]) for c in cases)
    report.parts.append(table(["Suite", "Family", "Cases"],[[*key,n] for key,n in sorted(family_counts.items())]))
    report.data(cases)

    report.heading("Complete response source and score appendix")
    report.add("All 2048 draws are included in deterministic sample-ID order. Each record preserves its exact "
               "generation metadata and every saved score field, followed by extracted source and the original "
               "response. Markdown fences are escaped structurally so generated content remains literal data. "
               "Sources that repeat are intentionally retained as separate draws, not additional independent policies.")
    for index,sid in enumerate(sorted(rows),1):
        sample = generations[sid]
        report.heading(f"Evaluation response {index:04d}",3)
        report.data({"record_type":"evaluation_response", "sample_id":sid,
                     "generation":{k:v for k,v in sample.items() if k not in {"source","raw"}},
                     "scores":rows[sid]})
        report.add("Extracted source, preserved as data.")
        report.parts.append(fence(sample["source"],"python"))
        report.add("Original generated response, preserved as data.")
        report.parts.append(fence(sample["raw"]))

    report.heading("Original protected input records for every investigated false acceptance")
    report.add("This appendix embeds all 38 locally exported program documents: 287 protected records per program, "
               "10906 records in total. JSON fields, including base64 output and supervisor metadata, are preserved "
               "without interpreting candidate text as instructions. These are original execution records, not "
               "reexecutions. The rest of the full raw archive remains external; it is not claimed to be embedded here.")
    for index,(sid,document) in enumerate(sorted(data["failure_documents"].items()),1):
        report.heading(f"Protected result document {index:02d}",3)
        report.add("Sample: `"+sid+"`.")
        report.data({"sample_id":sid, "document":document},compact=True)

    report.heading("Final recovery provenance and reproduction")
    report.data(config, "Complete saved configuration, including model revision, prompt hash, decoding, trainer settings, "
                "repair membership, resource rates, source fingerprints and approved runtime.")
    report.data(a["workload"], "Planned workload. Input slots are a design count, not independent samples or necessarily executed inputs.")
    report.data({k:v for k,v in result.items() if k not in {"policies","training_metrics","analysis"}},
                "Final result metadata. Per-policy score rows and training metrics were embedded in preceding sections.")
    report.data(result["analysis"],"Complete original cloud paired analysis, retained separately from later local analysis.")
    report.data(sources.read(LOCAL/"modal-results/recovery.json"),"Final reviewed recovery, including the original retained outcome identities.",compact=True)
    report.data(sources.read(LOCAL/"modal-results/permit-control.json"),"Final clock and permit control receipt.")
    report.data({"config_hash":inv["config_hash"], "retained_batch_ids":sorted(inv["completed"]),
                 "missing_at_recovery_not_at_completion":[m["key"] for m in inv["missing"]]},
                "Inventory partition at the last recovery. These identifiers describe the historical snapshot, not current unfinished work.")
    report.add("To regenerate this record from the same saved local evidence, use the command below. It performs "
               "local checks and writes a derived report; it does not contact Modal. No command quoted in the "
               "historical records should be run as part of reproduction.")
    report.parts.append(fence(".venv/bin/python -m verifier_rl.booking_complete_report\n"
                              ".venv/bin/python -m unittest tests.test_booking_complete_report -v","bash"))
    report.add("Numerical source validation covers frozen configuration/source bindings, every saved score/source "
               "identity, all 38 targeted protected program documents, and the separately validated pilot. This is "
               "not a fresh local replay of every cloud test or a rehash of model tensors. Report-generation tests "
               "and hashes are distinct from the historical infrastructure test counts quoted above.")
    sources.read_bytes(Path(__file__))
    report.heading("Source file index and integrity hashes")
    report.add("The index covers files read or verified while assembling this report. Some very large supporting "
               "pilot input journals and original implementation snapshots are hash-indexed rather than duplicated. "
               "Every replication evaluation response and score, every checkpoint receipt, and every targeted "
               "replication protected document is embedded above. The adjacent machine-readable manifest records "
               "this report's own SHA256 and counts; hashing a document inside itself would be self-referential.")
    report.parts.append(table(["Repository relative source", "Bytes", "SHA256"],
        [[path,meta["bytes"],meta["sha256"]] for path,meta in sorted(sources.entries.items())]))
    toc = "\n".join(f"- [{title}](#{title.lower().replace(' ','-')})" for title in report.sections)+"\n"
    text = "".join(report.parts).replace("CONTENTS_PLACEHOLDER",toc)
    counts = {"evaluation_cohorts":len(groups),"evaluation_responses":len(rows),"training_policies":len(inv["saved_arms"]),
              "training_updates":len(a["training_updates"]),"checkpoint_receipts":288,"post_training_batches":len(batch_rows),
              "false_acceptance_documents":len(data["failure_documents"]),
              "protected_input_records":sum(len(d["records"]) for d in data["failure_documents"].values()),
              "logical_test_cases":len(cases),"unique_test_inputs":287,"pilot_evaluation_cohorts":5,
              "new_cloud_calls":0,"new_candidate_executions":0,"new_downloads":0}
    return text,counts


def validate_rendered(text, counts):
    primary.require(len(re.findall(r"^### Evaluation response \d{4}$",text,re.M)) == counts["evaluation_responses"],
                    "response records missing from document")
    primary.require(text.count('"record_type": "checkpoint_receipt"') == counts["checkpoint_receipts"],
                    "checkpoint receipts missing from document")
    primary.require(len(re.findall(r"^### Protected result document \d{2}$",text,re.M)) == counts["false_acceptance_documents"],
                    "protected documents missing from report")
    primary.require("CONTENTS_PLACEHOLDER" not in text, "unresolved report placeholder")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--manifest",type=Path,default=MANIFEST)
    args = parser.parse_args()
    data = load()
    text,counts = render(data)
    validate_rendered(text,counts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text)
    manifest = {"report":str(args.output.relative_to(ROOT)),"report_sha256":sha(text.encode()),
                "report_bytes":len(text.encode()),"counts":counts,"sources":data["sources"].entries,
                "primary_result_hash":data["primary"]["result_hash"],"candidate_execution":False}
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k != "sources"},indent=2))


if __name__ == "__main__":
    main()
