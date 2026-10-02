"""Verify the published booking study and rebuild results using only the standard library.

Run from any directory: python /path/to/verifier-rl/scripts/reproduce.py
Inputs in docs/ and reports/ are read-only; derived files go to build/reproduction/.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import booking_publication as portable
from verifier_rl import booking_publication_analysis as archive


INPUTS = (
    "docs/booking_complete_study_record.md",
    "reports/booking-complete/manifest.json",
    "reports/booking-blog/program_scores.csv",
    "reports/booking-blog/source_manifest.json",
    "reports/booking-blog/analysis.json",
    "reports/booking-publication/analysis.json",
)
ROW_FIELDS = (
    "sample_id", "training_seed", "arm", "step", "draw_seed", "source_hash", "tokens",
    "syntax_valid", "hit_token_cap", "inclusive_signature", "weak_only_acceptance",
    "reference_passed", "endpoint_omission_passed", "repaired_passed", "audit_passed",
    "reference_full_pass", "endpoint_omission_full_pass", "repaired_full_pass", "audit_full_pass",
    "audit_case_accuracy",
)


def compare(actual, expected, location="result"):
    """Compare complete JSON results, tolerating only floating-point roundoff."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{location}: result fields differ")
        for key in actual:
            compare(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{location}: result lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{location}[{index}]")
    elif type(actual) is float and type(expected) is float:
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{location}: numeric result differs ({actual!r} vs {expected!r})")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{location}: result differs ({actual!r} vs {expected!r})")


def json_value(value):
    return json.loads(json.dumps(value))


def run(root=ROOT):
    root = Path(root).resolve()
    for name in INPUTS:
        if not (root / name).is_file():
            raise ValueError(f"Missing public input: {name}. Use a complete Git checkout.")
    fingerprints = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in INPUTS}
    score_dir = root / "reports/booking-blog"
    manifest = json.loads((score_dir / "source_manifest.json").read_text())
    score_rows = portable.load_rows(score_dir / "program_scores.csv", manifest)
    score_result = json_value(portable.compute(score_rows))
    compare(score_result, json.loads((score_dir / "analysis.json").read_text()), "portable analysis")

    rows, cases, provenance = archive.load(
        root / "docs/booking_complete_study_record.md", root / "reports/booking-complete/manifest.json"
    )
    indexed = {row["sample_id"]: row for row in score_rows}
    if set(indexed) != {row["sample_id"] for row in rows}:
        raise ValueError("Portable CSV and public archive contain different sample identities")
    for row in rows:
        peer = indexed[row["sample_id"]]
        compare({key: row[key] for key in ROW_FIELDS}, {key: peer[key] for key in ROW_FIELDS},
                f"CSV/archive binding {row['sample_id']}")
    result = archive.analyze(rows, cases, provenance)
    compare(result, json.loads((root / "reports/booking-publication/analysis.json").read_text()),
            "archive analysis")

    # Only write after all checks succeed; published inputs and reports stay untouched.
    output = root / "build/reproduction"
    output.mkdir(parents=True, exist_ok=True)
    archive.save(result, rows, output / "archive")
    (output / "portable-analysis.json").write_text(json.dumps(score_result, indent=2, sort_keys=True) + "\n")
    verification = {
        "status": "verified",
        "scope": "Published score arithmetic and public archive consistency; no candidate execution or training.",
        "evaluation_draws": len(rows),
        "training_seed_triplets": len(archive.SEEDS),
        "checkpoint_receipts_read": provenance["checkpoint_receipts_read"],
        "training_logs_read": provenance["training_logs_read"],
        "input_sha256": fingerprints,
        "python_version": sys.version.split()[0],
        "implementation_sha256": {
            "scripts/reproduce.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scripts/booking_publication.py": hashlib.sha256(Path(portable.__file__).read_bytes()).hexdigest(),
            "verifier_rl/booking_publication_analysis.py": hashlib.sha256(Path(archive.__file__).read_bytes()).hexdigest(),
        },
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    summary = [
        "# Reproduced booking study results", "",
        f"Verified {len(rows):,} evaluation draws, {len(archive.SEEDS)} paired training seeds, "
        f"{provenance['checkpoint_receipts_read']} checkpoint receipts, and {provenance['training_logs_read']} training logs.", "",
        "Both analyses match the checked-in results. The portable CSV also matches the public archive sample by sample.", "",
        "| Final training condition | Full audit passes | Target-bug draws |",
        "| --- | ---: | ---: |",
    ]
    for arm in archive.ARMS:
        profile = result["final_arms"][arm]
        summary.append(f"| {archive.LABELS[arm]} | {profile['audit_full_count']}/{profile['n']} | "
                       f"{profile['inclusive_count']}/{profile['n']} |")
    false_accepts = result["false_acceptance"]["draws"]
    retained = result["confusion"]["repaired"]["true_accept"]
    summary += [
        "", f"The repaired grader rejected all {false_accepts} observed weak false acceptances "
        f"and retained all {retained} audit-passing draws.",
        "The four-seed comparison did not establish amplification or a training benefit from the repair.", "",
        "- [Portable result tables](portable-analysis.json)",
        "- [Archive reanalysis and uncertainty](archive/analysis.json)",
        "- [Paired effects figure](archive/paired_effects.svg)",
        "- [Outcome distribution](archive/outcome_distribution.svg)",
        "- [Audit thresholds](archive/audit_thresholds.svg)",
        "- [Verified input fingerprints](verification.json)", "",
        "This verifies consistency of published saved evidence. It does not rerun candidate code, train a model,",
        "validate model tensors, or replace the original private-bundle checks.", "",
    ]
    (output / "README.md").write_text("\n".join(summary))
    return result, verification, output


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        result, verification, output = run()
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"Reproduction failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified {verification['evaluation_draws']:,} draws against the public archive and published results.")
    print(f"Training: {verification['training_seed_triplets']} paired seeds; "
          f"{verification['checkpoint_receipts_read']} checkpoint receipts; {verification['training_logs_read']} logs.")
    counts = [f"{archive.LABELS[arm]} {profile['audit_full_count']}/{profile['n']}"
              for arm, profile in result["final_arms"].items()]
    print("Final full audit passes: " + "; ".join(counts) + ".")
    print(f"Grader repair: {result['false_acceptance']['draws']} observed weak false acceptances; "
          f"{result['confusion']['repaired']['false_accept']} remain after repair; "
          f"{result['confusion']['repaired']['true_accept']} audit passes retained.")
    print(f"Results and figures: {output / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
