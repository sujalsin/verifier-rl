"""Recompute a completed paired measurement from saved data only, without Modal."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .cli import create_run_directory, write_private
from .measurement_v3 import budget_check, measurement_summary, unique_outcomes
from .model_trial import validate_run_id
from .smoke import require_current_conformance
from .suites import canonical_json, digest


def verify_documents(documents, repo_root):
    """No eval/import/exec of saved source; snapshot text is only hashed."""
    summary, setup = documents["summary"], documents["setup"]
    validate_run_id(summary["run_id"])
    require_current_conformance(documents["conformance"], setup["sandbox_image_id"])
    snapshot = documents["source_snapshot"]
    files = {name: name for name in snapshot
             if len(Path(name).parts) == 2 and Path(name).parts[0] == "verifier_rl" and name.endswith(".py")}
    for name in ("modal_measure_v3.py", "measurement_v3_protocol.txt"):
        matches = [key for key in snapshot if Path(key).name == name]
        if len(matches) != 1:
            raise ValueError("missing/ambiguous measurement snapshot file")
        files[name] = matches[0]
    for name in ("measurement_v3.py", "reward_v2.py", "reward_v3.py", "grading.py", "modal_backend.py"):
        if "verifier_rl/" + name not in files:
            raise ValueError("required implementation snapshot missing")
    for local, saved in files.items():
        if (repo_root / local).read_text() != snapshot[saved]:
            raise ValueError(f"current implementation differs from frozen run: {local}")
    recomputed = measurement_summary(documents["generation"], documents["prior_reports"],
                                     documents["reports"], setup["sandbox_image_id"])
    recomputed["run_id"] = summary["run_id"]
    if recomputed != summary or summary["plan"] != documents["plan"]:
        raise ValueError("saved summary/plan differs from recomputation")
    budget = documents["budget"]
    if budget_check(documents["plan"], budget["billing_before"], budget["rates"]) != budget:
        raise ValueError("budget evidence mismatch")
    statuses, stdout_evidence, cleanup = Counter(), Counter(), Counter()
    for row in summary["rows"]:
        stdout_evidence.update(row["diagnostic"]["stdout_evidence"])
    recorded_lifecycle_seconds = 0.0
    for report in documents["reports"]:
        for outcome in unique_outcomes(report).values():
            for attempt in outcome["attempts"]:
                statuses[attempt["status"]] += 1
                if attempt["status"] != "extraction_rejected":
                    metadata = attempt["metadata"]
                    cleanup[metadata["cleanup"]] += 1
                    recorded_lifecycle_seconds += metadata.get("total_seconds", 0)
    return {"run_id": summary["run_id"], "verified": True, "summary_recomputed_exactly": True,
            "frozen_implementation_files_checked": len(files),
            "document_hashes": {k: digest(canonical_json(v)) for k, v in documents.items()},
            "recorded_sandbox_executions": summary["recorded_sandbox_executions"],
            "unique_input_attempt_statuses": dict(statuses), "stdout_evidence": dict(stdout_evidence),
            "cleanup": dict(cleanup), "recorded_lifecycle_seconds": recorded_lifecycle_seconds,
            "lifecycle_seconds_are_billed_duration": False,
            "controls_passed": summary["controls_passed"], "v2_repeatability_passed": summary["v2_repeatability_passed"],
            "candidate_source_executed": False, "cloud_calls": 0, "new_model_samples": 0,
            "original_artifacts_changed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", required=True, help="new verification sidecar directory")
    args = parser.parse_args(argv)
    names = ("generation", "prior_reports", "reports", "setup", "conformance", "source_snapshot",
             "budget", "plan", "summary")
    documents = {name: json.loads((args.run / f"{name}.json").read_text()) for name in names}
    result = verify_documents(documents, Path(__file__).resolve().parent.parent)
    directory = create_run_directory(args.out)
    write_private(directory / "verification.json", canonical_json(result))
    print(canonical_json({k: v for k, v in result.items() if k != "document_hashes"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
