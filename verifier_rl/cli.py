"""Offline fixture checks and explicitly opted-in remote candidate evaluation."""

import argparse
import asyncio
from pathlib import Path
import platform
import sys

from .fixtures import FAULTS, fixture_report, source_for
from .grading import evaluate_candidate
from .suites import DEFAULT_SEED, build_suites, canonical_json, coverage


def write_private(path: Path, content: str):
    # Exclusive creation prevents accidentally replacing prior experiment evidence.
    with path.open("x", encoding="utf-8") as stream:
        path.chmod(0o600)
        stream.write(content)
        stream.write("\n")


def create_run_directory(path: str):
    directory = Path(path)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    return directory


def save_suites(directory, suites):
    for suite in suites:
        manifest = suite.manifest()
        manifest["fingerprint"] = suite.fingerprint
        manifest["coverage"] = coverage(suite)
        write_private(directory / f"{suite.name}.suite.json", canonical_json(manifest))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="score only built-in trusted fixtures; no sandbox or cloud")
    demo.add_argument("--seed", type=int, default=DEFAULT_SEED)
    demo.add_argument("--out", help="new directory for full reports, fixture sources, and suite manifests")
    export = sub.add_parser("export-suites", help="save reproducible development suites locally")
    export.add_argument("--seed", type=int, default=DEFAULT_SEED)
    export.add_argument("--out", required=True)
    grade = sub.add_parser("grade", help="execute one candidate in Modal; requires cloud opt-in")
    grade.add_argument("candidate", type=Path)
    grade.add_argument("--app", required=True, help="existing Modal app")
    grade.add_argument("--image-id", required=True, help="approved resolved image ID, im-...")
    grade.add_argument("--allow-cloud", action="store_true", help="acknowledge remote execution/billing")
    grade.add_argument("--suites", nargs="+", choices=("g1", "g2", "g3", "audit"), default=["g1"])
    grade.add_argument("--seed", type=int, default=DEFAULT_SEED)
    grade.add_argument("--concurrency", type=int, default=4)
    grade.add_argument("--max-retries", type=int, default=1)
    grade.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "grade" and not args.allow_cloud:
        parser.error("grade requires --allow-cloud; the offline demo never executes candidate files")
    suites = build_suites(args.seed)
    try:
        if args.command == "demo":
            report = fixture_report(suites)
            print("Author-written fixtures only; these are not model or sandbox results.")
            print(f"{'fixture':22} {'G1':>9} {'G2':>9} {'G3':>9} {'Audit':>9}")
            for row in report["fixtures"]:
                cells = [f"{r['passed_count']}/{r['total']}" for r in row["suites"]]
                print(f"{row['fixture']:22} " + " ".join(f"{c:>9}" for c in cells))
            if args.out:
                directory = create_run_directory(args.out)
                save_suites(directory, suites)
                report["controller_python"] = platform.python_version()
                write_private(directory / "fixture_report.json", canonical_json(report))
                for fault in FAULTS:
                    write_private(directory / f"fixture_{fault}.py", source_for(fault))
        elif args.command == "export-suites":
            directory = create_run_directory(args.out)
            save_suites(directory, suites)
            for suite in suites:
                print(suite.name, suite.fingerprint, canonical_json(coverage(suite)))
        else:
            from .modal_backend import ModalBackend
            selected = tuple(s for s in suites if s.name in args.suites)
            source = args.candidate.read_text(encoding="utf-8")
            backend = ModalBackend(args.app, args.image_id)
            directory = create_run_directory(args.out)
            save_suites(directory, selected)
            write_private(directory / "candidate.py", source)
            config = {"app": args.app, "image_id": args.image_id,
                      "concurrency": args.concurrency, "max_retries": args.max_retries,
                      "suites": args.suites, "seed": args.seed,
                      "controller_python": platform.python_version()}
            write_private(directory / "config.json", canonical_json(config))
            report = asyncio.run(evaluate_candidate(source, selected, backend,
                                                     args.concurrency, args.max_retries,
                                                     stop_on_infrastructure_error=True))
            write_private(directory / "result.json", canonical_json(report))
            for result in report["suites"]:
                print(f"{result['suite']}: {result['passed_count']}/{result['total']} "
                      f"reward={result['reward']} infra={result['infrastructure_errors']}")
            return 2 if any(r["infrastructure_errors"] for r in report["suites"]) else 0
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
