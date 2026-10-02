"""Read-only checks for files eligible for Git, without loading project code.

This is a small repository hygiene gate, not a comprehensive secret scanner.
It never reads ignored run data unless --require-evidence is explicitly given.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 50 * 1024 * 1024  # Project policy; the 38 MiB evidence archive is retained.
PRIVATE_PARTS = {".git", ".aws", ".venv", "venv", "runs", "__pycache__", "checkpoints", "wandb"}
PRIVATE_NAMES = {".env", ".modal.toml", ".DS_Store", "id_rsa", "id_ed25519", ".coverage"}
BINARY_SUFFIXES = {".pem", ".key", ".safetensors", ".ckpt", ".pt", ".pth", ".pyc", ".pyo"}
TOKEN_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    ("GitHub token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("OpenAI project token", re.compile(rb"sk-(?:proj|svcacct)-[A-Za-z0-9_-]{40,}")),
    ("AWS access key", re.compile(rb"AKIA[0-9A-Z]{16}")),
)


def git_paths(root):
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True,
    )
    return sorted({p.decode("utf-8") for p in result.stdout.split(b"\0") if p})


def private_path(name):
    path = Path(name)
    return (path.is_absolute() or ".." in path.parts
            or bool(set(path.parts) & PRIVATE_PARTS)
            or any(p.startswith(".venv-") for p in path.parts)
            or path.name in PRIVATE_NAMES or path.name.startswith(".env.")
            or path.suffix.lower() in BINARY_SUFFIXES)


def inspect_files(root, paths):
    """Return issue descriptions, never matched credential bytes."""
    root = root.resolve()
    issues, count, total = [], 0, 0
    for name in paths:
        if private_path(name):
            issues.append(f"{name}: private/generated file is eligible for Git")
            continue
        path = root / name
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            issues.append(f"{name}: symlink points outside this repository")
            continue
        if not path.exists():  # A tracked removal in the worktree is not new content.
            continue
        if not path.is_file():
            issues.append(f"{name}: expected a regular file")
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            issues.append(f"{name}: exceeds the project's 50 MiB per-file policy")
            continue
        raw = path.read_bytes()
        count += 1
        total += len(raw)
        for label, pattern in TOKEN_PATTERNS:
            if pattern.search(raw):
                issues.append(f"{name}: possible {label}; inspect privately before staging")
        if path.suffix == ".py":
            try:
                ast.parse(raw, filename=name)
            except (SyntaxError, ValueError) as exc:
                issues.append(f"{name}: invalid Python syntax ({type(exc).__name__})")
    return issues, count, total


def evidence_issues(root):
    """Full evidence mode fails on absent or changed files; it never downloads them."""
    path = root / "reports/booking-complete/manifest.json"
    if not path.is_file():
        return ["complete-evidence manifest is missing"]
    manifest = json.loads(path.read_text())
    entries = dict(manifest["sources"])
    entries[manifest["report"]] = {"bytes": manifest["report_bytes"], "sha256": manifest["report_sha256"]}
    issues = []
    for name, expected in entries.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            issues.append(f"{name}: evidence path is outside repository")
        elif not path.is_file():
            issues.append(f"{name}: evidence prerequisite missing")
        else:
            raw = path.read_bytes()
            if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                issues.append(f"{name}: evidence fingerprint changed")
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-evidence", action="store_true",
                        help="require every local artifact named by the complete-study manifest")
    args = parser.parse_args(argv)
    issues, count, total = inspect_files(args.root, git_paths(args.root))
    if args.require_evidence:
        issues.extend(evidence_issues(args.root))
    for issue in issues:
        print(issue)
    print(f"Checked {count} Git-eligible files ({total / 1024 / 1024:.1f} MiB); {len(issues)} issues.")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
