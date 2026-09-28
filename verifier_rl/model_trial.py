"""Small model-trial contracts; importing this never loads or executes a model."""

import re
from dataclasses import dataclass

from .grading import MAX_SOURCE_BYTES

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
LEGACY_EXTRACTION_VERSION = "single-python-fence-or-verbatim-0.1"
EXTRACTION_VERSION = "single-python-block-or-raw-0.2"
MAX_COMPLETION_TOKENS = 512


def canonical_prompt(document: str) -> str:
    start = "CANONICAL PROMPT SHOWN TO THE MODEL\n-----------------------------------\n"
    end = "\nEND OF CANONICAL PROMPT\n"
    if document.count(start) != 1 or document.count(end) != 1:
        raise ValueError("canonical task prompt markers missing or ambiguous")
    return document.split(start, 1)[1].split(end, 1)[0].strip()


@dataclass(frozen=True)
class Extraction:
    source: str
    status: str
    version: str


def extract_completion(completion: str, *, version: str = EXTRACTION_VERSION) -> Extraction:
    """Select one complete Python block or raw source; never repair its contents.

    V0.2 permits surrounding prose but rejects multiple, incomplete, unsupported,
    or malformed fenced blocks. Rejection becomes a candidate failure, not an
    infrastructure failure. V0.1 is retained for reproducing the original trial.
    The caller must retain the raw completion and extraction status/version.
    """
    source = completion.strip()
    status = "raw"
    if version == LEGACY_EXTRACTION_VERSION:
        match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", source, re.DOTALL)
        if match and "```" not in match.group(1):
            source, status = match.group(1), "python_block"
        if not source:
            source, status = "# Empty model completion; required function absent.", "rejected_empty"
    elif version == EXTRACTION_VERSION:
        if len(completion.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise ValueError("completion exceeds the execution source limit")
        fences = list(re.finditer(r"(?m)^[ \t]*```([^\r\n]*)\r?$", source))
        if fences:
            if len(fences) != 2:
                status = "rejected_fence_count"
            elif fences[0].group(1).strip() not in ("python", "py", ""):
                status = "rejected_language"
            elif fences[1].group(1).strip():
                status = "rejected_closing_fence"
            else:
                # Remove only the fence framing newline, not code indentation,
                # imports, demonstrations, print calls, or syntax errors.
                source = source[fences[0].end():fences[1].start()]
                source = source.removeprefix("\n").removesuffix("\n").removesuffix("\r")
                status = "python_block"
        if not source.strip():
            status = "rejected_empty"
        if status.startswith("rejected_"):
            source = f"# Extraction rejected: {status}; required function absent."
    else:
        raise ValueError("unknown extraction version")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("completion exceeds the execution source limit")
    return Extraction(source, status, version)


def extract_source(completion: str, *, version: str = EXTRACTION_VERSION) -> str:
    return extract_completion(completion, version=version).source


def submission_from_completion(raw: str):
    extraction = extract_completion(raw)
    return {"raw": raw, "source": extraction.source, "extraction_status": extraction.status,
            "extraction_version": extraction.version}


def validate_submission(submission):
    """Recompute extraction so a rejected completion cannot be relabelled as code."""
    expected = submission_from_completion(submission["raw"])
    if any(submission.get(k) != value for k, value in expected.items()):
        raise ValueError("submission does not match the current extraction policy")
    return expected


async def evaluate_submission(submission, suites, backend, *, concurrency=4):
    from .grading import evaluate_candidate, rejected_extraction_report
    entry = validate_submission(submission)
    if entry["extraction_status"].startswith("rejected_"):
        return rejected_extraction_report(entry["source"], entry["extraction_status"], suites)
    report = await evaluate_candidate(entry["source"], suites, backend, concurrency=concurrency,
                                      max_retries=0, stop_on_infrastructure_error=True)
    report["extraction"] = {"status": entry["extraction_status"], "accepted": True,
                            "version": entry["extraction_version"]}
    return report


def validate_run_id(run_id: str):
    if re.fullmatch(r"qwen-[A-Za-z0-9-]{1,80}", run_id) is None:
        raise ValueError("invalid model trial run ID")


def require_scored_rewards(reports):
    rewards = []
    for report in reports:
        if len(report["suites"]) != 1 or report["suites"][0]["suite"] != "g3":
            raise ValueError("training must use only G3")
        result = report["suites"][0]
        if result["infrastructure_errors"] or result["reward"] not in (0, 1):
            raise RuntimeError("unscored rollout: abort training, never convert infrastructure failure to zero")
        rewards.append(float(result["reward"]))
    return rewards
