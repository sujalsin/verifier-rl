"""Task schema, loader, and split management."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from verifiers.base import TestCase


@dataclass
class CodingTask:
    """A coding task with train and hidden test suites."""

    task_id: str
    category: str
    prompt: str
    entry_point: str
    train_tests: List[TestCase]
    hidden_tests: List[TestCase]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CodingTask:
        train_tests = [
            TestCase(
                input_args=tc["args"],
                expected_output=tc["expected"],
                is_trivial=tc.get("is_trivial", False),
            )
            for tc in data["train_tests"]
        ]
        hidden_tests = [
            TestCase(
                input_args=tc["args"],
                expected_output=tc["expected"],
                is_trivial=False,
            )
            for tc in data["hidden_tests"]
        ]
        return cls(
            task_id=data["task_id"],
            category=data.get("category", "general"),
            prompt=data["prompt"],
            entry_point=data["entry_point"],
            train_tests=train_tests,
            hidden_tests=hidden_tests,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "prompt": self.prompt,
            "entry_point": self.entry_point,
            "train_tests": [
                {"args": t.input_args, "expected": t.expected_output, "is_trivial": t.is_trivial}
                for t in self.train_tests
            ],
            "hidden_tests": [
                {"args": t.input_args, "expected": t.expected_output}
                for t in self.hidden_tests
            ],
        }


def load_tasks(path_str: str | Path) -> List[CodingTask]:
    """Loads all tasks from JSON or JSONL file or directory."""
    path = Path(path_str)
    tasks: List[CodingTask] = []
    if not path.exists():
        return tasks

    files = [path] if path.is_file() else sorted(list(path.glob("*.json")) + list(path.glob("*.jsonl")))

    for file in files:
        if file.suffix == ".json":
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    tasks.extend(CodingTask.from_dict(d) for d in data)
                else:
                    tasks.append(CodingTask.from_dict(data))
        elif file.suffix == ".jsonl":
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        tasks.append(CodingTask.from_dict(json.loads(line)))

    return tasks
