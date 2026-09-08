"""Base interface and data models for verifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from environments.sandbox import ExecutionResult, SandboxRunner


@dataclass
class TestCase:
    """Individual test case for a coding task."""

    __test__ = False  # Prevent pytest from collecting this as a test class
    input_args: List[Any]
    expected_output: Any
    is_trivial: bool = False  # For Condition B3 undercoverage labeling


@dataclass
class VerificationResult:
    """Standardized output of a verifier evaluation."""

    reward: float  # Reward value normalized to [0.0, 1.0]
    passed_count: int
    total_count: int
    is_exploit: bool
    exploit_type: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)
    execution_result: Optional[ExecutionResult] = None


class BaseVerifier(ABC):
    """Abstract base class for all task verifiers."""

    def __init__(self, sandbox_runner: Optional[SandboxRunner] = None):
        self.sandbox = sandbox_runner or SandboxRunner()

    @abstractmethod
    def verify(
        self,
        candidate_code: str,
        entry_point: str,
        test_cases: List[TestCase],
    ) -> VerificationResult:
        """Executes candidate code against test cases and returns VerificationResult."""
        pass
