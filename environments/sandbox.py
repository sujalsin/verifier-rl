"""Execution sandbox for running generated code in isolated subprocesses.

Provides process timeouts, memory limits, temporary directory isolation,
and capture of stdout, stderr, and exit codes.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution."""

    exit_code: int
    stdout: str
    stderr: str
    timeout: bool = False
    memory_exceeded: bool = False
    duration_seconds: float = 0.0
    telemetry: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0 and not self.timeout and not self.memory_exceeded


class SandboxRunner:
    """Executes Python code in an isolated temporary directory with resource bounds."""

    def __init__(
        self,
        default_timeout_seconds: float = 3.0,
        max_memory_mb: int = 512,
    ):
        self.default_timeout_seconds = default_timeout_seconds
        self.max_memory_mb = max_memory_mb

    def _set_limits(self) -> None:
        """Sets pre-exec resource limits in the child process."""
        # Memory limit in bytes (RLIMIT_AS / address space)
        if hasattr(resource, "RLIMIT_AS") and sys.platform != "darwin":
            # On macOS, RLIMIT_AS can cause memory allocator crashes in Python runtime;
            # enable on Linux / production GPU containers.
            bytes_limit = self.max_memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
            except (ValueError, OSError):
                pass

        # Limit CPU time slightly above wall-clock timeout as a fallback
        cpu_seconds = int(self.default_timeout_seconds) + 2
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError):
            pass

    def run_script(
        self,
        script_content: str,
        timeout: Optional[float] = None,
        extra_files: Optional[Dict[str, str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Executes a Python script in a dedicated, isolated temporary directory.

        Args:
            script_content: The full Python source code to execute.
            timeout: Wall-clock timeout in seconds.
            extra_files: Mapping of relative file paths to string contents to place in cwd.
            env_vars: Extra environment variables to pass.

        Returns:
            ExecutionResult containing exit code, stdout, stderr, and telemetry.
        """
        timeout_seconds = timeout or self.default_timeout_seconds

        with tempfile.TemporaryDirectory(prefix="verifier_sandbox_") as tmpdir:
            work_dir = Path(tmpdir)
            script_path = work_dir / "run_eval.py"
            script_path.write_text(script_content, encoding="utf-8")

            if extra_files:
                for rel_path, content in extra_files.items():
                    target_file = work_dir / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    target_file.write_text(content, encoding="utf-8")

            run_env = os.environ.copy()
            # Clean python cache and ensure isolated import path
            run_env["PYTHONDONTWRITEBYTECODE"] = "1"
            run_env["PYTHONUNBUFFERED"] = "1"
            run_env["PYTHONPATH"] = str(work_dir)
            if env_vars:
                run_env.update(env_vars)

            cmd = [sys.executable, str(script_path)]

            try:
                import time

                t0 = time.perf_counter()
                process = subprocess.run(
                    cmd,
                    cwd=str(work_dir),
                    env=run_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    preexec_fn=self._set_limits if sys.platform != "win32" else None,
                )
                duration = time.perf_counter() - t0

                # Check for telemetry output written by execution harness
                telemetry_file = work_dir / ".telemetry.json"
                telemetry: Dict[str, Any] = {}
                if telemetry_file.exists():
                    try:
                        telemetry = json.loads(telemetry_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                return ExecutionResult(
                    exit_code=process.returncode,
                    stdout=process.stdout,
                    stderr=process.stderr,
                    timeout=False,
                    memory_exceeded=False,
                    duration_seconds=duration,
                    telemetry=telemetry,
                )

            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    exit_code=-1,
                    stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                    stderr=exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "Execution timed out"),
                    timeout=True,
                    memory_exceeded=False,
                    duration_seconds=timeout_seconds,
                    telemetry={},
                )
            except Exception as e:
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Sandbox execution error: {str(e)}",
                    timeout=False,
                    memory_exceeded=False,
                    duration_seconds=0.0,
                    telemetry={},
                )
