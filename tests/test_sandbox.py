"""Unit tests for SandboxRunner process isolation, limits, and cleanup."""

import pytest
from environments.sandbox import SandboxRunner


def test_sandbox_executes_valid_code():
    runner = SandboxRunner()
    result = runner.run_script('print("Hello from sandbox")')
    assert result.is_success
    assert result.exit_code == 0
    assert "Hello from sandbox" in result.stdout
    assert result.stderr == ""
    assert not result.timeout


def test_sandbox_captures_stderr_on_syntax_error():
    runner = SandboxRunner()
    result = runner.run_script("def bad_syntax(:")
    assert not result.is_success
    assert result.exit_code != 0
    assert "SyntaxError" in result.stderr


def test_sandbox_enforces_timeout():
    runner = SandboxRunner(default_timeout_seconds=0.5)
    result = runner.run_script("import time; time.sleep(2.0)")
    assert not result.is_success
    assert result.timeout
    assert result.exit_code == -1


def test_sandbox_extra_files():
    runner = SandboxRunner()
    extra_files = {"data/input.txt": "sample payload"}
    script = '''
with open("data/input.txt", "r") as f:
    print(f.read())
'''
    result = runner.run_script(script, extra_files=extra_files)
    assert result.is_success
    assert "sample payload" in result.stdout.strip()
