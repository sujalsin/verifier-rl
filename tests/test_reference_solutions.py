"""Tests that every task in the 35-task benchmark has a working reference solution (100% pass rate)."""

import pytest
from data.loader import load_tasks
from data.reference_solutions import REFERENCE_SOLUTIONS
from verifiers.strict import StrictVerifier


def test_all_reference_solutions_pass_strict_evaluation():
    train_tasks = load_tasks("data/tasks/train.json")
    heldout_instances = load_tasks("data/tasks/heldout_instances.json")
    heldout_families = load_tasks("data/tasks/heldout_families.json")
    all_tasks = train_tasks + heldout_instances + heldout_families

    verifier = StrictVerifier()

    for task in all_tasks:
        code = REFERENCE_SOLUTIONS.get(task.task_id)
        assert code is not None, f"Missing reference solution for {task.task_id}"

        # Test train suite
        res_train = verifier.verify(code, task.entry_point, task.train_tests)
        assert res_train.reward == 1.0, (
            f"Reference solution for {task.task_id} failed train tests: "
            f"{res_train.passed_count}/{res_train.total_count}, stderr: {res_train.execution_result.stderr if res_train.execution_result else ''}"
        )

        # Test hidden suite
        res_hidden = verifier.verify(code, task.entry_point, task.hidden_tests)
        assert res_hidden.reward == 1.0, (
            f"Reference solution for {task.task_id} failed hidden tests: "
            f"{res_hidden.passed_count}/{res_hidden.total_count}, stderr: {res_hidden.execution_result.stderr if res_hidden.execution_result else ''}"
        )
