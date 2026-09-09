"""Validation tests for the curated task dataset (train, heldout instances, heldout families)."""

from pathlib import Path
from data.loader import load_tasks


def test_all_tasks_conform_to_schema():
    train_tasks = load_tasks("data/tasks/train.json")
    heldout_instances = load_tasks("data/tasks/heldout_instances.json")
    heldout_families = load_tasks("data/tasks/heldout_families.json")

    assert len(train_tasks) == 20, f"Expected 20 train tasks, found {len(train_tasks)}"
    assert len(heldout_instances) == 10, f"Expected 10 heldout instances, found {len(heldout_instances)}"
    assert len(heldout_families) == 5, f"Expected 5 heldout families, found {len(heldout_families)}"

    all_tasks = train_tasks + heldout_instances + heldout_families
    assert len(all_tasks) == 35

    # Check global uniqueness of task IDs
    task_ids = [t.task_id for t in all_tasks]
    assert len(task_ids) == len(set(task_ids)), "Duplicate task_id detected across datasets!"

    # Check structure of each task
    for task in all_tasks:
        assert task.prompt, f"Task {task.task_id} missing prompt"
        assert task.entry_point, f"Task {task.task_id} missing entry_point"
        assert len(task.train_tests) >= 2, f"Task {task.task_id} train_tests too short"
        assert len(task.hidden_tests) >= 2, f"Task {task.task_id} hidden_tests too short"

        # Check trivial test annotations for Condition B3 undercoverage
        trivial_count = sum(1 for tc in task.train_tests if tc.is_trivial)
        assert trivial_count >= 1, f"Task {task.task_id} must have at least one trivial test"
