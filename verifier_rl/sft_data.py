"""Fixed, project-authored SFT targets; no cache solutions or arbitrary exec."""

import ast
from copy import deepcopy
from dataclasses import dataclass
import inspect

from .suites import canonical_json, digest

VERSION = "non-cache-functions-sft-0.1"


def running_totals(values):
    total = 0
    results = []
    for value in values:
        total += value
        results.append(total)
    return results


def clamp_values(values, lower, upper):
    results = []
    for value in values:
        results.append(max(lower, min(upper, value)))
    return results


def occurrence_counts(keys):
    counts = {}
    results = []
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
        results.append(counts[key])
    return results


def first_indices(items, queries):
    positions = {}
    for index, item in enumerate(items):
        if item not in positions:
            positions[item] = index
    results = []
    for query in queries:
        results.append(positions.get(query))
    return results


def balance_reads(operations):
    balance = 0
    results = []
    for operation in operations:
        if operation["op"] == "add":
            balance += operation["value"]
        else:
            results.append(balance)
    return results


def counter_reads(operations):
    counter = 0
    results = []
    for operation in operations:
        if operation["op"] == "add":
            counter += operation["value"]
        elif operation["op"] == "reset":
            counter = 0
        else:
            results.append(counter)
    return results


def replace_negatives(values):
    results = []
    for value in values:
        results.append(0 if value < 0 else value)
    return results


def even_positions(values):
    results = []
    for index, value in enumerate(values):
        if index % 2 == 0:
            results.append(value)
    return results


def encode_runs(values):
    results = []
    for value in values:
        if results and results[-1][0] == value:
            results[-1][1] += 1
        else:
            results.append([value, 1])
    return results


def merge_sorted(left, right):
    i = 0
    j = 0
    results = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            results.append(left[i])
            i += 1
        else:
            results.append(right[j])
            j += 1
    results.extend(left[i:])
    results.extend(right[j:])
    return results


@dataclass(frozen=True)
class Family:
    function: object
    split: str
    description: str
    checks: tuple


def families():
    return (
        Family(running_totals, "train", "values is a list of integers. Return the running sum after each element, in input order. An empty list returns [].", (
            (([],), []), (([2, -3, 4],), [2, -1, 3]), (([0],), [0]), (([-2, -2],), [-2, -4]))),
        Family(clamp_values, "train", "values is a list of integers; lower and upper are integers with lower <= upper. Return a new list replacing values below lower with lower and above upper with upper. Preserve length and order.", (
            (([], 0, 1), []), (([-4, 0, 2, 8], 0, 3), [0, 0, 2, 3]), (([1, 7], 4, 4), [4, 4]), (([-6, -3, 0], -5, -1), [-5, -3, -1]))),
        Family(occurrence_counts, "train", "keys is a list of strings. For each key, return how many times it has appeared up to and including that position. Return one integer per input element, or [] for empty input.", (
            (([],), []), ((["a", "b", "a", "a"],), [1, 1, 2, 3]), (([""],), [1]), ((["x", "y", "z"],), [1, 1, 1]))),
        Family(first_indices, "train", "items and queries are lists of strings. For each query return the zero-based first index in items, or None if absent. Preserve query order and return exactly one result per query.", (
            (([], []), []), (([], ["x"]), [None]), ((["a", "b", "a"], ["a", "z", "b"]), [0, None, 1]), (([""], ["", ""]), [0, 0]))),
        Family(balance_reads, "train", "operations is a list of dictionaries in processing order. Initially the integer balance is 0. An add operation has exactly op='add' and integer value: add value to the balance. A read operation has only op='read', no value field. Return a list of balances at reads only. Do not modify operations.", (
            (([],), []), (([{"op": "read"}],), [0]), (([{"op": "add", "value": 5}, {"op": "read"}, {"op": "add", "value": -7}, {"op": "read"}],), [5, -2]), (([{"op": "add", "value": 2}],), []))),
        Family(counter_reads, "train", "operations is an ordered list of dictionaries. Start a counter at 0. {'op':'add','value':n} adds integer n; {'op':'reset'} sets it to 0; {'op':'read'} records its current value. Reset and read have no value field. Return one list containing the read results only, without modifying input.", (
            (([],), []), (([{"op": "read"}, {"op": "reset"}, {"op": "read"}],), [0, 0]), (([{"op": "add", "value": 4}, {"op": "read"}, {"op": "reset"}, {"op": "add", "value": -1}, {"op": "read"}],), [4, -1]), (([{"op": "add", "value": 2}, {"op": "reset"}],), []))),
        Family(replace_negatives, "train", "values is a list of integers. Return a new list replacing negative integers with 0 and leaving other integers unchanged. Preserve order and length.", (
            (([],), []), (([-2, 0, 3, -1],), [0, 0, 3, 0]), (([4],), [4]), (([-1, -5],), [0, 0]))),
        Family(even_positions, "train", "values is a list of integers. Return elements at even zero-based positions (0, 2, 4, ...) in their original order. Select by position, not by whether the value is even.", (
            (([],), []), (([9],), [9]), (([5, 2, 7, 4, 9],), [5, 7, 9]), (([0, -1, -2, -3],), [0, -2]))),
        Family(encode_runs, "development", "values is a list of strings. Return a list of [value, count] lists for consecutive equal runs. Keep separate nonadjacent runs and preserve order. Empty input returns [].", (
            (([],), []), ((["a", "a", "b", "a"],), [["a", 2], ["b", 1], ["a", 1]]), (([""],), [["", 1]]), ((["x", "x", "x"],), [["x", 3]]))),
        Family(merge_sorted, "development", "left and right are lists of integers already sorted in nondecreasing order. Return one sorted list containing all elements from both inputs, retaining duplicates and not modifying inputs.", (
            (([], []), []), (([1, 3], [1, 2]), [1, 1, 2, 3]), (([], [-2, 0]), [-2, 0]), (([-3, -1], [-2, 4]), [-3, -2, -1, 4]))),
    )


INSTRUCTIONS = (
    "Implement the Python function {signature}.\n{description}\nReturn only the function definition; no examples, tests, print calls, or prose.",
    "Write Python code defining {signature}.\nContract: {description}\nSubmit executable function code only, without demonstrations or explanation.",
    "Provide a Python implementation of {signature}.\nRequirements: {description}\nYour answer must contain the function code only; do not call it or print anything.",
    "Complete this programming task with a function named {signature}.\n{description}\nOutput only Python function code, with no example invocations or surrounding text.",
)


def dataset_manifest():
    examples, specs = [], []
    for family in families():
        fn = family.function
        source = inspect.getsource(fn).strip()
        signature = fn.__name__ + str(inspect.signature(fn))
        specs.append({"family": fn.__name__, "split": family.split, "signature": signature,
                      "description": family.description, "checks": family.checks, "source_hash": digest(source)})
        for index, wording in enumerate(INSTRUCTIONS):
            examples.append({"id": f"{fn.__name__}-{index}", "family": fn.__name__, "split": family.split,
                             "prompt": wording.format(signature=signature, description=family.description),
                             "completion": source, "source_hash": digest(source)})
    return {"version": VERSION, "authorship": "project-authored, assistant-written fixed functions; no external corpus",
            "families": specs, "examples": examples}


def validate_dataset():
    checks = 0
    for family in families():
        for args, expected in family.checks:
            args = deepcopy(args)
            before = deepcopy(args)
            actual = family.function(*args)  # Trusted fixed functions only; never exec a source string.
            if canonical_json(actual) != canonical_json(expected) or args != before:
                raise AssertionError(f"target mismatch/input mutation: {family.function.__name__}")
            checks += 1
    manifest = dataset_manifest()
    examples = manifest["examples"]
    for example in examples:
        tree = ast.parse(example["completion"])
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != example["family"]:
            raise AssertionError("target must be exactly its declared function")
        if any(word in (example["prompt"] + example["completion"]).lower() for word in ("cache", "ttl", "expiration")):
            raise AssertionError("cache material must not enter SFT")
    by_split = {split: {e["family"] for e in examples if e["split"] == split} for split in ("train", "development")}
    if by_split["train"] & by_split["development"] or [len(by_split[s]) for s in by_split] != [8, 2]:
        raise AssertionError("family split mismatch")
    return {"passed": True, "version": VERSION, "manifest_hash": digest(canonical_json(manifest)),
            "target_checks": checks, "train_examples": 32, "development_examples": 8,
            "train_families": 8, "development_families": 2}


def encode_example(tokenizer, example, max_tokens=1024):
    messages = [{"role": "user", "content": example["prompt"]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": example["completion"]}],
                                         tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if not full.startswith(prefix) or ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template/tokenizer prefix mismatch; refusing incorrect labels")
    if len(ids) > max_tokens or len(ids) <= len(prompt_ids):
        raise ValueError("empty or overlength target; no silent truncation")
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    if tokenizer.eos_token_id is None or tokenizer.eos_token_id not in labels[len(prompt_ids):]:
        raise ValueError("assistant EOS supervision is missing")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def collate_rows(rows, pad_token_id):
    if not rows or pad_token_id is None:
        raise ValueError("nonempty batch and explicit pad token required")
    width = max(len(r["input_ids"]) for r in rows)
    batch = {k: [] for k in ("input_ids", "attention_mask", "labels")}
    for row in rows:
        n = len(row["input_ids"])
        if len(row["labels"]) != n or row["attention_mask"] != [1] * n:
            raise ValueError("invalid unpadded training row")
        batch["input_ids"].append(row["input_ids"] + [pad_token_id] * (width - n))
        batch["attention_mask"].append(row["attention_mask"] + [0] * (width - n))
        batch["labels"].append(row["labels"] + [-100] * (width - n))
    return batch
