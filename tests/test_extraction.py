"""Unit tests for extract_python_code."""

from training.extraction import extract_python_code

PROMPT = "def sum_even(arr: list[int]) -> int:\n    \"\"\"Sum even numbers.\"\"\"\n"


def test_extract_markdown_fence():
    raw = """Here is the solution:
```python
def sum_even(arr: list[int]) -> int:
    return sum(x for x in arr if x % 2 == 0)
```
Hope this helps!"""
    code = extract_python_code(raw, PROMPT, "sum_even")
    assert "def sum_even" in code
    assert "Hope this helps" not in code
    assert "```" not in code


def test_extract_continuation():
    raw = "    return sum(x for x in arr if x % 2 == 0)"
    code = extract_python_code(raw, PROMPT, "sum_even")
    assert "def sum_even" in code
    assert "return sum(" in code


def test_extract_direct_code():
    raw = "def sum_even(arr):\n    return sum(x for x in arr if x % 2 == 0)"
    code = extract_python_code(raw, PROMPT, "sum_even")
    assert code == raw
