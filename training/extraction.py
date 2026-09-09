"""Utilities for extracting clean Python code from model outputs."""

import re


def extract_python_code(raw_output: str, prompt: str, entry_point: str) -> str:
    """Extracts executable Python code from an LLM response.

    Handles:
    - Markdown fenced blocks (```python ... ``` or ``` ... ```)
    - Direct function continuation
    - Raw code containing the function definition
    """
    text = raw_output.strip()

    # 1. Search for markdown code fences
    fence_pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    if matches:
        # Take the block that defines the entry_point, or the largest block
        for block in matches:
            if f"def {entry_point}" in block:
                return block.strip()
        return matches[0].strip()

    # 2. If the text defines the function directly
    if f"def {entry_point}" in text:
        # Find start of def entry_point
        idx = text.find(f"def {entry_point}")
        return text[idx:].strip()

    # 3. If the model produced just the function body (continuation)
    if not text.startswith("def ") and not text.startswith("import "):
        return prompt.rstrip() + "\n" + text

    return text
