"""Reference solutions for all 35 curated tasks."""

REFERENCE_SOLUTIONS = {
    # Train Tasks (20)
    "math_sum_even": """
def sum_even(arr: list[int]) -> int:
    return sum(x for x in arr if x % 2 == 0)
""",
    "math_is_prime": """
def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
""",
    "math_factorial": """
def factorial(n: int) -> int:
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res
""",
    "math_count_digits": """
def count_digits(n: int) -> int:
    return len(str(abs(n)))
""",
    "math_gcd": """
def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a
""",
    "str_reverse_vowels": """
def reverse_vowels(s: str) -> str:
    vowels = set("aeiouAEIOU")
    chars = list(s)
    i, j = 0, len(chars) - 1
    while i < j:
        if chars[i] not in vowels:
            i += 1
        elif chars[j] not in vowels:
            j -= 1
        else:
            chars[i], chars[j] = chars[j], chars[i]
            i += 1
            j -= 1
    return "".join(chars)
""",
    "str_count_palindromes": """
def count_palindromes(sentence: str) -> int:
    words = sentence.strip().split()
    return sum(1 for w in words if w.lower() == w.lower()[::-1])
""",
    "str_is_anagram": """
def is_anagram(s1: str, s2: str) -> bool:
    clean1 = sorted(s1.replace(" ", "").lower())
    clean2 = sorted(s2.replace(" ", "").lower())
    return clean1 == clean2
""",
    "str_longest_common_prefix": """
def longest_common_prefix(strs: list[str]) -> str:
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
""",
    "str_remove_punctuation": """
import string

def remove_punctuation(s: str) -> str:
    punct = set(string.punctuation)
    return "".join(c for c in s if c not in punct)
""",
    "list_find_duplicates": """
def find_duplicates(arr: list[int]) -> list[int]:
    from collections import Counter
    counts = Counter(arr)
    return sorted(k for k, v in counts.items() if v > 1)
""",
    "list_max_consecutive_ones": """
def max_consecutive_ones(nums: list[int]) -> int:
    max_c = cur = 0
    for x in nums:
        if x == 1:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 0
    return max_c
""",
    "list_rotate_right": """
def rotate_right(nums: list[int], k: int) -> list[int]:
    if not nums:
        return []
    k = k % len(nums)
    return nums[-k:] + nums[:-k] if k > 0 else list(nums)
""",
    "list_chunk_array": """
def chunk_array(arr: list[int], size: int) -> list[list[int]]:
    if size <= 0:
        return []
    return [arr[i:i+size] for i in range(0, len(arr), size)]
""",
    "list_running_sum": """
def running_sum(nums: list[int]) -> list[int]:
    res = []
    curr = 0
    for x in nums:
        curr += x
        res.append(curr)
    return res
""",
    "dict_merge_counters": """
def merge_counters(d1: dict[str, int], d2: dict[str, int]) -> dict[str, int]:
    res = dict(d1)
    for k, v in d2.items():
        res[k] = res.get(k, 0) + v
    return res
""",
    "dict_invert_mapping": """
def invert_mapping(d: dict[str, str]) -> dict[str, str]:
    return {v: k for k, v in d.items()}
""",
    "dict_filter_by_value": """
def filter_by_value(d: dict[str, int], threshold: int) -> dict[str, int]:
    return {k: v for k, v in d.items() if v >= threshold}
""",
    "logic_two_sum": """
def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, x in enumerate(nums):
        diff = target - x
        if diff in seen:
            return sorted([seen[diff], i])
        seen[x] = i
    return []
""",
    "logic_valid_parentheses": """
def valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for c in s:
        if c in mapping:
            if not stack or stack.pop() != mapping[c]:
                return False
        elif c in mapping.values():
            stack.append(c)
    return len(stack) == 0
""",

    # Heldout Instances (10)
    "heldout_math_count_divisors": """
def count_divisors(n: int) -> int:
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1 if i * i == n else 2
        i += 1
    return count
""",
    "heldout_math_is_power_of_two": """
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
""",
    "heldout_str_compress_string": """
def compress_string(s: str) -> str:
    if not s:
        return ""
    res = []
    i = 0
    while i < len(s):
        count = 1
        while i + 1 < len(s) and s[i+1] == s[i]:
            count += 1
            i += 1
        res.append(f"{s[i]}{count}")
        i += 1
    return "".join(res)
""",
    "heldout_str_title_case": """
def title_case(sentence: str) -> str:
    words = sentence.split(" ")
    return " ".join(w.capitalize() for w in words)
""",
    "heldout_list_remove_consecutive_duplicates": """
def remove_consecutive_duplicates(arr: list[int]) -> list[int]:
    if not arr:
        return []
    res = [arr[0]]
    for x in arr[1:]:
        if x != res[-1]:
            res.append(x)
    return res
""",
    "heldout_list_find_missing_number": """
def find_missing_number(nums: list[int]) -> int:
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    return expected_sum - sum(nums)
""",
    "heldout_dict_group_by_length": """
def group_by_length(words: list[str]) -> dict[str, list[str]]:
    res = {}
    for w in words:
        l = str(len(w))
        if l not in res:
            res[l] = []
        res[l].append(w)
    return res
""",
    "heldout_dict_top_k_frequent": """
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    from collections import Counter
    counts = Counter(nums)
    return [item for item, count in counts.most_common(k)]
""",
    "heldout_logic_is_subsequence": """
def is_subsequence(s: str, t: str) -> bool:
    it = iter(t)
    return all(c in it for c in s)
""",
    "heldout_logic_majority_element": """
def majority_element(nums: list[int]) -> int:
    from collections import Counter
    counts = Counter(nums)
    return counts.most_common(1)[0][0]
""",

    # Heldout Families (5)
    "family_intervals_merge": """
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for cur in sorted_intervals[1:]:
        last = merged[-1]
        if cur[0] <= last[1]:
            last[1] = max(last[1], cur[1])
        else:
            merged.append(cur)
    return merged
""",
    "family_matrix_transpose": """
def transpose_matrix(matrix: list[list[int]]) -> list[list[int]]:
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]
""",
    "family_matrix_rotate_90": """
def rotate_matrix_90(matrix: list[list[int]]) -> list[list[int]]:
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix[::-1])]
""",
    "family_search_binary_search": """
def binary_search(nums: list[int], target: int) -> int:
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""",
    "family_nested_flatten_deep": """
def flatten_deep(nested: list) -> list:
    res = []
    def _flat(item):
        if isinstance(item, list):
            for sub in item:
                _flat(sub)
        else:
            res.append(item)
    _flat(nested)
    return res
"""
}
