"""A tiny offline benchmark used by the demo and the test-suite.

Each problem mimics a HumanEval+/EvalPlus item: a docstring spec, a reference
solution, a handful of visible tests and a larger hidden test set. ``BUGS``
holds one injected bug per problem; ``SCRIPTED_CANDIDATES`` holds the kind of
patches an LLM typically produces (correct, overfitting and divergent ones).
"""

from __future__ import annotations

from acrdl.types import Problem, TestCase

T = TestCase

AVERAGE_CLAUSE = "Return 0.0 for an empty list."

PROBLEMS: list[Problem] = [
    Problem(
        task_id="toy/average",
        entry_point="average",
        prompt=(
            "def average(numbers):\n"
            '    """Return the arithmetic mean of a list of numbers. '
            f'{AVERAGE_CLAUSE}"""\n'
        ),
        reference=(
            "def average(numbers):\n"
            "    if not numbers:\n"
            "        return 0.0\n"
            "    return sum(numbers) / len(numbers)\n"
        ),
        visible_tests=[T(([1, 2],), 1.5), T(([0, 4],), 2.0), T(([],), 0.0)],
        hidden_tests=[
            T(([5],), 5.0), T(([1, 2, 3, 4],), 2.5), T(([-2, 2],), 0.0),
            T(([10, 20, 30],), 20.0), T(([7, 7],), 7.0), T(([],), 0.0),
            T(([3, 1],), 2.0), T(([0, 0, 9],), 3.0),
        ],
    ),
    Problem(
        task_id="toy/clip_percent",
        entry_point="clip_percent",
        prompt=(
            "def clip_percent(x):\n"
            '    """Return x limited to the closed interval [0, 100]."""\n'
        ),
        reference="def clip_percent(x):\n    return max(0, min(x, 100))\n",
        visible_tests=[T((50,), 50), T((-3,), 0), T((120,), 100)],
        hidden_tests=[
            T((0,), 0), T((100,), 100), T((101,), 100), T((-1,), 0),
            T((99,), 99), T((7,), 7), T((250,), 100), T((-90,), 0),
        ],
    ),
    Problem(
        task_id="toy/count_vowels",
        entry_point="count_vowels",
        prompt=(
            "def count_vowels(s):\n"
            '    """Count the vowels (a, e, i, o, u, case-insensitive) in s."""\n'
        ),
        reference=(
            "def count_vowels(s):\n"
            "    return sum(1 for c in s.lower() if c in 'aeiou')\n"
        ),
        visible_tests=[T(("hello",), 2), T(("sky",), 0), T(("AEIOU",), 5)],
        hidden_tests=[
            T(("",), 0), T(("queue",), 4), T(("Umbrella",), 3), T(("xyz",), 0),
            T(("aaa",), 3), T(("Python",), 1), T(("uU",), 2), T(("rhythm",), 0),
        ],
    ),
    Problem(
        task_id="toy/is_palindrome",
        entry_point="is_palindrome",
        prompt=(
            "def is_palindrome(s):\n"
            '    """Return True if s reads the same forwards and backwards."""\n'
        ),
        reference="def is_palindrome(s):\n    return s == s[::-1]\n",
        visible_tests=[T(("aba",), True), T(("ab",), False), T(("racecar",), True)],
        hidden_tests=[
            T(("",), True), T(("a",), True), T(("abca",), False), T(("abba",), True),
            T(("xyzzyx",), True), T(("xy",), False), T(("aab",), False), T(("noon",), True),
        ],
    ),
    Problem(
        task_id="toy/max_gap",
        entry_point="max_gap",
        prompt=(
            "def max_gap(nums):\n"
            '    """Return the largest difference between consecutive elements of nums."""\n'
        ),
        reference=(
            "def max_gap(nums):\n"
            "    return max(b - a for a, b in zip(nums, nums[1:]))\n"
        ),
        visible_tests=[T(([1, 4, 6],), 3), T(([5, 6],), 1), T(([1, 2, 10],), 8)],
        hidden_tests=[
            T(([0, 0],), 0), T(([3, 1],), -2), T(([1, 5, 2, 9],), 7), T(([2, 2, 2],), 0),
            T(([-5, 5],), 10), T(([10, 9, 8],), -1), T(([0, 100, 101],), 100),
            T(([1, 3, 6, 10],), 4),
        ],
    ),
]

BUGS: dict[str, str] = {
    "toy/average": (
        "def average(numbers):\n"
        "    if not numbers:\n"
        "        return 0.0\n"
        "    return sum(numbers[1:]) / len(numbers)\n"
    ),
    "toy/clip_percent": "def clip_percent(x):\n    return max(0, max(x, 100))\n",
    "toy/count_vowels": (
        "def count_vowels(s):\n"
        "    return sum(1 for c in s if c in 'aeiou')\n"
    ),
    "toy/is_palindrome": "def is_palindrome(s):\n    return s == s[::-1][1:]\n",
    "toy/max_gap": (
        "def max_gap(nums):\n"
        "    return max(b - a for a, b in zip(nums, nums[2:]))\n"
    ),
}

SCRIPTED_CANDIDATES: dict[str, list[str]] = {
    "toy/average": [
        # Three behaviourally distinct but spec-consistent treatments of [] once
        # the empty-list clause is removed, plus one overfitting patch.
        "def average(numbers):\n    if not numbers:\n        return 0.0\n"
        "    return sum(numbers) / len(numbers)\n",
        "def average(numbers):\n    if len(numbers) == 0:\n        return 0.0\n"
        "    total = 0\n    for x in numbers:\n        total += x\n"
        "    return total / len(numbers)\n",
        "def average(numbers):\n    return sum(numbers) / len(numbers)\n",
        "def average(numbers):\n    if not numbers:\n        return None\n"
        "    return sum(numbers) / len(numbers)\n",
        "def average(numbers):\n    if numbers == [1, 2]:\n        return 1.5\n"
        "    if not numbers:\n        return 0.0\n    return sum(numbers[1:]) / len(numbers)\n",
    ],
    "toy/clip_percent": [
        "def clip_percent(x):\n    return max(0, min(x, 100))\n",
        "def clip_percent(x):\n    if x < 0:\n        return 0\n"
        "    if x > 100:\n        return 100\n    return x\n",
        "def clip_percent(x):\n    if x == 120:\n        return 100\n    return max(0, x)\n",
        "def clip_percent(x):\n    return min(max(x, 0), 100) if x != 120 else 100\n",
    ],
    "toy/count_vowels": [
        "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiouAEIOU')\n",
        "def count_vowels(s):\n    if s == 'AEIOU':\n        return 5\n"
        "    return sum(1 for c in s if c in 'aeiou')\n",
        "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeioAEIO')\n",
    ],
    "toy/is_palindrome": [
        "def is_palindrome(s):\n    return s == s[::-1]\n",
        "def is_palindrome(s):\n    return all(s[i] == s[-1 - i] for i in range(len(s) // 2))\n",
        "def is_palindrome(s):\n    return len(s) % 2 == 1\n",
    ],
    "toy/max_gap": [
        "def max_gap(nums):\n    return max(b - a for a, b in zip(nums, nums[1:]))\n",
        "def max_gap(nums):\n    return max(nums[i + 1] - nums[i] for i in range(len(nums) - 1))\n",
        "def max_gap(nums):\n    return max(abs(b - a) for a, b in zip(nums, nums[1:]))\n",
    ],
}
