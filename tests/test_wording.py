"""Tests for the wording intervention and the wording-dependence statistic."""

from __future__ import annotations

import ast
import random

from acrdl.data.wording import make_wording, rename_params, reorder_docstring_text
from acrdl.evidence.wording import mutual_information, permutation_test
from acrdl.types import BuggyProgram, Problem, TestCase

PROMPT = (
    "def average(numbers):\n"
    '    """Return the mean of a list of numbers. The list holds floats or ints.\n'
    "    The result is a float. Rounding is not applied.\n"
    "    >>> average([1, 2])\n    1.5\n"
    '    """\n'
)
REF = PROMPT + "    if not numbers:\n        return 0.0\n    return sum(numbers) / len(numbers)\n"
BUGGY = PROMPT + "    # crashes on []\n    return sum(numbers) / len(numbers)\n"


def bug() -> BuggyProgram:
    visible = [TestCase(([1, 2],), 1.5), TestCase(([],), 0.0)]
    hidden = visible + [TestCase(([4, 6],), 5.0)]
    return BuggyProgram(Problem("toy/avg", "average", PROMPT, REF, visible, hidden), BUGGY, "b0")


def test_rename_changes_names_everywhere_but_keeps_behaviour():
    out = rename_params(REF, "average", {"numbers": "values"})
    assert "numbers" not in out.split('"""')[0] and "def average(values)" in out
    assert "mean of a list of values" in out  # docstring mention renamed too
    ast.parse(out)


def test_each_wording_preserves_meaning_and_bug():
    for w in ("w1", "w2", "w3"):
        v = make_wording(bug(), w)
        assert v is not None, w
        assert v.bug_id == f"b0~{w}" and v.problem.task_id == f"toy/avg~{w}"


def test_reorder_keeps_first_sentence_and_examples():
    doc = PROMPT.split('"""')[1]
    new = reorder_docstring_text(doc)
    assert new is not None
    assert new.strip().startswith("Return the mean")
    assert ">>> average([1, 2])" in new
    assert new.index("Rounding") < new.index("The list holds")


def test_mutual_information_and_permutation_test():
    dependent = {"w0": ["a"] * 10, "w1": ["b"] * 10, "w2": ["a"] * 10}
    noise = {"w0": ["a"] * 9 + ["b"], "w1": ["a"] * 9 + ["b"], "w2": ["a"] * 9 + ["b"]}
    assert mutual_information(dependent) > 0.9
    assert mutual_information(noise) == 0.0
    assert permutation_test(dependent, 500)[1] < 0.01
    assert permutation_test(noise, 500)[1] > 0.5


def test_permutation_test_false_alarm_rate_is_controlled():
    rng = random.Random(1)
    rejections = 0
    for trial in range(200):
        groups = {w: [rng.choice("aab") for _ in range(10)] for w in ("w0", "w1", "w2")}
        rejections += permutation_test(groups, 200, seed=trial)[1] <= 0.05
    assert rejections / 200 <= 0.09  # nominal 5%, slack for 200 trials
