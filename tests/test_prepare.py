"""Tests for real-data preparation (no network: synthetic inputs only)."""

from __future__ import annotations

import math

from acrdl.data.prepare import (
    bug_from_json,
    bug_to_json,
    find_guards,
    parse_evalplus_tests,
    remove_lines,
    underspecify_by_edge_class,
    underspecify_by_guard,
)
from acrdl.evidence.sandbox import run_function
from acrdl.evidence.testing import NO_CRASH, describe_failures, run_tests
from acrdl.types import BuggyProgram, Problem, TestCase

EVALPLUS_SCRIPT = '''
import numpy as np
def check(candidate):
    inputs = [[[1, 2]], [[]], [[float("inf")]]]
    results = [1.5, 0.0, inf]
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assertion(candidate(*inp), exp, 0)
'''

AVG_PROMPT = (
    "def average(numbers):\n"
    '    """Return the mean of a list of numbers. Return 0.0 for an empty list.\n'
    "    >>> average([1, 2])\n    1.5\n    >>> average([])\n    0.0\n"
    '    """\n'
)
AVG_REF = AVG_PROMPT + (
    "    if not numbers:\n        return 0.0\n    return sum(numbers) / len(numbers)\n"
)


def avg_problem() -> Problem:
    visible = [TestCase(([1, 2],), 1.5), TestCase(([4],), 4.0), TestCase(([],), 0.0)]
    hidden = visible + [TestCase(([2, 4, 6],), 4.0), TestCase(([0, 0],), 0.0)]
    return Problem("toy/avg", "average", AVG_PROMPT, AVG_REF, visible, hidden)


def test_parse_evalplus_tests_handles_special_floats():
    ins, outs = parse_evalplus_tests(EVALPLUS_SCRIPT)
    assert ins[:2] == [([1, 2],), ([],)] and math.isinf(ins[2][0][0])
    assert outs[:2] == [1.5, 0.0] and math.isinf(outs[2])


def test_no_crash_expectation_and_failure_text():
    src = "def f(x):\n    return 1 / x\n"
    tests = [TestCase((0,), NO_CRASH), TestCase((2,), 0.5)]
    assert run_tests(src, "f", tests) == [False, True]
    lines = describe_failures(src, "f", tests)
    assert lines == ["f(0) raises ZeroDivisionError; should return normally"]


def test_remove_guard_keeps_else_branch():
    src = ("def g(x):\n    if x < 0:\n        return 0\n"
           "    else:\n        y = x * 2\n        return y\n")
    (guard,) = find_guards(src, "g")
    out = remove_lines(src, guard)
    assert run_function(out, "g", [(3,), (-1,)])[0].value == "6"
    assert "if x < 0" not in out


def test_underspecify_by_guard_builds_exact_ambiguity():
    v = underspecify_by_guard(avg_problem())
    assert v is not None
    p = v.bug.problem
    assert p.ambiguous_inputs == [([],)]
    assert "Return 0.0 for an empty list." not in p.prompt
    assert ">>> average([])" not in p.prompt and ">>> average([1, 2])" in p.prompt
    assert all(t.args != ([],) for t in p.hidden_tests)
    assert TestCase(([],), NO_CRASH) in p.visible_tests
    # the buggy program crashes on the now-unspecified input; the reference does not
    assert not all(run_tests(v.bug.source, "average", p.visible_tests))
    assert all(run_tests(p.reference, "average", p.visible_tests + p.hidden_tests))


def test_underspecify_by_edge_class_keeps_original_bug():
    bug_src = AVG_REF.replace("sum(numbers) / len(numbers)", "sum(numbers[1:]) / len(numbers)")
    v = underspecify_by_edge_class(BuggyProgram(avg_problem(), bug_src, "b"))
    assert v is not None and v.bug.problem.task_id.endswith("~silent-empty")
    assert v.bug.problem.ambiguous_inputs == [([],)]
    assert "sum(numbers[1:])" in v.bug.source


def test_json_roundtrip_preserves_tuples_and_no_crash():
    v = underspecify_by_guard(avg_problem())
    back = bug_from_json(bug_to_json(v.bug))
    assert back.problem.visible_tests == v.bug.problem.visible_tests
    assert back.problem.ambiguous_inputs == v.bug.problem.ambiguous_inputs
    assert any(t.expected is NO_CRASH for t in back.problem.visible_tests)


def test_values_equal_handles_huge_integers():
    from acrdl.evidence.testing import values_equal

    big = 10 ** 400
    assert values_equal(big, big)
    assert not values_equal(big, 1.5)
    assert values_equal(2, 2.0)


def test_apply_unified_diff_roundtrip():
    import random

    from acrdl.data.rebuild import apply_unified_diff, diff_of
    from acrdl.evidence.code_text import code_text

    rng = random.Random(0)
    base = [f"line {i}" for i in range(30)]
    for _ in range(50):
        new = list(base)
        for _ in range(rng.randint(1, 4)):
            j = rng.randrange(len(new))
            op = rng.choice(["del", "ins", "rep"])
            if op == "del" and len(new) > 1:
                del new[j]
            elif op == "ins":
                new.insert(j, f"new {rng.random():.3f}")
            else:
                new[j] = f"changed {rng.random():.3f}"
        src, patch = "\n".join(base) + "\n", "\n".join(new) + "\n"
        assert apply_unified_diff(src, diff_of(code_text(src, patch))) == patch
