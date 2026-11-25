"""Test execution helpers built on the sandbox."""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from typing import Any

from acrdl.evidence.sandbox import run_function
from acrdl.types import ExecResult, TestCase


def values_equal(a: Any, b: Any, rel_tol: float = 1e-6) -> bool:
    """Structural equality with float tolerance (bools are not numbers here)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b  # exact, and safe for integers too large for a float
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)
        except OverflowError:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(values_equal(x, y, rel_tol) for x, y in zip(a, b)))
    return a == b


class _NoCrash:
    """Expected value meaning "returns normally, any value is acceptable".

    Used for crash reports on inputs where the specification is silent: the
    bug is the crash, and no particular return value is required.
    """

    def __repr__(self) -> str:
        return "NO_CRASH"


NO_CRASH = _NoCrash()


def result_matches(result: ExecResult, expected: Any) -> bool:
    if result.kind != "ok":
        return False
    if expected is NO_CRASH:
        return True
    try:
        actual = ast.literal_eval(result.value)
    except (ValueError, SyntaxError):
        return False
    return values_equal(actual, expected)


def run_tests(source: str, entry_point: str, tests: Sequence[TestCase],
              timeout_s: float = 2.0, memory_limit_mb: int = 512) -> list[bool]:
    """Per-test pass/fail for ``source``."""
    results = run_function(source, entry_point, [t.args for t in tests],
                           timeout_s, memory_limit_mb)
    return [result_matches(r, t.expected) for r, t in zip(results, tests)]


def _call_text(entry_point: str, args: tuple, limit: int = 160) -> str:
    text = f"{entry_point}({', '.join(repr(a) for a in args)})"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe_failures(source: str, entry_point: str, tests: Sequence[TestCase],
                      timeout_s: float = 2.0, memory_limit_mb: int = 512,
                      max_items: int = 3) -> list[str]:
    """Human-readable failing-test lines for the repair prompt."""
    lines: list[str] = []
    results = run_function(source, entry_point, [t.args for t in tests], timeout_s,
                           memory_limit_mb)
    for r, t in zip(results, tests):
        if result_matches(r, t.expected):
            continue
        call = _call_text(entry_point, t.args)
        if r.kind == "error":
            got = f"raises {r.value}"
        elif r.kind == "timeout":
            got = "times out"
        else:
            got = f"returns {r.value[:80]}"
        want = ("should return normally" if t.expected is NO_CRASH
                else f"expected {t.expected!r}"[:100])
        lines.append(f"{call} {got}; {want}")
        if len(lines) >= max_items:
            break
    return lines
