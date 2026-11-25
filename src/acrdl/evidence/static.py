"""Static "overfitting smell" features computed from the buggy -> patch change.

All counts refer to constructs that are *new* in the patch (present in the
patch but not in the buggy program), compared as multisets of AST dumps.
"""

from __future__ import annotations

import ast
import difflib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from acrdl.types import TestCase


@dataclass
class StaticFeatures:
    diff_lines: int = 0
    node_delta: float = 0.0
    hardcoded_test_literal: int = 0
    special_case_eq: int = 0
    removed_guards: int = 0
    added_broad_except: int = 0
    added_return_const: int = 0
    added_branches: int = 0
    parse_error: int = 0

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def _nodes(tree: ast.AST, kind: type) -> Counter:
    return Counter(ast.dump(n) for n in ast.walk(tree) if isinstance(n, kind))


def _new(buggy: ast.AST, patch: ast.AST, kind: type) -> list[ast.AST]:
    old = _nodes(buggy, kind)
    out = []
    for n in ast.walk(patch):
        if isinstance(n, kind):
            d = ast.dump(n)
            if old[d] > 0:
                old[d] -= 1
            else:
                out.append(n)
    return out


def _literal(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _NOT_LITERAL


_NOT_LITERAL = object()


def static_features(buggy_src: str, patch_src: str,
                    visible_tests: Sequence[TestCase] = ()) -> StaticFeatures:
    f = StaticFeatures()
    f.diff_lines = sum(
        1 for ln in difflib.unified_diff(buggy_src.splitlines(), patch_src.splitlines(),
                                         lineterm="", n=0)
        if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
    )
    try:
        bt, pt = ast.parse(buggy_src), ast.parse(patch_src)
    except SyntaxError:
        f.parse_error = 1
        return f
    nb = sum(1 for _ in ast.walk(bt))
    f.node_delta = abs(sum(1 for _ in ast.walk(pt)) - nb) / max(1, nb)

    test_values = {repr(v) for t in visible_tests for v in (*t.args, t.expected)}
    for cmp in _new(bt, pt, ast.Compare):
        if not any(isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.In)) for op in cmp.ops):
            continue
        sides = [cmp.left, *cmp.comparators]
        lits = [v for v in map(_literal, sides) if v is not _NOT_LITERAL]
        names = [s for s in sides if isinstance(s, (ast.Name, ast.Attribute, ast.Subscript))]
        if lits and names:
            f.special_case_eq += 1
            if any(repr(v) in test_values for v in lits):
                f.hardcoded_test_literal += 1

    f.added_return_const = sum(
        1 for r in _new(bt, pt, ast.Return)
        if r.value is not None and _literal(r.value) is not _NOT_LITERAL
    )
    f.added_branches = len(_new(bt, pt, ast.If)) + len(_new(bt, pt, ast.IfExp))

    removed = sum((_nodes(bt, ast.Raise) - _nodes(pt, ast.Raise)).values())
    removed += sum((_nodes(bt, ast.Assert) - _nodes(pt, ast.Assert)).values())
    f.removed_guards = removed
    for h in _new(bt, pt, ast.ExceptHandler):
        if h.type is None or (isinstance(h.type, ast.Name)
                              and h.type.id in ("Exception", "BaseException")):
            f.added_broad_except += 1
    return f


def squash(count: float, scale: float = 3.0) -> float:
    """Map a non-negative count into [0, 1)."""
    return 1.0 - math.exp(-count / scale)
