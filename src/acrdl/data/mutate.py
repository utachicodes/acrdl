"""AST mutation operators for injecting bugs into reference solutions.

Used to scale the dataset beyond HumanEvalFix: every operator enumerates its
mutation sites in a deterministic AST order; ``mutate`` picks sites with a
seeded RNG so a given (source, seed) always yields the same bugs.
"""

from __future__ import annotations

import ast
import copy
import random
from dataclasses import dataclass
from typing import Callable, Optional

_CMP_FLIP: dict[type, type] = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}
_ARITH_SWAP: dict[type, type] = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv,
    ast.Div: ast.FloorDiv, ast.FloorDiv: ast.Div, ast.Mod: ast.FloorDiv,
}


@dataclass(frozen=True)
class Mutant:
    source: str
    operator: str
    site: int


def _sites(tree: ast.AST, pred: Callable[[ast.AST], bool]) -> list[ast.AST]:
    return [n for n in ast.walk(tree) if pred(n)]


def _comparison_flip(node: ast.AST) -> None:
    assert isinstance(node, ast.Compare)
    node.ops[0] = _CMP_FLIP[type(node.ops[0])]()


def _arith_swap(node: ast.AST) -> None:
    assert isinstance(node, ast.BinOp)
    node.op = _ARITH_SWAP[type(node.op)]()


def _off_by_one(node: ast.AST) -> None:
    assert isinstance(node, ast.Constant)
    node.value = node.value + 1


def _negate_condition(node: ast.AST) -> None:
    assert isinstance(node, (ast.If, ast.While, ast.IfExp))
    node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)


def _return_change(node: ast.AST) -> None:
    assert isinstance(node, ast.Return) and node.value is not None
    v = node.value
    if isinstance(v, ast.Constant) and isinstance(v.value, bool):
        node.value = ast.Constant(value=not v.value)
    else:
        node.value = ast.Constant(value=None)


OPERATORS: dict[str, tuple[Callable[[ast.AST], bool], Callable[[ast.AST], None]]] = {
    "comparison_flip": (
        lambda n: isinstance(n, ast.Compare) and type(n.ops[0]) in _CMP_FLIP,
        _comparison_flip,
    ),
    "arith_swap": (
        lambda n: isinstance(n, ast.BinOp) and type(n.op) in _ARITH_SWAP,
        _arith_swap,
    ),
    "off_by_one": (
        lambda n: isinstance(n, ast.Constant) and type(n.value) is int,
        _off_by_one,
    ),
    "negate_condition": (
        lambda n: isinstance(n, (ast.If, ast.While, ast.IfExp)),
        _negate_condition,
    ),
    "return_change": (
        lambda n: isinstance(n, ast.Return) and n.value is not None,
        _return_change,
    ),
}


def enumerate_mutants(source: str, operators: Optional[list[str]] = None) -> list[Mutant]:
    """All single-site mutants of ``source``, deduplicated, excluding no-ops."""
    tree = ast.parse(source)
    original = ast.unparse(tree)
    seen: set[str] = set()
    out: list[Mutant] = []
    for name in operators or list(OPERATORS):
        pred, apply = OPERATORS[name]
        for i in range(len(_sites(tree, pred))):
            clone = copy.deepcopy(tree)
            apply(_sites(clone, pred)[i])
            ast.fix_missing_locations(clone)
            text = ast.unparse(clone)
            if text != original and text not in seen:
                seen.add(text)
                out.append(Mutant(text + "\n", name, i))
    return out


def mutate(source: str, n: int = 1, seed: int = 0,
           operators: Optional[list[str]] = None) -> list[Mutant]:
    """Pick ``n`` distinct mutants deterministically."""
    mutants = enumerate_mutants(source, operators)
    rng = random.Random(seed)
    return rng.sample(mutants, k=min(n, len(mutants)))
