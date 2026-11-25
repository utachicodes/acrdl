"""Controlled underspecification.

To obtain ground-truth AMBIGUOUS labels we take a fully specified problem,
delete one clause of its docstring together with every test that exercises
that clause, and record the affected inputs as ``ambiguous_inputs``. On those
inputs the reduced specification no longer determines a single behaviour.
"""

from __future__ import annotations

import copy
import re
from typing import Callable

from acrdl.types import Problem

# Sentences that typically pin down edge-case behaviour.
_EDGE_HINTS = re.compile(
    r"\b(empty|none|null|zero|negative|if|otherwise|unless|when|raise|error|default)\b",
    re.IGNORECASE,
)


def candidate_clauses(prompt: str) -> list[str]:
    """Docstring sentences that look like edge-case clauses (removal candidates)."""
    m = re.search(r'"""(.*?)"""', prompt, re.DOTALL) or re.search(r"'''(.*?)'''", prompt,
                                                                    re.DOTALL)
    if not m:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", m.group(1).strip())
    return [s.strip() for s in sentences if s.strip() and _EDGE_HINTS.search(s)]


def underspecify(problem: Problem, clause: str, covers: Callable[[tuple], bool]) -> Problem:
    """Return a copy of ``problem`` with ``clause`` and its covering tests removed.

    ``covers(args)`` decides whether a test input exercises the removed clause.
    """
    if clause not in problem.prompt:
        raise ValueError(f"Clause not found in prompt: {clause!r}")
    p = copy.deepcopy(problem)
    p.prompt = re.sub(r"[ \t]*" + re.escape(clause), "", p.prompt, count=1)
    p.reference = p.reference.replace(clause, "", 1)
    removed: list[tuple] = []
    for attr in ("visible_tests", "hidden_tests"):
        kept = []
        for t in getattr(p, attr):
            if covers(t.args):
                if t.args not in removed:
                    removed.append(t.args)
            else:
                kept.append(t)
        setattr(p, attr, kept)
    if not removed:
        raise ValueError("No test exercises the removed clause; nothing became ambiguous")
    p.ambiguous_inputs = removed
    p.task_id = f"{problem.task_id}~underspec"
    return p
