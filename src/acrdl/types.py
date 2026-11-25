"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class Label(IntEnum):
    """Three-way verdict for a plausible patch."""

    CORRECT = 0
    OVERFITTING = 1
    AMBIGUOUS = 2


@dataclass(frozen=True)
class TestCase:
    """A single input/expected-output pair. ``args`` is a tuple of positional args."""

    args: tuple
    expected: Any

    __test__ = False  # stop pytest from collecting this class


@dataclass
class Problem:
    """A programming task: specification plus reference behaviour."""

    task_id: str
    entry_point: str
    prompt: str  # signature + docstring (the natural-language spec)
    reference: str  # full reference solution source
    visible_tests: list[TestCase] = field(default_factory=list)
    hidden_tests: list[TestCase] = field(default_factory=list)
    ambiguous_inputs: list[tuple] = field(default_factory=list)


@dataclass
class BuggyProgram:
    """A buggy version of a problem's reference solution."""

    problem: Problem
    source: str
    bug_id: str = ""
    mutation: str = ""


@dataclass
class Patch:
    """A candidate repair of a buggy program."""

    patch_id: str
    bug: BuggyProgram
    source: str
    plausible: Optional[bool] = None


@dataclass(frozen=True)
class ExecResult:
    """Outcome of running a function on one input.

    ``kind`` is ``"ok"`` (``value`` holds ``repr`` of the return value),
    ``"error"`` (``value`` holds the exception type name) or ``"timeout"``.
    """

    kind: str
    value: str = ""

    @property
    def signature(self) -> str:
        """Hashable behaviour signature used for equivalence comparisons."""
        return f"{self.kind}:{self.value}"


@dataclass
class Evidence:
    """All evidence collected for one patch."""

    patch_id: str
    vector: list[float]
    names: list[str]
    cluster_id: int = -1
    cluster_features: list[list[float]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.vector))
