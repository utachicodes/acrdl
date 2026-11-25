"""Differential execution of the buggy program and all candidate patches.

The behaviour matrix records, for every input, the behaviour signature of the
buggy program and of each candidate. From it we derive reference-free signals:

* **consensus** - per input, the majority behaviour among candidates;
* **stable inputs** - inputs where the buggy program already agrees with the
  consensus, i.e. where the bug does not manifest;
* **preservation** - how often a patch keeps the buggy program's behaviour on
  stable inputs (a correct fix should change behaviour only where the bug bites).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from acrdl.evidence.sandbox import run_function


@dataclass
class BehaviourMatrix:
    inputs: list[tuple]
    buggy: list[str]
    candidates: dict[str, list[str]]

    @property
    def patch_ids(self) -> list[str]:
        return list(self.candidates)


def build_matrix(buggy_source: str, entry_point: str, candidates: Mapping[str, str],
                 inputs: Sequence[tuple], timeout_s: float = 2.0,
                 memory_limit_mb: int = 512) -> BehaviourMatrix:
    def sigs(src: str) -> list[str]:
        return [r.signature for r in run_function(src, entry_point, inputs, timeout_s,
                                                    memory_limit_mb)]

    return BehaviourMatrix(
        inputs=list(inputs),
        buggy=sigs(buggy_source),
        candidates={pid: sigs(src) for pid, src in candidates.items()},
    )


def consensus(matrix: BehaviourMatrix) -> list[Optional[str]]:
    """Majority signature per input; ``None`` on ties or with no candidates."""
    out: list[Optional[str]] = []
    for j in range(len(matrix.inputs)):
        counts = Counter(row[j] for row in matrix.candidates.values()).most_common(2)
        if not counts or (len(counts) == 2 and counts[0][1] == counts[1][1]):
            out.append(None)
        else:
            out.append(counts[0][0])
    return out


def stable_inputs(matrix: BehaviourMatrix, cons: Optional[list[Optional[str]]] = None
                  ) -> list[int]:
    cons = cons if cons is not None else consensus(matrix)
    return [j for j, (b, c) in enumerate(zip(matrix.buggy, cons)) if c is not None and b == c]


def preservation_score(matrix: BehaviourMatrix, pid: str,
                       stable: Optional[list[int]] = None) -> tuple[float, int]:
    """(fraction of stable inputs where the patch matches the buggy program, #stable)."""
    stable = stable if stable is not None else stable_inputs(matrix)
    if not stable:
        return 1.0, 0
    row = matrix.candidates[pid]
    same = sum(row[j] == matrix.buggy[j] for j in stable)
    return same / len(stable), len(stable)


def consensus_agreement(matrix: BehaviourMatrix, pid: str,
                        cons: Optional[list[Optional[str]]] = None) -> float:
    cons = cons if cons is not None else consensus(matrix)
    idx = [j for j, c in enumerate(cons) if c is not None]
    if not idx:
        return 1.0
    row = matrix.candidates[pid]
    return sum(row[j] == cons[j] for j in idx) / len(idx)


def changed_fraction(matrix: BehaviourMatrix, pid: str) -> float:
    row = matrix.candidates[pid]
    return sum(a != b for a, b in zip(row, matrix.buggy)) / max(1, len(row))


def outcome_rates(matrix: BehaviourMatrix, pid: str) -> tuple[float, float]:
    """(error rate, timeout rate) of a candidate over all inputs."""
    row = matrix.candidates[pid]
    n = max(1, len(row))
    return (sum(s.startswith("error:") for s in row) / n,
            sum(s.startswith("timeout:") for s in row) / n)
