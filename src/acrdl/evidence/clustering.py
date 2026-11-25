"""Behavioural equivalence classes over candidate patches.

Two candidates are equivalent if they produce identical behaviour signatures
on every considered input. Multiple clusters mean the candidates *disagree*;
whether that disagreement reflects wrong patches or an ambiguous
specification is exactly what the downstream model has to decide.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from acrdl.evidence.differential import BehaviourMatrix


@dataclass
class Clustering:
    clusters: list[list[str]]
    assignment: dict[str, int]
    disagreement_inputs: list[int] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def entropy(self) -> float:
        """Normalised Shannon entropy of cluster sizes (0 = full agreement)."""
        n = sum(len(c) for c in self.clusters)
        if n <= 1 or len(self.clusters) <= 1:
            return 0.0
        h = -sum((len(c) / n) * math.log(len(c) / n) for c in self.clusters)
        return h / math.log(n)


def cluster(matrix: BehaviourMatrix, patch_ids: Optional[Sequence[str]] = None,
            input_idx: Optional[Sequence[int]] = None) -> Clustering:
    """Group candidates by exact behaviour on the selected inputs.

    Clusters are ordered by size (largest first) and then by first member.
    """
    pids = list(patch_ids) if patch_ids is not None else matrix.patch_ids
    idx = list(input_idx) if input_idx is not None else list(range(len(matrix.inputs)))
    groups: dict[tuple, list[str]] = {}
    for pid in pids:
        key = tuple(matrix.candidates[pid][j] for j in idx)
        groups.setdefault(key, []).append(pid)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), pids.index(g[0])))
    assignment = {pid: i for i, g in enumerate(ordered) for pid in g}
    disagreement = [j for j in idx
                    if len({matrix.candidates[p][j] for p in pids}) > 1] if pids else []
    return Clustering(ordered, assignment, disagreement)
