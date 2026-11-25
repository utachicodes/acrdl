"""Semantic-entropy baseline (the ambiguity signal used by SpecFix and Code-MUE style work).

Candidates of a bug are grouped into behaviour clusters (identical outputs on all
executed inputs). The semantic entropy of the bug is the entropy of the cluster
sizes. SpecFix treats high entropy over sampled programs as a sign that the task
description is ambiguous. As a three-way rule:

* entropy >= tau: every candidate of the bug is AMBIGUOUS;
* otherwise: candidates in the largest cluster are CORRECT, the rest OVERFITTING.

``tau`` is tuned on the training split for macro-F1, so the baseline gets the
same data the learned models get.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from acrdl.training.dataset import Record
from acrdl.training.metrics import macro_f1
from acrdl.types import Label


def bug_entropy(record: Record) -> float:
    """Entropy (bits) of the behaviour-cluster sizes of the record's bug."""
    sizes = [row[0] for row in record.clusters if row and row[0] > 0]
    total = sum(sizes)
    return -sum(s / total * math.log2(s / total) for s in sizes) if total else 0.0


def _largest_cluster(record: Record) -> int:
    return int(np.argmax([row[0] for row in record.clusters])) if record.clusters else 0


def _rule(records: Sequence[Record], tau: float) -> np.ndarray:
    out = []
    for r in records:
        if bug_entropy(r) >= tau:
            out.append(int(Label.AMBIGUOUS))
        elif r.own_cluster == _largest_cluster(r):
            out.append(int(Label.CORRECT))
        else:
            out.append(int(Label.OVERFITTING))
    return np.array(out)


class SemanticEntropy:
    def __init__(self, grid: Sequence[float] = tuple(np.round(np.arange(0.05, 3.0, 0.05), 2))):
        self.grid = grid
        self.tau = float("inf")

    def fit(self, records: Sequence[Record]) -> SemanticEntropy:
        y = np.array([r.label for r in records])
        best = -1.0
        for tau in list(self.grid) + [float("inf")]:
            score = macro_f1(y, _rule(records, tau))
            if score > best + 1e-12:
                best, self.tau = score, float(tau)
        return self

    def predict_records(self, records: Sequence[Record]) -> np.ndarray:
        return _rule(records, self.tau)


def entropies_by_bug(records: Sequence[Record]) -> dict[str, float]:
    by_bug: dict[str, float] = defaultdict(float)
    for r in records:
        by_bug[r.bug_id] = bug_entropy(r)
    return dict(by_bug)
