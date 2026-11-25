"""Transparent disagreement rule (also the pipeline's fallback predictor).

* OVERFITTING if the patch disagrees with the candidate consensus on more
  than ``tau`` of inputs, or has a test-literal smell and any disagreement.
* AMBIGUOUS if another non-overfitting candidate of the same bug behaves
  differently (they fall in different behavioural clusters).
* CORRECT otherwise.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from acrdl.evidence.features import EVIDENCE_NAMES
from acrdl.training.dataset import Record
from acrdl.types import Evidence, Label, Patch

_AGREE = EVIDENCE_NAMES.index("consensus_agreement")
_SMELL = EVIDENCE_NAMES.index("hardcoded_test_literal")


def _overfit(vec: Sequence[float], tau: float) -> bool:
    agree = vec[_AGREE]
    return (1 - agree) > tau or (vec[_SMELL] > 0 and agree < 1.0)


def _rule(items: list[tuple[str, Sequence[float], int]], tau: float) -> dict[str, Label]:
    over = {pid for pid, vec, _ in items if _overfit(vec, tau)}
    clusters = {cid for pid, _, cid in items if pid not in over}
    return {pid: Label.OVERFITTING if pid in over
            else Label.AMBIGUOUS if len(clusters) >= 2 else Label.CORRECT
            for pid, _, _ in items}


class HeuristicPredictor:
    def __init__(self, tau: float = 0.05):
        self.tau = tau

    def predict(self, patches: list[Patch], evidence: Mapping[str, Evidence]
                ) -> dict[str, tuple[Label, list[float]]]:
        items = [(p.patch_id, evidence[p.patch_id].vector, evidence[p.patch_id].cluster_id)
                 for p in patches]
        out = {}
        for pid, lab in _rule(items, self.tau).items():
            probs = [0.0, 0.0, 0.0]
            probs[int(lab)] = 1.0
            out[pid] = (lab, probs)
        return out


def predict_records(records: Sequence[Record], tau: float = 0.05) -> np.ndarray:
    by_bug: dict[str, list[tuple[str, Sequence[float], int]]] = defaultdict(list)
    for r in records:
        by_bug[r.bug_id].append((r.patch_id, r.evidence, r.own_cluster))
    labels: dict[str, Label] = {}
    for items in by_bug.values():
        labels.update(_rule(items, tau))
    return np.array([int(labels[r.patch_id]) for r in records])
