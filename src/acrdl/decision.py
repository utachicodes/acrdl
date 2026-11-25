"""Turn calibrated class probabilities into review decisions.

Decision rule for one plausible patch with probabilities
``p = (P(correct), P(overfitting), P(ambiguous))``:

1. **ABSTAIN** (report AMBIGUOUS) when ``P(ambiguous) >= amb_threshold``, or,
   if no such threshold was fitted, when ``P(ambiguous)`` is the largest.
2. **ACCEPT** (report CORRECT) when ``P(correct) >= threshold``.
3. **REJECT** (report OVERFITTING) otherwise.

The accept threshold is fitted on validation data as the smallest value whose
false-accept rate (overfitting patches accepted) stays at or below the target
(5% by default). The abstention threshold is fitted on validation data to
maximise F1 on the ambiguous class (rare classes rarely win an argmax).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from acrdl.types import Label


def decide(probs: np.ndarray, threshold: float,
           amb_threshold: Optional[float] = None) -> np.ndarray:
    """Vectorised decision rule; ``probs`` has shape (N, 3)."""
    probs = np.asarray(probs, dtype=float)
    out = np.full(len(probs), int(Label.OVERFITTING))
    out[probs[:, Label.CORRECT] >= threshold] = int(Label.CORRECT)
    if amb_threshold is None:
        abstain = probs.argmax(1) == Label.AMBIGUOUS
    else:
        abstain = probs[:, Label.AMBIGUOUS] >= amb_threshold
    out[abstain] = int(Label.AMBIGUOUS)
    return out


def false_accept_rate(pred: np.ndarray, labels: np.ndarray) -> float:
    """Share of truly overfitting patches that were accepted as correct."""
    bad = np.asarray(labels) == Label.OVERFITTING
    if not bad.any():
        return 0.0
    return float((np.asarray(pred)[bad] == Label.CORRECT).mean())


def fit_ambiguous_threshold(probs: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Threshold on P(ambiguous) that maximises validation F1 of the ambiguous class.

    Returns ``None`` (fall back to argmax) when validation has no ambiguous patch.
    """
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels)
    pos = labels == Label.AMBIGUOUS
    if not pos.any():
        return None
    best_t, best_f1 = None, -1.0
    for t in np.unique(probs[:, Label.AMBIGUOUS]):
        pred = probs[:, Label.AMBIGUOUS] >= t
        tp = float((pred & pos).sum())
        f1 = 2 * tp / max(float(pred.sum() + pos.sum()), 1.0)
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return best_t


def fit_accept_threshold(probs: np.ndarray, labels: np.ndarray,
                         target_far: float = 0.05,
                         amb_threshold: Optional[float] = None) -> float:
    """Smallest threshold on P(correct) whose validation FAR <= ``target_far``."""
    probs = np.asarray(probs, dtype=float)
    candidates = np.unique(np.concatenate([probs[:, Label.CORRECT], [1.0 + 1e-9]]))
    for t in candidates:
        if false_accept_rate(decide(probs, t, amb_threshold), labels) <= target_far:
            return float(t)
    return float(candidates[-1])
