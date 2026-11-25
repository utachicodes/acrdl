"""Evaluation metrics (pure numpy).

Labels follow :class:`acrdl.types.Label`: 0 CORRECT, 1 OVERFITTING, 2 AMBIGUOUS.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from acrdl.decision import false_accept_rate
from acrdl.types import Label

__all__ = [
    "abstention_precision_recall", "accuracy", "auroc", "auroc_ovr",
    "expected_calibration_error", "false_accept_rate", "macro_f1", "paired_bootstrap",
    "summarize",
]


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float((np.asarray(y) == np.asarray(pred)).mean()) if len(y) else 0.0


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int = 3) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    f1s = []
    for k in range(n_classes):
        tp = np.sum((pred == k) & (y == k))
        fp = np.sum((pred == k) & (y != k))
        fn = np.sum((pred != k) & (y == k))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def auroc(y_bin: np.ndarray, score: np.ndarray) -> float:
    """Binary AUROC via the Mann-Whitney U statistic (ties get half credit)."""
    y_bin, score = np.asarray(y_bin).astype(bool), np.asarray(score, dtype=float)
    pos, neg = score[y_bin], score[~y_bin]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def auroc_ovr(y: np.ndarray, probs: np.ndarray) -> float:
    """Macro one-vs-rest AUROC over classes present in ``y``."""
    y, probs = np.asarray(y), np.asarray(probs)
    scores = [auroc(y == k, probs[:, k]) for k in range(probs.shape[1])]
    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)) if scores else float("nan")


def abstention_precision_recall(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Precision/recall of the AMBIGUOUS (abstain) decision."""
    y, pred = np.asarray(y), np.asarray(pred)
    a = Label.AMBIGUOUS
    tp = np.sum((pred == a) & (y == a))
    prec = tp / max(1, np.sum(pred == a))
    rec = tp / max(1, np.sum(y == a))
    return float(prec), float(rec)


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    y, probs = np.asarray(y), np.asarray(probs, dtype=float)
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def summarize(y: np.ndarray, pred: np.ndarray, probs: np.ndarray | None = None
              ) -> dict[str, float]:
    prec, rec = abstention_precision_recall(y, pred)
    out = {
        "accuracy": accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
        "false_accept_rate": false_accept_rate(pred, y),
        "correct_recall": float(np.mean(np.asarray(pred)[np.asarray(y) == Label.CORRECT]
                                        == Label.CORRECT))
        if np.any(np.asarray(y) == Label.CORRECT) else float("nan"),
        "abstain_precision": prec,
        "abstain_recall": rec,
    }
    if probs is not None:
        out["auroc_ovr"] = auroc_ovr(y, probs)
        out["ece"] = expected_calibration_error(y, probs)
    return out


def paired_bootstrap(metric: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray,
                     pred_a: np.ndarray, pred_b: np.ndarray, n: int = 2000, seed: int = 0,
                     groups: np.ndarray | None = None) -> dict[str, float]:
    """Paired bootstrap of ``metric(a) - metric(b)``.

    If ``groups`` (e.g. problem ids) is given, whole groups are resampled so
    that patches of the same problem stay together.
    """
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    deltas = np.empty(n)
    for i in range(n):
        idx = np.concatenate([index[g] for g in rng.choice(uniq, size=len(uniq))])
        deltas[i] = metric(y[idx], pred_a[idx]) - metric(y[idx], pred_b[idx])
    observed = metric(y, pred_a) - metric(y, pred_b)
    return {
        "delta": float(observed),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_value": float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))),
    }
