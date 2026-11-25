"""Static-feature gradient-boosted trees (ODS-style; needs scikit-learn)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from acrdl.evidence.features import EVIDENCE_NAMES
from acrdl.training.dataset import Record

STATIC_FEATURES = [
    "diff_lines", "node_delta", "hardcoded_test_literal", "special_case_eq",
    "removed_guards", "added_broad_except", "added_return_const", "added_branches",
    "parse_error",
]
_IDX = [EVIDENCE_NAMES.index(n) for n in STATIC_FEATURES]


def _x(records: Sequence[Record]) -> np.ndarray:
    return np.array([[r.evidence[i] for i in _IDX] for r in records])


class StaticGBDT:
    def __init__(self, seed: int = 0, **kwargs: Any):
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError as e:  # pragma: no cover - depends on extras
            raise ImportError("StaticGBDT needs the 'ml' extra (scikit-learn)") from e
        self.clf = GradientBoostingClassifier(random_state=seed, **kwargs)

    def fit(self, records: Sequence[Record]) -> StaticGBDT:
        self.clf.fit(_x(records), [r.label for r in records])
        return self

    def predict_records(self, records: Sequence[Record]) -> np.ndarray:
        return self.clf.predict(_x(records))
