"""PATCH-SIM-style baseline (after Xiong et al., ICSE 2018).

Reject a patch whose behaviour on inputs where the bug does not manifest
(stable inputs) drifts from the original program. Two-way only: it never
abstains.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from acrdl.evidence.features import EVIDENCE_NAMES
from acrdl.training.dataset import Record
from acrdl.types import Label

_PRES = EVIDENCE_NAMES.index("preservation")


def predict_records(records: Sequence[Record], threshold: float = 0.95) -> np.ndarray:
    return np.array([int(Label.CORRECT) if r.evidence[_PRES] >= threshold
                     else int(Label.OVERFITTING) for r in records])
