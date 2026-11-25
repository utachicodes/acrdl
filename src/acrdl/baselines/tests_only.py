"""Status quo: accept every plausible patch."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from acrdl.training.dataset import Record
from acrdl.types import Label


def predict_records(records: Sequence[Record]) -> np.ndarray:
    return np.full(len(records), int(Label.CORRECT))
