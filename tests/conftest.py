from __future__ import annotations

import pytest

from acrdl.cli import toy_bugs
from acrdl.config import Config


@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.evidence.n_fuzz_inputs = 60
    c.evidence.exec_timeout_s = 1.0
    return c


@pytest.fixture
def average_bug():
    return next(b for b in toy_bugs(True) if b.problem.task_id.startswith("toy/average"))
