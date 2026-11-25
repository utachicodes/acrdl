"""Optional symbolic divergence check via CrossHair ``diffbehavior``.

CrossHair (``analysis`` extra) searches for an input on which two functions
behave differently. It works best on type-annotated code; for unannotated
functions it usually returns ``UNKNOWN``. Both functions run in a separate
CrossHair subprocess, never in-process.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path


class SymbolicVerdict(str, Enum):
    DIFF = "diff"
    NO_DIFF_FOUND = "no_diff_found"
    UNKNOWN = "unknown"

    def as_feature(self) -> float:
        return {"diff": 1.0, "no_diff_found": 0.0, "unknown": 0.5}[self.value]


def crosshair_available() -> bool:
    return importlib.util.find_spec("crosshair") is not None


def diff_behavior(source_a: str, source_b: str, entry_point: str,
                  timeout_s: float = 10.0) -> SymbolicVerdict:
    """Search for an input where ``source_a`` and ``source_b`` disagree."""
    if not crosshair_available():
        return SymbolicVerdict.UNKNOWN
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "acr_sym_a.py").write_text(source_a)
        Path(tmp, "acr_sym_b.py").write_text(source_b)
        cmd = [sys.executable, "-m", "crosshair", "diffbehavior",
               f"acr_sym_a.{entry_point}", f"acr_sym_b.{entry_point}",
               "--per_path_timeout", "2"]
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return SymbolicVerdict.UNKNOWN
    out = proc.stdout + proc.stderr
    if "Given:" in out or "differ" in out.lower():
        return SymbolicVerdict.DIFF
    if proc.returncode == 0:
        return SymbolicVerdict.NO_DIFF_FOUND
    return SymbolicVerdict.UNKNOWN
