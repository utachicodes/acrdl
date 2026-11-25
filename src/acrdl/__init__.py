"""Automated Code Review and Program Repair System Using Deep Learning.

Ambiguity-aware assessment of LLM-generated patches: every plausible patch is
classified as CORRECT, OVERFITTING or AMBIGUOUS (the specification does not
determine a single correct behaviour).
"""

from acrdl.types import Evidence, ExecResult, Label, Patch, Problem

__all__ = ["Evidence", "ExecResult", "Label", "Patch", "Problem"]
__version__ = "0.1.0"
