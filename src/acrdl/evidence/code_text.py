"""Text given to the code encoder (dependency-free)."""

from __future__ import annotations

import difflib


def code_text(buggy: str, patch: str, spec: str = "") -> str:
    """Spec, buggy function and unified diff, separated by marker lines."""
    diff = "\n".join(difflib.unified_diff(buggy.splitlines(), patch.splitlines(),
                                          lineterm="", n=1))
    return f"{spec.strip()}\n<buggy>\n{buggy.strip()}\n<diff>\n{diff}"
