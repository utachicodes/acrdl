"""Rebuild full patch sources from training records.

A record stores the encoder text: spec, buggy program and a unified diff
(``acrdl.evidence.code_text``). Together with the buggy source from the
prepared bug files, the diff gives back the exact patch, which baselines such
as the LLM judge need.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Union

from acrdl.types import BuggyProgram

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_unified_diff(source: str, diff: str) -> str:
    """Apply a ``difflib.unified_diff`` (any context size) to ``source``."""
    old = source.splitlines()
    out: list[str] = []
    i = 0  # next unconsumed line of ``old`` (0-based)
    for line in diff.splitlines():
        if line.startswith(("---", "+++")):
            continue
        m = _HUNK.match(line)
        if m:
            start = int(m.group(1)) - (0 if m.group(2) == "0" else 1)
            out.extend(old[i:start])
            i = start
            continue
        if line.startswith("-"):
            i += 1
        elif line.startswith("+"):
            out.append(line[1:])
        elif line.startswith(" ") or line == "":
            out.append(old[i] if i < len(old) else "")
            i += 1
    out.extend(old[i:])
    return "\n".join(out) + "\n"


def diff_of(code_text: str) -> str:
    return code_text.split("\n<diff>\n", 1)[1] if "\n<diff>\n" in code_text else ""


def base_bug_id(bug_id: str) -> str:
    """Bug id without a sampling-pass tag such as ``@t1``."""
    return bug_id.split("@", 1)[0]


def load_bug_index(files: Iterable[Union[str, Path]]) -> dict[str, BuggyProgram]:
    from acrdl.data.prepare import load_prepared

    return {b.bug_id: b for f in files for b in load_prepared(f)}


def patch_source(record_code: str, bug_id: str, bugs: Mapping[str, BuggyProgram]) -> str:
    """Exact patch source for a record (raises ``KeyError`` for an unknown bug)."""
    bug = bugs[base_bug_id(bug_id)]
    return apply_unified_diff(bug.source, diff_of(record_code))
