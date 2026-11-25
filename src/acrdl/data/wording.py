"""Meaning-preserving rewordings of a bug report (the wording intervention).

Idea: if the specification determines the output on an input, a rewording that
cannot change the specification's meaning should not change what the generator
writes for that input. If the specification is silent there, the generator
guesses, and the guess may follow the wording. Sampling candidates under several
rewordings therefore separates "candidates disagree because some are wrong"
(noise within a wording) from "candidates disagree because the spec is silent"
(a dependence between behaviour and wording); see ``acrdl.evidence.wording``.

Each transformation is meaning-preserving by construction, and is also checked:

* ``w1`` reformat: the buggy implementation is re-printed from its AST, so the
  code is identical as a program (same AST) but looks different.
* ``w2`` rename: parameters of the entry function get synonymous names in the
  signature, body and docstring. Tests call the function positionally, so every
  test still applies.
* ``w3`` reorder: independent prose sentences of the docstring after the first
  appear in reverse order; sentences that refer back ("Otherwise", "If not") stay
  attached to the sentence before them, and examples stay in place.
"""

from __future__ import annotations

import ast
import copy
import io
import re
import tokenize
from typing import Optional

from acrdl.evidence.testing import run_tests
from acrdl.types import BuggyProgram, Problem

WORDINGS = ("w1", "w2", "w3")

SYNONYMS = {
    "numbers": "values", "number": "value", "nums": "values", "num": "value",
    "lst": "items", "list1": "items", "arr": "array", "array": "seq", "string": "text",
    "strings": "texts", "text": "string", "txt": "text", "word": "token", "words": "tokens",
    "sentence": "phrase", "values": "numbers", "value": "number", "items": "elements",
    "threshold": "limit", "delimeter": "separator", "delimiter": "separator",
    "operations": "steps", "paren_string": "brackets", "substring": "fragment",
    "prefix": "start", "data": "records", "test_tup1": "first_tuple",
    "test_tup2": "second_tuple", "test_tup": "tuple_in", "test_list": "input_list",
}

_BACKREF = re.compile(r"^(otherwise|else|then|it|this|that|these|those|if not|also|however|"
                      r"in that case|in this case|and|but|or)\b", re.IGNORECASE)


# ----------------------------------------------------------------- helpers

def _entry_params(source: str, entry: str) -> list[str]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            a = node.args
            return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    return []


def _rename_tokens(source: str, mapping: dict[str, str]) -> str:
    """Rename NAME tokens and whole words inside string tokens (docstrings)."""
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, int, int, str]] = []
    word = {k: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(k)}(?![A-Za-z0-9_])") for k in mapping}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.NAME and tok.string in mapping:
            edits.append((*tok.start, *tok.end, mapping[tok.string]))
        elif tok.type == tokenize.STRING:
            new = tok.string
            for k, v in mapping.items():
                if len(k) >= 3:
                    new = word[k].sub(v, new)
            if new != tok.string:
                edits.append((*tok.start, *tok.end, new))
    for (r0, c0, r1, c1, new) in sorted(edits, reverse=True):
        if r0 == r1:
            ln = lines[r0 - 1]
            lines[r0 - 1] = ln[:c0] + new + ln[c1:]
        else:  # multi-line string token
            head, tail = lines[r0 - 1][:c0], lines[r1 - 1][c1:]
            lines[r0 - 1:r1] = [head + new + tail]
    return "".join(lines)


def rename_params(source: str, entry: str, mapping: dict[str, str]) -> str:
    return _rename_tokens(source, mapping) if mapping else source


def synonym_mapping(params: list[str]) -> dict[str, str]:
    taken = set(params)
    out = {}
    for p in params:
        new = SYNONYMS.get(p)
        if new and new not in taken:
            out[p] = new
            taken.add(new)
    return out


def _docstring_bounds(source: str) -> Optional[tuple[int, int]]:
    m = re.search(r'("""|\'\'\')(.*?)(\1)', source, re.DOTALL)
    return (m.start(2), m.end(2)) if m else None


def reorder_docstring_text(doc: str) -> Optional[str]:
    """Reverse independent prose sentences after the first; None if nothing to do."""
    cut = len(doc)
    for marker in (">>>", "Example", "example", "For example", "Input:", "Examples"):
        i = doc.find(marker)
        if 0 <= i < cut:
            cut = i
    prose, rest = doc[:cut], doc[cut:]
    indent_m = re.search(r"\n([ \t]+)\S", doc)
    indent = indent_m.group(1) if indent_m else "    "
    flat = " ".join(prose.split())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", flat) if s]
    units: list[str] = []
    for s in sentences:
        if units and _BACKREF.match(s):
            units[-1] = units[-1] + " " + s
        else:
            units.append(s)
    if len(units) < 3:
        return None
    new_units = [units[0]] + units[1:][::-1]
    if new_units == units:
        return None
    lead = re.match(r"\s*", doc).group(0)
    body = ("\n" + indent).join(new_units)
    tail = "\n" + indent if rest.strip() else ("\n" + indent if doc.endswith("\n" + indent) else "")
    return f"{lead}{body}{tail}{rest.lstrip() if rest.strip() else ''}"


def _replace_docstring(source: str, old: str, new: str) -> str:
    return source.replace(old, new, 1) if old and old in source else source


# ----------------------------------------------------------------- variants

def make_wording(bug: BuggyProgram, which: str, timeout_s: float = 2.0
                 ) -> Optional[BuggyProgram]:
    """Return the reworded bug, or None if the rewording does not apply or fails checks."""
    p = bug.problem
    prompt, source, reference = p.prompt, bug.source, p.reference
    try:
        if which == "w1":
            tree = ast.parse(source)
            new_source = ast.unparse(tree) + "\n"
            if ast.dump(ast.parse(new_source)) != ast.dump(tree) or new_source == source:
                return None
            source = new_source
        elif which == "w2":
            mapping = synonym_mapping(_entry_params(source, p.entry_point))
            if not mapping:
                return None
            prompt = rename_params(prompt, p.entry_point, mapping)
            source = rename_params(source, p.entry_point, mapping)
            reference = rename_params(reference, p.entry_point, mapping)
        elif which == "w3":
            b = _docstring_bounds(prompt)
            if not b:
                return None
            old = prompt[b[0]:b[1]]
            new = reorder_docstring_text(old)
            if new is None:
                return None
            prompt = prompt[:b[0]] + new + prompt[b[1]:]
            source = _replace_docstring(source, old, new)
            reference = _replace_docstring(reference, old, new)
        else:
            raise ValueError(f"unknown wording {which!r}")
        for s in (prompt, source, reference):
            ast.parse(s)
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
        return None

    q = copy.deepcopy(p)
    q.prompt, q.reference = prompt, reference
    q.task_id = f"{p.task_id}~{which}"
    # meaning preserved: the reference must still pass every test, the bug must still show
    tests = q.visible_tests + q.hidden_tests
    if tests and not all(run_tests(reference, q.entry_point, tests, timeout_s)):
        return None
    if all(run_tests(source, q.entry_point, q.visible_tests, timeout_s)):
        return None
    return BuggyProgram(q, source, f"{bug.bug_id}~{which}", f"{bug.mutation}+{which}")


def wording_problem(problem: Problem) -> str:
    """Base task id of a reworded problem (``X~underspec~w2`` -> ``X~underspec``)."""
    parts = problem.task_id.split("~")
    return "~".join(x for x in parts if x not in WORDINGS)
