"""Prepare the real benchmarks for patch generation.

This module does every step that does not need the LLM:

1. HumanEvalFix bugs, with visible tests from HumanEval and hidden tests from
   HumanEval+ (EvalPlus). Hidden expected values come from EvalPlus and are
   kept only where the reference solution reproduces them in our sandbox.
2. MBPP+ problems with bugs injected by AST mutation. Visible tests come from
   the original MBPP asserts, hidden tests from MBPP+.
3. Ambiguous variants by controlled underspecification. We delete one
   edge-case guard (an ``if`` whose body is a single ``return`` or ``raise``)
   from a correct solution. The inputs whose behaviour changes are the inputs
   that guard governed. We remove the docstring sentences and examples that
   describe them and every test that covers them, and keep a crash on one of
   them as the bug report. On those inputs the reduced specification no
   longer decides the behaviour, so ``ambiguous_inputs`` is exact.

Everything is written as JSON Lines so later stages can reload it without
network access (:func:`load_prepared`).
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from acrdl.data.loaders import _parse_assert_tests
from acrdl.data.mutate import mutate
from acrdl.evidence.sandbox import run_function
from acrdl.evidence.testing import NO_CRASH, result_matches, run_tests
from acrdl.types import BuggyProgram, Problem, TestCase

NO_CRASH_MARK = "__NO_CRASH__"


# --------------------------------------------------------------------------
# EvalPlus test parsing
# --------------------------------------------------------------------------

class _SpecialFloats(ast.NodeTransformer):
    """Turn ``inf``, ``nan`` and ``float('inf')`` into constants for literal_eval."""

    _NAMES = {"inf": math.inf, "nan": math.nan}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._NAMES:
            return ast.copy_location(ast.Constant(self._NAMES[node.id]), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == "float" and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            try:
                return ast.copy_location(ast.Constant(float(node.args[0].value)), node)
            except ValueError:
                return node
        return node


def _literal(node: ast.AST) -> Any:
    return ast.literal_eval(_SpecialFloats().visit(node))


def parse_evalplus_tests(test_src: str) -> tuple[list[tuple], list[Any]]:
    """Extract the ``inputs`` and ``results`` lists from an EvalPlus test script."""
    found: dict[str, Any] = {}
    for node in ast.walk(ast.parse(test_src)):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("inputs", "results")
                and node.targets[0].id not in found):
            try:
                found[node.targets[0].id] = _literal(node.value)
            except (ValueError, SyntaxError, TypeError):
                found[node.targets[0].id] = None
    inputs, results = found.get("inputs"), found.get("results")
    if not isinstance(inputs, list) or not isinstance(results, list):
        return [], []
    n = min(len(inputs), len(results))
    return [tuple(a) if isinstance(a, (list, tuple)) else (a,) for a in inputs[:n]], results[:n]


def _roundtrips(value: Any) -> bool:
    """True if ``repr`` -> ``literal_eval`` reproduces the value (sandbox requirement)."""
    try:
        back = ast.literal_eval(repr(value))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False
    try:
        return back == value
    except Exception:  # noqa: BLE001
        return False


def _usable(t: TestCase, max_repr: int = 4000) -> bool:
    return (_roundtrips(t.args) and _roundtrips(t.expected)
            and len(repr(t.args)) <= max_repr and len(repr(t.expected)) <= 1500)


def _keep_reproduced(reference: str, entry: str, tests: list[TestCase],
                     timeout_s: float) -> list[TestCase]:
    """Keep only tests whose expected value the reference reproduces in the sandbox."""
    if not tests:
        return []
    ok = run_tests(reference, entry, tests, timeout_s)
    return [t for t, good in zip(tests, ok) if good]


def _sample(tests: list[TestCase], k: int, seed: int) -> list[TestCase]:
    if len(tests) <= k:
        return tests
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(tests)), k))
    return [tests[i] for i in idx]


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def _enc_tests(tests: list[TestCase]) -> list[list[str]]:
    return [[repr(tuple(t.args)), NO_CRASH_MARK if t.expected is NO_CRASH else repr(t.expected)]
            for t in tests]


def _dec_tests(rows: list[list[str]]) -> list[TestCase]:
    return [TestCase(ast.literal_eval(a), NO_CRASH if e == NO_CRASH_MARK else ast.literal_eval(e))
            for a, e in rows]


def bug_to_json(bug: BuggyProgram, meta: Optional[dict] = None) -> dict:
    p = bug.problem
    return {
        "bug_id": bug.bug_id, "mutation": bug.mutation, "source": bug.source,
        "problem": {
            "task_id": p.task_id, "entry_point": p.entry_point, "prompt": p.prompt,
            "reference": p.reference, "visible_tests": _enc_tests(p.visible_tests),
            "hidden_tests": _enc_tests(p.hidden_tests),
            "ambiguous_inputs": [repr(tuple(a)) for a in p.ambiguous_inputs],
        },
        "meta": meta or {},
    }


def bug_from_json(d: dict) -> BuggyProgram:
    q = d["problem"]
    problem = Problem(
        task_id=q["task_id"], entry_point=q["entry_point"], prompt=q["prompt"],
        reference=q["reference"], visible_tests=_dec_tests(q["visible_tests"]),
        hidden_tests=_dec_tests(q["hidden_tests"]),
        ambiguous_inputs=[ast.literal_eval(a) for a in q["ambiguous_inputs"]],
    )
    return BuggyProgram(problem, d["source"], d["bug_id"], d["mutation"])


def save_bugs(rows: list[dict], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_prepared(path: Union[str, Path], limit: Optional[int] = None) -> list[BuggyProgram]:
    """Reload bugs written by :func:`prepare_all`."""
    out = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                out.append(bug_from_json(json.loads(line)))
                if limit and len(out) >= limit:
                    break
    return out


# --------------------------------------------------------------------------
# HumanEvalFix + HumanEval+
# --------------------------------------------------------------------------

@dataclass
class PrepStats:
    seen: int = 0
    kept: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    visible_tests: int = 0
    hidden_tests: int = 0

    def skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def as_dict(self) -> dict:
        return {"seen": self.seen, "kept": self.kept, "skipped": self.skipped,
                "avg_visible_tests": round(self.visible_tests / max(self.kept, 1), 1),
                "avg_hidden_tests": round(self.hidden_tests / max(self.kept, 1), 1)}


def _datasets() -> Any:
    try:
        import datasets  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - depends on extras
        raise ImportError("Preparing data needs the 'data' extra: pip install -e '.[data]'") from e
    return datasets


def prepare_humanevalfix(cache_dir: Optional[str] = None, max_hidden: int = 1000,
                         timeout_s: float = 2.0, seed: int = 13,
                         limit: Optional[int] = None) -> tuple[list[BuggyProgram], PrepStats]:
    ds = _datasets()
    hep = ds.load_dataset("bigcode/humanevalpack", "python", split="test", cache_dir=cache_dir)
    plus = ds.load_dataset("evalplus/humanevalplus", split="test", cache_dir=cache_dir)
    plus_by_num = {r["task_id"].split("/")[-1]: r for r in plus}
    stats, bugs = PrepStats(), []
    for row in hep:
        if limit and stats.seen >= limit:
            break
        stats.seen += 1
        num = row["task_id"].split("/")[-1]
        entry = row["entry_point"]
        reference = row["prompt"] + row["canonical_solution"]
        buggy = row["prompt"] + row["buggy_solution"]
        visible = [t for t in _parse_assert_tests(row["test"], entry) if _usable(t)]
        if not visible:
            stats.skip("no parseable visible tests")
            continue
        if not all(run_tests(reference, entry, visible, timeout_s)):
            stats.skip("reference fails visible tests")
            continue
        if all(run_tests(buggy, entry, visible, timeout_s)):
            stats.skip("buggy passes all visible tests")
            continue
        hidden: list[TestCase] = []
        if num in plus_by_num:
            ins, outs = parse_evalplus_tests(plus_by_num[num]["test"])
            hidden = [t for t in (TestCase(a, r) for a, r in zip(ins, outs)) if _usable(t)]
            hidden = _keep_reproduced(reference, entry, _sample(hidden, max_hidden, seed),
                                      timeout_s)
        if len(hidden) < len(visible):
            stats.skip("too few usable hidden tests")
            continue
        problem = Problem(task_id=f"HumanEval/{num}", entry_point=entry, prompt=row["prompt"],
                          reference=reference, visible_tests=visible, hidden_tests=hidden)
        bugs.append(BuggyProgram(problem, buggy, f"HumanEval/{num}#hef", row["bug_type"]))
        stats.kept += 1
        stats.visible_tests += len(visible)
        stats.hidden_tests += len(hidden)
    return bugs, stats


# --------------------------------------------------------------------------
# MBPP+ with injected bugs
# --------------------------------------------------------------------------

def _defined_functions(source: str) -> list[ast.FunctionDef]:
    return [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]


def _mbpp_entry(code: str, test_list: list[str]) -> Optional[str]:
    names = [f.name for f in _defined_functions(code)]
    joined = "\n".join(test_list)
    called = [n for n in names if re.search(rf"\b{re.escape(n)}\(", joined)]
    return called[-1] if called else (names[-1] if names else None)


def _mbpp_prompt(code: str, entry: str, text: str) -> str:
    fn = next(f for f in _defined_functions(code) if f.name == entry)
    header = ast.get_source_segment(code, fn).splitlines()[0] if fn else f"def {entry}(...):"
    return f'{header}\n    """{text.strip()}\n    """\n'


def prepare_mbpp(cache_dir: Optional[str] = None, mutants_per_problem: int = 2,
                 max_hidden: int = 1000, timeout_s: float = 2.0, seed: int = 13,
                 limit: Optional[int] = None
                 ) -> tuple[list[BuggyProgram], list[Problem], PrepStats]:
    ds = _datasets()
    mb = ds.load_dataset("evalplus/mbppplus", split="test", cache_dir=cache_dir)
    stats, bugs, problems = PrepStats(), [], []
    for row in mb:
        if limit and stats.seen >= limit:
            break
        stats.seen += 1
        code = row["code"].strip() + "\n"
        try:
            test_list = ast.literal_eval(row["test_list"]) if isinstance(row["test_list"], str) \
                else list(row["test_list"])
            entry = _mbpp_entry(code, test_list)
        except (ValueError, SyntaxError):
            stats.skip("unparseable source or tests")
            continue
        if entry is None:
            stats.skip("no function found")
            continue
        ins, outs = parse_evalplus_tests(row["test"])
        hidden = [t for t in (TestCase(a, r) for a, r in zip(ins, outs)) if _usable(t)]
        hidden = _keep_reproduced(code, entry, _sample(hidden, max_hidden, seed), timeout_s)
        visible = [t for t in _parse_assert_tests("\n".join(test_list), entry) if _usable(t)]
        visible = _keep_reproduced(code, entry, visible, timeout_s) or hidden[:3]
        if len(hidden) < 3:
            stats.skip("too few usable hidden tests")
            continue
        problem = Problem(task_id=f"Mbpp/{row['task_id']}", entry_point=entry,
                          prompt=_mbpp_prompt(code, entry, row["prompt"]), reference=code,
                          visible_tests=visible, hidden_tests=hidden)
        problems.append(problem)
        made = 0
        for i, m in enumerate(mutate(code, n=mutants_per_problem * 4, seed=seed)):
            if made >= mutants_per_problem:
                break
            if all(run_tests(m.source, entry, visible, timeout_s)):
                continue  # the visible tests do not reveal this mutant
            bugs.append(BuggyProgram(problem, m.source, f"{problem.task_id}+m{i}", m.operator))
            made += 1
        if made == 0:
            stats.skip("no mutant fails the visible tests")
            continue
        stats.kept += 1
        stats.visible_tests += len(visible)
        stats.hidden_tests += len(hidden)
    return bugs, problems, stats


# --------------------------------------------------------------------------
# QuixBugs (out-of-distribution test set)
# --------------------------------------------------------------------------

def _module_doc(source: str) -> str:
    """QuixBugs keeps its spec in a module-level string after the function."""
    for node in reversed(ast.parse(source).body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            return node.value.value.strip()
    return ""


def prepare_quixbugs(root: Union[str, Path], max_hidden: int = 1000, timeout_s: float = 2.0,
                     seed: int = 13) -> tuple[list[BuggyProgram], PrepStats]:
    """QuixBugs Python bugs with a real docstring spec and one failing visible test."""
    from acrdl.data.loaders import load_quixbugs

    stats, bugs = PrepStats(), []
    for raw in load_quixbugs(root):
        stats.seen += 1
        p = raw.problem
        try:
            doc = _module_doc(raw.source)
            fn = next(f for f in _defined_functions(raw.source) if f.name == p.entry_point)
        except (SyntaxError, StopIteration):
            stats.skip("unparseable program")
            continue
        header = raw.source.splitlines()[fn.lineno - 1]
        body = "\n".join("    " + ln if ln.strip() else "" for ln in doc.splitlines())
        prompt = f'{header}\n    """\n{body}\n    """\n'
        tests = [t for t in p.hidden_tests if _usable(t)]
        tests = _keep_reproduced(p.reference, p.entry_point, _sample(tests, max_hidden, seed),
                                 timeout_s)
        if len(tests) < 3:
            stats.skip("too few usable tests")
            continue
        buggy_ok = run_tests(raw.source, p.entry_point, tests, timeout_s)
        failing = [t for t, ok in zip(tests, buggy_ok) if not ok]
        if not failing:
            stats.skip("buggy passes every test")
            continue
        visible = [failing[0]] + [t for t in tests if t != failing[0]][:2]
        problem = Problem(task_id=p.task_id, entry_point=p.entry_point, prompt=prompt,
                          reference=p.reference, visible_tests=visible, hidden_tests=tests)
        bugs.append(BuggyProgram(problem, raw.source, raw.bug_id, "quixbugs"))
        stats.kept += 1
        stats.visible_tests += len(visible)
        stats.hidden_tests += len(tests)
    return bugs, stats


# --------------------------------------------------------------------------
# Controlled underspecification by guard removal
# --------------------------------------------------------------------------

_EDGE_WORDS = re.compile(
    r"\b(empty|none|null|zero|0|negative|positive|nothing|no\s+\w+|if|otherwise|unless|"
    r"return\s+(?:none|-1|0|false|true|an?\s+empty)|raise|error|invalid|less than|greater than)\b",
    re.IGNORECASE,
)


@dataclass
class Variant:
    bug: BuggyProgram
    guard: str
    removed_sentences: list[str]
    removed_examples: list[str]
    crash: str
    needs_review: bool


def find_guards(source: str, entry: str) -> list[ast.If]:
    """``if`` statements in ``entry`` whose body is a single return or raise.

    Guards anywhere in the function qualify (top level first). A guard may have
    an ``else``/``elif`` branch; removing it keeps that branch.
    """
    for fn in _defined_functions(source):
        if fn.name == entry:
            guards = [s for s in ast.walk(fn)
                      if isinstance(s, ast.If) and len(s.body) == 1
                      and isinstance(s.body[0], (ast.Return, ast.Raise))]
            top = {id(s) for s in fn.body}
            return sorted(guards, key=lambda g: (id(g) not in top, g.lineno))
    return []


def remove_lines(source: str, node: ast.stmt) -> str:
    """Delete a guard; if it has an else/elif branch, keep that branch in its place."""
    import textwrap

    lines = source.splitlines(keepends=True)
    end = getattr(node, "end_lineno", node.lineno)
    replacement: list[str] = []
    if isinstance(node, ast.If) and node.orelse:
        first, last = node.orelse[0], node.orelse[-1]
        block = lines[first.lineno - 1:getattr(last, "end_lineno", last.lineno)]
        indent = " " * node.col_offset
        if block and block[0].lstrip().startswith("elif"):
            # "elif c:" becomes "if c:" at the guard's own level
            block[0] = block[0].replace("elif", "if", 1)
            body = textwrap.dedent("".join(block))
        else:
            body = textwrap.dedent("".join(block))
        replacement = [textwrap.indent(body, indent)]
    return "".join(lines[: node.lineno - 1] + replacement + lines[end:])


def _arg_keywords(args_list: list[tuple]) -> set[str]:
    """Words a docstring would use to describe these inputs."""
    words: set[str] = set()
    for args in args_list:
        for a in args:
            if isinstance(a, (str, list, tuple, dict, set)) and len(a) == 0:
                words |= {"empty", "nothing", "no "}
            elif isinstance(a, bool):
                continue
            elif isinstance(a, (int, float)):
                if a == 0:
                    words |= {"zero", "0"}
                elif a < 0:
                    words |= {"negative"}
    return words


def _docstring_span(prompt: str) -> Optional[tuple[int, int]]:
    m = re.search(r'("""|\'\'\')(.*?)(\1)', prompt, re.DOTALL)
    return (m.start(2), m.end(2)) if m else None


def _strip_examples(doc: str, entry: str, covered: set[str]) -> tuple[str, list[str]]:
    """Remove ``>>> entry(...)`` examples (and their output line) on covered inputs."""
    out, removed, lines, i = [], [], doc.splitlines(keepends=True), 0
    while i < len(lines):
        line = lines[i]
        m = re.search(rf"{re.escape(entry)}\((.*)\)", line)
        if m:
            try:
                call = ast.parse(f"{entry}({m.group(1)})", mode="eval").body
                args = tuple(ast.literal_eval(a) for a in call.args)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                args = None
            if args is not None and repr(args) in covered:
                removed.append(line.strip())
                i += 1
                # a doctest's expected output sits on the next non-empty, non->>> line
                if line.lstrip().startswith(">>>") and i < len(lines) \
                        and lines[i].strip() and not lines[i].lstrip().startswith(">>>"):
                    removed.append(lines[i].strip())
                    i += 1
                continue
        out.append(line)
        i += 1
    return "".join(out), removed


def _strip_sentences(doc: str, keywords: set[str]) -> tuple[str, list[str], bool]:
    """Remove docstring sentences that describe the covered edge case.

    Returns the new text, the removed sentences and whether a sentence that
    still mentions a keyword was left behind (flag for manual review).
    """
    removed: list[str] = []
    if not keywords:
        return doc, removed, False
    kw = re.compile("|".join(rf"\b{re.escape(k.strip())}\b" if k.strip().isalnum()
                             else re.escape(k) for k in keywords), re.IGNORECASE)
    parts = re.split(r"(?<=[.!?])(\s+)", doc)
    kept = []
    for part in parts:
        if part.strip() and kw.search(part) and _EDGE_WORDS.search(part) \
                and ">>>" not in part and len(part) < 400:
            removed.append(part.strip())
            continue
        kept.append(part)
    new = "".join(kept)
    leftover = bool(kw.search(re.sub(r">>>.*", "", new)))
    return new, removed, leftover


def underspecify_by_guard(problem: Problem, timeout_s: float = 2.0,
                          max_ambiguous: int = 25,
                          max_covered_frac: float = 0.25) -> Optional[Variant]:
    """Build one ambiguous variant of ``problem`` (or ``None`` if no guard qualifies)."""
    entry = problem.entry_point
    all_tests = problem.visible_tests + [t for t in problem.hidden_tests
                                         if t not in problem.visible_tests]
    if not all_tests:
        return None
    ref_runs = run_function(problem.reference, entry, [t.args for t in all_tests], timeout_s)
    for guard in find_guards(problem.reference, entry):
        variant_src = remove_lines(problem.reference, guard)
        try:
            ast.parse(variant_src)
        except SyntaxError:
            continue
        var_runs = run_function(variant_src, entry, [t.args for t in all_tests], timeout_s)
        covered = [t for t, r, v in zip(all_tests, ref_runs, var_runs)
                   if r.signature != v.signature]
        crashes = [(t, v) for t, r, v in zip(all_tests, ref_runs, var_runs)
                   if r.signature != v.signature and v.kind == "error"]
        if not covered or not crashes:
            continue
        if len(covered) > max(1, int(max_covered_frac * len(all_tests))):
            continue  # the guard governs ordinary inputs, not an edge case
        remaining_visible = [t for t in problem.visible_tests if t not in covered]
        if not remaining_visible:
            continue
        covered_args = sorted({repr(t.args): t.args for t in covered}.values(),
                              key=lambda a: len(repr(a)))
        covered_keys = {repr(a) for a in covered_args}
        crash_t, crash_r = min(crashes, key=lambda tv: len(repr(tv[0].args)))

        # Reduce the specification.
        prompt = problem.prompt
        span = _docstring_span(prompt)
        removed_sentences: list[str] = []
        removed_examples: list[str] = []
        leftover = False
        if span:
            doc = prompt[span[0]:span[1]]
            doc, removed_examples = _strip_examples(doc, entry, covered_keys)
            doc, removed_sentences, leftover = _strip_sentences(doc, _arg_keywords(covered_args))
            prompt = prompt[:span[0]] + doc + prompt[span[1]:]
        reference = problem.reference.replace(problem.prompt, prompt, 1) \
            if problem.prompt in problem.reference else problem.reference
        buggy = variant_src.replace(problem.prompt, prompt, 1) \
            if problem.prompt in variant_src else variant_src

        reduced = Problem(
            task_id=f"{problem.task_id}~underspec",
            entry_point=entry,
            prompt=prompt,
            reference=reference,
            visible_tests=remaining_visible + [TestCase(crash_t.args, NO_CRASH)],
            hidden_tests=[t for t in problem.hidden_tests if repr(t.args) not in covered_keys],
            ambiguous_inputs=covered_args[:max_ambiguous],
        )
        guard_text = ast.get_source_segment(problem.reference, guard) or ""
        bug = BuggyProgram(reduced, buggy, f"{reduced.task_id}#guard{guard.lineno}",
                           "guard_removal")
        return Variant(bug, guard_text.strip(), removed_sentences, removed_examples,
                       f"{entry}{crash_t.args!r} raises {crash_r.value}",
                       needs_review=leftover)
    return None


_EDGE_CLASSES = ("empty", "zero", "negative")


def _in_class(args: tuple, cls: str) -> bool:
    for a in args:
        if isinstance(a, bool):
            continue
        if cls == "empty" and isinstance(a, (str, list, tuple, dict, set)) and len(a) == 0:
            return True
        if cls == "zero" and isinstance(a, (int, float)) and a == 0:
            return True
        if cls == "negative" and isinstance(a, (int, float)) and a < 0:
            return True
    return False


def _reduce_spec(problem: Problem, covered_args: list[tuple]
                 ) -> tuple[str, list[str], list[str], bool]:
    prompt = problem.prompt
    span = _docstring_span(prompt)
    if not span:
        return prompt, [], [], False
    doc = prompt[span[0]:span[1]]
    doc, examples = _strip_examples(doc, problem.entry_point, {repr(a) for a in covered_args})
    doc, sentences, leftover = _strip_sentences(doc, _arg_keywords(covered_args))
    return prompt[:span[0]] + doc + prompt[span[1]:], sentences, examples, leftover


def underspecify_by_edge_class(bug: BuggyProgram, timeout_s: float = 2.0,
                               max_ambiguous: int = 25,
                               max_covered_frac: float = 0.25) -> Optional[Variant]:
    """Make the spec silent on one edge-input class while keeping the original bug.

    Unlike guard removal the bug is unrelated to the edge class, so whether the
    candidates disagree there is observed at labelling time, not guaranteed.
    """
    problem = bug.problem
    all_tests = problem.visible_tests + [t for t in problem.hidden_tests
                                         if t not in problem.visible_tests]
    for cls in _EDGE_CLASSES:
        covered = [t for t in all_tests if _in_class(t.args, cls)]
        if not covered or len(covered) > max(1, int(max_covered_frac * len(all_tests))):
            continue
        keys = {repr(t.args) for t in covered}
        remaining_visible = [t for t in problem.visible_tests if repr(t.args) not in keys]
        if not remaining_visible or all(run_tests(bug.source, problem.entry_point,
                                                  remaining_visible, timeout_s)):
            continue  # the original bug must still be visible without the edge tests
        covered_args = sorted({repr(t.args): t.args for t in covered}.values(),
                              key=lambda a: len(repr(a)))
        prompt, sentences, examples, leftover = _reduce_spec(problem, covered_args)
        reduced = Problem(
            task_id=f"{problem.task_id}~silent-{cls}",
            entry_point=problem.entry_point, prompt=prompt,
            reference=problem.reference.replace(problem.prompt, prompt, 1),
            visible_tests=remaining_visible,
            hidden_tests=[t for t in problem.hidden_tests if repr(t.args) not in keys],
            ambiguous_inputs=covered_args[:max_ambiguous],
        )
        source = bug.source.replace(problem.prompt, prompt, 1)
        return Variant(BuggyProgram(reduced, source, f"{bug.bug_id}~silent-{cls}",
                                    f"{bug.mutation}+silent-{cls}"),
                       guard="", removed_sentences=sentences, removed_examples=examples,
                       crash="", needs_review=leftover)
    return None


def _check_variant(v: Variant, timeout_s: float) -> bool:
    """Sanity: buggy fails the crash report, reference passes everything kept."""
    p = v.bug.problem
    if all(run_tests(v.bug.source, p.entry_point, p.visible_tests, timeout_s)):
        return False
    tests = p.visible_tests + p.hidden_tests
    ref = run_function(p.reference, p.entry_point, [t.args for t in tests], timeout_s)
    return all(result_matches(r, t.expected) for r, t in zip(ref, tests))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _review_markdown(variants: list[Variant], original: dict[str, Problem]) -> str:
    out = ["# Ambiguous variants for review", "",
           "Each variant removes one edge-case guard from a correct solution, the docstring",
           "text and examples describing it, and every test covering it. Check that the",
           "reduced docstring really no longer decides the behaviour on the listed inputs.",
           "Variants marked REVIEW still mention a keyword related to those inputs.", ""]
    for v in variants:
        p = v.bug.problem
        orig = original.get(p.task_id.split("~")[0])
        head = ([f"- Removed guard: `{v.guard.splitlines()[0]}`", f"- Bug report: `{v.crash}`"]
                if v.guard else ["- Original bug kept; spec made silent on this input class"])
        out += [f"## {p.task_id}{'  (REVIEW)' if v.needs_review else ''}", "", *head,
                f"- Ambiguous inputs ({len(p.ambiguous_inputs)}): "
                + ", ".join(f"`{a!r}`" for a in p.ambiguous_inputs[:5]),
                f"- Removed sentences: {v.removed_sentences or 'none'}",
                f"- Removed examples: {v.removed_examples or 'none'}", ""]
        if orig is not None:
            out += ["Reduced docstring:", "", "```python", p.prompt.strip(), "```", ""]
    return "\n".join(out)


def prepare_all(out_dir: Union[str, Path], cache_dir: Optional[str] = None,
                max_hidden: int = 1000, mbpp_mutants: int = 1, timeout_s: float = 2.0,
                seed: int = 13, limit: Optional[int] = None, max_silent: int = 150,
                log=print) -> dict:
    """Run every preparation step and write ``*.jsonl``, ``stats.json`` and a review file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log("[1/3] HumanEvalFix + HumanEval+ ...")
    hef, hef_stats = prepare_humanevalfix(cache_dir, max_hidden, timeout_s, seed, limit)
    save_bugs([bug_to_json(b) for b in hef], out / "humanevalfix.jsonl")
    log(f"      kept {hef_stats.kept} of {hef_stats.seen}")

    log("[2/3] MBPP+ with injected bugs ...")
    mbpp, mbpp_problems, mbpp_stats = prepare_mbpp(cache_dir, mbpp_mutants, max_hidden,
                                                    timeout_s, seed, limit)
    save_bugs([bug_to_json(b) for b in mbpp], out / "mbpp_mutants.jsonl")
    log(f"      kept {mbpp_stats.kept} of {mbpp_stats.seen} problems, {len(mbpp)} bugs")

    log("[3/3] Ambiguous variants (controlled underspecification) ...")
    originals = {b.problem.task_id: b.problem for b in hef}
    originals.update({p.task_id: p for p in mbpp_problems})
    variants: list[Variant] = []
    rejected = 0
    # (a) guard removal: the bug itself sits on the now-unspecified inputs
    for prob in originals.values():
        v = underspecify_by_guard(prob, timeout_s)
        if v is None:
            continue
        if _check_variant(v, timeout_s):
            variants.append(v)
        else:
            rejected += 1
    n_guard = len(variants)
    # (b) silent edge class: one variant per problem, original bug kept
    done: set[str] = set()
    for bug in hef + mbpp:
        if bug.problem.task_id in done or len(variants) - n_guard >= max_silent:
            continue
        v = underspecify_by_edge_class(bug, timeout_s)
        if v is not None:
            variants.append(v)
            done.add(bug.problem.task_id)
    save_bugs([bug_to_json(v.bug, {"guard": v.guard, "crash": v.crash,
                                   "removed_sentences": v.removed_sentences,
                                   "removed_examples": v.removed_examples,
                                   "needs_review": v.needs_review}) for v in variants],
              out / "underspec.jsonl")
    (out / "underspec_review.md").write_text(_review_markdown(variants, originals))
    log(f"      {n_guard} guard-removal + {len(variants) - n_guard} silent-edge variants "
        f"({sum(v.needs_review for v in variants)} flagged for review, "
        f"{rejected} failed sanity checks)")

    stats = {
        "humanevalfix": hef_stats.as_dict(),
        "mbpp": {**mbpp_stats.as_dict(), "bugs": len(mbpp)},
        "underspec": {"variants": len(variants), "guard_removal": n_guard,
                      "silent_edge": len(variants) - n_guard,
                      "from_humaneval": sum(v.bug.problem.task_id.startswith("HumanEval")
                                            for v in variants),
                      "from_mbpp": sum(v.bug.problem.task_id.startswith("Mbpp")
                                       for v in variants),
                      "needs_review": sum(v.needs_review for v in variants),
                      "failed_sanity_check": rejected,
                      "avg_ambiguous_inputs": round(
                          sum(len(v.bug.problem.ambiguous_inputs) for v in variants)
                          / max(len(variants), 1), 1)},
        "total_bugs_for_generation": len(hef) + len(mbpp) + len(variants),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
