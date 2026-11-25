"""Dataset loaders.

``load_toy`` is offline and always available. ``load_humanevalfix`` and
``load_evalplus`` need the ``data`` extra (``pip install -e ".[data]"``) and
network access on first use; ``load_quixbugs`` reads a local clone of the
QuixBugs repository. All map public benchmarks onto :class:`Problem`.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any, Optional, Union

from acrdl.data import toy
from acrdl.types import BuggyProgram, Problem, TestCase


def load_toy() -> list[BuggyProgram]:
    """Return the built-in toy benchmark as buggy programs."""
    return [
        BuggyProgram(problem=copy.deepcopy(p), source=toy.BUGS[p.task_id],
                     bug_id=f"{p.task_id}#0", mutation="manual")
        for p in toy.PROBLEMS
    ]


def _require_datasets() -> Any:
    try:
        import datasets  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - depends on extras
        raise ImportError(
            "Loading public benchmarks needs the 'data' extra: pip install -e '.[data]'"
        ) from e
    return datasets


def _parse_assert_tests(test_src: str, entry_point: str) -> list[TestCase]:
    """Extract ``assert f(args) == expected`` style tests from a test module."""
    tests: list[TestCase] = []
    for node in ast.walk(ast.parse(test_src)):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        cmp = node.test
        if len(cmp.ops) != 1 or not isinstance(cmp.ops[0], ast.Eq):
            continue
        call, rhs = cmp.left, cmp.comparators[0]
        if not (isinstance(call, ast.Call) and getattr(call.func, "id", None)
                in (entry_point, "candidate")):
            continue
        try:
            args = tuple(ast.literal_eval(a) for a in call.args)
            expected = ast.literal_eval(rhs)
        except ValueError:
            continue
        tests.append(TestCase(args, expected))
    return tests


def load_humanevalfix(limit: Optional[int] = None) -> list[BuggyProgram]:
    """HumanEvalFix (Python split of HumanEvalPack): buggy solution + tests."""
    ds = _require_datasets().load_dataset("bigcode/humanevalpack", "python", split="test")
    out: list[BuggyProgram] = []
    for row in ds:
        entry = row["entry_point"]
        tests = _parse_assert_tests(row["test"], entry)
        problem = Problem(
            task_id=row["task_id"],
            entry_point=entry,
            prompt=row["prompt"],
            reference=row["prompt"] + row["canonical_solution"],
            visible_tests=tests[:3],
            hidden_tests=tests,
        )
        out.append(BuggyProgram(problem, row["prompt"] + row["buggy_solution"],
                                bug_id=f"{row['task_id']}#hef",
                                mutation=row.get("bug_type", "")))
        if limit and len(out) >= limit:
            break
    return out


_EVALPLUS = {"humaneval": "evalplus/humanevalplus", "mbpp": "evalplus/mbppplus"}


def _first_def(source: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise ValueError("no function definition found")


def load_evalplus(benchmark: str = "humaneval", limit: Optional[int] = None) -> list[Problem]:
    """HumanEval+ or MBPP+ problems (``benchmark`` is ``"humaneval"`` or ``"mbpp"``).

    Hidden tests are left empty: EvalPlus ships inputs without expected
    outputs, so expected values are computed by running the reference in the
    sandbox (``acrdl.data.labeling.materialize_expected``).
    """
    if benchmark not in _EVALPLUS:
        raise ValueError(f"benchmark must be one of {sorted(_EVALPLUS)}")
    ds = _require_datasets().load_dataset(_EVALPLUS[benchmark], split="test")
    out: list[Problem] = []
    for row in ds:
        if benchmark == "humaneval":
            prompt = row["prompt"]
            reference = row["prompt"] + row["canonical_solution"]
            entry = row["entry_point"]
        else:
            reference = row["code"]
            entry = row.get("entry_point") or _first_def(reference)
            prompt = f'def {entry}(...):\n    """{row["prompt"].strip()}"""\n'
        out.append(Problem(
            task_id=str(row["task_id"]),
            entry_point=entry,
            prompt=prompt,
            reference=reference,
            visible_tests=_parse_assert_tests(row.get("test", ""), entry)[:3],
        ))
        if limit and len(out) >= limit:
            break
    return out


def load_quixbugs(root: Union[str, Path], limit: Optional[int] = None) -> list[BuggyProgram]:
    """QuixBugs Python programs from a local clone of the QuixBugs repository.

    Expects ``python_programs/``, ``correct_python_programs/`` and
    ``json_testcases/`` (one ``[args, expected]`` JSON array per line).
    The first three tests are visible, all tests are hidden.
    """
    root = Path(root)
    out: list[BuggyProgram] = []
    for case_file in sorted((root / "json_testcases").glob("*.json")):
        name = case_file.stem
        buggy = root / "python_programs" / f"{name}.py"
        fixed = root / "correct_python_programs" / f"{name}.py"
        if not (buggy.exists() and fixed.exists()):
            continue
        tests = []
        for line in case_file.read_text().splitlines():
            if not line.strip():
                continue
            args, expected = json.loads(line)
            tests.append(TestCase(tuple(args) if isinstance(args, list) else (args,), expected))
        problem = Problem(task_id=f"quixbugs/{name}", entry_point=name,
                          prompt=f"def {name}(...):\n    \"\"\"QuixBugs program {name}.\"\"\"\n",
                          reference=fixed.read_text(), visible_tests=tests[:3],
                          hidden_tests=tests)
        out.append(BuggyProgram(problem, buggy.read_text(), f"quixbugs/{name}#0", "quixbugs"))
        if limit and len(out) >= limit:
            break
    return out
