import pytest

from acrdl.data import toy
from acrdl.data.labeling import label_patches, materialize_expected
from acrdl.data.loaders import _parse_assert_tests, load_toy
from acrdl.data.mutate import OPERATORS, enumerate_mutants, mutate
from acrdl.data.underspec import candidate_clauses, underspecify
from acrdl.evidence.testing import run_tests
from acrdl.types import Label

REF = "def f(xs, k):\n    if len(xs) > k:\n        return xs[0] + 1\n    return True\n"


def test_toy_bugs_fail_and_references_pass():
    for bug in load_toy():
        p = bug.problem
        assert all(run_tests(p.reference, p.entry_point, p.visible_tests + p.hidden_tests))
        assert not all(run_tests(bug.source, p.entry_point, p.visible_tests))


def test_every_operator_produces_mutants():
    for name in OPERATORS:
        assert enumerate_mutants(REF, [name]), name


def test_mutate_is_deterministic_and_changes_code():
    a = mutate(REF, n=3, seed=7)
    assert a == mutate(REF, n=3, seed=7)
    assert all(m.source.strip() != REF.strip() for m in a)


def test_underspecify_removes_clause_and_tests():
    p = toy.PROBLEMS[0]
    q = underspecify(p, toy.AVERAGE_CLAUSE, lambda a: a == ([],))
    assert toy.AVERAGE_CLAUSE not in q.prompt
    assert q.ambiguous_inputs == [([],)]
    assert all(t.args != ([],) for t in q.visible_tests + q.hidden_tests)
    assert toy.AVERAGE_CLAUSE in p.prompt  # original untouched
    with pytest.raises(ValueError):
        underspecify(p, "not a clause", lambda a: True)


def test_candidate_clauses_finds_edge_sentence():
    assert toy.AVERAGE_CLAUSE in candidate_clauses(toy.PROBLEMS[0].prompt)


def test_labeling_rule(average_bug):
    cands = dict(enumerate(toy.SCRIPTED_CANDIDATES["toy/average"]))
    labels = label_patches(average_bug.problem, {str(k): v for k, v in cands.items()})
    assert [labels[str(i)] for i in range(5)] == [Label.AMBIGUOUS] * 4 + [Label.OVERFITTING]


def test_labeling_full_spec_is_correct():
    p = toy.PROBLEMS[0]
    labels = label_patches(p, {"a": toy.SCRIPTED_CANDIDATES["toy/average"][0]})
    assert labels == {"a": Label.CORRECT}


def test_materialize_expected_and_assert_parser():
    p = toy.PROBLEMS[0]
    tests = materialize_expected(p, [([2, 4],), ([],)])
    assert [t.expected for t in tests] == [3.0, 0.0]
    parsed = _parse_assert_tests("assert candidate([1], 2) == 3\nassert x\n", "f")
    assert parsed[0].args == ([1], 2) and parsed[0].expected == 3


def test_load_quixbugs_from_local_clone(tmp_path):
    import json

    from acrdl.data.loaders import _first_def, load_quixbugs

    for d in ("python_programs", "correct_python_programs", "json_testcases"):
        (tmp_path / d).mkdir()
    (tmp_path / "python_programs" / "inc.py").write_text("def inc(x):\n    return x + 2\n")
    (tmp_path / "correct_python_programs" / "inc.py").write_text(
        "def inc(x):\n    return x + 1\n")
    (tmp_path / "json_testcases" / "inc.json").write_text(
        "\n".join(json.dumps([[i], i + 1]) for i in range(5)) + "\n")
    bugs = load_quixbugs(tmp_path)
    assert len(bugs) == 1
    p = bugs[0].problem
    assert len(p.visible_tests) == 3 and len(p.hidden_tests) == 5
    assert all(run_tests(p.reference, p.entry_point, p.hidden_tests))
    assert not all(run_tests(bugs[0].source, p.entry_point, p.visible_tests))
    assert _first_def("x = 1\ndef g():\n    pass\n") == "g"
