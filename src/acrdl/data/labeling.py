"""Ground-truth labels for plausible patches.

Labelling rule (applied only to *plausible* patches, i.e. patches that pass
every visible test):

1. **OVERFITTING** - the patch fails at least one hidden test.
2. **AMBIGUOUS** - the patch passes all hidden tests, the problem has
   ``ambiguous_inputs`` (created by controlled underspecification), and the
   hidden-test-passing candidates split into >= 2 behavioural clusters on
   those inputs. Every such cluster is consistent with the reduced spec, so
   no single behaviour is "the" correct one.
3. **CORRECT** - otherwise.

Ambiguity is labelled when it is *observed* among candidates: if all
candidates happen to agree on the ambiguous inputs, they are CORRECT with
respect to every reading the generator produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from acrdl.evidence.sandbox import run_function
from acrdl.evidence.testing import run_tests
from acrdl.types import Label, Problem, TestCase


def label_patches(problem: Problem, candidates: Mapping[str, str],
                  timeout_s: float = 2.0, memory_limit_mb: int = 512) -> dict[str, Label]:
    """Label plausible candidates (``patch_id -> source``) of one problem."""
    labels: dict[str, Label] = {}
    passing: list[str] = []
    for pid, src in candidates.items():
        ok = all(run_tests(src, problem.entry_point, problem.hidden_tests, timeout_s,
                           memory_limit_mb))
        if ok:
            passing.append(pid)
        else:
            labels[pid] = Label.OVERFITTING

    split = False
    if problem.ambiguous_inputs and len(passing) >= 2:
        behaviours = {
            tuple(r.signature for r in run_function(candidates[pid], problem.entry_point,
                                                     problem.ambiguous_inputs, timeout_s,
                                                     memory_limit_mb))
            for pid in passing
        }
        split = len(behaviours) >= 2
    for pid in passing:
        labels[pid] = Label.AMBIGUOUS if split else Label.CORRECT
    return labels


def materialize_expected(problem: Problem, inputs: Sequence[tuple],
                         timeout_s: float = 2.0) -> list[TestCase]:
    """Turn bare inputs (e.g. EvalPlus) into tests by running the reference.

    Inputs on which the reference raises or times out are dropped.
    """
    import ast

    tests = []
    for args, r in zip(inputs, run_function(problem.reference, problem.entry_point, inputs,
                                            timeout_s)):
        if r.kind == "ok":
            try:
                tests.append(TestCase(tuple(args), ast.literal_eval(r.value)))
            except (ValueError, SyntaxError):
                continue
    return tests
