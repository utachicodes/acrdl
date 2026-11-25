"""Input generation: visible-test inputs plus type-guided random fuzzing.

The fuzzer infers a value template for each argument position from example
inputs and samples new values of the same shape, always including classic
edge cases (empty containers, zero, negative numbers). It is deliberately
dependency-free; ``hypothesis`` (``analysis`` extra) can be plugged in via
``extra_inputs`` for richer strategies.
"""

from __future__ import annotations

import random
import string
from collections.abc import Sequence
from typing import Any, Optional

_EDGE: dict[type, list[Any]] = {
    int: [0, 1, -1],
    float: [0.0, -1.5],
    str: ["", "a", "A"],
    list: [[]],
    tuple: [()],
}


def _sample_like(examples: list[Any], rng: random.Random, alphabet: str) -> Any:
    ex = rng.choice(examples)
    if isinstance(ex, bool):
        return rng.random() < 0.5
    if isinstance(ex, int):
        span = max([10, *(abs(e) * 2 for e in examples if isinstance(e, int))])
        return rng.randint(-span, span)
    if isinstance(ex, float):
        return round(rng.uniform(-100, 100), 2)
    if isinstance(ex, str):
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
    if isinstance(ex, (list, tuple)):
        elems = [e for x in examples if isinstance(x, (list, tuple)) for e in x] or [0, 1, 2]
        out = [_sample_like(elems, rng, alphabet) for _ in range(rng.randint(0, 6))]
        return out if isinstance(ex, list) else tuple(out)
    return ex  # dicts, None, other literals: reuse as-is


def fuzz_inputs(examples: Sequence[tuple], n: int, seed: int = 0) -> list[tuple]:
    """Generate up to ``n`` new argument tuples shaped like ``examples``."""
    if not examples:
        return []
    rng = random.Random(seed)
    arity = len(examples[0])
    columns = [[ex[i] for ex in examples if len(ex) == arity] for i in range(arity)]
    chars = {c for col in columns for v in col if isinstance(v, str) for c in v}
    alphabet = "".join(sorted(chars)) + string.ascii_letters[:6] + "AEIOUxyz "
    out: list[tuple] = []
    # edge cases first: vary one position at a time
    for i, col in enumerate(columns):
        for edge in _EDGE.get(type(col[0]), []):
            base = list(examples[0])
            base[i] = edge
            out.append(tuple(base))
    while len(out) < n:
        out.append(tuple(_sample_like(col, rng, alphabet) for col in columns))
    return out[:n]


def dedup(inputs: Sequence[tuple]) -> list[tuple]:
    seen: set[str] = set()
    out = []
    for x in inputs:
        key = repr(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def build_inputs(visible_args: Sequence[tuple], n_fuzz: int, seed: int = 0,
                 extra_inputs: Optional[Sequence[tuple]] = None) -> list[tuple]:
    """Visible-test inputs first, then fuzzed inputs, deduplicated."""
    pool = list(visible_args) + fuzz_inputs(visible_args, n_fuzz, seed) + list(extra_inputs or [])
    return dedup(pool)


def is_edge_input(args: tuple) -> bool:
    """Heuristic: does the input contain an empty/zero/negative edge value?"""
    for a in args:
        if isinstance(a, bool):
            continue
        if isinstance(a, (int, float)) and a <= 0:
            return True
        if isinstance(a, (str, list, tuple, dict, set)) and len(a) == 0:
            return True
    return False
