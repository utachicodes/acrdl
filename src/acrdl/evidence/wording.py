"""Wording dependence: is a candidate's behaviour on an input tied to the wording?

For one input x, collect the behaviour (output signature) of every plausible
candidate, grouped by the wording of the bug report it was generated from.

    I(B; W | x) = H(B | x) - H(B | W, x)

is the mutual information between behaviour B and wording W. Disagreement among
candidates contributes to H(B | x) in two ways: wrong candidates add noise inside
every wording, while a silent specification lets the wording steer the guess,
which shows up as dependence between B and W.

Guarantee (the only assumption is A1 below). A1: on an input the specification
determines, the rewordings are meaning-preserving, so which wording a candidate
came from is exchangeable with respect to its behaviour. Under A1 the permutation
test below rejects with probability at most ``alpha``: a determined input is
flagged as wording-dependent at most at the chosen false-alarm rate. Whether a
silent input is flagged (power) depends on how strongly the generator follows the
wording, which we measure on inputs whose silence is known by construction
(controlled underspecification).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence


def entropy(counts: Sequence[int]) -> float:
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c) if n else 0.0


def mutual_information(groups: Mapping[str, Sequence[str]]) -> float:
    """Plug-in I(B; W) in bits from behaviours grouped by wording."""
    pooled = [b for bs in groups.values() for b in bs]
    n = len(pooled)
    if n == 0:
        return 0.0
    h_b = entropy(list(Counter(pooled).values()))
    h_b_given_w = sum(len(bs) / n * entropy(list(Counter(bs).values()))
                      for bs in groups.values() if bs)
    return max(0.0, round(h_b - h_b_given_w, 12))  # clear floating-point dust


def permutation_test(groups: Mapping[str, Sequence[str]], n_perm: int = 2000,
                     seed: int = 0) -> tuple[float, float]:
    """Return (observed MI, p-value) for independence of behaviour and wording.

    The p-value uses the standard (count + 1) / (n_perm + 1) form, so it is a
    valid level-alpha test under exchangeability.
    """
    keys = [k for k, bs in groups.items() if bs]
    sizes = [len(groups[k]) for k in keys]
    pooled = [b for k in keys for b in groups[k]]
    observed = mutual_information({k: groups[k] for k in keys})
    if len(keys) < 2 or len(set(pooled)) < 2:
        return observed, 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm, i = {}, 0
        for k, s in zip(keys, sizes):
            perm[k] = pooled[i:i + s]
            i += s
        if mutual_information(perm) >= observed - 1e-12:
            hits += 1
    return observed, (hits + 1) / (n_perm + 1)


def disagreement(behaviours: Sequence[str]) -> float:
    """Share of candidates outside the majority behaviour (0 when all agree)."""
    if not behaviours:
        return 0.0
    return 1.0 - max(Counter(behaviours).values()) / len(behaviours)
