"""Assemble fixed-length evidence vectors for the fusion model.

``EVIDENCE_NAMES`` defines the 30-dimensional per-patch vector, grouped as
dynamic (differential execution), cluster-level, static and symbolic
features. ``CLUSTER_FEATURE_NAMES`` defines one row per behavioural cluster
of the same bug; the model's cluster branch reads these rows to reason about
*how* the candidates disagree.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Optional

from acrdl.evidence import differential as D
from acrdl.evidence.clustering import Clustering
from acrdl.evidence.inputs import is_edge_input
from acrdl.evidence.static import StaticFeatures, squash
from acrdl.evidence.symbolic import SymbolicVerdict
from acrdl.types import Evidence

EVIDENCE_NAMES: list[str] = [
    # dynamic: tests and differential execution
    "fail_to_pass",
    "visible_fixed_frac",
    "preservation",
    "stable_coverage",
    "consensus_agreement",
    "changed_frac",
    "change_edge_frac",
    "error_rate",
    "timeout_rate",
    "none_return_rate",
    "buggy_error_rate",
    # cluster-level: how candidates of the same bug disagree
    "cluster_size_frac",
    "n_clusters_norm",
    "cluster_entropy",
    "disagreement_input_frac",
    "minority_frac",
    "tie_frac",
    "edge_disagreement_frac",
    "error_on_disagreement",
    "plausible_frac",
    # static: overfitting smells of the buggy -> patch change
    "diff_lines",
    "node_delta",
    "hardcoded_test_literal",
    "special_case_eq",
    "removed_guards",
    "added_broad_except",
    "added_return_const",
    "added_branches",
    "parse_error",
    # symbolic
    "symbolic_diff",
]

CLUSTER_FEATURE_NAMES: list[str] = [
    "size_frac",
    "consensus_agreement",
    "changed_frac",
    "error_rate",
    "edge_disagreement_frac",
    "is_own_cluster",
]


def _frac(xs: Sequence[bool]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def build_evidence(
    matrix: D.BehaviourMatrix,
    clustering: Clustering,
    static: Mapping[str, StaticFeatures],
    fail_to_pass: Mapping[str, bool],
    symbolic: Optional[Mapping[str, SymbolicVerdict]] = None,
    visible_fixed: Optional[Mapping[str, float]] = None,
    n_generated: Optional[int] = None,
) -> dict[str, Evidence]:
    """Evidence for every candidate in ``matrix``.

    ``visible_fixed`` maps a patch to the fraction of initially failing
    visible tests it fixes; ``n_generated`` is the number of candidates
    sampled before the plausibility filter.
    """
    cons = D.consensus(matrix)
    stable = D.stable_inputs(matrix, cons)
    n_in = max(1, len(matrix.inputs))
    n_cand = max(1, len(matrix.candidates))
    disagree = set(clustering.disagreement_inputs)
    buggy_err = _frac([s.startswith("error:") for s in matrix.buggy])
    counts = [Counter(r[j] for r in matrix.candidates.values()) for j in range(len(matrix.inputs))]
    ties = _frac([c is None for c in cons])

    per: dict[str, dict[str, float]] = {}
    for pid, row in matrix.candidates.items():
        pres, n_stable = D.preservation_score(matrix, pid, stable)
        err, tout = D.outcome_rates(matrix, pid)
        s = static[pid]
        sym = (symbolic or {}).get(pid, SymbolicVerdict.UNKNOWN)
        changed = [j for j in range(len(row)) if row[j] != matrix.buggy[j]]
        off_cons = [j for j, c in enumerate(cons) if c is not None and row[j] != c]
        minority = [counts[j][row[j]] < max(counts[j].values()) for j in range(len(row))]
        per[pid] = {
            "fail_to_pass": float(fail_to_pass.get(pid, False)),
            "visible_fixed_frac": float((visible_fixed or {}).get(pid, 1.0)),
            "preservation": pres,
            "stable_coverage": n_stable / n_in,
            "consensus_agreement": D.consensus_agreement(matrix, pid, cons),
            "changed_frac": len(changed) / n_in,
            "change_edge_frac": _frac([is_edge_input(matrix.inputs[j]) for j in changed]),
            "error_rate": err,
            "timeout_rate": tout,
            "none_return_rate": _frac([x == "ok:None" for x in row]),
            "buggy_error_rate": buggy_err,
            "cluster_size_frac": len(clustering.clusters[clustering.assignment[pid]]) / n_cand,
            "n_clusters_norm": (clustering.n_clusters - 1) / max(1, n_cand - 1),
            "cluster_entropy": clustering.entropy,
            "disagreement_input_frac": len(disagree) / n_in,
            "minority_frac": _frac(minority),
            "tie_frac": ties,
            "edge_disagreement_frac": _frac([is_edge_input(matrix.inputs[j]) for j in off_cons]),
            "error_on_disagreement": _frac([row[j].startswith("error:") for j in disagree]),
            "plausible_frac": len(matrix.candidates) / max(1, n_generated or n_cand),
            "diff_lines": math.log1p(s.diff_lines) / 5.0,
            "node_delta": min(1.0, s.node_delta),
            "hardcoded_test_literal": squash(s.hardcoded_test_literal, 1.0),
            "special_case_eq": squash(s.special_case_eq, 1.0),
            "removed_guards": squash(s.removed_guards, 1.0),
            "added_broad_except": squash(s.added_broad_except, 1.0),
            "added_return_const": squash(s.added_return_const, 1.0),
            "added_branches": squash(s.added_branches, 1.0),
            "parse_error": float(s.parse_error),
            "symbolic_diff": sym.as_feature(),
        }

    cluster_rows = []
    for members in clustering.clusters:
        k = len(members)
        cluster_rows.append([
            k / n_cand,
            sum(per[p]["consensus_agreement"] for p in members) / k,
            sum(per[p]["changed_frac"] for p in members) / k,
            sum(per[p]["error_rate"] for p in members) / k,
            sum(per[p]["edge_disagreement_frac"] for p in members) / k,
        ])

    out: dict[str, Evidence] = {}
    for pid, feats in per.items():
        own = clustering.assignment[pid]
        rows = [r + [float(i == own)] for i, r in enumerate(cluster_rows)]
        out[pid] = Evidence(
            patch_id=pid,
            vector=[feats[n] for n in EVIDENCE_NAMES],
            names=list(EVIDENCE_NAMES),
            cluster_id=own,
            cluster_features=rows,
        )
    return out
