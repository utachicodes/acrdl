from acrdl.evidence import differential as D
from acrdl.evidence.clustering import cluster
from acrdl.evidence.features import CLUSTER_FEATURE_NAMES, EVIDENCE_NAMES, build_evidence
from acrdl.evidence.inputs import build_inputs, fuzz_inputs, is_edge_input
from acrdl.evidence.static import static_features
from acrdl.evidence.symbolic import SymbolicVerdict, crosshair_available, diff_behavior
from acrdl.types import TestCase

BUGGY = "def f(x):\n    return x + 2\n"
CANDS = {
    "good1": "def f(x):\n    return x + 1\n",
    "good2": "def f(x):\n    y = x\n    return y + 1\n",
    "overfit": "def f(x):\n    if x == 3:\n        return 4\n    return x + 2\n",
}
INPUTS = [(i,) for i in range(-3, 7)]


def _matrix():
    return D.build_matrix(BUGGY, "f", CANDS, INPUTS)


def test_fuzz_shapes_edges_and_determinism():
    xs = fuzz_inputs([([1, 2], "ab")], 30, seed=1)
    assert xs == fuzz_inputs([([1, 2], "ab")], 30, seed=1)
    assert all(isinstance(a, list) and isinstance(b, str) for a, b in xs)
    assert ([], "ab") in xs
    assert len(build_inputs([(1,), (1,)], 10)) <= 11
    assert is_edge_input(([],)) and not is_edge_input(([1],))


def test_differential_signals():
    m = _matrix()
    cons = D.consensus(m)
    assert cons[0] == "ok:-2"  # good1/good2 outvote the overfit patch
    assert D.consensus_agreement(m, "good1") == 1.0
    assert D.consensus_agreement(m, "overfit") < 0.2
    assert D.changed_fraction(m, "good1") == 1.0
    pres, n = D.preservation_score(m, "good1")
    assert n == 0 and pres == 1.0  # bug manifests everywhere


def test_clustering():
    c = cluster(_matrix())
    assert c.n_clusters == 2
    assert c.assignment["good1"] == c.assignment["good2"] == 0
    assert len(c.disagreement_inputs) == len(INPUTS) - 1  # all but x == 3
    assert 0 < c.entropy <= 1


def test_static_smells():
    s = static_features(BUGGY, CANDS["overfit"], [TestCase((3,), 4)])
    assert s.hardcoded_test_literal == 1 and s.special_case_eq == 1
    assert s.added_branches == 1 and s.added_return_const == 1
    g = static_features(BUGGY, CANDS["good1"], [TestCase((3,), 4)])
    assert g.hardcoded_test_literal == 0 and g.diff_lines == 2
    assert static_features(BUGGY, "def f(:", []).parse_error == 1
    r = static_features("def f(x):\n    raise ValueError\n",
                        "def f(x):\n    try:\n        pass\n    except Exception:\n        pass\n")
    assert r.removed_guards == 1 and r.added_broad_except == 1


def test_feature_vector_shapes():
    m = _matrix()
    c = cluster(m)
    stat = {pid: static_features(BUGGY, src) for pid, src in CANDS.items()}
    ev = build_evidence(m, c, stat, {pid: True for pid in CANDS})
    assert len(EVIDENCE_NAMES) == 30
    for e in ev.values():
        assert len(e.vector) == 30
        assert len(e.cluster_features) == c.n_clusters
        assert all(len(row) == len(CLUSTER_FEATURE_NAMES) for row in e.cluster_features)
        assert sum(row[-1] for row in e.cluster_features) == 1.0
        assert all(0.0 <= v <= 1.0 for v in e.vector)


def test_symbolic_unknown_without_crosshair():
    v = diff_behavior(BUGGY, CANDS["good1"], "f", timeout_s=5)
    if not crosshair_available():
        assert v is SymbolicVerdict.UNKNOWN
    assert 0.0 <= v.as_feature() <= 1.0
