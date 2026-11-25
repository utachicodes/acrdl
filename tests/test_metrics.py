import numpy as np
import pytest

from acrdl.decision import decide, false_accept_rate, fit_accept_threshold
from acrdl.training.metrics import (
    abstention_precision_recall,
    auroc,
    auroc_ovr,
    expected_calibration_error,
    macro_f1,
    paired_bootstrap,
    summarize,
)


def test_basic_metrics():
    y = np.array([0, 1, 2, 0])
    assert macro_f1(y, y) == 1.0
    assert auroc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert auroc([1, 0], [0.5, 0.5]) == 0.5
    assert false_accept_rate(np.array([0, 1]), np.array([1, 1])) == 0.5
    assert abstention_precision_recall(np.array([2, 2, 0]), np.array([2, 0, 2])) == (0.5, 0.5)
    probs = np.eye(3)[y]
    assert auroc_ovr(y, probs) == 1.0
    assert expected_calibration_error(y, probs) == pytest.approx(0.0)
    s = summarize(y, y, probs)
    assert s["macro_f1"] == 1.0 and s["false_accept_rate"] == 0.0


def test_decision_rule_and_threshold():
    probs = np.array([[0.9, 0.05, 0.05], [0.6, 0.4, 0.0], [0.3, 0.2, 0.5], [0.4, 0.6, 0.0]])
    assert decide(probs, 0.5).tolist() == [0, 0, 2, 1]
    labels = np.array([0, 1, 2, 1])
    t = fit_accept_threshold(probs, labels, target_far=0.0)
    assert false_accept_rate(decide(probs, t), labels) == 0.0
    assert t > 0.6


def test_paired_bootstrap():
    y = np.array([0, 1] * 20)
    res = paired_bootstrap(lambda a, b: float((a == b).mean()), y, y, 1 - y, n=200,
                           groups=np.arange(40) // 2)
    assert res["delta"] == 1.0 and res["ci_low"] > 0.9 and res["p_value"] < 0.05


def test_ambiguous_threshold_and_balanced_weights():
    import numpy as np

    from acrdl.decision import decide, fit_ambiguous_threshold
    from acrdl.types import Label

    probs = np.array([[0.8, 0.1, 0.1], [0.7, 0.05, 0.25], [0.6, 0.1, 0.3], [0.9, 0.05, 0.05]])
    labels = np.array([Label.CORRECT, Label.AMBIGUOUS, Label.AMBIGUOUS, Label.CORRECT])
    t = fit_ambiguous_threshold(probs, labels)
    assert t == 0.25  # catches both ambiguous rows although P(ambiguous) never wins argmax
    assert list(decide(probs, 0.5, t)) == [0, 2, 2, 0]
    assert fit_ambiguous_threshold(probs, np.zeros(4)) is None


def test_balanced_class_weights():
    import pytest

    pytest.importorskip("torch")
    from acrdl.training.train import resolve_class_weights

    class R:
        def __init__(self, label):
            self.label = label

    w = resolve_class_weights("balanced", [R(0)] * 8 + [R(1)] + [R(2)])
    assert w[0] < 1 < w[2] and w[1] == w[2]
    assert resolve_class_weights([1.0, 2.0, 3.0], []) == [1.0, 2.0, 3.0]


def test_semantic_entropy_baseline():
    from acrdl.baselines.semantic_entropy import SemanticEntropy, bug_entropy
    from acrdl.training.dataset import Record

    def rec(bug, pid, clusters, own, label):
        return Record("p" + bug, bug, pid, "", [0.0] * 30, clusters, own, label)

    split = [[0.5, 0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0]]   # two equal clusters: 1 bit
    lopsided = [[0.9, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]]
    data = [rec("a", "a1", split, 0, 2), rec("a", "a2", split, 1, 2),
            rec("b", "b1", lopsided, 0, 0), rec("b", "b2", lopsided, 1, 1)]
    assert abs(bug_entropy(data[0]) - 1.0) < 1e-9
    model = SemanticEntropy().fit(data)
    assert list(model.predict_records(data)) == [2, 2, 0, 1]
