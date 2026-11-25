"""Neural components; skipped when the 'ml' extra (torch) is not installed."""

import pytest

torch = pytest.importorskip("torch")

from acrdl.config import Config  # noqa: E402
from acrdl.model.calibration import TemperatureScaler  # noqa: E402
from acrdl.model.encoder import HashingEncoder  # noqa: E402
from acrdl.model.fusion import FusionModel  # noqa: E402
from acrdl.model.losses import FusionLoss, supervised_contrastive  # noqa: E402


def _batch(b=6, k=3):
    cfg = Config().model
    mask = torch.zeros(b, k, dtype=torch.bool)
    mask[0, 2] = True
    return cfg, (torch.randn(b, cfg.code_dim), torch.rand(b, cfg.evidence_dim),
                 torch.rand(b, k, cfg.cluster_feat_dim), mask,
                 torch.tensor([0, 1, 0, 2, 1, 0]))


def test_fusion_forward_backward():
    cfg, inputs = _batch()
    model = FusionModel(cfg)
    logits, z = model(*inputs)
    assert logits.shape == (6, 3) and z.shape == (6, 3 * cfg.hidden_dim)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    loss = FusionLoss([1.0, 1.0, 2.0], 0.1)(logits, z, labels, groups)
    loss.backward()
    assert all(p.grad is not None for p in model.head.parameters())


def test_ablation_switches_change_output():
    cfg, inputs = _batch()
    torch.manual_seed(0)
    full = FusionModel(cfg).eval()
    torch.manual_seed(0)
    no_clusters = FusionModel(cfg, use_clusters=False).eval()
    assert not torch.allclose(full(*inputs)[0], no_clusters(*inputs)[0])


def test_supcon_only_uses_same_bug_pairs():
    z = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 0, 1])
    assert supervised_contrastive(z, labels, torch.tensor([0, 1, 2, 3])) == 0
    assert supervised_contrastive(z, labels, torch.tensor([0, 0, 0, 0])) > 0


def test_temperature_scaling_reduces_overconfidence():
    torch.manual_seed(0)
    labels = torch.randint(0, 3, (200,))
    logits = torch.randn(200, 3) * 10
    t = TemperatureScaler().fit(logits, labels)
    assert t.temperature > 1.0


def test_hashing_encoder_is_deterministic():
    enc = HashingEncoder(64)
    a, b = enc(["def f(x): return x"]), enc(["def f(x): return x"])
    assert torch.equal(a, b) and a.shape == (1, 64)


def test_train_and_evaluate_on_toy(tmp_path):
    from acrdl.cli import main
    from acrdl.training.dataset import load_jsonl
    from acrdl.training.evaluate import ModelPredictor, evaluate_model
    from acrdl.training.train import load_checkpoint, save_checkpoint, train_model

    path = tmp_path / "toy.jsonl"
    assert main(["build-dataset", "--source", "toy", "--out", str(path), "--mutants", "1"]) == 0
    records = load_jsonl(path)
    cfg = Config()
    cfg.training.epochs = 3
    res = train_model(records, records, cfg)
    assert 0.0 <= res.threshold <= 1.0 + 1e-6
    metrics = evaluate_model(res, records)
    assert set(metrics) >= {"macro_f1", "false_accept_rate", "auroc_ovr", "ece"}
    save_checkpoint(res, cfg, str(tmp_path / "ckpt"), "hashing", "full")
    again = load_checkpoint(str(tmp_path / "ckpt"), cfg)
    assert evaluate_model(again, records) == pytest.approx(metrics)

    from acrdl.cli import toy_bugs
    from acrdl.data import toy
    from acrdl.generation.generator import MockGenerator
    from acrdl.pipeline import Pipeline

    rep = Pipeline(cfg, MockGenerator(toy.SCRIPTED_CANDIDATES), ModelPredictor(res)).run(
        toy_bugs()[0])
    assert all(r.verdict is not None for r in rep.patches if r.plausible)


def test_frozen_cached_encoder_reuses_cache(tmp_path, monkeypatch):
    import torch

    from acrdl.model.encoder import FrozenCachedEncoder

    calls = []

    class Fake(torch.nn.Module):
        dim = 4

        def forward(self, texts):
            calls.append(list(texts))
            return torch.ones(len(texts), 4)

    enc = FrozenCachedEncoder(cache_path=tmp_path / "c.pt")
    monkeypatch.setattr(enc, "_encoder", lambda: Fake())
    a = enc(["x", "y", "x"])
    b = enc(["y"])
    assert a.shape == (3, 4) and b.shape == (1, 4)
    assert calls == [["x", "y"]]  # each distinct text encoded once
    again = FrozenCachedEncoder(cache_path=tmp_path / "c.pt")
    assert set(again.cache) == set(enc.cache)  # cache persisted to disk


def test_crossfit_folds_partition_problems_and_training_runs(tmp_path):
    from acrdl.cli import main
    from acrdl.training.dataset import load_jsonl
    from acrdl.training.train import _crossfit_folds, train_crossfit

    path = tmp_path / "toy.jsonl"
    assert main(["build-dataset", "--source", "toy", "--out", str(path), "--mutants", "1"]) == 0
    records = load_jsonl(path)
    folds = _crossfit_folds(records, 3, seed=0)
    problems = {r.problem_id for r in records}
    assert set().union(*folds) == problems
    assert sum(len(f) for f in folds) == len(problems)  # disjoint
    cfg = Config()
    cfg.training.epochs = 2
    res = train_crossfit(records, [], cfg, k=3, min_rare=1)
    assert 0.0 <= res.threshold <= 1.0 + 1e-6
