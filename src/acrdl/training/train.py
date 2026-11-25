"""Training loop for the fusion model (requires the ``ml`` extra).

Steps: bug-grouped batches -> encoder + FusionModel -> CE + lambda * SupCon
-> temperature scaling on validation logits -> accept threshold fitted for
the target false-accept rate -> checkpoint.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from acrdl.config import Config
from acrdl.decision import fit_accept_threshold, fit_ambiguous_threshold
from acrdl.model.calibration import TemperatureScaler
from acrdl.model.encoder import CodeEncoder, FrozenCachedEncoder, HashingEncoder
from acrdl.model.fusion import FusionModel
from acrdl.model.losses import FusionLoss
from acrdl.training.dataset import Record, bug_batches, collate

ABLATIONS = {
    "full": {},
    "code_only": {"use_evidence": False, "use_clusters": False},
    "no_code": {"use_code": False},
    "no_evidence": {"use_evidence": False},
    "no_clusters": {"use_clusters": False},
    "no_contrastive": {},
}


@dataclass
class TrainResult:
    model: FusionModel
    encoder: nn.Module
    scaler: TemperatureScaler
    threshold: float
    history: list[dict[str, float]] = field(default_factory=list)
    amb_threshold: Optional[float] = None


def resolve_class_weights(spec, records: Sequence[Record], n_classes: int = 3,
                          cap: float = 20.0) -> Optional[list[float]]:
    """``"balanced"`` -> inverse class frequency on ``records``, capped; lists pass through."""
    if spec is None or isinstance(spec, (list, tuple)):
        return list(spec) if spec is not None else None
    if spec != "balanced":
        raise ValueError(f"class_weights must be a list or 'balanced', got {spec!r}")
    counts = np.bincount([int(r.label) for r in records], minlength=n_classes).astype(float)
    total = counts.sum()
    return [float(min(cap, total / (n_classes * c))) if c else 1.0 for c in counts]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_encoder(kind: str, cfg: Config) -> nn.Module:
    if kind == "hashing":
        return HashingEncoder(cfg.model.code_dim)
    if kind == "hf":
        return CodeEncoder(cfg.model.encoder_name, cfg.model.encoder_trainable_layers)
    if kind == "hf-frozen":
        safe = cfg.model.encoder_name.replace("/", "_")
        return FrozenCachedEncoder(cfg.model.encoder_name,
                                   cache_path=f"data/processed/emb_cache_{safe}.pt")
    raise ValueError(f"Unknown encoder kind: {kind}")


def forward_batch(model: FusionModel, encoder: nn.Module, batch: dict[str, Any]
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    dev = next(model.parameters()).device
    code = encoder(batch["texts"]).to(dev)
    return model(code, batch["evidence"].to(dev), batch["clusters"].to(dev),
                 batch["cluster_mask"].to(dev), batch["own_index"].to(dev))


@torch.no_grad()
def collect_logits(model: FusionModel, encoder: nn.Module, records: Sequence[Record],
                   batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    encoder.eval()
    logits, labels = [], []
    for chunk in bug_batches(records, batch_size, shuffle=False):
        b = collate(chunk)
        out, _ = forward_batch(model, encoder, b)
        logits.append(out.cpu())
        labels.append(b["labels"])
    return torch.cat(logits), torch.cat(labels)


def train_model(train: Sequence[Record], val: Sequence[Record], cfg: Config,
                encoder: Optional[nn.Module] = None, ablation: str = "full",
                seed: Optional[int] = None, device: str = "cpu") -> TrainResult:
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation {ablation!r}; choose from {sorted(ABLATIONS)}")
    tc = cfg.training
    seed = cfg.seed if seed is None else seed
    set_seed(seed)
    encoder = encoder if encoder is not None else HashingEncoder(cfg.model.code_dim)
    model = FusionModel(cfg.model, **ABLATIONS[ablation]).to(device)
    encoder.to(device)
    loss_fn = FusionLoss(resolve_class_weights(tc.class_weights, train, cfg.model.n_classes),
                         0.0 if ablation == "no_contrastive" else tc.contrastive_weight,
                         tc.contrastive_temperature).to(device)
    groups = [{"params": model.parameters(), "lr": tc.head_lr}]
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    if enc_params:
        groups.append({"params": enc_params, "lr": tc.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    history = []
    for epoch in range(tc.epochs):
        model.train()
        encoder.train()
        total, n = 0.0, 0
        for batch_records in bug_batches(train, tc.batch_size, seed=seed + epoch):
            b = collate(batch_records)
            logits, z = forward_batch(model, encoder, b)
            loss = loss_fn(logits, z, b["labels"].to(device), b["groups"].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch_records)
            n += len(batch_records)
        history.append({"epoch": epoch, "train_loss": total / max(1, n)})

    scaler = TemperatureScaler()
    threshold, amb_threshold = 0.5, None
    if val:
        logits, labels = collect_logits(model, encoder, val, tc.batch_size)
        scaler.fit(logits, labels)
        probs = torch.softmax(scaler(logits), -1).detach().numpy()
        amb_threshold = fit_ambiguous_threshold(probs, labels.numpy())
        threshold = fit_accept_threshold(probs, labels.numpy(),
                                         cfg.decision.target_false_accept_rate, amb_threshold)
    return TrainResult(model, encoder, scaler, threshold, history, amb_threshold)


def save_checkpoint(result: TrainResult, cfg: Config, out_dir: str, encoder_kind: str,
                    ablation: str) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "model": result.model.state_dict(),
        # a frozen encoder is rebuilt from its pretrained weights; store only trained ones
        "encoder": ({k: v for k, v in result.encoder.state_dict().items()}
                    if any(p.requires_grad for p in result.encoder.parameters()) else {}),
        "temperature_log": result.scaler.log_t.detach().cpu(),
    }
    torch.save(state, path / "checkpoint.pt")
    meta = {"config": asdict(cfg), "encoder_kind": encoder_kind, "ablation": ablation,
            "threshold": result.threshold, "amb_threshold": result.amb_threshold,
            "temperature": result.scaler.temperature,
            "history": result.history}
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    return path


def load_checkpoint(out_dir: str, cfg: Config) -> TrainResult:
    path = Path(out_dir)
    meta = json.loads((path / "meta.json").read_text())
    state = torch.load(path / "checkpoint.pt", map_location="cpu")
    encoder = make_encoder(meta["encoder_kind"], cfg)
    if state["encoder"]:
        encoder.load_state_dict(state["encoder"])
    model = FusionModel(cfg.model, **ABLATIONS[meta["ablation"]])
    model.load_state_dict(state["model"])
    scaler = TemperatureScaler()
    scaler.log_t.data = state["temperature_log"]
    return TrainResult(model, encoder, scaler, float(meta["threshold"]), meta["history"],
                       meta.get("amb_threshold"))


def _crossfit_folds(records: Sequence[Record], k: int, seed: int) -> list[set[str]]:
    """Problem-level folds, stratified by which rare labels a problem contains."""
    labels_of: dict[str, set[int]] = {}
    for r in records:
        labels_of.setdefault(r.problem_id, set()).add(int(r.label))
    strata: dict[tuple[bool, bool], list[str]] = {}
    for pid in sorted(labels_of):
        ls = labels_of[pid]
        strata.setdefault((2 in ls, 1 in ls), []).append(pid)
    rng = random.Random(seed)
    folds: list[set[str]] = [set() for _ in range(k)]
    offset = 0
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        for i, pid in enumerate(group):
            folds[(i + offset) % k].add(pid)
        offset += len(group)
    return folds


def train_crossfit(train: Sequence[Record], val: Sequence[Record], cfg: Config,
                   encoder: Optional[nn.Module] = None, ablation: str = "full",
                   seed: Optional[int] = None, k: int = 5, min_rare: int = 10) -> TrainResult:
    """Train on train + val; fit calibration and thresholds on out-of-fold predictions.

    With only a few rare examples in a validation split, thresholds tuned on it are
    unstable. Cross-fitting uses every training example once as held-out data: k models
    are trained on k-1 folds and predict the remaining fold; temperature scaling and both
    decision thresholds are fitted on these out-of-fold predictions; the final model is
    trained on all of train + val. A threshold is only fitted when its class has at least
    ``min_rare`` out-of-fold examples (otherwise the plain argmax rule is used).
    """
    seed = cfg.seed if seed is None else seed
    pool = list(train) + list(val)
    oof_logits, oof_labels = [], []
    for fold in _crossfit_folds(pool, k, seed):
        fit = [r for r in pool if r.problem_id not in fold]
        held = [r for r in pool if r.problem_id in fold]
        if not held or not fit:
            continue
        res = train_model(fit, [], cfg, encoder, ablation, seed)
        logits, labels = collect_logits(res.model, res.encoder, held, cfg.training.batch_size)
        oof_logits.append(logits)
        oof_labels.append(labels)
    final = train_model(pool, [], cfg, encoder, ablation, seed)
    if not oof_logits:
        return final
    logits, labels = torch.cat(oof_logits), torch.cat(oof_labels)
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    probs = torch.softmax(scaler(logits), -1).detach().numpy()
    y = labels.numpy()
    amb = fit_ambiguous_threshold(probs, y) if (y == 2).sum() >= min_rare else None
    thr = (fit_accept_threshold(probs, y, cfg.decision.target_false_accept_rate, amb)
           if (y == 1).sum() >= min_rare else 0.5)
    return TrainResult(final.model, final.encoder, scaler, thr, final.history, amb)
