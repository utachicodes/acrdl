"""Prediction and evaluation with a trained model (requires the ``ml`` extra)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from acrdl.decision import decide
from acrdl.evidence.code_text import code_text
from acrdl.training.dataset import Record, problem_id_of
from acrdl.training.metrics import summarize
from acrdl.training.train import TrainResult, collect_logits
from acrdl.types import Evidence, Label, Patch


def predict_probs(result: TrainResult, records: Sequence[Record], batch_size: int = 32
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated probabilities and labels, in ``bug_batches`` order."""
    logits, labels = collect_logits(result.model, result.encoder, records, batch_size)
    with torch.no_grad():
        probs = torch.softmax(result.scaler(logits), -1).numpy()
    return probs, labels.numpy()


def evaluate_model(result: TrainResult, records: Sequence[Record]) -> dict[str, float]:
    probs, y = predict_probs(result, records)
    return summarize(y, decide(probs, result.threshold, result.amb_threshold), probs)


class ModelPredictor:
    """Adapter so a trained model can drive :class:`acrdl.pipeline.Pipeline`."""

    def __init__(self, result: TrainResult):
        self.result = result

    def predict(self, patches: list[Patch], evidence: Mapping[str, Evidence]
                ) -> dict[str, tuple[Label, list[float]]]:
        records = [
            Record(problem_id=problem_id_of(p.bug.problem.task_id), bug_id=p.bug.bug_id,
                   patch_id=p.patch_id,
                   code=code_text(p.bug.source, p.source, p.bug.problem.prompt),
                   evidence=evidence[p.patch_id].vector,
                   clusters=evidence[p.patch_id].cluster_features,
                   own_cluster=evidence[p.patch_id].cluster_id, label=0)
            for p in patches
        ]
        probs, _ = predict_probs(self.result, records)
        labels = decide(probs, self.result.threshold, self.result.amb_threshold)
        return {r.patch_id: (Label(int(lab)), pr.tolist())
                for r, lab, pr in zip(records, labels, probs)}
