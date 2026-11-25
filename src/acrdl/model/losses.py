"""Training objectives: weighted cross-entropy + same-bug supervised contrastive."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def supervised_contrastive(z: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor,
                           temperature: float = 0.1) -> torch.Tensor:
    """SupCon restricted to pairs from the same bug (``groups``).

    Positives share bug and label; negatives share the bug but not the label.
    Contrasting candidates *of the same bug* forces the model to learn
    semantic differences instead of "this bug is easy".
    """
    z = nn.functional.normalize(z, dim=-1)
    sim = z @ z.T / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same_group = (groups[:, None] == groups[None, :]) & ~eye
    pos = same_group & (labels[:, None] == labels[None, :])
    valid = pos.any(1) & (same_group & ~pos).any(1)
    if not valid.any():
        return z.sum() * 0.0
    sim = sim.masked_fill(~same_group, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    log_prob = log_prob.masked_fill(~pos, 0.0)
    per_anchor = -log_prob.sum(1) / pos.sum(1).clamp(min=1)
    return per_anchor[valid].mean()


class FusionLoss(nn.Module):
    def __init__(self, class_weights: Optional[list[float]] = None,
                 contrastive_weight: float = 0.2, temperature: float = 0.1):
        super().__init__()
        w = torch.tensor(class_weights) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=w)
        self.lam = contrastive_weight
        self.t = temperature

    def forward(self, logits: torch.Tensor, z: torch.Tensor, labels: torch.Tensor,
                groups: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, labels)
        if self.lam > 0:
            loss = loss + self.lam * supervised_contrastive(z, labels, groups, self.t)
        return loss
