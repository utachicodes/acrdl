"""Post-hoc temperature scaling (Guo et al., 2017)."""

from __future__ import annotations

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100
            ) -> TemperatureScaler:
        """Fit T on held-out logits by minimising NLL."""
        logits, labels = logits.detach(), labels.detach()
        opt = torch.optim.LBFGS([self.log_t], lr=0.1, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = nll(self(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self
