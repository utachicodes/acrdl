"""Evidence-fusion classifier (the neural reviewer).

Inputs per patch (B = batch, K = max clusters per bug):
  * ``code``         (B, code_dim)  code embedding of (spec, buggy program, diff)
  * ``evidence``     (B, E)         fixed-length evidence vector, E = 30
  * ``clusters``     (B, K, C)      one feature row per behavioural cluster, C = 6
  * ``cluster_mask`` (B, K)         True marks padding rows
  * ``own_index``    (B,)           index of the patch's own cluster

Branches:
  * code branch: linear projection of the encoder embedding;
  * evidence branch: 2-layer MLP;
  * cluster branch: 2-layer Transformer encoder over the cluster set with no
    positional encoding (clusters are an unordered set); the output token of
    the patch's own cluster is the query that summarises how it relates to
    the competing behaviours.
The three vectors are concatenated, passed through a sigmoid gate and a
LayerNorm, and a 2-layer head emits logits over {CORRECT, OVERFITTING, AMBIGUOUS}.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from acrdl.config import ModelConfig


class FusionModel(nn.Module):
    def __init__(self, cfg: ModelConfig, use_code: bool = True, use_evidence: bool = True,
                 use_clusters: bool = True):
        super().__init__()
        h = cfg.hidden_dim
        self.use_code, self.use_evidence, self.use_clusters = use_code, use_evidence, use_clusters
        self.code_proj = nn.Sequential(nn.Linear(cfg.code_dim, h), nn.GELU(),
                                       nn.Dropout(cfg.dropout))
        self.ev_mlp = nn.Sequential(nn.Linear(cfg.evidence_dim, h), nn.GELU(),
                                    nn.Linear(h, h))
        self.cluster_in = nn.Linear(cfg.cluster_feat_dim, h)
        layer = nn.TransformerEncoderLayer(h, cfg.n_heads, dim_feedforward=2 * h,
                                           dropout=cfg.dropout, batch_first=True)
        self.cluster_tf = nn.TransformerEncoder(layer, cfg.cluster_layers,
                                                enable_nested_tensor=False)
        self.gate = nn.Linear(3 * h, 3 * h)
        self.norm = nn.LayerNorm(3 * h)
        self.head = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(h, cfg.n_classes))

    def forward(self, code: torch.Tensor, evidence: torch.Tensor, clusters: torch.Tensor,
                cluster_mask: Optional[torch.Tensor] = None,
                own_index: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits (B, 3), fused embedding (B, 3h))``."""
        b = code.shape[0]
        c = self.code_proj(code)
        e = self.ev_mlp(evidence)
        tokens = self.cluster_tf(self.cluster_in(clusters), src_key_padding_mask=cluster_mask)
        if own_index is None:
            own_index = torch.zeros(b, dtype=torch.long, device=code.device)
        a = tokens[torch.arange(b, device=code.device), own_index]
        if not self.use_code:
            c = torch.zeros_like(c)
        if not self.use_evidence:
            e = torch.zeros_like(e)
        if not self.use_clusters:
            a = torch.zeros_like(a)
        z = torch.cat([c, e, a], -1)
        z = self.norm(z * torch.sigmoid(self.gate(z)))
        return self.head(z), z
