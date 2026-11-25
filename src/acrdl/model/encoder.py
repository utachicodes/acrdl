"""Code encoders producing one vector per (spec, buggy program, diff) text.

``CodeEncoder`` wraps a pretrained Hugging Face encoder (default UniXcoder)
with only its top ``trainable_layers`` transformer layers unfrozen (4 by
default). ``FrozenCachedEncoder`` keeps the whole encoder frozen and caches
every embedding (in memory and on disk), so each text is encoded once and
reused across epochs, seeds and ablations: the fast option on a laptop.
``HashingEncoder`` is a parameter-free, offline stand-in (hashed bag of
tokens) for tests and CPU smoke runs.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from acrdl.evidence.code_text import code_text

__all__ = ["CodeEncoder", "FrozenCachedEncoder", "HashingEncoder", "code_text"]

_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|\S")


class HashingEncoder(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), self.dim)
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall(t):
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return nn.functional.normalize(out, dim=-1)


class CodeEncoder(nn.Module):
    """Mean-pooled pretrained transformer encoder."""

    def __init__(self, name: str = "microsoft/unixcoder-base", trainable_layers: int = 4,
                 max_length: int = 512):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.max_length = max_length
        self.dim = int(self.model.config.hidden_size)
        for p in self.model.parameters():
            p.requires_grad = False
        layers = getattr(getattr(self.model, "encoder", None), "layer", [])
        top = list(layers)[len(layers) - trainable_layers:] if trainable_layers > 0 else []
        for layer in top:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, texts: list[str]) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        batch = self.tok(texts, padding=True, truncation=True, max_length=self.max_length,
                         return_tensors="pt").to(dev)
        hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class FrozenCachedEncoder(nn.Module):
    """Frozen pretrained encoder with an embedding cache keyed by text hash."""

    def __init__(self, name: str = "microsoft/unixcoder-base",
                 cache_path: Optional[Union[str, Path]] = None, device: str = "cpu",
                 batch_size: int = 16, max_length: int = 512):
        super().__init__()
        self.name = name
        self.cache_path = Path(cache_path) if cache_path else None
        self.device_name = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._enc: Optional[CodeEncoder] = None
        self.cache: dict[str, torch.Tensor] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = torch.load(self.cache_path)
        self.dim = 768

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    def _encoder(self) -> CodeEncoder:
        if self._enc is None:
            self._enc = CodeEncoder(self.name, trainable_layers=0, max_length=self.max_length)
            self._enc.to(self.device_name).eval()
            self.dim = self._enc.dim
        return self._enc

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        keys = [self.key(t) for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.cache]
        if missing:
            enc = self._encoder()
            for i in range(0, len(missing), self.batch_size):
                chunk = missing[i:i + self.batch_size]
                vecs = enc([t for _, t in chunk]).float().cpu()
                for (k, _), v in zip(chunk, vecs):
                    self.cache[k] = v
            self.save_cache()
        return torch.stack([self.cache[k] for k in keys])

    def save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.cache, self.cache_path)
