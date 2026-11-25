"""YAML configuration loaded into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class GeneratorConfig:
    backend: str = "mock"
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    n_candidates: int = 10
    temperature: float = 0.8
    max_new_tokens: int = 512
    cache_dir: Optional[str] = None  # where Hugging Face model weights are stored


@dataclass
class EvidenceConfig:
    n_fuzz_inputs: int = 200
    exec_timeout_s: float = 2.0
    memory_limit_mb: int = 512
    use_symbolic: bool = False
    symbolic_timeout_s: float = 10.0


@dataclass
class ModelConfig:
    encoder_name: str = "microsoft/unixcoder-base"
    encoder_trainable_layers: int = 4
    code_dim: int = 768
    evidence_dim: int = 30
    cluster_feat_dim: int = 6
    hidden_dim: int = 256
    n_heads: int = 4
    cluster_layers: int = 2
    dropout: float = 0.1
    n_classes: int = 3


@dataclass
class TrainingConfig:
    batch_size: int = 16
    lr: float = 2e-5
    head_lr: float = 1e-3
    epochs: int = 10
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.1
    # "balanced": inverse class frequency on the training split (capped at 20)
    class_weights: Union[str, list[float]] = "balanced"
    output_dir: str = "runs/default"


@dataclass
class DecisionConfig:
    target_false_accept_rate: float = 0.05


@dataclass
class Config:
    seed: int = 13
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: Union[str, Path, None] = None) -> Config:
    """Load a config file; missing sections fall back to defaults."""
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sections = {
        "generator": GeneratorConfig,
        "evidence": EvidenceConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "decision": DecisionConfig,
    }
    kwargs: dict[str, Any] = {k: _build(c, raw.get(k) or {}) for k, c in sections.items()}
    extra = set(raw) - set(sections) - {"seed"}
    if extra:
        raise ValueError(f"Unknown top-level config keys: {sorted(extra)}")
    return Config(seed=int(raw.get("seed", 13)), **kwargs)
