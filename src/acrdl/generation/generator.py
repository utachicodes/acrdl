"""Patch generators.

``MockGenerator`` replays scripted candidates (offline demo and tests).
``HFGenerator`` samples from a Hugging Face causal LM (``ml`` extra, GPU
recommended); the model is loaded lazily on first use.
"""

from __future__ import annotations

from typing import Optional, Protocol

from acrdl.generation.prompts import extract_code, repair_prompt
from acrdl.types import BuggyProgram, Patch


class PatchGenerator(Protocol):
    def generate(self, bug: BuggyProgram, n: int, failing: Optional[list[str]] = None
                 ) -> list[Patch]: ...


class MockGenerator:
    """Returns pre-scripted candidate sources keyed by ``task_id``."""

    def __init__(self, scripted: dict[str, list[str]]):
        self.scripted = scripted

    def generate(self, bug: BuggyProgram, n: int, failing: Optional[list[str]] = None
                 ) -> list[Patch]:
        key = bug.problem.task_id.split("~")[0]
        sources = self.scripted.get(key, [])[:n]
        return [Patch(f"{bug.bug_id}/c{i}", bug, s) for i, s in enumerate(sources)]


def _fence_stop(tokenizer, prompt_len: int):
    """Per-sample stopping rule: stop once a fenced code block has opened and closed."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class FenceStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            texts = tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)
            return torch.tensor([t.count("```") >= 2 for t in texts],
                                dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([FenceStop()])


class HFGenerator:
    """Sample repairs from a Hugging Face instruction-tuned code model."""

    def __init__(self, model_name: str, temperature: float = 0.8, max_new_tokens: int = 512,
                 cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tok = None
        self._fp32 = False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on extras
            raise ImportError("HFGenerator needs the 'ml' extra: pip install -e '.[ml]'") from e
        import torch

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():  # Apple Silicon GPU
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype,
                                                           cache_dir=self.cache_dir)
        self._model.to(device).eval()

    def generate(self, bug: BuggyProgram, n: int, failing: Optional[list[str]] = None
                 ) -> list[Patch]:
        self._load()
        assert self._tok is not None and self._model is not None
        messages = [{"role": "user", "content": repair_prompt(bug, failing or [])}]
        ids = self._tok.apply_chat_template(messages, add_generation_prompt=True,
                                            return_tensors="pt").to(self._model.device)
        texts: list[str] = []
        batch = n
        while len(texts) < n:
            k = min(batch, n - len(texts))
            try:
                texts += self._sample(ids, k)
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg and batch > 1:  # retry in smaller batches
                    batch = max(1, batch // 2)
                elif "probability tensor" in msg and not self._fp32:
                    # 16-bit overflow produced inf/nan logits: switch to 32-bit maths
                    import torch

                    self._model.to(torch.float32)
                    self._fp32 = True
                else:
                    raise
            finally:
                self._free()
        return [Patch(f"{bug.bug_id}/c{i}", bug, extract_code(t)) for i, t in enumerate(texts)]

    def _sample(self, ids, k: int) -> list[str]:
        import torch

        assert self._tok is not None and self._model is not None
        with torch.no_grad():
            outs = self._model.generate(
                ids, attention_mask=torch.ones_like(ids), do_sample=True,
                temperature=self.temperature, max_new_tokens=self.max_new_tokens,
                num_return_sequences=k, pad_token_id=self._tok.eos_token_id,
                # stop a sample once its code block closes; extra prose only costs time
                stopping_criteria=_fence_stop(self._tok, ids.shape[1]),
            )
        return self._tok.batch_decode(outs[:, ids.shape[1]:], skip_special_tokens=True)

    @staticmethod
    def _free() -> None:
        """Release cached GPU memory (PyTorch keeps freed blocks reserved otherwise)."""
        import gc

        import torch

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
