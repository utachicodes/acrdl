"""LLM-as-judge baseline: ask the generator model for a 3-way verdict on test patches.

Usage (after patch generation has finished, so the GPU is free):
    .venv/bin/python scripts/run_llm_judge.py \\
        --data data/processed/records.jsonl,data/processed/records_underspec_t1.jsonl \\
        --bugs data/processed/humanevalfix.jsonl,data/processed/mbpp_mutants.jsonl,\\
data/processed/underspec.jsonl --seeds 0 1 2 --out results/llm_judge.json

Only patches that land in a test split for one of the seeds are judged. The
output maps patch id to the predicted label; ``run_experiments.py --llm-judge``
scores it like any other baseline. The file is updated as it goes, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from acrdl.baselines.llm_judge import judge_prompt, parse_verdict
from acrdl.config import load_config
from acrdl.data.rebuild import base_bug_id, load_bug_index, patch_source
from acrdl.training.dataset import load_jsonl, split_by_problem
from acrdl.types import Patch


def make_complete(model_name: str, cache_dir: str | None, max_new_tokens: int = 8):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, cache_dir=cache_dir)
    model.to(device).eval()

    @torch.no_grad()
    def complete(prompt: str) -> str:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt").to(device)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    return complete


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--bugs", required=True, help="prepared bug files, comma separated")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--config", default="configs/mac.yaml")
    ap.add_argument("--out", default="results/llm_judge.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    records = [r for f in args.data.split(",") for r in load_jsonl(f.strip())]
    wanted = {}
    for seed in args.seeds:
        for r in split_by_problem(records, seed=seed)[2]:
            wanted[r.patch_id] = r
    bugs = load_bug_index([f.strip() for f in args.bugs.split(",")])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, int] = json.loads(out.read_text()) if out.exists() else {}
    todo = [r for pid, r in wanted.items() if pid not in done]
    print(f"{len(wanted)} test patches, {len(done)} already judged, {len(todo)} to go", flush=True)

    complete = make_complete(cfg.generator.model_name, cfg.generator.cache_dir)
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        bug = bugs[base_bug_id(r.bug_id)]
        patch = Patch(r.patch_id, bug, patch_source(r.code, r.bug_id, bugs))
        done[r.patch_id] = int(parse_verdict(complete(judge_prompt(patch))))
        if i % 25 == 0 or i == len(todo):
            out.write_text(json.dumps(done))
            rate = (time.time() - t0) / i
            print(f"[{i}/{len(todo)}] {rate:.2f}s per patch", flush=True)
    out.write_text(json.dumps(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
