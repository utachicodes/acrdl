# Development

## Setup

```bash
make setup                  # .venv + pip install -e ".[dev]"
make setup-ml               # adds torch, transformers, scikit-learn
.venv/bin/pip install -e ".[analysis]"   # crosshair-tool, hypothesis, mypy
.venv/bin/pip install -e ".[data]"       # Hugging Face datasets
```

| Extra | Packages | Needed for |
|---|---|---|
| (core) | numpy, pyyaml, tqdm | pipeline, evidence, heuristic baselines, demo |
| `dev` | pytest, ruff | tests and lint |
| `ml` | torch, transformers, scikit-learn | model, training, HF generator, GBDT baseline |
| `analysis` | hypothesis, crosshair-tool, mypy | CrossHair symbolic check (`evidence.use_symbolic`); hypothesis and mypy are reserved for planned richer fuzzing and type checks |
| `data` | datasets | HumanEvalFix and EvalPlus loaders |
| `all` | all of the above | |

The package imports without any extra. Heavy dependencies are imported lazily inside
the functions that need them.

## Tests and lint

```bash
make test       # pytest; tests/test_model.py is skipped without torch
make lint       # ruff check src tests
make format     # ruff check --fix src tests
```

CI runs lint, tests and `acrdl demo` on Python 3.10 and 3.12; the code also runs on 3.9.

## Adding an evidence collector

1. Implement it under `src/acrdl/evidence/`. It should take sources or a
   `BehaviourMatrix` and return plain numbers per patch.
2. Call it in `Pipeline.run` (`src/acrdl/pipeline.py`) and pass the result to
   `build_evidence`.
3. Append the feature name to `EVIDENCE_NAMES` and compute it in `build_evidence`
   (`src/acrdl/evidence/features.py`). Keep values roughly in [0, 1].
4. Update `model.evidence_dim` in `configs/default.yaml` and the `ModelConfig`
   default in `src/acrdl/config.py`.
5. Add a unit test in `tests/test_evidence.py`. Rebuild datasets: old JSONL records
   have the old vector length.

## Sandbox safety

Candidate patches are LLM output and must be treated as untrusted code.

- Execute code only through `acrdl.evidence.sandbox.run_function` (or helpers built
  on it). Never `exec` a candidate in the main process.
- The sandbox uses a separate `python -I` process, per-input `SIGALRM` timeouts, an
  overall wall-clock timeout and `RLIMIT_AS`/`RLIMIT_DATA` memory limits where the
  OS supports them. macOS ignores some memory limits.
- It does **not** block file system or network access. For large-scale generation
  with real LLMs, run the pipeline inside a container or VM with no network and a
  read-only file system.
- Inputs cross the process boundary as `repr` strings and are rebuilt with
  `ast.literal_eval`, so only literal values are supported.
