# Automated Code Review and Program Repair System Using Deep Learning

[![CI](https://github.com/utachicodes/acrdl/actions/workflows/ci.yml/badge.svg)](https://github.com/utachicodes/acrdl/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-53%20passing-brightgreen)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Course](https://img.shields.io/badge/CS%204212-Deep%20Learning%20%26%20Neural%20Networks-8a2be2)

Abdoullah Ndao, CS 4212 Deep Learning & Neural Networks, final project (2026).

An LLM proposes fixes for a buggy Python function. Many of those fixes pass the
visible tests and are still wrong. This system acts as an automated code reviewer
for those candidate patches. It runs every plausible patch against the buggy
program and against the other candidates on hundreds of generated inputs, adds
static and symbolic evidence, and a neural model then puts each patch into one of
three classes: **CORRECT**, **OVERFITTING** (passes the tests but is wrong) or
**AMBIGUOUS** (the specification does not decide which behaviour is right, so the
system abstains and shows the input where the spec is silent).

Progress report: [report/progress/](report/progress/) (PDF and Word).

## The problem

Test-based program repair accepts any patch that makes the tests pass. That has two
failure modes:

1. **Plausible but wrong patches.** A patch can special-case the failing test input
   (`if numbers == [1, 2]: return 1.5`) or break behaviour that used to work. The
   tests stay green.
2. **Underspecified bugs.** Take

   ```python
   def average(numbers):
       """Return the arithmetic mean of a list of numbers."""
   ```

   What should `average([])` do? Return `0.0`, return `None`, or raise
   `ZeroDivisionError`? The docstring does not say. Candidate patches that differ only
   on `[]` are each consistent with the spec, so picking one is a guess. The right
   output of a reviewer is "the spec is ambiguous at input `[]`", not a verdict.

Earlier work on patch correctness (PATCH-SIM, ODS, APPT, Invalidator, Prism) handles
failure mode 1. It does not separate "candidates disagree because some are wrong"
from "candidates disagree because the spec is silent". This project focuses on that
separation.

## How it works

```
 buggy program + spec + visible tests
                 |
   +-------------v--------------+
   | 1. Patch generation        |  LLM samples N candidates (acrdl.generation)
   +-------------+--------------+
                 |  keep patches that pass every visible test
   +-------------v--------------+
   | 2. Plausibility filter     |  sandboxed test run (acrdl.evidence.testing)
   +-------------+--------------+
                 |
   +-------------v-----------------------------------------------+
   | 3. Evidence collection (acrdl.evidence)                     |
   |    sandbox     run untrusted code in an isolated subprocess |
   |    inputs      visible inputs + type-guided fuzzing         |
   |    differential behaviour matrix: buggy vs every candidate  |
   |    clustering  behavioural equivalence classes              |
   |    static      overfitting smells in the diff               |
   |    symbolic    optional CrossHair divergence search         |
   |    -> 30-dim evidence vector + one feature row per cluster  |
   +-------------+-----------------------------------------------+
                 |
   +-------------v--------------+
   | 4. Neural reviewer         |  code encoder + evidence MLP + cluster
   |    (acrdl.model)           |  Transformer -> gated fusion -> 3-way softmax
   +-------------+--------------+
                 |
        ACCEPT (CORRECT) | REJECT (OVERFITTING) | ABSTAIN (AMBIGUOUS + witness input)
```

Until a model is trained, stage 4 uses a transparent rule
(`acrdl.baselines.disagreement.HeuristicPredictor`), so the whole pipeline runs
offline out of the box.

### The three verdicts

| Verdict | Meaning | Decision rule (trained model) |
|---|---|---|
| CORRECT | Patch matches the intended behaviour | `P(correct) >= threshold`, threshold fitted for a 5% false-accept rate |
| OVERFITTING | Passes the visible tests but is wrong | otherwise |
| AMBIGUOUS | Spec does not determine one behaviour; abstain | `P(ambiguous)` is the largest probability |

## Quickstart

Requires Python 3.9 or newer.

```bash
make setup        # creates .venv and installs the package with dev tools
make test         # runs the test-suite
make demo         # same as: .venv/bin/acrdl demo
```

Sample output (`acrdl demo --task toy/average`):

```
[toy/average~underspec] bug toy/average#0
  note: an edge-case clause was removed from this spec (controlled underspecification)
  candidates: 5   plausible: 5   inputs executed: 98   behaviour clusters: 4
  patch  plausible  preserve  agree cluster smell  verdict
  c0     yes            1.00   1.00       0     0  AMBIGUOUS
  c1     yes            1.00   1.00       0     0  AMBIGUOUS
  c2     yes            0.89   0.99       1     0  AMBIGUOUS
  c3     yes            0.89   0.99       2     0  AMBIGUOUS
  c4     yes            1.00   0.10       3     1  OVERFITTING
  spec ambiguous at input ([],):
    ok:0.0                       <- c0, c1
    error:ZeroDivisionError      <- c2
    ok:None                      <- c3
```

`c4` special-cases the visible test input `[1, 2]` and is rejected. The other four
are each consistent with the reduced spec, and the system reports the exact input
where they disagree instead of choosing one.

To train the neural reviewer, install the `ml` extra:

```bash
make setup-ml
.venv/bin/acrdl build-dataset --source toy --mutants 2 --out data/processed/toy.jsonl
.venv/bin/acrdl train --data data/processed/toy.jsonl --epochs 5 --out runs/toy
.venv/bin/acrdl evaluate --data data/processed/toy.jsonl --checkpoint runs/toy --split all
```

The toy benchmark has only 5 problems, so it is a smoke test, not an experiment.

### Real data on a Mac (no CUDA GPU)

`configs/mac.yaml` runs patch generation with Qwen2.5-Coder-1.5B-Instruct on the Apple
Silicon GPU (PyTorch `mps` backend).

```bash
.venv/bin/pip install -e ".[data,ml]"
.venv/bin/acrdl prepare-data --out data/processed        # downloads data, builds bugs and variants
.venv/bin/acrdl --config configs/mac.yaml build-dataset --source prepared \
    --bugs data/processed/humanevalfix.jsonl,data/processed/underspec.jsonl \
    --out data/processed/records.jsonl --resume
```

See [docs/data-and-labeling.md](docs/data-and-labeling.md) for what each step produces.

## CLI reference

Global option: `--config PATH` (YAML, defaults to the built-in values in
`acrdl.config`).

| Command | Purpose | Options |
|---|---|---|
| `acrdl demo` | Offline end-to-end run on the toy benchmark | `--task PREFIX`, `--fuzz N` (default 120), `--no-underspec` |
| `acrdl prepare-data` | Download HumanEvalFix, HumanEval+ and MBPP+; build bugs and ambiguous variants (no LLM) | `--out DIR`, `--cache-dir DIR`, `--max-hidden N`, `--mbpp-mutants N`, `--max-silent N`, `--limit N`, `--quixbugs CLONE_DIR` (QuixBugs only) |
| `acrdl build-dataset` | Generate candidates, collect evidence, label, write JSONL | `--source {toy,humanevalfix,prepared}`, `--bugs FILES`, `--out PATH`, `--resume`, `--mutants N`, `--limit N` |
| `acrdl train` | Train the fusion model (needs `ml` extra) | `--data PATH`, `--encoder {hashing,hf,hf-frozen}`, `--ablation {full,code_only,no_code,no_evidence,no_clusters,no_contrastive}`, `--epochs N`, `--seed N`, `--out DIR` |
| `acrdl evaluate` | Evaluate a checkpoint or a baseline | `--data PATH`, `--checkpoint DIR` or `--baseline {tests_only,disagreement,patch_sim,static_gbdt}`, `--split {all,train,val,test}`, `--seed N` |

## Configuration

`configs/default.yaml` lists every key; `configs/mac.yaml` is the Apple Silicon setup. Sections:

| Section | Keys |
|---|---|
| top level | `seed` |
| `generator` | `backend` (`mock` or `hf`), `model_name`, `n_candidates`, `temperature`, `max_new_tokens`, `cache_dir` |
| `evidence` | `n_fuzz_inputs`, `exec_timeout_s`, `memory_limit_mb`, `use_symbolic`, `symbolic_timeout_s` |
| `model` | `encoder_name`, `encoder_trainable_layers`, `code_dim`, `evidence_dim`, `cluster_feat_dim`, `hidden_dim`, `n_heads`, `cluster_layers`, `dropout`, `n_classes` |
| `training` | `batch_size`, `lr` (encoder), `head_lr` (fusion model), `epochs`, `contrastive_weight`, `contrastive_temperature`, `class_weights`, `output_dir` |
| `decision` | `target_false_accept_rate` |

Unknown keys are rejected, so typos fail fast.

## Repository map

```
configs/default.yaml          all hyperparameters
configs/mac.yaml              Apple Silicon setup (1.5B generator on mps)
configs/mac_diverse.yaml      second, more diverse sampling pass for ambiguous variants
scripts/                      run_generation.sh, run_diverse_pass.sh, run_quixbugs.sh,
                              run_experiments.py, run_llm_judge.py, make_figures.py
data/                         raw/ interim/ processed/ (git-ignored contents)
docs/                         architecture, walkthrough, data, model, experiments, development
src/acrdl/
  types.py                    Problem, BuggyProgram, Patch, ExecResult, Evidence, Label
  config.py                   YAML -> typed dataclasses
  pipeline.py                 end-to-end orchestration (stages 1 to 4)
  decision.py                 accept / reject / abstain rule and threshold fitting
  cli.py                      acrdl command
  data/                       toy benchmark, loaders, mutation, underspecification, labelling
  generation/                 prompts, MockGenerator, HFGenerator
  evidence/                   sandbox, testing, inputs, differential, clustering,
                              static, symbolic, features, code_text
  model/                      encoder, fusion, losses, calibration (torch)
  training/                   dataset, train, evaluate, metrics
  baselines/                  tests_only, disagreement, patch_sim, static_gbdt, llm_judge
tests/                        pytest suite (torch tests skip without the ml extra)
```

## Testing

```bash
make test    # pytest
make lint    # ruff
```

Without the `ml` extra, the model tests are skipped. With it installed, the suite
also trains a small model end to end on the toy data. CI
(`.github/workflows/ci.yml`) runs lint, tests and the demo on Python 3.10 and 3.12 (the code also runs on 3.9).

## Project status

| Component | Status |
|---|---|
| Sandbox, fuzzing, differential matrix, clustering, static smells | implemented and tested |
| Labelling (overfitting / correct / ambiguous), mutation, underspecification | implemented and tested |
| Fusion model, SupCon loss, temperature scaling, decision threshold | implemented and tested on CPU |
| Toy benchmark and offline demo | implemented |
| Real data preparation (HumanEvalFix, HumanEval+, MBPP+, ambiguous variants) | implemented and tested; see `acrdl prepare-data` |
| HF patch generator | implemented; runs on CUDA, Apple Silicon (`mps`) or CPU |
| UniXcoder encoder | implemented, needs the `ml` extra |
| CrossHair symbolic check | optional; returns `unknown` when not installed |
| LLM-judge baseline | prompt and parser implemented; plug in any completion function |
| Full-scale experiments | not run yet (see [docs/experiments.md](docs/experiments.md)) |

## Documentation

- [docs/architecture.md](docs/architecture.md): components and data flow
- [docs/how-it-works.md](docs/how-it-works.md): one bug end to end
- [docs/data-and-labeling.md](docs/data-and-labeling.md): datasets, labels, splits
- [docs/model.md](docs/model.md): neural reviewer and tensor shapes
- [docs/experiments.md](docs/experiments.md): baselines, metrics, ablations
- [docs/development.md](docs/development.md): setup, extending, sandbox safety

## License

MIT
