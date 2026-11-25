# Data

```
data/raw/        downloaded benchmarks (git-ignored)
data/interim/    generated candidates and behaviour matrices (git-ignored)
data/processed/  labelled JSONL records for training (git-ignored)
```

## Fetching the benchmarks

HumanEvalFix and EvalPlus come from the Hugging Face Hub through the `data` extra:

```bash
.venv/bin/pip install -e ".[data]"
.venv/bin/python -c "from acrdl.data.loaders import load_humanevalfix; print(len(load_humanevalfix()))"
.venv/bin/python -c "from acrdl.data.loaders import load_evalplus; print(len(load_evalplus('mbpp')))"
```

- HumanEvalFix: `bigcode/humanevalpack`, config `python`.
- HumanEval+: `evalplus/humanevalplus`. MBPP+: `evalplus/mbppplus`.

QuixBugs is read from a local clone:

```bash
git clone https://github.com/jkoppel/QuixBugs data/raw/QuixBugs
.venv/bin/python -c "from acrdl.data.loaders import load_quixbugs; print(len(load_quixbugs('data/raw/QuixBugs')))"
```

## Building training records

```bash
.venv/bin/acrdl build-dataset --source toy --mutants 2 --out data/processed/toy.jsonl
.venv/bin/acrdl --config configs/default.yaml build-dataset --source humanevalfix \
    --out data/processed/humanevalfix.jsonl
```

`--source humanevalfix` needs `generator.backend: hf` in the config (the `ml` extra
and preferably a GPU). See [../docs/data-and-labeling.md](../docs/data-and-labeling.md)
for the labelling rules.
