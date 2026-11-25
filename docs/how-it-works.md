# How it works: one bug end to end

This walkthrough follows the `average` problem from the toy benchmark
(`src/acrdl/data/toy.py`). Run it yourself with `acrdl demo --task toy/average`.

## 0. The problem and the bug

Full specification:

```python
def average(numbers):
    """Return the arithmetic mean of a list of numbers. Return 0.0 for an empty list."""
```

Injected bug (skips the first element):

```python
def average(numbers):
    if not numbers:
        return 0.0
    return sum(numbers[1:]) / len(numbers)
```

To get a ground-truth AMBIGUOUS case, the demo applies **controlled
underspecification** (`acrdl.data.underspec.underspecify`):

- it deletes the clause `Return 0.0 for an empty list.`;
- it removes every test whose input is `[]`;
- it records `ambiguous_inputs = [([],)]`.

The task id becomes `toy/average~underspec`. The remaining visible tests are
`average([1, 2]) == 1.5` and `average([0, 4]) == 2.0`.

## 1. Generation

`MockGenerator` returns five candidates, standing in for LLM samples:

| id | patch idea |
|---|---|
| c0 | `if not numbers: return 0.0`, then `sum / len` |
| c1 | same behaviour, written with an explicit loop |
| c2 | plain `sum(numbers) / len(numbers)` (raises on `[]`) |
| c3 | `if not numbers: return None`, then `sum / len` |
| c4 | `if numbers == [1, 2]: return 1.5`, otherwise the buggy code |

## 2. Plausibility filter

All five pass both visible tests, so all five are plausible. Tests-only repair would
accept all of them, including c4.

## 3. Evidence

**Inputs.** The two visible inputs plus fuzzed lists built from the example
elements. Edge cases come first, including `[]`. With `--fuzz 120` this gives 98
unique inputs after deduplication.

**Behaviour matrix.** Each cell is a behaviour signature, for example:

| input | buggy | c0 | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|---|
| `[1, 2]` | `ok:1.0` | `ok:1.5` | `ok:1.5` | `ok:1.5` | `ok:1.5` | `ok:1.5` |
| `[0, 4]` | `ok:2.0` | `ok:2.0` | `ok:2.0` | `ok:2.0` | `ok:2.0` | `ok:2.0` |
| `[]` | `ok:0.0` | `ok:0.0` | `ok:0.0` | `error:ZeroDivisionError` | `ok:None` | `ok:0.0` |
| `[4, 1]` | `ok:0.5` | `ok:2.5` | `ok:2.5` | `ok:2.5` | `ok:2.5` | `ok:0.5` |

**Consensus and preservation.**

- The consensus per input is the majority signature: `ok:0.0` on `[]`, the true mean
  elsewhere.
- Stable inputs are those where the buggy program already matches the consensus:
  `[]`, `[0, 4]` and other lists starting with 0.
- c2 and c3 change behaviour on `[]`, which is a stable input, so their preservation
  is 0.89.
- c4 keeps the buggy behaviour almost everywhere, so its preservation is 1.00 but its
  consensus agreement is only 0.10.

**Clusters.** Four behavioural clusters: `{c0, c1}`, `{c2}`, `{c3}`, `{c4}`.

**Static smells.** c4 adds `numbers == [1, 2]`, and `[1, 2]` is a visible-test input,
so `hardcoded_test_literal = 1`.

**Vectors.** Each patch gets a 30-dim evidence vector and a 4 x 6 matrix of cluster
rows, with its own cluster flagged.

## 4. Verdicts

The demo has no trained model, so it uses the disagreement rule
(`acrdl.baselines.disagreement.HeuristicPredictor`). A trained `FusionModel` replaces
it through `acrdl.training.evaluate.ModelPredictor`.

| id | verdict | why |
|---|---|---|
| c0, c1 | AMBIGUOUS | spec-consistent, but other spec-consistent candidates behave differently |
| c2, c3 | AMBIGUOUS | they disagree with the majority only on one edge input |
| c4 | OVERFITTING | disagrees with the consensus on 90% of inputs and hard-codes a test input |

The report then prints the witness:

```
spec ambiguous at input ([],):
  ok:0.0                       <- c0, c1
  error:ZeroDivisionError      <- c2
  ok:None                      <- c3
```

This is the output a code reviewer needs. It names the exact question the developer
has to answer ("what should `average([])` do?") instead of silently shipping one
guess.

## Same bug, full spec

`acrdl demo --task toy/average --no-underspec` keeps the clause and the `[]` tests:

- c2 and c3 fail the visible test `average([]) == 0.0` and are filtered out;
- c0 and c1 are CORRECT;
- c4 is still OVERFITTING;
- no ambiguity is reported.

## Labels for training

`acrdl.data.labeling.label_patches` assigns ground truth to the same patches using
the hidden tests and `ambiguous_inputs`:

- c4 fails hidden tests: OVERFITTING;
- c0 to c3 pass every hidden test and split into three behaviours on `[]`: AMBIGUOUS.

The model learns to reproduce these labels from evidence alone, without hidden tests.
