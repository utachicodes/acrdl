# Data and labelling

## Datasets

| Dataset | Role | Loader |
|---|---|---|
| Toy benchmark (5 problems) | offline demo and tests | `acrdl.data.loaders.load_toy` |
| HumanEvalFix (HumanEvalPack, Python) | 164 real injected bugs with tests | `acrdl.data.loaders.load_humanevalfix` |
| EvalPlus HumanEval+ / MBPP+ | extended hidden test inputs (about 80x more than the originals) | `acrdl.data.loaders.load_evalplus(benchmark="humaneval" or "mbpp")` |
| Mutated MBPP+ | extra bugs from AST mutation operators, unseen by any LLM | `acrdl.data.mutate.mutate` on MBPP+ references |
| QuixBugs (Python) | out-of-distribution test set | `acrdl.data.loaders.load_quixbugs(root)` from a local clone |

The `data` extra (`pip install -e ".[data]"`) provides the Hugging Face `datasets`
library for the HumanEvalFix and EvalPlus loaders. EvalPlus provides inputs without
outputs. `acrdl.data.labeling.materialize_expected` turns them into tests by running
the reference solution in the sandbox.

## Bug injection

`acrdl.data.mutate` implements five deterministic single-site operators:

| Operator | Example |
|---|---|
| `comparison_flip` | `<` to `<=`, `==` to `!=` |
| `arith_swap` | `+` to `-`, `/` to `//` |
| `off_by_one` | integer constant `k` to `k + 1` |
| `negate_condition` | `if c:` to `if not c:` |
| `return_change` | `return expr` to `return None` (booleans are flipped) |

`mutate(source, n, seed)` samples `n` distinct, non-trivial mutants with a seeded RNG.
`acrdl build-dataset --mutants N` adds `N` mutants per problem.

## Preparing the real data

`acrdl prepare-data` (module `acrdl.data.prepare`) does every step that needs no LLM
and writes JSON Lines files to `data/processed/`:

| File | Contents |
|---|---|
| `humanevalfix.jsonl` | HumanEvalFix bugs. Visible tests: the original HumanEval asserts. Hidden tests: HumanEval+ inputs and expected values (up to `--max-hidden`, default 1000, per problem) |
| `mbpp_mutants.jsonl` | MBPP+ problems with `--mbpp-mutants` (default 1) injected bugs each. Only mutants that fail a visible test are kept |
| `underspec.jsonl` | ambiguous variants (next section) |
| `underspec_review.md` | every variant with its reduced docstring, for a manual check |
| `stats.json` | counts, skip reasons and average test sizes |

`acrdl prepare-data --quixbugs data/raw/QuixBugs` prepares the out-of-distribution set
from a local clone of QuixBugs (`quixbugs.jsonl`): the spec is the program's module
docstring, all JSON test cases are hidden tests, and the visible tests are one test the
bug fails plus two others.

Hidden expected values come from EvalPlus. A hidden test is kept only if the
reference solution reproduces it in our sandbox and its input and output survive a
`repr` round trip. `acrdl.data.prepare.load_prepared(path)` reloads any of the files.

Patch generation then runs on these files:

```bash
.venv/bin/acrdl --config configs/mac.yaml build-dataset --source prepared \
    --bugs data/processed/humanevalfix.jsonl,data/processed/underspec.jsonl \
    --out data/processed/records.jsonl --resume
```

`--resume` appends one bug at a time and skips bugs already processed, so a long run
can be stopped and restarted.

## Controlled underspecification

Ground-truth AMBIGUOUS labels need inputs on which the specification is known to be
silent. `acrdl.data.prepare` creates them in two ways.

**Guard removal** (`underspecify_by_guard`). Take a correct solution and delete one
edge-case guard: an `if` whose body is a single `return` or `raise` (an `else` branch
is kept). The inputs whose behaviour changes are exactly the inputs that guard
governed. Then:

1. remove the docstring sentences and examples that describe those inputs;
2. remove every visible and hidden test on those inputs;
3. record them as `problem.ambiguous_inputs`;
4. add one crash report as a visible test: the input must "return normally"
   (`acrdl.evidence.testing.NO_CRASH`), with no particular value required.

This mirrors the `average([])` example: the program crashes on an input the
specification no longer covers, so the repair model has to choose a behaviour. A
variant is kept only if the removed guard makes the program crash on at least one
input and governs at most 25% of the tests (an edge case, not core logic).

**Silent edge class** (`underspecify_by_edge_class`). For problems without a suitable
guard, pick an input class the tests use (empty containers or strings, zero, or
negative numbers), remove the docstring text, examples and tests that cover it, and
keep the original bug. Whether the candidates then disagree on that class is observed
at labelling time, not guaranteed.

Silent-edge labels are noisier than guard-removal labels. Some docstrings still imply
the edge behaviour logically (for example "are any two numbers closer than the
threshold" implies `False` for an empty list), and a candidate that crashes there may
be wrong rather than an equally valid reading. Results are therefore reported
separately for the two variant types, and `underspec_review.md` supports a manual
check.

**Manual review and exclusions.** `data/excluded_variants.tsv` lists variants removed
after reading them by hand, with a reason each: the reduced docstring still decides the
behaviour (for example an "Input: [] Output: []" example the stripper does not
recognise), the sentence stripper removed a whole description or definition, or the
ambiguous input is an artefact. `scripts/run_experiments.py` and
`scripts/make_figures.py` drop these bug ids (and their `@t1` second-pass copies) by
default; pass `--exclude ''` to keep them.

The older `acrdl.data.underspec.underspecify(problem, clause, covers)` removes a given
docstring clause and the tests matching a predicate. It is used by the toy benchmark.

## Labelling rule

`acrdl.data.labeling.label_patches` labels **plausible** patches (patches that pass
every visible test):

1. **OVERFITTING**: the patch fails at least one hidden test.
2. **AMBIGUOUS**: all three of the following hold:
   - the patch passes all hidden tests;
   - the problem has `ambiguous_inputs`;
   - the hidden-test-passing candidates split into two or more behavioural clusters
     on those inputs.

   Every such cluster is consistent with the reduced spec.
3. **CORRECT**: otherwise.

Ambiguity is labelled when it is observed among the candidates. If all candidates
happen to agree on the ambiguous inputs, they are CORRECT with respect to every
reading the generator produced.

## Records and splits

`acrdl build-dataset` writes one JSON line per labelled plausible patch
(`acrdl.training.dataset.Record`):

| Field | Content |
|---|---|
| `problem_id` | source problem id; underspecified variants share it with the original |
| `bug_id`, `patch_id` | identifiers |
| `code` | encoder input: spec, buggy function, unified diff (`acrdl.evidence.code_text`) |
| `evidence` | 30 floats in `EVIDENCE_NAMES` order |
| `clusters` | K rows of 6 floats in `CLUSTER_FEATURE_NAMES` order |
| `own_cluster` | index of the patch's cluster |
| `label` | 0 CORRECT, 1 OVERFITTING, 2 AMBIGUOUS |

`split_by_problem` makes a **70/15/15 train/validation/test split by problem**. Every
patch of a problem, including its mutants and underspecified variants, lands in
exactly one split. This prevents leakage between near-identical patches.

## Contamination control

HumanEval and MBPP are probably in the training data of any modern code LLM. All
results are reported separately on the mutated MBPP+ bugs, which no model can have
seen.
