# Architecture

The system has four stages. Each stage below names the module that implements it.
Orchestration is in `src/acrdl/pipeline.py` (`Pipeline.run`).

```
             +-------------------+
 BuggyProgram|  1. Generation    |  acrdl.generation.generator
 (spec, code,|  N candidates     |  MockGenerator | HFGenerator
  tests)     +---------+---------+
                       |
             +---------v---------+
             | 2. Plausibility   |  acrdl.evidence.testing.run_tests
             |    filter         |  (sandboxed)
             +---------+---------+
                       | plausible patches only
   +-------------------v----------------------------------------------+
   | 3. Evidence                                                      |
   |   inputs.build_inputs --> differential.build_matrix --+          |
   |                                                       v          |
   |   static.static_features   clustering.cluster   (consensus,      |
   |   symbolic.diff_behavior        |                preservation)   |
   |            \                    |                    /           |
   |             +---------> features.build_evidence <---+            |
   |                          30-dim vector + K x 6 cluster rows      |
   +-------------------+----------------------------------------------+
                       |
             +---------v---------+
             | 4. Neural reviewer|  acrdl.model.fusion.FusionModel
             |    + decision     |  acrdl.decision.decide
             +---------+---------+
                       v
           CORRECT | OVERFITTING | AMBIGUOUS (+ witness inputs)
```

## Stage 1: patch generation

- `acrdl.generation.prompts.repair_prompt` builds the prompt: spec, buggy code and
  failing tests. `extract_code` pulls the fenced code block out of the completion.
- `HFGenerator` samples `generator.n_candidates` completions from
  `generator.model_name` at `generator.temperature`. It loads the model lazily
  (`ml` extra).
- `MockGenerator` replays scripted candidates. The offline demo and the tests use it.

## Stage 2: plausibility filter

`acrdl.evidence.testing.run_tests` runs each candidate on the visible tests inside
the sandbox. A patch is **plausible** if it passes all of them. The filter also
records:

- `fail_to_pass`: the patch fixes at least one visible test the buggy program fails;
- `visible_fixed_frac`: the fraction of initially failing visible tests it fixes.

Only plausible patches go on to stage 3. They are the patches that test-based repair
would ship.

## Stage 3: evidence

### Sandbox (`acrdl.evidence.sandbox`)

Candidate code is untrusted, so it never runs in-process. `run_function`:

- spawns `python -I`;
- sets `RLIMIT_AS`/`RLIMIT_DATA` to `evidence.memory_limit_mb` where the OS allows it;
- arms a per-input `SIGALRM` timer (`evidence.exec_timeout_s`) and an overall
  wall-clock timeout;
- exchanges JSON over stdin/stdout.

Each input yields an `ExecResult` with signature `ok:<repr>`, `error:<ExceptionName>`
or `timeout:`.

### Inputs (`acrdl.evidence.inputs`)

`build_inputs` starts from the visible-test inputs and adds `evidence.n_fuzz_inputs`
fuzzed inputs. `fuzz_inputs` infers a shape template for each argument position
(int, float, str, list, tuple, bool) and samples new values. Edge cases come first:
empty containers, `0`, `1`, `-1`, `""`. Inputs are deduplicated by `repr`.

### Differential matrix (`acrdl.evidence.differential`)

`build_matrix` runs the buggy program and every plausible candidate on every input
and stores a `BehaviourMatrix`. Reference-free signals derived from it:

| Function | Signal |
|---|---|
| `consensus` | majority behaviour per input (None on ties) |
| `stable_inputs` | inputs where the buggy program already agrees with the consensus, i.e. the bug does not manifest |
| `preservation_score` | share of stable inputs where the patch keeps the buggy program's behaviour |
| `consensus_agreement` | share of inputs where the patch matches the consensus |
| `changed_fraction`, `outcome_rates` | behaviour change vs buggy; error and timeout rates |

### Clustering (`acrdl.evidence.clustering`)

`cluster` groups candidates whose behaviour signatures are identical on every input.
Clusters are ordered by size. The result also records the **disagreement inputs**
and the normalised entropy of the cluster sizes. Multiple clusters mean the
candidates disagree. Stage 4 has to decide whether that is because some are wrong
or because the spec is silent.

### Static smells (`acrdl.evidence.static`)

`static_features` compares the buggy and patched ASTs and counts constructs that are
new in the patch:

- equality checks against literals (`special_case_eq`), and those whose literal comes
  from a visible test (`hardcoded_test_literal`);
- removed `raise`/`assert` (`removed_guards`);
- new bare or `Exception` handlers (`added_broad_except`);
- new constant returns and new branches;
- diff size and AST size change.

### Symbolic check (`acrdl.evidence.symbolic`)

When `evidence.use_symbolic` is true, `diff_behavior` runs CrossHair `diffbehavior`
in a subprocess on the buggy and patched functions. It returns `diff`,
`no_diff_found` or `unknown`. It returns `unknown` if CrossHair (`analysis` extra) is
missing or times out. It works best on type-annotated code.

### Feature assembly (`acrdl.evidence.features`)

`build_evidence` produces, per patch, an `Evidence` object containing:

- `vector`: 30 features named by `EVIDENCE_NAMES`:
  - 11 dynamic;
  - 9 cluster-level;
  - 9 static;
  - 1 symbolic.
- `cluster_features`: one row per behavioural cluster of the same bug, 6 features
  named by `CLUSTER_FEATURE_NAMES`. The last feature flags the patch's own cluster.
- `cluster_id`: index of the patch's own cluster.

## Stage 4: neural reviewer

Full details and tensor shapes are in [model.md](model.md).

```
 spec + buggy + diff --> UniXcoder (top 4 layers fine-tuned) --> Linear --> c (256)
 evidence (30) -------------------------> MLP ------------------------> e (256)
 cluster rows (K x 6) --> Linear --> 2-layer Transformer, no positions
                                     --> token of own cluster --------> a (256)
                     [c ; e ; a] (768) --> sigmoid gate --> LayerNorm
                                       --> MLP --> logits (3) --> softmax / T
```

| Piece | Module |
|---|---|
| Code encoder (UniXcoder, top `model.encoder_trainable_layers` = 4 layers trainable; `HashingEncoder` offline stand-in) | `acrdl.model.encoder` |
| Fusion model (evidence MLP, cluster Transformer with `model.cluster_layers` = 2, gated fusion, 3-way head) | `acrdl.model.fusion` |
| Loss: weighted cross-entropy + `training.contrastive_weight` (0.1) x supervised contrastive over same-bug pairs | `acrdl.model.losses` |
| Temperature scaling on validation logits | `acrdl.model.calibration` |
| Accept threshold for `decision.target_false_accept_rate` (5%) and the abstain rule | `acrdl.decision` |
| Training loop with bug-grouped batches, checkpointing | `acrdl.training.train` |
| Pipeline adapter for a trained model | `acrdl.training.evaluate.ModelPredictor` |

Decision rule (`acrdl.decision.decide`):

1. abstain (AMBIGUOUS) if `P(ambiguous)` is the largest;
2. accept (CORRECT) if `P(correct) >= threshold`;
3. reject (OVERFITTING) otherwise.

`fit_accept_threshold` picks the smallest threshold whose validation false-accept
rate is at or below 5%.

Before a model is trained, `Pipeline` falls back to
`acrdl.baselines.disagreement.HeuristicPredictor`.

## Outputs

`Pipeline.run` returns a `BugReport` containing:

- one `PatchReport` per candidate: plausibility, evidence, static features, verdict
  and probabilities;
- the behaviour matrix and clustering;
- `ambiguity_witnesses`: every input on which the non-rejected candidates disagree,
  with the candidates grouped by behaviour.
