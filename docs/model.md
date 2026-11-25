# Model

The neural reviewer is `acrdl.model.fusion.FusionModel`, fed by a code encoder from
`acrdl.model.encoder`. Default sizes come from the `model` section of
`configs/default.yaml`.

## Inputs and shapes

`B` is the batch size. `K` is the maximum number of behavioural clusters of any bug
in the batch; rows beyond a bug's own cluster count are padding.

| Tensor | Shape | Source |
|---|---|---|
| `texts` | B strings | `acrdl.evidence.code_text.code_text(buggy, patch, spec)` |
| `code` | (B, 768) | encoder output, `model.code_dim` |
| `evidence` | (B, 30) | `Evidence.vector`, `model.evidence_dim` |
| `clusters` | (B, K, 6) | `Evidence.cluster_features`, `model.cluster_feat_dim` |
| `cluster_mask` | (B, K) bool | True marks padding (built by `training.dataset.collate`) |
| `own_index` | (B,) long | index of the patch's own cluster |

## Architecture

With `h = model.hidden_dim = 256`:

```
code (B,768) --Linear(768,256)-GELU-Dropout-------------------> c (B,256)
evidence (B,30) --Linear(30,256)-GELU-Linear(256,256)----------> e (B,256)
clusters (B,K,6) --Linear(6,256)--> TransformerEncoder
                                   2 layers, 4 heads, ffn 512,
                                   key padding mask, NO positional encoding
                                --> tokens (B,K,256) --gather own_index--> a (B,256)

z = concat(c, e, a)                    (B,768)
z = LayerNorm(z * sigmoid(Linear(768,768)(z)))       gated fusion
logits = Linear(768,256)-GELU-Dropout-Linear(256,3)(z)   (B,3)
```

Design choices:

- **No positional encoding in the cluster branch.** Clusters are an unordered set,
  so the branch is permutation-equivariant over clusters.
- **Own cluster as the query.** Self-attention lets every cluster token see the
  competing behaviours. The output token at `own_index` summarises how this patch's
  behaviour relates to the alternatives. A lone outlier cluster with low consensus
  agreement suggests OVERFITTING. Several sizeable, spec-consistent clusters that
  differ only on edge inputs suggest AMBIGUOUS.
- **Gated fusion.** The sigmoid gate lets the model reweight branches per example,
  for instance relying on the code branch when execution evidence is thin.
- **Ablation switches.** `use_code`, `use_evidence` and `use_clusters` zero out a
  branch. The `--ablation` CLI flag maps to these through
  `acrdl.training.train.ABLATIONS`.

## Code encoder

- `CodeEncoder` loads `model.encoder_name` (default `microsoft/unixcoder-base`) and
  freezes all weights except the top `model.encoder_trainable_layers` (4)
  transformer layers.
- It mean-pools the last hidden states over non-padding tokens.
- The encoder learning rate is `training.lr` (2e-5). The fusion model uses
  `training.head_lr` (1e-3).
- `HashingEncoder` is a parameter-free hashed bag-of-tokens stand-in, used for CPU
  smoke tests and offline runs. Select it with `--encoder hashing` (the default) and
  the real encoder with `--encoder hf`.

## Loss

`acrdl.model.losses.FusionLoss`:

```
L = CE(logits, y; class_weights) + lambda * SupCon(z, y, bug)
```

- `class_weights` defaults to `[1.0, 1.0, 2.0]`, upweighting the rarer AMBIGUOUS class.
- `lambda = training.contrastive_weight = 0.1`, `tau = training.contrastive_temperature = 0.1`.
- `supervised_contrastive` only contrasts patches **of the same bug**:
  - positives have the same bug and the same label;
  - negatives have the same bug and a different label;
  - anchors without both are skipped.

  This forces the model to learn what separates candidates of one bug, rather than
  which bugs are easy.
- `acrdl.training.dataset.bug_batches` keeps all candidates of a bug in the same
  batch so these pairs exist.

## Calibration and decision

1. **Temperature scaling** (`acrdl.model.calibration.TemperatureScaler`): a single
   scalar `T`, fitted by LBFGS on validation logits to minimise NLL.
2. **Accept threshold** (`acrdl.decision.fit_accept_threshold`): the smallest
   threshold on calibrated `P(correct)` whose validation false-accept rate is at most
   `decision.target_false_accept_rate` (0.05).
3. **Decision** (`acrdl.decision.decide`):
   - AMBIGUOUS (abstain) if `P(ambiguous)` is the largest;
   - else CORRECT if `P(correct) >= threshold`;
   - else OVERFITTING.

## Checkpoints

`acrdl.training.train.save_checkpoint` writes two files:

- `checkpoint.pt`: model and encoder state dicts and the log-temperature;
- `meta.json`: config, encoder kind, ablation, threshold, temperature and loss
  history.

`load_checkpoint` restores both.
