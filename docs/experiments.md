# Experiments

## Research questions

- **RQ1.** Does the fused model reject overfitting patches that tests-only repair
  accepts, without rejecting correct ones?
- **RQ2.** Can it separate "candidates disagree because the spec is silent"
  (AMBIGUOUS) from "candidates disagree because some are wrong" (OVERFITTING)?
- **RQ3.** Which evidence matters most: code, execution evidence or cluster structure?
- **RQ4.** Does it generalise to bugs no LLM has seen (mutated MBPP+) and to another
  benchmark (QuixBugs)?

## Baselines

| Baseline | Idea | Where |
|---|---|---|
| Tests-only | accept every plausible patch (status quo) | `acrdl.baselines.tests_only` (`--baseline tests_only`) |
| Disagreement rule | reject consensus outliers and test-literal smells; abstain if the survivors disagree | `acrdl.baselines.disagreement` (`--baseline disagreement`) |
| PATCH-SIM-style | reject patches whose behaviour on stable inputs drifts (preservation < 0.95) | `acrdl.baselines.patch_sim` (`--baseline patch_sim`) |
| Static + GBDT | gradient-boosted trees on the 9 static features (ODS-style) | `acrdl.baselines.static_gbdt` (`--baseline static_gbdt`) |
| Diff-only encoder | FusionModel with only the code branch (APPT-style) | `acrdl train --ablation code_only` |
| LLM judge | ask an LLM for a 3-way verdict | `acrdl.baselines.llm_judge.LLMJudge` |

## Metrics

All metrics are in `acrdl.training.metrics`; `summarize` reports all of them.

| Metric | Why |
|---|---|
| False-accept rate (overfitting patches accepted as correct) | primary safety metric; the threshold targets 5% |
| Correct recall (correct patches accepted) | cost of being strict |
| Macro F1 over the three classes | overall quality |
| Macro one-vs-rest AUROC | threshold-free ranking quality |
| Abstention precision / recall (AMBIGUOUS) | is abstaining justified, and is ambiguity found |
| Expected calibration error | are the probabilities trustworthy |

End to end, the proposal also reports the share of bugs whose top accepted patch is
truly correct.

## Ablations

Run with `acrdl train --ablation NAME`:

| NAME | Removes |
|---|---|
| `code_only` | evidence and cluster branches |
| `no_code` | code branch |
| `no_evidence` | evidence MLP |
| `no_clusters` | cluster Transformer |
| `no_contrastive` | SupCon term (lambda = 0) |

Planned further analyses (not yet scripted):

- per evidence group, by zeroing feature groups at evaluation time;
- by bug type (mutation operator);
- by bug complexity.

## Protocol

- 70/15/15 split by problem (`acrdl.training.dataset.split_by_problem`), stratified by
  which rare labels a problem contains (ambiguous, overfitting, both, neither), so every
  split sees the rare classes. Ambiguous variants share the problem id of their source
  problem, so a problem and its variants never straddle two splits.
- Every configuration is trained with **3 seeds** (`--seed 0` to `--seed 2`). We
  report the mean and standard deviation.
- Significance: **paired bootstrap** (`acrdl.training.metrics.paired_bootstrap`)
  that resamples whole problems, with 2000 resamples. It reports the metric
  difference, a 95% confidence interval and a two-sided p-value.
- The threshold and the temperature are fitted on validation only. The test split is
  used once per configuration.
- Results are reported separately on the original HumanEval/MBPP bugs, the mutated
  MBPP+ bugs (contamination-free) and QuixBugs (out of distribution).

## Running everything

```bash
.venv/bin/python scripts/run_experiments.py --data data/processed/records.jsonl,data/processed/records_underspec_t1.jsonl \
    --out results --seeds 0 1 2 --encoder hf-frozen
```

This trains and scores every baseline and model variant for each seed and writes
`results/results.json` (all numbers) and `results/summary.md` (mean and standard
deviation over seeds, macro-F1 per subset, and a paired bootstrap of the full model
against each baseline). The `hf-frozen` encoder keeps UniXcoder frozen and caches its
embeddings in `data/processed/emb_cache_*.pt`, so each patch is encoded once on CPU and
reused across seeds and ablations. `--encoder hf` fine-tunes the top layers instead.

### A note on the disagreement baseline

The ambiguous label is defined by observed disagreement among hidden-test-passing
candidates on the unspecified inputs, so the disagreement rule recovers part of it by
construction. It is the strongest simple baseline. The question the learned reviewer
must answer is the harder one: separating disagreement on edge inputs the specification
leaves open (ambiguous) from disagreement on ordinary inputs (overfitting). Results are
therefore also reported on bugs whose candidates show both kinds of disagreement.

### Second sampling pass

A 1.5B generator at temperature 0.8 often produces candidates that agree with each
other, so few ambiguous variants show disagreement. `scripts/run_diverse_pass.sh`
samples the ambiguous variants again (20 candidates, temperature 1.0,
`configs/mac_diverse.yaml`) under bug ids tagged `@t1`.

### Figures

```bash
.venv/bin/pip install -e ".[report]"
.venv/bin/python scripts/make_figures.py --data data/processed/records.jsonl \
    --results results/results.json --out report/figures
```

Writes the label distribution per subset, macro-F1 per method, the false-accept
against correct-recall trade-off, the full model's confusion matrix and its
reliability diagram. `results.json` also contains a "candidates disagree (RQ2)"
subset: bugs whose plausible candidates fall into two or more behaviour clusters,
where the reviewer must explain the disagreement.

## Wording intervention (RQ5)

Meaning-preserving rewordings of the bug report (`acrdl.data.wording`: reformat, synonym
rename, sentence reorder) are sampled with the same settings as the original pass.
`acrdl.evidence.wording` measures the mutual information between a candidate's behaviour on
an input and the wording, with a permutation test whose false-alarm rate on inputs the
specification determines is at most alpha under exchangeability.

```bash
.venv/bin/python scripts/prepare_wordings.py          # writes data/processed/wording_w{1,2,3}.jsonl
scripts/run_diverse_pass.sh                           # generates and labels them (resumable)
.venv/bin/python scripts/wording_analysis.py --out results/wording
```

`results/wording/summary.json` compares disagreement and wording dependence as detectors of
specification silence (AUROC on inputs where candidates disagree), and reports the false-alarm
rate on determined inputs and the power on silent ones, per input and pooled per bug.

## Thresholds: cross-fitting

`run_experiments.py` fits calibration and the two decision thresholds with
`acrdl.training.train.train_crossfit`: five problem-level folds over train plus validation,
thresholds from out-of-fold predictions, final model on all of train plus validation. Tuning on
the validation split alone (`--no-crossfit`) is unstable when it holds only a few rare
examples. Every model is also reported as `model:<variant>@argmax`, using the plain argmax rule.
