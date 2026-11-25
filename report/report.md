# Automated Code Review and Program Repair System Using Deep Learning

*Ambiguity-aware assessment of LLM-generated patches*

Deep Learning final project report. Draft of 24 November 2026: sections marked
**[RESULTS PENDING]** are filled in once the experiments finish.

## Abstract

Large language models now write a large share of new code and increasingly propose
fixes for bugs, but the main check applied to those fixes, the test suite, is weak: a
November 2026 study of coding agents found that about a quarter of patches that pass
the tests are wrong. We build a code review and repair system whose reviewer is a
trained neural network. For every LLM-generated patch that passes the visible tests,
the reviewer predicts one of three verdicts: **correct**, **overfitting** (passes the
tests but is wrong) or **spec-ambiguous** (the specification does not determine a
single correct fix, so the system should ask the developer instead of choosing). The
reviewer combines a pretrained code encoder (UniXcoder) with execution evidence
gathered by running every candidate against the buggy program and against the other
candidates on generated inputs, and it attends over groups of candidates that behave
identically. We obtain labels automatically: hidden EvalPlus tests separate correct
from overfitting patches, and *controlled underspecification* (deleting an edge-case
clause and its tests from a specification) creates inputs on which the specification
is known to be silent. On 682 bugs from HumanEvalFix, HumanEval+ and MBPP+, with
patches generated on a laptop by Qwen2.5-Coder-1.5B, **[RESULTS PENDING]**.

## 1. Introduction

Test-based program repair accepts a patch once the failing test passes. Two different
problems hide behind a passing test.

- **Overfitting.** The patch passes the visible tests but breaks intended behaviour,
  for example by hard-coding a test value or changing outputs on inputs that already
  worked.
- **Underspecification.** The specification never says what should happen on some
  input. Consider a function `average(numbers)` that crashes on an empty list. Returning
  `0.0`, returning `None` and raising `ValueError` all make the crash go away, and the
  docstring does not say which is intended. Accepting any of them silently makes a
  design decision on the developer's behalf.

Prior work on patch correctness treats the problem as binary (correct or overfitting).
Work on ambiguity detection clusters sampled programs to detect ambiguous task
descriptions, but only for code generation. Neither separates the two reasons why
candidate patches can disagree: some of them are wrong, or the specification is
silent. That separation is the subject of this project.

**Research questions.**

| ID | Question |
|---|---|
| RQ1 | Does a reviewer that fuses code and execution evidence accept fewer overfitting patches than test-only acceptance, at a comparable rate of keeping correct patches? |
| RQ2 | Can it separate spec ambiguity from wrong patches when candidates disagree, better than a rule that treats any disagreement as ambiguity? |
| RQ3 | Which evidence matters most (code, execution features, cluster structure, contrastive training)? |
| RQ4 | Does the reviewer transfer to bugs the generator cannot have memorised (mutated MBPP+) and to a different benchmark (QuixBugs)? |
| RQ5 | Does the dependence between a candidate's behaviour and meaning-preserving rewordings of the bug report localise the inputs on which the specification is silent, better than disagreement alone? |

**Contributions.**

1. A three-way formulation of patch assessment (correct, overfitting, spec-ambiguous)
   for LLM-generated repairs, without access to a developer reference fix.
2. Controlled underspecification by guard removal and by silent edge classes, which
   yields exact ground truth for where a specification is silent.
3. A neural reviewer with a cluster-attention branch over behaviourally equivalent
   candidates, trained with cross-entropy and supervised contrastive loss and
   calibrated by temperature scaling.
4. A complete, laptop-scale pipeline (data preparation, sandboxed execution, patch
   generation, labelling, training and evaluation) released as the `acrdl` package.

## 2. Background and related work

**Why the problem matters in 2026.** The Stack Overflow Developer Survey 2026 reports
that 83% of developers use at least one AI tool, yet only 6.6% trust AI with important
work decisions and 48% trust its output only when they can easily verify it.
Veracode's July 2026 report found that about 44% of AI code generation tasks produced
code with a known security vulnerability. For repair specifically, a November 2026
study of coding agents on SWE-Gym found that about a quarter of patches that pass the
tests are wrong, and a November 2026 audit reports that 260 of 877 plausible SWE-bench
patches behave differently from the developer's fix, with human reviewers unable to
decide on 51 of 77 sampled cases.

**Patch correctness assessment.** PATCH-SIM (Xiong et al., 2018) compares execution
traces of the buggy and patched programs. ODS uses static features; APPT fine-tunes a
pretrained encoder on the code change; Crex learns embeddings of execution traces;
Invalidator combines inferred invariants with a learned classifier; Prism (OOPSLA 2025)
trains a classifier on differential static-analysis features; RePaCA uses a reasoning
LLM. All predict a binary label.

**Agreement among candidates.** xTestCluster groups patches by their outputs on
generated tests to reduce review effort; PatchFusion (2026) selects patches by how many
candidates agree with them. SpecFix (ASE 2025) and ClarifyGPT detect ambiguous task
descriptions by clustering sampled programs, but for code generation, and they repair
the description rather than judging a patch. Abstain and Validate (2025) abstains on
bugs a repair agent is unlikely to fix, not on ambiguous ones.

**Positioning.** Behaviour preservation and candidate agreement are known signals; we
use them as model inputs, not as contributions. What is new is predicting ambiguity as
its own class in program repair and learning to tell legitimate disagreement (the
specification is silent) from disagreement caused by wrong patches.

## 3. System

The system has four stages; only the last is trained.

1. **Patch generator.** Qwen2.5-Coder-1.5B-Instruct samples 10 candidate repairs per
   bug (temperature 0.8, at most 384 new tokens), prompted with the specification, the
   buggy function and the failing tests in plain words (for example
   `f([]) raises IndexError; should return normally`). It runs on the GPU of an Apple
   M3 Pro laptop through PyTorch's `mps` backend, at about 45 seconds per bug.
2. **Plausibility filter.** Candidates that fail any visible test are discarded.
3. **Evidence.** Every plausible candidate, the buggy program and the other candidates
   are executed in a sandbox (a separate isolated Python process with time and memory
   limits) on the visible test inputs plus about 100 fuzzed inputs. From the resulting
   behaviour matrix we compute a 30-dimensional evidence vector per patch: dynamic
   features (fixes the failing tests, preservation of the buggy program's behaviour on
   inputs where it agrees with the candidates' consensus, share of changed outputs and
   how many of them fall on edge inputs, error and timeout rates), cluster features
   (behavioural clusters of identical outputs, cluster size, entropy, the share of
   disagreement on edge inputs), static overfitting smells of the change (hard-coded
   test literals, special-casing equality checks, removed guards, broad `except`), and
   an optional symbolic check. Each cluster also gets a 6-dimensional feature row.
4. **Neural reviewer** (Section 4).

The output is a calibrated verdict: accept a patch as correct only if
P(correct) exceeds a threshold fitted on validation data for a 5% false-accept rate,
and abstain when P(ambiguous) is the largest, reporting the inputs on which the
candidates disagree.

## 4. Model

**Code branch.** UniXcoder (125M parameters) encodes the specification, the buggy
function and the unified diff of the patch; the mean-pooled final layer is the code
embedding *c* (768 dimensions). On a laptop we keep the encoder frozen and cache its
embeddings, so each patch is encoded once and reused across epochs, seeds and
ablations; fine-tuning the top four layers is supported when a GPU is available.

**Evidence branch.** A two-layer MLP embeds the 30-dimensional evidence vector into *e*.

**Cluster branch.** Each behavioural cluster of the bug becomes one token built from
its feature row. A two-layer Transformer encoder without positional encoding (the
order of clusters is meaningless) contextualises the tokens, and the output token of
the patch's own cluster is the summary *k*. This lets the model compare how the
patch's cluster differs from the others: on edge inputs only, or on ordinary inputs.

**Fusion and head.** A gated fusion of *c*, *e* and *k* feeds a linear layer with a
three-way softmax.

**Loss.** Class-weighted cross-entropy plus a supervised contrastive term over patches
of the same bug:

L = L_CE + 0.1 * L_SupCon

The contrastive term pulls together same-label patches of a bug and pushes apart
different-label ones, so the model must learn what separates them rather than which
bugs are easy. Batches keep all candidates of a bug together.

**Calibration and decision.** Temperature scaling is fitted on the validation split;
the acceptance threshold is chosen on validation for a 5% false-accept rate.

## 5. Data and labels

**Sources.** All data was prepared by `acrdl prepare-data`.

| Set | Bugs | Visible tests per problem | Hidden tests per problem | Origin of the bug |
|---|---|---|---|---|
| HumanEvalFix (HumanEvalPack, Python) | 148 of 164 | 7.0 | 755.9 (HumanEval+) | human-written |
| MBPP+ with injected bugs | 372 | 3.1 | 106.9 (MBPP+) | AST mutation, unseen by any LLM |
| Ambiguous variants | 162 | as source problem | as source problem | controlled underspecification |
| QuixBugs (out of distribution) | 28 of 31 | 3 | 7.7 | classic algorithm bugs |

Hidden expected values come from EvalPlus and are kept only if the reference solution
reproduces them in our sandbox. Problems were skipped when no visible test could be
parsed or too few hidden tests were usable.

**Labelling** (applied to plausible patches only).

1. A patch that fails any hidden test is **overfitting**.
2. A patch that passes every hidden test is **correct**, unless rule 3 applies.
3. If the problem has inputs on which its specification is known to be silent, and the
   hidden-test-passing candidates split into two or more behaviours on those inputs,
   those candidates are **ambiguous**.

**Controlled underspecification.** *Guard removal* deletes one edge-case guard (an `if`
whose body is a single `return` or `raise`) from a correct solution. The inputs whose
behaviour changes are exactly the inputs that guard governed; we delete the docstring
sentences and examples that describe them and every test that covers them, and keep a
crash on one of them as the bug report ("must return normally", with no particular
value required). This reproduces the `average([])` situation: the model has to choose a
behaviour the specification no longer fixes. *Silent edge classes* apply to problems
without a suitable guard: the specification is made silent on one input class (empty
inputs, zero or negative numbers) and the original bug is kept, so whether candidates
disagree there is observed rather than guaranteed. 12 guard-removal and 150
silent-edge variants were produced.

**Manual review.** Every guard-removal variant and every variant flagged by the
automatic check was read by hand, and all variants were scanned for two failure modes
of the automatic spec reduction: removed "sentences" longer than 160 characters (which
turned out to be whole descriptions or definitions) and examples on an ambiguous input
written in a format the stripper does not recognise ("Input: [] Output: []"). Twelve
variants were excluded, each with its reason recorded in `data/excluded_variants.tsv`:
four where the remaining docstring still determines the behaviour, six where the
specification was damaged, one whose ambiguous input is an artefact (a boolean passed
as a number) and one where "balanced or not" determines the answer. Three kept
guard-removal variants are borderline (for example, whether `prime_num(-1)` should
return `False` or raise an error); they are kept and discussed in Section 8. This
leaves 150 ambiguous variants (8 guard-removal, 142 silent-edge).

**Second sampling pass.** A 1.5B model at temperature 0.8 often produces candidates that
agree, so the ambiguous variants were sampled again with 20 candidates at temperature
1.0 under separate bug ids.

**Dataset statistics.** **[RESULTS PENDING: number of candidates, plausible patches and
label counts per set.]**

**Splits.** 70/15/15 by problem, stratified by which rare labels a problem contains,
so that every split sees ambiguous and overfitting patches. Ambiguous variants share
the problem id of their source problem and therefore never straddle two splits.

## 5b. The wording intervention

**Idea.** When candidates disagree on an input, the disagreement has two possible causes:
some candidates are wrong, or the specification is silent there. Disagreement alone cannot
tell them apart. We intervene on the wording of the bug report while keeping its meaning
fixed. If the specification determines the output on input *x*, a meaning-preserving
rewording cannot change what is correct on *x*, so a candidate's behaviour there should not
depend on the wording (mistakes still occur, but at the same rate under every wording). If
the specification is silent on *x*, the generator guesses, and its guess may follow the
wording.

**Rewordings.** Three transformations that preserve meaning by construction, each checked
(the reference still passes every test, the bug still fails a visible test):

| Wording | Transformation | Bugs |
|---|---|---|
| w1 | the buggy function re-printed from its syntax tree (identical program, different text) | 144 |
| w2 | parameters renamed to synonyms in signature, body and docstring (`numbers` to `values`) | 88 |
| w3 | independent docstring sentences after the first reversed, examples kept in place | 37 |

They were applied to the 150 reviewed ambiguous variants and, as controls with a complete
specification, to 50 HumanEvalFix bugs. Each rewording is sampled with exactly the settings
of the original pass (10 candidates, temperature 0.8), so the wording is the only change.

**Statistic.** For input *x*, let *B* be a candidate's behaviour (output signature) and *W*
the wording it was generated from. We measure

I(B; W | x) = H(B | x) - H(B | W, x)

Wrong candidates raise H(B | x) inside every wording; a silent specification adds dependence
between *B* and *W*. A permutation test over wording labels gives a p-value.

**Guarantee.** Assume (A1) that on inputs the specification determines, the wording label is
exchangeable with respect to behaviour. Under A1 the permutation test flags a determined
input with probability at most alpha, the chosen false-alarm rate. Whether silent inputs are
flagged (power) depends on how strongly the generator follows the wording; it is measured on
inputs whose silence is known exactly from controlled underspecification. Both A1 (via the
false-alarm rate on determined inputs) and the power are reported.

**Relation to ReCode.** The closest use of the same tool is ReCode (Wang et al., 2022,
[arXiv:2212.10264](https://arxiv.org/abs/2212.10264)), which applies more than 30
meaning-preserving transformations to code-generation prompts (docstring rewording, renaming,
reformatting) on HumanEval and MBPP and measures the drop in accuracy, treating any change in
output as a lack of robustness of the model. We use the same kind of intervention for the
opposite purpose: a change in behaviour on a particular input is evidence about the
specification, not only about the model. ReCode also gives a warning we take seriously: models
react to such changes even where the answer is fixed, most of all to reformatting. That
violates our exchangeability assumption on determined inputs, so we report the false-alarm
rate on determined inputs for each rewording separately (`per_rewording_vs_original` in the
analysis output) and drop any rewording that raises false alarms on its own.

**Relation to prior work.** Behaviour-based uncertainty for code (Code-MUE, ensemble semantic
entropy) and prompt rewording for LLM uncertainty exist, and SpecFix detects ambiguous task
descriptions from disagreement. To our knowledge no prior work uses meaning-preserving
rewording as a controlled intervention to decide, per input, whether disagreement among repair
candidates reflects wrong patches or a silent specification, with a false-alarm guarantee and
exact ground truth.

**Analysis.** `scripts/wording_analysis.py` reports, on inputs where candidates disagree, the
AUROC of disagreement, of I(B; W) and of 1 minus the p-value for "silent versus determined",
the false-alarm rate and power at alpha = 0.05, and a pooled per-bug version (Fisher's method).

**[RESULTS PENDING]**

## 6. Experimental setup

**Baselines.**

| Baseline | Idea |
|---|---|
| Tests only | Accept every plausible patch (the status quo) |
| Disagreement rule | Ambiguous whenever candidates split into several behaviours; majority cluster correct |
| PATCH-SIM style | Reject patches whose behaviour on stable inputs deviates from the buggy program |
| Static features + gradient boosting | ODS/Prism-style classifier on the evidence vector, no code |
| LLM as judge | The same 1.5B model asked for a one-word three-way verdict |

**Metrics.** Macro-F1 over the three classes; false-accept rate (overfitting patches
accepted as correct, the safety metric); recall of correct patches; abstention
precision and recall on the ambiguous class; one-vs-rest AUROC and expected
calibration error for the model.

**Ablations.** Code branch only; no code branch; no evidence branch; no cluster branch;
no contrastive loss.

**Protocol.** Three seeds, each with its own stratified split. Mean and standard
deviation are reported; the full model is compared with each baseline by a paired
bootstrap over problems (2,000 resamples). Results are also reported per subset:
HumanEvalFix, MBPP+ mutants (contamination check), each kind of ambiguous variant, and
QuixBugs.

**A note on the disagreement baseline.** Because the ambiguous label is defined by
observed disagreement, the disagreement rule recovers part of it by construction. The
learned reviewer adds value only if it separates edge-input disagreement from
overfitting, which is why per-subset results and bugs with both kinds of disagreement
are reported.

## 6b. A lesson on decision thresholds

The first full run exposed a methodological problem worth reporting. The two decision
thresholds (accept at a 5% false-accept rate; abstain above a tuned P(ambiguous)) were tuned on
the validation split, which held only 3 to 10 overfitting or ambiguous patches. The tuned
abstention threshold landed between 0.02 and 0.10 (the model abstained on almost everything) and
the accept threshold between 0.94 and 1.0 (most correct patches were rejected), giving a
seed-to-seed spread of plus or minus 0.20 in macro-F1. We therefore cross-fit: five models are
trained on problem-level folds of train plus validation, calibration and both thresholds are
fitted on their out-of-fold predictions, and the final model is trained on all of train plus
validation; a threshold is fitted only when its class has at least 10 out-of-fold examples.
This cut the spread to plus or minus 0.06. We also report every model with plain argmax
decisions, which is the fair comparison with the baselines (they predict by argmax too), and
the false-accept rate against correct-patch recall as a curve rather than a single point,
because the safe operating point (no overfitting patch accepted) costs about 40% of correct
patches on the ambiguous-variant set.

## 7. Results

**[RESULTS PENDING]** Main table (macro-F1, false-accept rate, correct recall,
abstention precision and recall), per-subset table, ablations, bootstrap tests,
calibration before and after temperature scaling.

### 7.1 A real ambiguous case

After the sentence "Empty sum should be equal to 0 and empty product should be equal to
1" was removed from `sum_product`, nine of ten candidates still returned `(0, 1)` for an
empty list and one raised `ValueError`. The labelling rule marks all ten ambiguous
because they disagree, yet `(0, 1)` is arguably the natural reading. Cases like this
show both why disagreement alone is not enough and where our labels are noisy.

## 8. Discussion and limitations

- **Small generator.** Patches come from a 1.5B model run on a laptop. Larger models
  overfit differently and may disagree less; results may not carry over.
- **Synthetic ambiguity.** Controlled underspecification is a stand-in for real
  ambiguity. Silent-edge labels are noisier than guard-removal labels, because some
  docstrings still imply the edge behaviour and a crashing candidate can be wrong rather
  than an equally valid reading. Results are reported separately for the two.
- **Automatic spec reduction is imperfect.** Manual review removed 12 of 162 variants
  in which the reduced docstring still decided the behaviour or had lost its main
  description. Borderline cases remain, such as negative inputs to `prime_num` or
  `is_perfect_square`, where mathematics suggests `False` but raising an error is also a
  reasonable design.
- **Observed ambiguity.** A bug counts as ambiguous only if the sampled candidates
  happen to disagree; with few samples, some ambiguous bugs are labelled correct.
- **Scope.** Single-function Python bugs; repository-level repair (SWE-bench) is out of
  scope.
- **Contamination.** HumanEval is likely in the generator's training data; the mutated
  MBPP+ subset is reported separately for this reason.
- **Sandbox.** Generated code runs in a separate process with time and memory limits but
  without file-system or network isolation.

## 9. Conclusion

**[RESULTS PENDING]**

## Reproducibility

Everything runs from the repository on an Apple Silicon laptop:

```bash
.venv/bin/pip install -e ".[data,ml,dev]"
.venv/bin/acrdl prepare-data --out data/processed
.venv/bin/acrdl prepare-data --out data/processed --quixbugs data/raw/QuixBugs
scripts/run_generation.sh            # about 10 hours, resumable
scripts/run_diverse_pass.sh          # second pass on ambiguous variants
scripts/run_quixbugs.sh
.venv/bin/python scripts/run_llm_judge.py --data ... --bugs ... --out results/llm_judge.json
.venv/bin/python scripts/run_experiments.py --data ... --llm-judge results/llm_judge.json
```

## References

1. Xiong et al. Identifying Patch Correctness in Test-Based Program Repair. ICSE 2018. arXiv:1706.09120.
2. Ye et al. Automated Classification of Overfitting Patches with Statically Extracted Code Features (ODS). arXiv:1910.12057.
3. Zhang et al. APPT: Boosting Automated Patch Correctness Prediction via Fine-tuning Pre-trained Models. arXiv:2301.12453.
4. Le-Cong et al. Invalidator: Automated Patch Correctness Assessment via Semantic and Syntactic Reasoning. arXiv:2301.01113.
5. Crex: Predicting Patch Correctness in Automated Repair of C Programs through Transfer Learning of Execution Semantics. Information and Software Technology.
6. A Semantic-Based Approach to Overfitting Patch Detection (Prism). OOPSLA 2025.
7. RePaCA: Leveraging Reasoning Large Language Models for Static Automated Patch Correctness Assessment. arXiv:2507.22580.
8. Test-based Patch Clustering for Automatically-Generated Patches Assessment (xTestCluster). arXiv:2207.11082.
9. A Single Patch Is Not Enough: Deterministic Fusion of Repair Candidates (PatchFusion). arXiv:2607.01597.
10. Automated Repair of Ambiguous Problem Descriptions for LLM-Based Code Generation (SpecFix). ASE 2025. arXiv:2505.07270.
11. Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair. arXiv:2510.03217.
12. Wu et al. Patch Correctness in Automated Program Repair: A Five-Dimensional Evidence Framework and Cross-Benchmark Audit. arXiv:2611.10837, November 2026.
13. What Makes Good Data for Training Coding Agents? A Systematic Study on SWE-Gym. arXiv:2611.30207, November 2026.
14. Muennighoff et al. OctoPack: Instruction Tuning Code Large Language Models (HumanEvalPack).
15. Liu et al. Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation (EvalPlus). NeurIPS 2023.
16. Lin et al. QuixBugs: A Multi-Lingual Program Repair Benchmark Set Based on the Quixey Challenge.
17. Guo et al. UniXcoder: Unified Cross-Modal Pre-training for Code Representation. ACL 2022.
18. Khosla et al. Supervised Contrastive Learning. NeurIPS 2020.
19. Stack Overflow Developer Survey 2026, AI section (October 2026).
20. Veracode. 2026 GenAI Code Security Report (July 2026).
21. Wang et al. ReCode: Robustness Evaluation of Code Generation Models. ACL 2023. arXiv:2212.10264.
