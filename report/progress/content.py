"""Text of the progress report, shared by the Word and PDF builders.

Inline markup: **bold**, *italic*, `code`, [link text](url).
Blocks: ("h1", text), ("h2", text), ("p", text), ("bullets", [text]),
("table", header, rows, widths_in, caption, bold_rows), ("figure", path, caption),
("eq", runs), ("lit",), ("refs", [text]).
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
TITLE = "Automated Code Review and Program Repair System Using Deep Learning"
SUBTITLE = "Ambiguity-aware assessment of LLM-generated patches: progress report"
AUTHOR = "Abdoullah Ndao"
COURSE = "CS 4212 · Deep Learning & Neural Networks · Final project progress report"
DATE = "26 November 2026"
LIT = json.loads((HERE / "literature.json").read_text())

ABSTRACT = (
    "AI tools now write a lot of our code, and more and more they also try to fix our bugs. The "
    "problem that got me interested is simple: we usually trust a fix because it passes the tests, "
    "and that is often not enough. A study published this month found that about a quarter of "
    "patches that pass the tests are still wrong. For this project I am building a code reviewer, "
    "a neural network, that looks at every fix an LLM proposes and gives one of three answers: the "
    "fix is *correct*, it is *overfitting* (it passes the tests but is wrong), or the "
    "*specification itself does not say* what the right behaviour is, in which case the reviewer "
    "should ask the developer instead of guessing. So far I have prepared 682 bugs from "
    "HumanEvalFix, HumanEval+ and MBPP+, generated more than a thousand candidate fixes on my own "
    "laptop with Qwen2.5-Coder-1.5B, and trained a first version of the reviewer. On the first 547 "
    "labelled patches it accepted none of the overfitting patches in the test split, while trusting "
    "the tests alone accepts all of them. A simple rule based on disagreement between candidates is "
    "still better at spotting ambiguity, and none of the differences is statistically significant "
    "yet. Along the way I came up with a new way to tell the two kinds of disagreement apart, which "
    "I call the wording test; it is running on my laptop as I write this."
)

BLOCKS = [
    ("h1", "1  Why I Picked This Problem"),
    ("p", "Coding assistants are now everywhere. In the Stack Overflow Developer Survey 2026, 83% of "
          "developers said they use at least one AI tool, but only 6.6% trust it with important "
          "decisions and 48% trust its output only when they can easily check it [19]. That last "
          "number is the one that stuck with me: the real bottleneck is checking. And checking is "
          "weakest exactly where AI is being pushed hardest, in automatic bug fixing, where a fix is "
          "usually accepted as soon as the failing test passes."),
    ("p", "There are two very different ways a fix that passes the tests can still be a problem."),
    ("bullets", [
        "**Overfitting.** The fix games the tests. It might hard-code a test value or quietly break "
        "inputs that used to work. About one in four test-passing agent patches on SWE-Gym is wrong "
        "[13].",
        "**An unclear specification.** Sometimes the description never says what should happen. "
        "Take `average(numbers)`, which crashes on `[]`. Returning `0.0`, returning `None` and "
        "raising `ValueError` all make the crash go away, and the docstring does not say which one "
        "is wanted. Picking one silently is making a design decision for the developer. This is not "
        "rare: human reviewers could not decide on 51 of 77 sampled SWE-bench patches that behaved "
        "differently from the developer's fix [12].",
    ]),
    ("p", "What makes this hard is that both cases look the same from the outside: the candidate "
          "fixes disagree with each other. So the questions I am trying to answer are:"),
    ("bullets", [
        "**RQ1.** Can a reviewer that combines code and execution evidence accept fewer overfitting "
        "fixes than \"tests pass\", without throwing away good ones?",
        "**RQ2.** When the candidates disagree, can it tell \"some fixes are wrong\" apart from "
        "\"the specification is silent\", better than a rule that calls every disagreement "
        "ambiguous?",
        "**RQ3.** Which kind of evidence matters most?",
        "**RQ4.** Does it still work on bugs the model cannot have memorised (freshly mutated "
        "MBPP+) and on a different benchmark (QuixBugs)?",
        "**RQ5.** Does the way a model's answer changes when I reword the bug report, without "
        "changing its meaning, show where the specification is silent?",
    ]),
    ("p", "**What I am contributing.** (1) A three-way way of judging fixes (correct, overfitting, "
          "or ambiguous) that needs no reference fix from a developer. (2) *Controlled "
          "underspecification*, a way to build ambiguous bugs on purpose so I know exactly where the "
          "specification is silent. (3) A neural reviewer with a branch that looks at groups of "
          "candidates that behave the same way. (4) The wording test (Section 5). (5) The whole "
          "pipeline as a working package, `acrdl`, that runs on a laptop."),

    ("h1", "2  What I Read"),
    ("lit",),

    ("h1", "3  How the System Works"),
    ("figure", "figure1_architecture.png",
     "**Figure 1.** The system has four stages and only the last one is trained. The cluster "
     "branch (blue) is the main new piece of the architecture."),
    ("p", "**Generating fixes.** I use Qwen2.5-Coder-1.5B-Instruct to write 10 candidate fixes per "
          "bug. I do not have access to a cloud GPU, so it runs on my laptop's Apple M3 Pro GPU, at "
          "about 45 seconds per bug. The prompt gives the model the specification, the buggy "
          "function and the failing tests written out in words (for example "
          "`f([]) raises IndexError; should return normally`)."),
    ("p", "**Collecting evidence.** Every fix that passes the visible tests is run, together with "
          "the buggy program and the other candidates, on the visible inputs plus about 100 "
          "generated inputs. All of this happens in a separate, isolated Python process with time "
          "and memory limits, because LLM-written code should never run inside my own program. "
          "From the outputs I compute a 30-number evidence vector per fix (does it keep the old "
          "behaviour where the old behaviour was fine, how many outputs it changes and whether "
          "those are edge cases, error rates, how the candidates cluster, and code smells such as "
          "hard-coded test values) and a 6-number summary of every group of candidates that behave "
          "identically."),
    ("p", "**The model.** UniXcoder reads the specification, the buggy code and the diff of the "
          "fix. A small MLP reads the evidence vector. A two-layer Transformer, with no positional "
          "encoding because the groups have no order, reads the behaviour groups and asks how the "
          "fix's own group compares with the others. A gated fusion combines the three and outputs "
          "three probabilities. I train it with class-weighted cross-entropy plus a supervised "
          "contrastive loss over fixes of the same bug, so the model has to learn what separates a "
          "good fix from a bad one for the same problem:"),
    ("eq", [("L", True, False), (" = ", False, False), ("L", True, False), ("CE", False, True),
            (" + 0.1 · ", False, False), ("L", True, False), ("SupCon", False, True)]),
    ("p", "After training I calibrate the probabilities with temperature scaling. The reviewer "
          "accepts a fix only when it is confident enough to keep the false-accept rate at 5% on "
          "validation data, and it says \"ambiguous\" when that probability passes a second "
          "threshold tuned on validation data."),

    ("h1", "4  Data and Labels"),
    ("table", ["Set", "Bugs", "Visible tests", "Hidden tests"],
     [["HumanEvalFix [14] + HumanEval+ [15]", "148", "7.0", "755.9"],
      ["MBPP+ with bugs I injected", "372", "3.1", "106.9"],
      ["Ambiguous variants (after my review)", "150", "as source", "as source"],
      ["QuixBugs [16] (different benchmark)", "28", "3.0", "7.7"]],
     [3.0, 0.8, 1.1, 1.1],
     "**Table 2.** The data I prepared; test counts are per problem. I only kept a hidden test if "
     "the reference solution reproduced it in my sandbox.", ()),
    ("p", "**Labels, without hand-labelling.** A fix that fails any hidden test is *overfitting*. "
          "A fix that passes everything is *correct*, unless its problem has inputs where I know "
          "the specification is silent and the passing candidates behave differently on them, in "
          "which case they are *ambiguous*."),
    ("p", "**Building ambiguous bugs on purpose.** To know exactly where a specification is silent, "
          "I create the silence myself. *Guard removal* takes a correct solution and deletes one "
          "edge-case check (an `if` that just returns or raises). The inputs whose behaviour "
          "changes are exactly the ones that check handled. I delete the parts of the docstring "
          "and the examples that describe those inputs, remove every test on them, and keep one "
          "crash as the bug report, which recreates the `average([])` situation. *Silent edge "
          "classes* do the same for empty, zero or negative inputs while keeping the original bug."),
    ("p", "I did not fully trust the automatic version, so I read the variants myself. 12 of 162 "
          "were broken: in some the docstring still gave the answer away, in others my script had "
          "cut out the whole description, and one used a boolean as a number. I excluded those and "
          "wrote down why. Data is split 70/15/15 by problem, so a problem and its variants never "
          "end up in both training and testing."),

    ("h1", "5  A New Idea: the Wording Test"),
    ("p", "While looking at real cases I kept getting stuck on the same question. When the "
          "candidates disagree, is it because some of them are wrong, or because the "
          "specification never said? Counting disagreement cannot tell. One real example from my "
          "data: after I removed the sentence \"Empty sum should be equal to 0 and empty product "
          "should be equal to 1\" from `sum_product`, nine of ten candidates still returned "
          "`(0, 1)` on `[]` and one raised an error. Is that an ambiguous specification, or one bad "
          "fix?"),
    ("p", "My idea is to treat this as an experiment. I reword the bug report *without changing "
          "what it means*, generate new candidates, and see whether their answers move."),
    ("bullets", [
        "If the specification decides what `f(x)` should be, rewording it cannot change the right "
        "answer, so the candidates' behaviour on `x` should not depend on the wording. Mistakes "
        "still happen, but equally often under every wording.",
        "If the specification is silent on `x`, the model is guessing, and a guess can follow small "
        "changes in the wording.",
    ]),
    ("p", "So instead of asking \"do the candidates disagree?\", I ask \"does their behaviour "
          "depend on wording that should not matter?\" I measure this as the mutual information "
          "between a candidate's behaviour *B* on an input and the wording *W* it was generated "
          "from:"),
    ("eq", [("I", True, False), ("(B; W | x) = H(B | x) − H(B | W, x)", False, False)]),
    ("p", "Wrong fixes add noise inside every wording; a silent specification shows up as a "
          "difference between wordings. I use three rewordings that cannot change the meaning, and "
          "I check every one automatically (the reference solution must still pass every test and "
          "the bug must still show):"),
    ("table", ["Rewording", "What changes", "Bugs"],
     [["w1, reformat", "the buggy function is re-printed from its syntax tree: same program, "
       "different text", "144"],
      ["w2, rename", "parameters get synonyms everywhere, e.g. `numbers` to `values`", "88"],
      ["w3, reorder", "independent docstring sentences after the first appear in reverse order", "37"]],
     [1.3, 4.0, 0.7],
     "**Table 3.** The rewordings, applied to the 150 ambiguous variants and to 50 HumanEvalFix "
     "bugs with complete specifications as controls (269 reworded bugs).", ()),
    ("p", "**What I can actually guarantee.** If, on inputs the specification decides, the wording "
          "really cannot matter (my one assumption), then a permutation test over the wording "
          "labels wrongly flags such an input as ambiguous at most 5% of the time. I checked this "
          "with a simulation in my test suite. What I cannot guarantee is that the model's guesses "
          "actually follow the wording; that is the experiment, and because I built the "
          "ambiguity myself, I know exactly which inputs are silent and can measure it precisely."),
    ("p", "**How new is it?** I searched for this specifically. People already measure "
          "uncertainty from how differently sampled programs behave (Code-MUE, ensemble semantic "
          "entropy), reword prompts to measure LLM uncertainty in general, and inject ambiguity "
          "into code generation tasks. I did not find anyone using a meaning-preserving rewording "
          "as a controlled experiment to decide, input by input, whether disagreement among repair "
          "candidates comes from wrong fixes or from a silent specification. So I would call it new "
          "to my knowledge, not something nobody could have thought of."),
    ("p", "**Status.** The rewordings are built and checked, the analysis code is tested (on fake "
          "data with identical wordings it correctly finds no dependence and no false alarms), and "
          "the generation run is scheduled for the night of 25 November."),

    ("h1", "6  Where I Am"),
    ("table", ["Part", "Status"],
     [["Reading and novelty check (18 papers)", "done"],
      ["Sandbox, input fuzzing, behaviour clustering, code smells", "done, tested"],
      ["Data: 682 bugs, ambiguous variants, manual review", "done"],
      ["Neural reviewer, losses, calibration, decision thresholds", "done, tested"],
      ["Five baselines and the experiment runner (3 seeds, bootstrap)", "done"],
      ["Wording test: rewordings, statistic, analysis", "done, tested"],
      ["Generating fixes for all bugs", "running on my laptop"],
      ["Full experiments and final report", "26 November to 3 December"]],
     [4.4, 1.9],
     "**Table 4.** Status on 25 November 2026. The `acrdl` package has 45 modules, about 4,300 "
     "lines of Python and 51 passing tests.", ()),
    ("p", "Not everything went smoothly. My first overnight run died when my development session "
          "closed, and running other GPU work at the same time slowed generation from 45 seconds to "
          "more than 13 minutes per bug. Everything now runs detached and saves after every bug, so "
          "nothing was lost, but it cost me most of a day."),

    ("h1", "7  First Results"),
    ("p", "The first 111 bugs (all ambiguous variants) gave 1,110 candidate fixes. 568 of them (51%) "
          "pass the visible tests, and 547 are labelled after my exclusions: 488 correct, 22 "
          "overfitting and 37 ambiguous, from 80 bugs that had at least one passing fix (Table 5)."),
    ("table", ["Variant type", "Bugs", "Correct", "Overfitting", "Ambiguous"],
     [["Guard removal", "6", "27", "3", "5"],
      ["Silent edge class", "74", "461", "19", "32"],
      ["Total", "80", "488", "22", "37"]],
     [2.2, 0.8, 1.0, 1.0, 1.0],
     "**Table 5.** Labels so far. The two rare classes are still small.", (2,)),
    ("table", ["Method", "Macro-F1", "False accepts", "Correct kept", "Ambiguous caught"],
     [["Tests only (what people do now)", "0.31", "100%", "100%", "0%"],
      ["PATCH-SIM style", "0.31", "100%", "100%", "0%"],
      ["Static features + gradient boosting", "0.31", "89%", "90%", "3%"],
      ["Disagreement rule", "0.60", "33%", "98%", "100%"],
      ["My neural reviewer", "0.38", "0%", "74%", "40%"]],
     [2.4, 0.9, 1.0, 1.0, 1.1],
     "**Table 6.** First results on the test split, averaged over 3 random seeds. \"False "
     "accepts\" is the share of overfitting fixes accepted as correct, so lower is safer.", (4,)),
    ("p", "**RQ1.** The reviewer accepted none of the overfitting fixes in the test split, while "
          "tests alone accepted all of them. The price is that it also rejected about a quarter of "
          "the correct fixes. Each test split only has a handful of overfitting fixes, so I see "
          "this as an early signal, not a result."),
    ("p", "**RQ2.** The disagreement rule is ahead on ambiguity. Part of that is built in: my "
          "ambiguous label is defined by disagreement, so the rule gets it right whenever the "
          "candidates only disagree on the silent inputs. My reviewer has to win on the harder "
          "cases, where both kinds of disagreement happen at once, and so far only 9 bugs have any "
          "overfitting at all. This is exactly why I came up with the wording test."),
    ("table", ["Version of the model", "Macro-F1", "False accepts"],
     [["Full model", "0.38", "0%"],
      ["Without the code branch", "0.48", "67%"],
      ["Without the contrastive loss", "0.41", "33%"],
      ["Without the cluster branch", "0.40", "0%"],
      ["Code branch only", "0.29", "67%"],
      ["Without the evidence branch", "0.22", "67%"]],
     [3.0, 1.0, 1.1],
     "**Table 7.** What happens when I remove parts of the model (3 seeds).", (0,)),
    ("p", "**RQ3.** Execution evidence matters most: without it macro-F1 drops to 0.22. One result "
          "surprised me. With only about 380 training examples, the model does better on macro-F1 "
          "without the code encoder, although it then loses its zero false-accept rate. I think "
          "the 768-dimensional code embedding is too much for this little data, and I expect it to "
          "change with the full dataset. I will report it either way."),
    ("p", "**Honest summary.** A paired bootstrap over problems shows no significant difference "
          "between my model and any baseline yet (p between 0.23 and 0.78). These numbers tell me "
          "where to push, not what the final answer is."),

    ("h1", "8  What Is Left Until 3 December"),
    ("bullets", [
        "**25 to 26 November.** Finish generating fixes for all 682 bugs and QuixBugs, then run the "
        "wording test overnight.",
        "**27 to 28 November.** Analyse the wording test, run the LLM-as-judge baseline and the "
        "full experiments with all baselines and ablations, and report results separately for "
        "HumanEvalFix, the mutated MBPP+ bugs and QuixBugs.",
        "**29 to 30 November.** If the wording test works, feed it into the reviewer as a new "
        "input; adjust the model using validation data only.",
        "**1 to 3 December.** Final report and presentation.",
    ]),

    ("h1", "9  Risks and Limitations"),
    ("bullets", [
        "**Few ambiguous examples.** A small model often agrees with itself. If the class stays "
        "small, I will report per-bug results and confidence intervals instead of overclaiming.",
        "**The wording test might not work.** A 1.5B model's guesses may simply not follow the "
        "wording. That would still be a clear finding, and the 5% false-alarm guarantee holds "
        "either way.",
        "**Synthetic ambiguity.** My ambiguous bugs are built on purpose. The silent-edge ones are "
        "noisier than the guard-removal ones, so I report them separately.",
        "**Small model, small programs.** Everything uses a 1.5B model on single Python functions, "
        "so the results may not carry over to bigger models or whole repositories.",
        "**Memorisation.** HumanEval is probably in the model's training data, which is why I "
        "report the freshly mutated MBPP+ bugs separately.",
    ]),
]

REFS = [
    "Xiong et al. Identifying Patch Correctness in Test-Based Program Repair. ICSE 2018.",
    "Ye et al. Automated Classification of Overfitting Patches with Statically Extracted Code "
    "Features. arXiv:1910.12057.",
    "Zhang et al. APPT: Boosting Automated Patch Correctness Prediction via Fine-tuning Pre-trained "
    "Models. arXiv:2301.12453.",
    "Le-Cong et al. Invalidator: Automated Patch Correctness Assessment via Semantic and Syntactic "
    "Reasoning. arXiv:2301.01113.",
    "Crex: Predicting Patch Correctness in Automated Repair of C Programs through Transfer Learning "
    "of Execution Semantics. Information and Software Technology.",
    "A Semantic-Based Approach to Overfitting Patch Detection (Prism). OOPSLA 2025.",
    "RePaCA: Leveraging Reasoning Large Language Models for Static Automated Patch Correctness "
    "Assessment. arXiv:2507.22580.",
    "Test-based Patch Clustering for Automatically-Generated Patches Assessment. arXiv:2207.11082.",
    "A Single Patch Is Not Enough: Deterministic Fusion of Repair Candidates. arXiv:2607.01597.",
    "Automated Repair of Ambiguous Problem Descriptions for LLM-Based Code Generation (SpecFix). "
    "ASE 2025.",
    "Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair. "
    "arXiv:2510.03217.",
    "Wu et al. Patch Correctness in Automated Program Repair: A Five-Dimensional Evidence Framework "
    "and Cross-Benchmark Audit. arXiv:2611.10837, November 2026.",
    "What Makes Good Data for Training Coding Agents? A Systematic Study on SWE-Gym. "
    "arXiv:2611.30207, November 2026.",
    "Muennighoff et al. OctoPack: Instruction Tuning Code Large Language Models (HumanEvalPack).",
    "Liu et al. Is Your Code Generated by ChatGPT Really Correct? (EvalPlus). NeurIPS 2023.",
    "Lin et al. QuixBugs: A Multi-Lingual Program Repair Benchmark Set.",
    "Guo et al. UniXcoder: Unified Cross-Modal Pre-training for Code Representation. ACL 2022.",
    "Khosla et al. Supervised Contrastive Learning. NeurIPS 2020.",
    "Stack Overflow Developer Survey 2026, AI section. October 2026.",
] + [p["cite"] + ". [" + p["link_text"] + "](" + p["url"] + ")" for p in LIT["groups"][0]["papers"]] + [
    "Code-MUE: Measuring Code LLMs' Uncertainty through Execution-based Semantic Interaction "
    "Graphs. [arXiv:2607.12273](https://arxiv.org/abs/2607.12273)",
    "Predicting Program Correctness By Ensemble Semantic Entropy. "
    "[arXiv:2603.27098](https://arxiv.org/abs/2603.27098)",
    "Clarity Is Not Assumed: Understanding LLM-Based Code Generation under Ambiguous Requirements. "
    "[arXiv:2604.21505](https://arxiv.org/abs/2604.21505)",
]
