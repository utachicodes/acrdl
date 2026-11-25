"""Run every baseline and model variant over several seeds and write a results table.

Usage:
    .venv/bin/python scripts/run_experiments.py --data data/processed/records.jsonl \\
        --out results --seeds 0 1 2 --encoder hf-frozen

For each seed the data is split 70/15/15 by problem (stratified, see
``split_by_problem``). Baselines that need training (static gradient boosting)
and every model variant are fit on train, calibrated on validation and scored
on test. Writes ``results.json`` (every number) and ``summary.md`` (mean and
standard deviation over seeds, per-subset scores and paired bootstrap tests
of the full model against each baseline on the first seed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from acrdl.config import load_config
from acrdl.decision import decide
from acrdl.training.dataset import Record, bug_batches, load_jsonl, split_by_problem
from acrdl.training.metrics import macro_f1, paired_bootstrap, summarize

BASELINES = ["tests_only", "disagreement", "semantic_entropy", "patch_sim", "static_gbdt"]
VARIANTS = ["full", "code_only", "no_code", "no_evidence", "no_clusters", "no_contrastive"]
HEADLINE = ["macro_f1", "false_accept_rate", "correct_recall", "abstain_precision",
            "abstain_recall", "accuracy"]


def subset_of(r: Record) -> str:
    if "~underspec" in r.bug_id:
        return "ambiguous: guard removal"
    if "~silent" in r.bug_id:
        return "ambiguous: silent edge"
    if r.bug_id.startswith("Mbpp/"):
        return "MBPP+ mutants (unseen bugs)"
    return "HumanEvalFix"


def ordered(records: list[Record], batch_size: int) -> list[Record]:
    """Records in the order the model scores them (bug-grouped)."""
    return [r for chunk in bug_batches(records, batch_size, shuffle=False) for r in chunk]


def load_exclusions(path: str) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    return {line.split("\t", 1)[0].strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def baseline_preds(name: str, train: list[Record], test: list[Record], seed: int,
                   judged: dict | None = None) -> np.ndarray:
    if name == "llm_judge":
        return np.array([judged[r.patch_id] for r in test])
    if name == "semantic_entropy":
        from acrdl.baselines.semantic_entropy import SemanticEntropy

        return SemanticEntropy().fit(train).predict_records(test)
    if name == "static_gbdt":
        from acrdl.baselines.static_gbdt import StaticGBDT

        return StaticGBDT(seed).fit(train).predict_records(test)
    import importlib

    return importlib.import_module(f"acrdl.baselines.{name}").predict_records(test)


def per_subset(y: np.ndarray, pred: np.ndarray, test: list[Record]) -> dict[str, dict]:
    out = {}
    subsets = np.array([subset_of(r) for r in test])
    for s in sorted(set(subsets)):
        m = subsets == s
        out[s] = {"n": int(m.sum()), **summarize(y[m], pred[m])}
    # RQ2: bugs whose plausible candidates fall into two or more behaviour clusters.
    # Here disagreement must be explained: wrong patches or a silent specification.
    m = np.array([len(r.clusters) >= 2 for r in test])
    if m.any():
        out["candidates disagree (RQ2)"] = {"n": int(m.sum()), **summarize(y[m], pred[m])}
    return out


def confusion(y: np.ndarray, pred: np.ndarray, k: int = 3) -> list[list[int]]:
    """Rows: true label (correct, overfitting, ambiguous); columns: predicted."""
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(y, pred):
        cm[int(a), int(b)] += 1
    return cm.tolist()


def reliability(y: np.ndarray, probs: np.ndarray, bins: int = 10) -> list[dict]:
    """Confidence vs accuracy per bin, for a reliability diagram."""
    conf, guess = probs.max(1), probs.argmax(1)
    out = []
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.any():
            out.append({"bin": (i + 0.5) / bins, "confidence": float(conf[m].mean()),
                        "accuracy": float((guess[m] == y[m]).mean()), "n": int(m.sum())})
    return out


def fmt(vals: list[float]) -> str:
    v = [x for x in vals if x == x]  # drop NaN
    if not v:
        return "n/a"
    return f"{np.mean(v):.3f} +/- {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="records JSONL (comma separated for several)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--encoder", default="hf-frozen", choices=["hashing", "hf", "hf-frozen"])
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-crossfit", action="store_true",
                    help="tune thresholds on the validation split instead of cross-fitting")
    ap.add_argument("--exclude", default="data/excluded_variants.tsv",
                    help="bug ids removed after manual review (TSV, first column); '' for none")
    ap.add_argument("--llm-judge", default=None,
                    help="verdicts from scripts/run_llm_judge.py, scored as a baseline")
    args = ap.parse_args(argv)

    from acrdl.training.evaluate import predict_probs
    from acrdl.training.train import make_encoder, train_crossfit, train_model

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    records = [r for f in args.data.split(",") for r in load_jsonl(f.strip())]
    excluded = load_exclusions(args.exclude)
    if excluded:
        before = len(records)
        records = [r for r in records if r.bug_id.split("@", 1)[0] not in excluded]
        print(f"excluded {before - len(records)} records from {len(excluded)} reviewed variants")
    encoder = make_encoder(args.encoder, cfg)  # shared: frozen embeddings are cached once
    bs = cfg.training.batch_size
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results: dict = {"n_records": len(records), "seeds": args.seeds, "encoder": args.encoder,
                     "excluded_variants": sorted(excluded),
                     "label_counts": {k: int(sum(r.label == i for r in records))
                                      for i, k in enumerate(["correct", "overfitting",
                                                             "ambiguous"])},
                     "runs": defaultdict(dict)}
    first_seed_preds: dict[str, np.ndarray] = {}
    judged = json.loads(Path(args.llm_judge).read_text()) if args.llm_judge else None
    baselines = BASELINES + (["llm_judge"] if judged else [])
    first_seed_y = first_seed_groups = None

    for seed in args.seeds:
        train, val, test = split_by_problem(records, seed=seed)
        test = ordered(test, bs)
        y = np.array([r.label for r in test])
        print(f"seed {seed}: train={len(train)} val={len(val)} test={len(test)}", flush=True)
        for name in baselines:
            if name == "llm_judge" and not all(r.patch_id in judged for r in test):
                print("  llm_judge: missing verdicts for this split, skipped", flush=True)
                continue
            pred = baseline_preds(name, train, test, seed, judged)
            results["runs"][f"baseline:{name}"][seed] = {
                "overall": summarize(y, pred), "subsets": per_subset(y, pred, test),
                "confusion": confusion(y, pred)}
            if seed == args.seeds[0]:
                first_seed_preds[f"baseline:{name}"] = pred
        for variant in args.variants:
            t0 = time.time()
            fit = train_model if args.no_crossfit else train_crossfit
            res = fit(train, val, cfg, encoder, variant, seed)
            probs, y_model = predict_probs(res, test, bs)
            assert np.array_equal(y_model, y), "prediction order must match the test order"
            pred = decide(probs, res.threshold, res.amb_threshold)
            results["runs"][f"model:{variant}"][seed] = {
                "overall": summarize(y, pred, probs), "subsets": per_subset(y, pred, test),
                "confusion": confusion(y, pred), "reliability": reliability(y, probs),
                "threshold": res.threshold, "amb_threshold": res.amb_threshold,
                "seconds": round(time.time() - t0, 1)}
            # the same model with the plain argmax rule (no tuned thresholds), comparable
            # with the baselines, which predict by argmax too
            arg = probs.argmax(1)
            results["runs"][f"model:{variant}@argmax"][seed] = {
                "overall": summarize(y, arg, probs), "subsets": per_subset(y, arg, test),
                "confusion": confusion(y, arg)}
            if seed == args.seeds[0]:
                first_seed_preds[f"model:{variant}"] = pred
            print(f"  {variant}: macro_f1={macro_f1(y, pred):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if seed == args.seeds[0]:
            first_seed_y = y
            first_seed_groups = np.array([r.problem_id for r in test])

    # Paired bootstrap on the first seed: full model vs each baseline.
    tests = {}
    if "model:full" in first_seed_preds:
        for name in [b for b in baselines if f"baseline:{b}" in first_seed_preds]:
            tests[name] = paired_bootstrap(macro_f1, first_seed_y,
                                           first_seed_preds["model:full"],
                                           first_seed_preds[f"baseline:{name}"],
                                           groups=first_seed_groups)
    results["bootstrap_full_vs_baselines_macro_f1"] = tests
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "summary.md").write_text(render_summary(results))
    print(f"wrote {out / 'results.json'} and {out / 'summary.md'}")
    return 0


def render_summary(results: dict) -> str:
    runs = results["runs"]
    lines = ["# Results", "",
             f"Records: {results['n_records']} labelled plausible patches "
             f"({results['label_counts']}). Seeds: {results['seeds']}. "
             f"Encoder: {results['encoder']}. Mean +/- standard deviation over seeds, test split.",
             "", "| Method | " + " | ".join(HEADLINE) + " |",
             "|---|" + "---|" * len(HEADLINE)]
    for name, per_seed in runs.items():
        row = [fmt([s["overall"][k] for s in per_seed.values()]) for k in HEADLINE]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    lines += ["", "## Macro-F1 by subset", ""]
    subsets = sorted({s for per_seed in runs.values() for run in per_seed.values()
                      for s in run["subsets"]})
    lines += ["| Method | " + " | ".join(subsets) + " |", "|---|" + "---|" * len(subsets)]
    for name, per_seed in runs.items():
        row = [fmt([run["subsets"][s]["macro_f1"] for run in per_seed.values()
                    if s in run["subsets"]]) for s in subsets]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    if results.get("bootstrap_full_vs_baselines_macro_f1"):
        lines += ["", "## Paired bootstrap, full model minus baseline (macro-F1, first seed)", "",
                  "| Baseline | delta | 95% CI | p |", "|---|---|---|---|"]
        for name, t in results["bootstrap_full_vs_baselines_macro_f1"].items():
            lines.append(f"| {name} | {t['delta']:+.3f} | [{t['ci_low']:+.3f}, "
                         f"{t['ci_high']:+.3f}] | {t['p_value']:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
