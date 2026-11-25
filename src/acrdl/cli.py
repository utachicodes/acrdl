"""Command-line interface: ``acrdl {demo,build-dataset,train,evaluate}``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, TextIO

from acrdl.config import Config, load_config
from acrdl.data import toy
from acrdl.data.labeling import label_patches
from acrdl.data.loaders import load_toy
from acrdl.data.mutate import mutate
from acrdl.data.underspec import underspecify
from acrdl.generation.generator import HFGenerator, MockGenerator, PatchGenerator
from acrdl.pipeline import BugReport, Pipeline
from acrdl.types import BuggyProgram, Label

TITLE = "Automated Code Review and Program Repair System Using Deep Learning"
BASELINES = ("tests_only", "disagreement", "patch_sim", "static_gbdt")


def _short(pid: str) -> str:
    return pid.rsplit("/", 1)[-1]


def toy_bugs(with_underspec: bool = True) -> list[BuggyProgram]:
    """Toy bugs; the ``average`` problem is underspecified (empty-list clause removed)."""
    bugs = load_toy()
    if with_underspec:
        for b in bugs:
            if b.problem.task_id == "toy/average":
                b.problem = underspecify(b.problem, toy.AVERAGE_CLAUSE,
                                         lambda args: args == ([],))
    return bugs


def make_generator(cfg: Config) -> PatchGenerator:
    if cfg.generator.backend == "mock":
        return MockGenerator(toy.SCRIPTED_CANDIDATES)
    if cfg.generator.backend == "hf":
        return HFGenerator(cfg.generator.model_name, cfg.generator.temperature,
                           cfg.generator.max_new_tokens, cfg.generator.cache_dir)
    raise ValueError(f"Unknown generator backend: {cfg.generator.backend}")


def print_report(rep: BugReport, out: Optional[TextIO] = None) -> None:
    p = rep.bug.problem
    w = (out or sys.stdout).write
    w(f"\n[{p.task_id}] bug {rep.bug.bug_id}\n")
    if p.ambiguous_inputs:
        w("  note: an edge-case clause was removed from this spec "
          "(controlled underspecification)\n")
    n_clusters = rep.clustering.n_clusters if rep.clustering else 0
    w(f"  candidates: {len(rep.patches)}   plausible: {sum(r.plausible for r in rep.patches)}"
      f"   inputs executed: {rep.n_inputs}   behaviour clusters: {n_clusters}\n")
    w(f"  {'patch':<6} {'plausible':<10} {'preserve':>8} {'agree':>6} {'cluster':>7} "
      f"{'smell':>5}  verdict\n")
    for r in rep.patches:
        if not r.plausible or r.evidence is None:
            w(f"  {_short(r.patch.patch_id):<6} {'no':<10} {'-':>8} {'-':>6} {'-':>7} "
              f"{'-':>5}  filtered (fails visible tests)\n")
            continue
        e = r.evidence.as_dict()
        smell = r.static.hardcoded_test_literal if r.static else 0
        verdict = r.verdict.name if r.verdict is not None else "?"
        w(f"  {_short(r.patch.patch_id):<6} {'yes':<10} {e['preservation']:>8.2f} "
          f"{e['consensus_agreement']:>6.2f} {r.evidence.cluster_id:>7d} {smell:>5d}  {verdict}\n")
    for args, groups in rep.ambiguity_witnesses:
        w(f"  spec ambiguous at input {args!r}:\n")
        for sig, pids in groups.items():
            w(f"    {sig:<28} <- {', '.join(_short(x) for x in pids)}\n")


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.generator.backend = "mock"
    cfg.evidence.n_fuzz_inputs = args.fuzz
    print(TITLE)
    print("demo: offline toy benchmark, scripted candidates, disagreement-rule predictor")
    pipe = Pipeline(cfg, make_generator(cfg))
    bugs = toy_bugs(not args.no_underspec)
    if args.task:
        bugs = [b for b in bugs if b.problem.task_id.startswith(args.task)]
    counts = {lab: 0 for lab in Label}
    for bug in bugs:
        rep = pipe.run(bug)
        print_report(rep)
        for r in rep.patches:
            if r.verdict is not None:
                counts[r.verdict] += 1
    print("\nsummary: " + ", ".join(f"{lab.name}={n}" for lab, n in counts.items()))
    return 0


def build_bugs(source: str, mutants: int, limit: Optional[int], seed: int,
               bug_files: Optional[str] = None) -> list[BuggyProgram]:
    if source == "prepared":
        from acrdl.data.prepare import load_prepared

        if not bug_files:
            raise ValueError("--source prepared needs --bugs FILE[,FILE...]")
        bugs = [b for f in bug_files.split(",") for b in load_prepared(f.strip())]
        return bugs[:limit] if limit else bugs
    if source == "toy":
        bugs = toy_bugs(True) + [b for b in load_toy() if b.problem.task_id == "toy/average"]
    elif source == "humanevalfix":
        from acrdl.data.loaders import load_humanevalfix

        bugs = load_humanevalfix(limit)
    else:
        raise ValueError(f"Unknown source: {source}")
    extra = []
    for b in bugs:
        for i, m in enumerate(mutate(b.problem.reference, n=mutants, seed=seed)):
            extra.append(BuggyProgram(b.problem, m.source, f"{b.bug_id}+m{i}", m.operator))
    return bugs + extra


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from acrdl.training.dataset import records_from_report, save_jsonl

    cfg = load_config(args.config)
    if args.source == "toy":
        cfg.generator.backend = "mock"
    elif cfg.generator.backend == "mock":
        print("error: --source humanevalfix needs generator.backend: hf in the config "
              "(the mock generator only knows the toy problems)", file=sys.stderr)
        return 2
    pipe = Pipeline(cfg, make_generator(cfg))
    bugs = build_bugs(args.source, args.mutants, args.limit, cfg.seed, args.bugs)
    skip_file = Path(args.skip_file) if args.skip_file else None
    if skip_file and skip_file.exists():
        skip = {ln.split("\t", 1)[0].strip() for ln in skip_file.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")}
        bugs = [b for b in bugs if b.bug_id not in skip]
    if args.tag:  # a second sampling pass gets its own bug ids (same problem ids)
        for b in bugs:
            b.bug_id = f"{b.bug_id}@{args.tag}"
    out = Path(args.out)
    done_file = out.with_suffix(out.suffix + ".done")
    done = set(done_file.read_text().split("\n")) if args.resume and done_file.exists() else set()
    if not args.resume:
        out.unlink(missing_ok=True)
        done_file.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = {lab.name: 0 for lab in Label}
    todo = [b for b in bugs if b.bug_id not in done]
    print(f"{len(bugs)} bugs, {len(bugs) - len(todo)} already done, {len(todo)} to run")
    if args.max_bugs and len(todo) > args.max_bugs:
        # Stop after max_bugs and let the caller relaunch: a fresh process resets GPU memory
        # (PyTorch on Apple GPUs keeps a growing cache of kernels for every new input shape).
        todo, more = todo[: args.max_bugs], True
    else:
        more = False
    for i, bug in enumerate(todo, 1):
        t0 = time.time()
        try:
            rep = pipe.run(bug)
        except Exception as e:  # noqa: BLE001  one bad bug must not stop a long run
            print(f"[{i}/{len(todo)}] SKIPPED {bug.bug_id}: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:200] if str(e) else ''}", flush=True)
            continue  # not marked done, so --resume retries it
        plaus = {r.patch.patch_id: r.patch.source for r in rep.patches if r.plausible}
        labels = label_patches(bug.problem, plaus, cfg.evidence.exec_timeout_s,
                               cfg.evidence.memory_limit_mb)
        recs = records_from_report(rep, labels)
        save_jsonl(recs, out, append=True)
        with done_file.open("a") as f:
            f.write(bug.bug_id + "\n")
        for r in recs:
            hist[Label(r.label).name] += 1
        print(f"[{i}/{len(todo)}] {bug.bug_id}: {len(rep.patches)} candidates, "
              f"{len(plaus)} plausible, {time.time() - t0:.0f}s  totals {hist}", flush=True)
    print(f"wrote records to {out}  {hist}")
    return 75 if more else 0  # 75 (EX_TEMPFAIL): stopped early on purpose, run again


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from acrdl.data.prepare import bug_to_json, prepare_all, prepare_quixbugs, save_bugs

    if args.quixbugs:  # out-of-distribution set only; leaves the other files untouched
        bugs, st = prepare_quixbugs(args.quixbugs)
        save_bugs([bug_to_json(b) for b in bugs], Path(args.out) / "quixbugs.jsonl")
        print(json.dumps({"quixbugs": st.as_dict()}, indent=2))
        return 0

    stats = prepare_all(args.out, cache_dir=args.cache_dir, max_hidden=args.max_hidden,
                        mbpp_mutants=args.mbpp_mutants, limit=args.limit,
                        max_silent=args.max_silent)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from acrdl.training.dataset import load_jsonl, split_by_problem
    from acrdl.training.evaluate import evaluate_model
    from acrdl.training.train import make_encoder, save_checkpoint, train_model

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    seed = cfg.seed if args.seed is None else args.seed
    train, val, test = split_by_problem(load_jsonl(args.data), seed=seed)
    encoder = make_encoder(args.encoder, cfg)
    res = train_model(train, val, cfg, encoder, args.ablation, seed)
    out = args.out or cfg.training.output_dir
    save_checkpoint(res, cfg, out, args.encoder, args.ablation)
    print(f"saved checkpoint to {out}  (train={len(train)} val={len(val)} test={len(test)})")
    if test:
        print(json.dumps(evaluate_model(res, test), indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    import numpy as np

    from acrdl.training.dataset import load_jsonl, split_by_problem
    from acrdl.training.metrics import summarize

    cfg = load_config(args.config)
    records = load_jsonl(args.data)
    if args.split != "all":
        train, val, test = split_by_problem(records, seed=cfg.seed if args.seed is None
                                            else args.seed)
        records = {"train": train, "val": val, "test": test}[args.split]
    y = np.array([r.label for r in records])
    if args.baseline:
        if args.baseline == "static_gbdt":
            from acrdl.baselines.static_gbdt import StaticGBDT

            train, _, _ = split_by_problem(load_jsonl(args.data), seed=cfg.seed)
            pred = StaticGBDT(cfg.seed).fit(train).predict_records(records)
        else:
            import importlib

            mod = importlib.import_module(f"acrdl.baselines.{args.baseline}")
            pred = mod.predict_records(records)
        print(json.dumps(summarize(y, pred), indent=2))
        return 0
    if not args.checkpoint:
        print("error: pass --checkpoint DIR or --baseline NAME", file=sys.stderr)
        return 2
    from acrdl.training.evaluate import evaluate_model
    from acrdl.training.train import load_checkpoint

    print(json.dumps(evaluate_model(load_checkpoint(args.checkpoint, cfg), records), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="acrdl", description=TITLE)
    p.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="offline end-to-end demo on the toy benchmark")
    d.add_argument("--task", default=None, help="only run tasks whose id starts with this")
    d.add_argument("--fuzz", type=int, default=120, help="fuzzed inputs per bug")
    d.add_argument("--no-underspec", action="store_true",
                   help="keep the full average() spec (no ambiguity expected)")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("build-dataset", help="generate, collect evidence and label patches")
    b.add_argument("--source", choices=["toy", "humanevalfix", "prepared"], default="toy")
    b.add_argument("--bugs", default=None,
                   help="prepared bug files (comma separated) for --source prepared")
    b.add_argument("--tag", default=None,
                   help="suffix for bug ids, to keep a second sampling pass separate")
    b.add_argument("--skip-file", default="data/skip_generation.tsv",
                   help="bug ids never sent to the generator (TSV, first column)")
    b.add_argument("--max-bugs", type=int, default=None,
                   help="process at most N bugs, then exit with code 75 if work remains")
    b.add_argument("--resume", action="store_true",
                   help="append to --out and skip bugs already processed")
    b.add_argument("--out", default="data/processed/toy.jsonl")
    b.add_argument("--mutants", type=int, default=0, help="extra mutation bugs per problem")
    b.add_argument("--limit", type=int, default=None)
    b.set_defaults(func=cmd_build_dataset)

    pr = sub.add_parser("prepare-data",
                        help="download benchmarks, build bugs and ambiguous variants (no LLM)")
    pr.add_argument("--out", default="data/processed")
    pr.add_argument("--cache-dir", default="data/raw/hf")
    pr.add_argument("--max-hidden", type=int, default=1000, help="hidden tests kept per problem")
    pr.add_argument("--mbpp-mutants", type=int, default=1, help="injected bugs per MBPP+ problem")
    pr.add_argument("--max-silent", type=int, default=150,
                    help="cap on silent-edge ambiguous variants")
    pr.add_argument("--limit", type=int, default=None, help="problems per benchmark (debug)")
    pr.add_argument("--quixbugs", default=None, metavar="CLONE_DIR",
                    help="only prepare QuixBugs from a local clone (out-of-distribution set)")
    pr.set_defaults(func=cmd_prepare_data)

    t = sub.add_parser("train", help="train the fusion model (needs the 'ml' extra)")
    t.add_argument("--data", required=True)
    t.add_argument("--encoder", choices=["hashing", "hf", "hf-frozen"], default="hashing")
    t.add_argument("--ablation", default="full",
                   choices=["full", "code_only", "no_code", "no_evidence", "no_clusters",
                            "no_contrastive"])
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--out", default=None)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a checkpoint or a baseline")
    e.add_argument("--data", required=True)
    e.add_argument("--checkpoint", default=None)
    e.add_argument("--baseline", choices=BASELINES, default=None)
    e.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    e.add_argument("--seed", type=int, default=None)
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
