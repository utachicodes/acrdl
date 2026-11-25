"""Does wording dependence localise specification silence better than disagreement?

Usage (after scripts/run_diverse_pass.sh has produced records_wording_*.jsonl):
    .venv/bin/python scripts/wording_analysis.py --out results/wording

For every bug with candidates from at least two wordings (the original w0 plus
meaning-preserving rewordings), every plausible candidate is executed on the
inputs where the specification is known to be silent (from controlled
underspecification) and on a sample of inputs the specification determines (hidden
tests). For each input we compute

* disagreement: share of candidates outside the majority behaviour (the signal
  clustering methods such as SpecFix use),
* I(B; W): mutual information between behaviour and wording, and its permutation
  p-value (acrdl.evidence.wording).

Outputs results/wording/inputs.csv (one row per bug and input) and summary.json:
AUROC of each signal for "silent vs determined" on inputs where candidates
disagree, the false-alarm rate of the test on determined inputs (checks the
exchangeability assumption A1) and its power on silent inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import load_exclusions  # noqa: E402

from acrdl.data.prepare import load_prepared  # noqa: E402
from acrdl.data.rebuild import apply_unified_diff, diff_of  # noqa: E402
from acrdl.evidence.sandbox import run_function  # noqa: E402
from acrdl.evidence.wording import disagreement, entropy, permutation_test  # noqa: E402
from acrdl.training.dataset import load_jsonl  # noqa: E402
from acrdl.training.metrics import auroc  # noqa: E402

WORDING_TAGS = ("w1", "w2", "w3")


def split_wording(bug_id: str) -> tuple[str, str]:
    for w in WORDING_TAGS:
        if bug_id.endswith("~" + w):
            return bug_id[: -len(w) - 1], w
    return bug_id, "w0"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results/wording")
    ap.add_argument("--determined", type=int, default=30, help="determined inputs per bug")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args(argv)
    d, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    excluded = load_exclusions("data/excluded_variants.tsv")

    bugs = {}
    for name in ["underspec", "humanevalfix"] + [f"wording_{w}" for w in WORDING_TAGS]:
        f = d / f"{name}.jsonl"
        if f.exists():
            bugs.update({b.bug_id: b for b in load_prepared(f)})
    records = []
    for name in ["underspec", "humanevalfix"] + [f"wording_{w}" for w in WORDING_TAGS]:
        f = d / f"records_{name}.jsonl"
        if f.exists():
            records += load_jsonl(f)

    # base bug -> wording -> list of patch sources
    patches: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        base, w = split_wording(r.bug_id)
        if base in excluded or r.bug_id not in bugs:
            continue
        patches[base][w].append(apply_unified_diff(bugs[r.bug_id].source, diff_of(r.code)))

    rows = []
    for base, by_w in sorted(patches.items()):
        by_w = {w: srcs for w, srcs in by_w.items() if len(srcs) >= 2}
        if len(by_w) < 2 or base not in bugs:
            continue
        prob = bugs[base].problem
        silent = [tuple(a) for a in prob.ambiguous_inputs]
        silent_keys = {repr(a) for a in silent}
        determined = [t.args for t in prob.hidden_tests if repr(t.args) not in silent_keys]
        determined = determined[: args.determined]
        inputs = silent + determined
        if not inputs:
            continue
        sig: dict[str, list[list[str]]] = {}
        for w, srcs in by_w.items():
            entry = bugs[base if w == "w0" else f"{base}~{w}"].problem.entry_point
            sig[w] = [[r.signature for r in run_function(s, entry, inputs, 2.0)] for s in srcs]
        for j, x in enumerate(inputs):
            groups = {w: [cand[j] for cand in sig[w]] for w in sig}
            pooled = [b for bs in groups.values() for b in bs]
            mi, p = permutation_test(groups, args.perm, seed=j)
            pair_p = {}
            for w in WORDING_TAGS:  # each rewording against the original wording alone
                if w in groups and "w0" in groups:
                    pair_p[f"p_w0_{w}"] = round(permutation_test(
                        {"w0": groups["w0"], w: groups[w]}, args.perm, seed=j)[1], 4)
            rows.append({**pair_p,
                "bug": base, "input": repr(x)[:120], "silent": int(j < len(silent)),
                "control_bug": int("~" not in base), "n_wordings": len(groups),
                "n_candidates": len(pooled), "disagreement": round(disagreement(pooled), 4),
                "entropy_bits": round(entropy(list(Counter(pooled).values())), 4),
                "mi_bits": round(mi, 4), "p_value": round(p, 4)})
        print(f"{base}: {len(by_w)} wordings, {len(inputs)} inputs", flush=True)

    if not rows:
        print("no bug has candidates from two wordings yet")
        return 1
    with (out / "inputs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    y = np.array([r["silent"] for r in rows])
    dis = np.array([r["disagreement"] for r in rows])
    mi = np.array([r["mi_bits"] for r in rows])
    ent = np.array([r["entropy_bits"] for r in rows])
    p = np.array([r["p_value"] for r in rows])
    hot = dis > 0  # only where candidates disagree is the question interesting
    det, sil = (y == 0), (y == 1)
    summary = {
        "bugs": len({r["bug"] for r in rows}), "inputs": len(rows),
        "silent_inputs": int(sil.sum()), "determined_inputs": int(det.sum()),
        "inputs_with_disagreement": int(hot.sum()),
        "silent_with_disagreement": int((hot & sil).sum()),
        "auroc_on_disagreeing_inputs": {
            "disagreement": auroc(y[hot], dis[hot]) if hot.any() else None,
            "semantic_entropy": auroc(y[hot], ent[hot]) if hot.any() else None,
            "wording_mi": auroc(y[hot], mi[hot]) if hot.any() else None,
            "wording_1_minus_p": auroc(y[hot], 1 - p[hot]) if hot.any() else None,
        },
        f"false_alarm_rate_determined_alpha_{args.alpha}":
            float((p[det] <= args.alpha).mean()) if det.any() else None,
        f"power_silent_alpha_{args.alpha}":
            float((p[sil] <= args.alpha).mean()) if sil.any() else None,
        f"power_silent_with_disagreement_alpha_{args.alpha}":
            float((p[sil & hot] <= args.alpha).mean()) if (sil & hot).any() else None,
    }
    # Pooled per bug: combine the per-input p-values with Fisher's method, separately for
    # the silent and the determined inputs of each bug (more power than one input at a time;
    # inputs of one bug are not independent, so treat these p-values as descriptive).
    from scipy.stats import combine_pvalues  # noqa: PLC0415

    per_bug = defaultdict(lambda: {"silent": [], "determined": []})
    for r in rows:
        if r["disagreement"] > 0:
            per_bug[r["bug"]]["silent" if r["silent"] else "determined"].append(r["p_value"])
    pooled = {k: [] for k in ("silent", "determined")}
    for groups in per_bug.values():
        for k, ps in groups.items():
            if ps:
                pooled[k].append(float(combine_pvalues(ps, method="fisher")[1]))
    summary["per_bug_fisher"] = {
        k: {"bugs": len(v), f"share_p_below_{args.alpha}":
            float(np.mean([x <= args.alpha for x in v])) if v else None}
        for k, v in pooled.items()}
    # Per rewording: models react to formatting changes even where the answer is fixed
    # (ReCode, arXiv:2212.10264), which would break exchangeability for that rewording and
    # show up as false alarms. Report each rewording against the original on its own.
    per_w = {}
    for w in WORDING_TAGS:
        key = f"p_w0_{w}"
        sub = [r for r in rows if key in r]
        if not sub:
            continue
        d_ = [r[key] for r in sub if not r["silent"]]
        s_ = [r[key] for r in sub if r["silent"] and r["disagreement"] > 0]
        per_w[w] = {"inputs": len(sub),
                    f"false_alarm_rate_alpha_{args.alpha}":
                        float(np.mean([x <= args.alpha for x in d_])) if d_ else None,
                    f"power_silent_with_disagreement_alpha_{args.alpha}":
                        float(np.mean([x <= args.alpha for x in s_])) if s_ else None}
    summary["per_rewording_vs_original"] = per_w
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
