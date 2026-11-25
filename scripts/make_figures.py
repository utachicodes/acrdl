"""Figures for the report, from the records and results/results.json.

Usage:
    .venv/bin/python scripts/make_figures.py --data data/processed/records.jsonl \\
        --results results/results.json --out report/figures

Writes PNG files (light background, print-ready):
  labels_by_subset.png   label counts per data subset (stacked bars)
  macro_f1.png           macro-F1 per method, mean and spread over seeds
  safety_tradeoff.png    false-accept rate against recall of correct patches
  confusion_full.png     confusion matrix of the full model (first seed)
  reliability.png        calibration of the full model (first seed)

Colours: the three classes use the first three slots of a validated categorical
palette in fixed order (blue, orange, aqua); every bar segment carries a visible
count, because aqua is below 3:1 contrast on the white surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import load_exclusions, subset_of  # noqa: E402

from acrdl.training.dataset import load_jsonl  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASS_NAMES = ["correct", "overfitting", "ambiguous"]
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
BASE_COLOR, MODEL_COLOR = "#8f8e88", "#2a78d6"
W = 8.0  # figure width in inches; --column sets it to one column of a two-column paper

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def labels_by_subset(records, path: Path) -> None:
    subsets = sorted({subset_of(r) for r in records})
    counts = {s: Counter(r.label for r in records if subset_of(r) == s) for s in subsets}
    biggest = max(sum(c.values()) for c in counts.values())
    fig, ax = plt.subplots(figsize=(W, 0.6 * len(subsets) + 1.6))
    ax.grid(axis="y", visible=False)
    for i, s in enumerate(subsets):
        left = 0
        total = sum(counts[s].values())
        for k, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            n = counts[s].get(k, 0)
            if n == 0:
                continue
            ax.barh(i, n, left=left, color=color, height=0.6, edgecolor=SURFACE, linewidth=2,
                    label=name if i == 0 or name not in ax.get_legend_handles_labels()[1] else "")
            if n / biggest > 0.035:  # label a segment only when it is wide enough
                ax.text(left + n / 2, i, str(n), ha="center", va="center",
                        fontsize=9 if W >= 5 else 6.5,
                        color="white" if k == 0 else INK)
            left += n
        ax.text(left + total * 0.01 + 1, i, f"{total}", va="center",
                fontsize=9 if W >= 5 else 6.5, color=INK2)
    ax.set_yticks(range(len(subsets)), [x.replace(": ", ":\n") if W < 5 else x for x in subsets])
    ax.invert_yaxis()
    ax.set_xlabel("labelled plausible patches")
    ax.set_title("Labels of plausible patches by subset")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _metric(runs: dict, name: str, key: str) -> list[float]:
    vals = [s["overall"][key] for s in runs[name].values()]
    return [v for v in vals if v == v]  # drop NaN


def macro_f1(results: dict, path: Path) -> None:
    runs = results["runs"]
    names = sorted(runs, key=lambda n: np.mean(_metric(runs, n, "macro_f1") or [0]))
    means = [np.mean(_metric(runs, n, "macro_f1") or [0]) for n in names]
    spread = [np.std(_metric(runs, n, "macro_f1") or [0]) for n in names]
    colors = [MODEL_COLOR if n.startswith("model:") else BASE_COLOR for n in names]
    fig, ax = plt.subplots(figsize=(W, 0.42 * len(names) + 1.4))
    ax.grid(axis="y", visible=False)
    ax.barh(range(len(names)), means, xerr=spread, color=colors, height=0.6,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    for i, (m, sd) in enumerate(zip(means, spread)):
        ax.text(m + sd + 0.015, i, f"{m:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(names)), [n.split(":", 1)[1] for n in names])
    ax.set_xlim(0, 1.05)
    n_seeds = len(results["seeds"])
    ax.set_xlabel(f"macro-F1 on the test split (mean, bars: std over {n_seeds} seeds)")
    ax.set_title("Macro-F1 by method")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MODEL_COLOR, label="neural reviewer variants"),
                       Patch(color=BASE_COLOR, label="baselines")],
              loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def safety_tradeoff(results: dict, path: Path) -> None:
    runs = results["runs"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    pts = []
    for n in runs:
        far = np.mean(_metric(runs, n, "false_accept_rate") or [np.nan])
        rec = np.mean(_metric(runs, n, "correct_recall") or [np.nan])
        if not (np.isnan(far) or np.isnan(rec)):
            pts.append((far, rec, n))
    placed: list[tuple[float, float]] = []
    for far, rec, n in sorted(pts, key=lambda p: (p[0], -p[1])):
        model = n.startswith("model:")
        ax.scatter(far, rec, s=64 if model else 48, color=MODEL_COLOR if model else BASE_COLOR,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ly = rec  # push a label down until it clears the labels already placed nearby
        while any(abs(far - px) < 0.25 and abs(ly - py) < 0.045 for px, py in placed):
            ly -= 0.045
        placed.append((far, ly))
        ax.annotate(n.split(":", 1)[1], (far, rec), xytext=(far + 0.02, ly), fontsize=8.5,
                    color=INK2, va="center",
                    arrowprops=None if abs(ly - rec) < 1e-9 else
                    {"arrowstyle": "-", "color": GRID, "linewidth": 0.8})
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("false-accept rate (overfitting patches accepted), lower is safer")
    ax.set_ylabel("recall of correct patches")
    ax.set_title("Safety against usefulness: top left is best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion_full(results: dict, path: Path) -> None:
    run = results["runs"].get("model:full")
    if not run:
        return
    first = run[next(iter(run))]
    cm = np.array(first["confusion"], dtype=float)
    row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.grid(False)
    ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i, j])}\n{row[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color="white" if row[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true label")
    ax.set_title("Full model, confusion (first seed)")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def reliability(results: dict, path: Path) -> None:
    run = results["runs"].get("model:full")
    if not run:
        return
    bins = run[next(iter(run))].get("reliability", [])
    if not bins:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--", label="perfect calibration")
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], color=MODEL_COLOR,
            linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
            label="full model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Calibration after temperature scaling")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="records JSONL, comma separated")
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--exclude", default="data/excluded_variants.tsv")
    ap.add_argument("--column", action="store_true",
                    help="size figures for one column of a two-column paper (3.4 in)")
    args = ap.parse_args(argv)
    if args.column:
        global W
        W = 3.4
        plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "xtick.labelsize": 6.5,
                             "ytick.labelsize": 6.5, "legend.fontsize": 6.5})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = [r for f in args.data.split(",") for r in load_jsonl(f.strip())]
    excluded = load_exclusions(args.exclude)
    records = [r for r in records if r.bug_id.split("@", 1)[0] not in excluded]
    labels_by_subset(records, out / "labels_by_subset.png")
    if Path(args.results).exists():
        results = json.loads(Path(args.results).read_text())
        macro_f1(results, out / "macro_f1.png")
        safety_tradeoff(results, out / "safety_tradeoff.png")
        confusion_full(results, out / "confusion_full.png")
        reliability(results, out / "reliability.png")
    print("figures written to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
