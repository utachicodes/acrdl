"""Write the reworded bug files for the wording-intervention experiment.

Usage: .venv/bin/python scripts/prepare_wordings.py [--controls 50]

For every ambiguous variant kept after review, and for the first N HumanEvalFix
bugs as controls (specification complete), writes data/processed/wording_<w>.jsonl
for w in w1 (reformat), w2 (rename), w3 (reorder). A bug is skipped for a wording
that does not apply or fails the meaning-preservation checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import load_exclusions  # noqa: E402

from acrdl.data.prepare import bug_to_json, load_prepared, save_bugs  # noqa: E402
from acrdl.data.wording import WORDINGS, make_wording  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--controls", type=int, default=50)
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args(argv)
    out = Path(args.out)
    excluded = load_exclusions("data/excluded_variants.tsv")
    bugs = [b for b in load_prepared(out / "underspec.jsonl") if b.bug_id not in excluded]
    bugs += load_prepared(out / "humanevalfix.jsonl")[: args.controls]
    made: dict[str, list] = {w: [] for w in WORDINGS}
    for i, bug in enumerate(bugs, 1):
        for w in WORDINGS:
            v = make_wording(bug, w)
            if v is not None:
                made[w].append(bug_to_json(v))
        if i % 25 == 0:
            print(f"{i}/{len(bugs)} bugs", flush=True)
    for w, rows in made.items():
        save_bugs(rows, out / f"wording_{w}.jsonl")
    stats = {w: len(rows) for w, rows in made.items()}
    per_bug = Counter(sum(1 for w in WORDINGS for r in made[w]
                          if r["bug_id"].rsplit("~", 1)[0] == b.bug_id) for b in bugs)
    print(json.dumps({"bugs": len(bugs), "per_wording": stats,
                      "bugs_by_number_of_wordings": dict(sorted(per_bug.items()))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
