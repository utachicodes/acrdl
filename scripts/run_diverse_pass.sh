#!/usr/bin/env bash
# Wording-intervention pass (replaces the earlier "more samples at higher
# temperature" pass, kept as run_diverse_sampling.sh). Generates 10 candidates for
# each meaning-preserving rewording of the ambiguous variants and the HumanEvalFix
# controls (data/processed/wording_w{1,2,3}.jsonl, from scripts/prepare_wordings.py),
# with exactly the same settings as the original pass, so the only thing that
# changes between passes is the wording. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/processed
LOG="$OUT/generation.log"
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
for w in w2 w3 w1; do
  [ -s "$OUT/wording_$w.jsonl" ] || continue
  echo "=== wording $w ($(date '+%Y-%m-%d %H:%M')) ===" | tee -a "$LOG"
  caffeinate -i .venv/bin/acrdl --config configs/mac.yaml build-dataset --source prepared \
    --bugs "$OUT/wording_$w.jsonl" --out "$OUT/records_wording_$w.jsonl" --resume 2>&1 \
    | grep --line-buffered -vE "deprecated|attention mask" | tee -a "$LOG"
done
echo "=== wording pass done ===" | tee -a "$LOG"
