#!/usr/bin/env bash
# Second, more diverse sampling pass over the ambiguous variants (20 candidates,
# temperature 1.0). Resumable like run_generation.sh. Bug ids get the suffix
# "@t1"; problem ids are unchanged, so both passes land in the same data split.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/processed
LOG="$OUT/generation.log"
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "=== underspec second pass, t=1.0 x20 ($(date '+%Y-%m-%d %H:%M')) ===" | tee -a "$LOG"
caffeinate -i .venv/bin/acrdl --config configs/mac_diverse.yaml build-dataset \
  --source prepared --bugs "$OUT/underspec.jsonl" --tag t1 \
  --out "$OUT/records_underspec_t1.jsonl" --resume 2>&1 \
  | grep --line-buffered -vE "deprecated|attention mask" | tee -a "$LOG"
echo "=== second pass done ===" | tee -a "$LOG"
