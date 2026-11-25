#!/usr/bin/env bash
# Generate and label patches for the QuixBugs out-of-distribution set. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/processed
LOG="$OUT/generation.log"
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "=== quixbugs ($(date '+%Y-%m-%d %H:%M')) ===" | tee -a "$LOG"
caffeinate -i .venv/bin/acrdl --config configs/mac.yaml build-dataset --source prepared \
  --bugs "$OUT/quixbugs.jsonl" --out "$OUT/records_quixbugs.jsonl" --resume 2>&1 \
  | grep --line-buffered -vE "deprecated|attention mask" | tee -a "$LOG"
echo "=== quixbugs done ===" | tee -a "$LOG"
