#!/usr/bin/env bash
# Generate, test and label patches for all prepared bugs on an Apple Silicon Mac.
#
# Runs the three bug sets in priority order (ambiguous variants first, then
# HumanEvalFix, then MBPP+ mutants). Every bug is saved as soon as it is done,
# so the script can be stopped with Ctrl+C and started again: finished bugs are
# skipped. caffeinate keeps the Mac awake while it runs (keep it plugged in).
#
# Usage: scripts/run_generation.sh [config]   (default: configs/mac.yaml)
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:-configs/mac.yaml}"
OUT=data/processed
LOG="$OUT/generation.log"
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

run_set() {
  local name="$1" bugs="$2"
  echo "=== $name ($(date '+%Y-%m-%d %H:%M')) ===" | tee -a "$LOG"
  caffeinate -i .venv/bin/acrdl --config "$CONFIG" build-dataset --source prepared \
    --bugs "$bugs" --out "$OUT/records_${name}.jsonl" --resume 2>&1 \
    | grep --line-buffered -vE "deprecated|attention mask" | tee -a "$LOG"
}

run_set underspec "$OUT/underspec.jsonl"
run_set humanevalfix "$OUT/humanevalfix.jsonl"
run_set mbpp "$OUT/mbpp_mutants.jsonl"

cat "$OUT"/records_underspec.jsonl "$OUT"/records_humanevalfix.jsonl \
    "$OUT"/records_mbpp.jsonl > "$OUT/records.jsonl"
echo "=== done: $(wc -l < "$OUT/records.jsonl") labelled patches in $OUT/records.jsonl ===" \
  | tee -a "$LOG"
