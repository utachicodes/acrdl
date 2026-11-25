#!/usr/bin/env bash
# More-randomness pass for the ambiguous variants: 20 candidates at temperature 1.0
# (configs/mac_diverse.yaml), under bug ids tagged "@t1" so they stay separate from the
# main pass while sharing its problem ids. Resumable; relaunches every 15 bugs to reset
# GPU memory, like scripts/run_all_generation.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=data/processed
LOG="$OUT/generation.log"
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "=== underspec_t1, temperature 1.0 x20 ($(date '+%Y-%m-%d %H:%M')) ===" | tee -a "$LOG"
rc=75
while [ "$rc" -eq 75 ]; do
  caffeinate -i .venv/bin/acrdl --config configs/mac_diverse.yaml build-dataset \
    --source prepared --bugs "$OUT/underspec.jsonl" --tag t1 \
    --out "$OUT/records_underspec_t1.jsonl" --resume --max-bugs 15 2>&1 \
    | grep --line-buffered -vE "deprecated|attention mask" | tee -a "$LOG"
  rc="${PIPESTATUS[0]}"
done
echo "=== underspec_t1 done ($(date '+%Y-%m-%d %H:%M')) ===" | tee -a "$LOG"
