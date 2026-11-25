#!/usr/bin/env bash
# Run every generation job in sequence. Each job is resumable, so this script can be
# started again at any time. Start it detached so it survives a closed terminal:
#   screen -dmS acrdl-gen scripts/run_all_generation.sh
#   screen -r acrdl-gen        # look at it; Ctrl+A then D to detach again
# Order: the wording test (the project's newest question) runs before the large MBPP+ set.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=data/processed
LOG="$OUT/generation.log"
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

run_set() {  # name, bug file(s)
  echo "=== $1 ($(date '+%Y-%m-%d %H:%M')) ===" | tee -a "$LOG"
  caffeinate -i .venv/bin/acrdl --config configs/mac.yaml build-dataset --source prepared \
    --bugs "$2" --out "$OUT/records_$1.jsonl" --resume 2>&1 \
    | grep --line-buffered -vE "deprecated|attention mask" | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

run_set underspec "$OUT/underspec.jsonl" || exit 1
run_set humanevalfix "$OUT/humanevalfix.jsonl" || exit 1
for w in w2 w3 w1; do
  [ -s "$OUT/wording_$w.jsonl" ] && { run_set "wording_$w" "$OUT/wording_$w.jsonl" || exit 1; }
done
run_set quixbugs "$OUT/quixbugs.jsonl" || exit 1
run_set mbpp "$OUT/mbpp_mutants.jsonl" || exit 1
echo "=== all generation done ($(date '+%Y-%m-%d %H:%M')) ===" | tee -a "$LOG"
