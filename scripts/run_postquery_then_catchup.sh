#!/usr/bin/env bash
# Grade currently-available responses for a model, then wait until the full
# eval set is queried and grade any remaining samples.
set -euo pipefail

MODEL="${1:?model required, e.g. qwen/qwen3.8-27b}"
RUN_MONTH="${2:-2026-07}"
TARGET_N="${3:-136}"
POLL_SECS="${4:-300}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/data_track/sciconbench_track.db"
LOG_DIR="$ROOT/data_track/logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
SAFE_MODEL="$(echo "$MODEL" | tr '/' '-')"
LOG="$LOG_DIR/${SAFE_MODEL}-postquery-catchup-${STAMP}.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PYTHONUNBUFFERED=1

count_responses() {
  sqlite3 "$DB" "SELECT COUNT(*) FROM model_responses WHERE run_month='$RUN_MONTH' AND model='$MODEL';"
}

echo "=== Pass 1: grade currently available $MODEL responses ===" | tee -a "$LOG"
python scripts/run_postquery_for_model.py --model "$MODEL" --run-month "$RUN_MONTH" 2>&1 | tee -a "$LOG"

echo "=== Waiting for $MODEL to reach $TARGET_N responses (poll ${POLL_SECS}s) ===" | tee -a "$LOG"
while true; do
  n="$(count_responses)"
  echo "$(date -Is) responses=$n/$TARGET_N" | tee -a "$LOG"
  if [[ "$n" -ge "$TARGET_N" ]]; then
    break
  fi
  sleep "$POLL_SECS"
done

echo "=== Pass 2: grade any remaining $MODEL samples ===" | tee -a "$LOG"
python scripts/run_postquery_for_model.py --model "$MODEL" --run-month "$RUN_MONTH" 2>&1 | tee -a "$LOG"
echo "=== Catch-up complete for $MODEL ===" | tee -a "$LOG"
