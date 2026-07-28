#!/usr/bin/env bash
# Sharded bulk atomic-fact generation (fixed settings).
#
# You only specify 4 things:
#   1) --model PATH        (model directory name under sciconharness/logs/)
#   2) --num-shards N
#   3) --api-key KEY
#   4) --base-url URL
#
# Everything else is fixed to match the pipeline defaults (config/model_config.yaml,
# same as in the paper):
#   --logs-dir <repo_root>/sciconharness/logs
#   all pipeline stages enabled (decomposition, decontextualization,
#     incomplete detection, irrelevant filtering, redundant filtering)
#   default per-component models from config/model_config.yaml
#
# Outputs:
#   scripts/data_preprocessing/atomic_fact_generation/example_scripts/output/
#     <model>_shard<k>of<N>_atomic_facts.jsonl
#     <model>_shard<k>of<N>_atomic_facts.jsonl.errors.jsonl
#     <model>_atomic_facts.json   (merged, written after all shards finish)

set -euo pipefail

# Record where the user invoked the script from (before we `cd`).
INVOCATION_CWD="$(pwd)"

# Script lives in `scripts/data_preprocessing/atomic_fact_generation/example_scripts`;
# repository root is four levels up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# Ensure CWD is `example_scripts` so relative script invocation works.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash run_bulk_atomic_facts_shards.sh \
    --model MODEL_DIR_NAME \
    --num-shards N \
    --api-key KEY \
    --base-url URL

Example:
  bash run_bulk_atomic_facts_shards.sh \
    --model qwen_qwen3.5-9b_tools_filter \
    --num-shards 6 \
    --api-key "$AZURE_OPENAI_KEY" \
    --base-url "https://YOUR_RESOURCE.openai.azure.com/"

Run 'python run_bulk_atomic_facts.py --logs-dir <logs-dir> --list-models' to see
available MODEL_DIR_NAME values.
EOF
}

MODEL=""
NUM_SHARDS=""
API_KEY=""
BASE_URL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --num-shards) NUM_SHARDS="${2:-}"; shift 2 ;;
    --api-key) API_KEY="${2:-}"; shift 2 ;;
    --base-url) BASE_URL="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${MODEL}" || -z "${NUM_SHARDS}" || -z "${API_KEY}" || -z "${BASE_URL}" ]]; then
  usage
  exit 2
fi

if ! [[ "${NUM_SHARDS}" =~ ^[0-9]+$ ]] || [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "--num-shards must be a positive integer, got: ${NUM_SHARDS}" >&2
  exit 1
fi

LOGS_DIR="${REPO_ROOT}/sciconharness/logs"
if [[ ! -d "${LOGS_DIR}" ]]; then
  echo "Logs directory not found: ${LOGS_DIR}" >&2
  exit 1
fi

if [[ ! -d "${LOGS_DIR}/${MODEL}" ]]; then
  echo "Model directory not found: ${LOGS_DIR}/${MODEL}" >&2
  echo "Available models:" >&2
  python3 run_bulk_atomic_facts.py --logs-dir "${LOGS_DIR}" --list-models >&2 || true
  exit 1
fi

OUTPUT_DIR="output"
mkdir -p "${OUTPUT_DIR}"

echo "Logs dir: ${LOGS_DIR}"
echo "Model: ${MODEL}"
echo "Shards: 0..$((NUM_SHARDS-1))"
echo "Output dir: ${OUTPUT_DIR}"

run_one() {
  local shard_id="$1"

  echo "=== shard ${shard_id}/${NUM_SHARDS} ==="
  python3 run_bulk_atomic_facts.py \
    --logs-dir "${LOGS_DIR}" \
    --model "${MODEL}" \
    --shard-id "${shard_id}" --num-shards "${NUM_SHARDS}" \
    --output-dir "${OUTPUT_DIR}" \
    --api-key "${API_KEY}" --base-url "${BASE_URL}"
}

# Default: run shards in parallel.
PARALLEL="${PARALLEL:-1}"
MAX_JOBS="${MAX_JOBS:-${NUM_SHARDS}}"

if [[ "${PARALLEL}" -eq 1 ]]; then
  for shard_id in $(seq 0 $((NUM_SHARDS-1))); do
    while [[ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_JOBS}" ]]; do
      sleep 2
    done
    run_one "${shard_id}" &
  done
  wait
else
  for shard_id in $(seq 0 $((NUM_SHARDS-1))); do
    run_one "${shard_id}"
  done
fi

echo "All shard runs finished. Merging into a single <model>_atomic_facts.json ..."

python3 run_bulk_atomic_facts.py \
  --model "${MODEL}" \
  --num-shards "${NUM_SHARDS}" \
  --output-dir "${OUTPUT_DIR}" \
  --merge

echo "Done."
