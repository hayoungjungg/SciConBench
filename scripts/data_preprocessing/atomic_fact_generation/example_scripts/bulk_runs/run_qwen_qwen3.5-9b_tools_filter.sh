#!/usr/bin/env bash
# Bulk atomic-fact generation for model: qwen_qwen3.5-9b_tools_filter
# Key group 3/4: SALAME_AZURE_OPENAI_KEY / SALAME_OPENAI_BASE_URL (4 shards)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

: "${SALAME_AZURE_OPENAI_KEY:?SALAME_AZURE_OPENAI_KEY not set (check .env)}"
: "${SALAME_OPENAI_BASE_URL:?SALAME_OPENAI_BASE_URL not set (check .env)}"

bash "${SCRIPT_DIR}/../run_bulk_atomic_facts_shards.sh" \
  --model "qwen_qwen3.5-9b_tools_filter" \
  --num-shards 4 \
  --api-key "${SALAME_AZURE_OPENAI_KEY}" \
  --base-url "${SALAME_OPENAI_BASE_URL}"
