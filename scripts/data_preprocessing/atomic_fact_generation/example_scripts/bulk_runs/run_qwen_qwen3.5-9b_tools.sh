#!/usr/bin/env bash
# Bulk atomic-fact generation for model: qwen_qwen3.5-9b_tools
# Key group 3/4: SALAME_AZURE_OPENAI_KEY / SALAME_OPENAI_BASE_URL (4 shards)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  _val="$(python3 "${SCRIPT_DIR}/_env_get.py" "${REPO_ROOT}/.env" SALAME_AZURE_OPENAI_KEY 2>/dev/null || true)"
  [[ -n "${_val}" ]] && SALAME_AZURE_OPENAI_KEY="${_val}"
  _val="$(python3 "${SCRIPT_DIR}/_env_get.py" "${REPO_ROOT}/.env" SALAME_OPENAI_BASE_URL 2>/dev/null || true)"
  [[ -n "${_val}" ]] && SALAME_OPENAI_BASE_URL="${_val}"
fi

: "${SALAME_AZURE_OPENAI_KEY:?SALAME_AZURE_OPENAI_KEY not set (check .env or export it)}"
: "${SALAME_OPENAI_BASE_URL:?SALAME_OPENAI_BASE_URL not set (check .env or export it)}"

bash "${SCRIPT_DIR}/../run_bulk_atomic_facts_shards.sh" \
  --model "qwen_qwen3.5-9b_tools" \
  --num-shards 4 \
  --api-key "${SALAME_AZURE_OPENAI_KEY}" \
  --base-url "${SALAME_OPENAI_BASE_URL}"
