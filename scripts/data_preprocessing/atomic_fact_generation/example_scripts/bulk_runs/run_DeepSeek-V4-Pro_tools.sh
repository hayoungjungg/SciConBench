#!/usr/bin/env bash
# Bulk atomic-fact generation for model: DeepSeek-V4-Pro_tools
# Key group 1/4: AZURE_OPENAI_KEY / OPENAI_BASE_URL (4 shards)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  _val="$(python3 "${SCRIPT_DIR}/_env_get.py" "${REPO_ROOT}/.env" AZURE_OPENAI_KEY 2>/dev/null || true)"
  [[ -n "${_val}" ]] && AZURE_OPENAI_KEY="${_val}"
  _val="$(python3 "${SCRIPT_DIR}/_env_get.py" "${REPO_ROOT}/.env" OPENAI_BASE_URL 2>/dev/null || true)"
  [[ -n "${_val}" ]] && OPENAI_BASE_URL="${_val}"
fi

: "${AZURE_OPENAI_KEY:?AZURE_OPENAI_KEY not set (check .env or export it)}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL not set (check .env or export it)}"

bash "${SCRIPT_DIR}/../run_bulk_atomic_facts_shards.sh" \
  --model "DeepSeek-V4-Pro_tools" \
  --num-shards 4 \
  --api-key "${AZURE_OPENAI_KEY}" \
  --base-url "${OPENAI_BASE_URL}"
