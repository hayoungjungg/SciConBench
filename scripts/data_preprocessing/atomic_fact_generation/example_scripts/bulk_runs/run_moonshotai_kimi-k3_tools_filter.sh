#!/usr/bin/env bash
# Bulk atomic-fact generation for model: moonshotai_kimi-k3_tools_filter
# Key group 2/4: COCHRANE_DASHBOARD_OPENAI_KEY / COCHRANE_DASHBOARD_BASE_URL (4 shards)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  _val="$(python3 "${SCRIPT_DIR}/_env_get.py" "${REPO_ROOT}/.env" COCHRANE_DASHBOARD_OPENAI_KEY 2>/dev/null || true)"
  [[ -n "${_val}" ]] && COCHRANE_DASHBOARD_OPENAI_KEY="${_val}"
  _val="$(python3 "${SCRIPT_DIR}/_env_get.py" "${REPO_ROOT}/.env" COCHRANE_DASHBOARD_BASE_URL 2>/dev/null || true)"
  [[ -n "${_val}" ]] && COCHRANE_DASHBOARD_BASE_URL="${_val}"
fi

: "${COCHRANE_DASHBOARD_OPENAI_KEY:?COCHRANE_DASHBOARD_OPENAI_KEY not set (check .env or export it)}"
: "${COCHRANE_DASHBOARD_BASE_URL:?COCHRANE_DASHBOARD_BASE_URL not set (check .env or export it)}"

bash "${SCRIPT_DIR}/../run_bulk_atomic_facts_shards.sh" \
  --model "moonshotai_kimi-k3_tools_filter" \
  --num-shards 4 \
  --api-key "${COCHRANE_DASHBOARD_OPENAI_KEY}" \
  --base-url "${COCHRANE_DASHBOARD_BASE_URL}"
