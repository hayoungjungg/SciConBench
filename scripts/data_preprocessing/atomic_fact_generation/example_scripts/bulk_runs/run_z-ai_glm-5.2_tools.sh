#!/usr/bin/env bash
# Bulk atomic-fact generation for model: z-ai_glm-5.2_tools
# Key group 4/4: SKYLOR_AZURE_OPENAI_KEY / SKYLOR_OPENAI_BASE_URL (4 shards)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

: "${SKYLOR_AZURE_OPENAI_KEY:?SKYLOR_AZURE_OPENAI_KEY not set (check .env)}"
: "${SKYLOR_OPENAI_BASE_URL:?SKYLOR_OPENAI_BASE_URL not set (check .env)}"

bash "${SCRIPT_DIR}/../run_bulk_atomic_facts_shards.sh" \
  --model "z-ai_glm-5.2_tools" \
  --num-shards 4 \
  --api-key "${SKYLOR_AZURE_OPENAI_KEY}" \
  --base-url "${SKYLOR_OPENAI_BASE_URL}"
