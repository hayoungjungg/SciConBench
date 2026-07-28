#!/usr/bin/env bash
# Key group 4/4: SKYLOR_AZURE_OPENAI_KEY / SKYLOR_OPENAI_BASE_URL
# Runs its 3 assigned models sequentially (each internally shards 4-way in parallel).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_z-ai_glm-5.2.sh"
bash "${SCRIPT_DIR}/run_z-ai_glm-5.2_tools.sh"
bash "${SCRIPT_DIR}/run_z-ai_glm-5.2_tools_filter.sh"
