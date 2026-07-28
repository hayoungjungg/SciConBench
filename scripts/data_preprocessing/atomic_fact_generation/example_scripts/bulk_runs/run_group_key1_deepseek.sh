#!/usr/bin/env bash
# Key group 1/4: AZURE_OPENAI_KEY / OPENAI_BASE_URL
# Runs its 3 assigned models sequentially (each internally shards 4-way in parallel).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_DeepSeek-V4-Pro.sh"
bash "${SCRIPT_DIR}/run_DeepSeek-V4-Pro_tools.sh"
bash "${SCRIPT_DIR}/run_DeepSeek-V4-Pro_tools_filter.sh"
