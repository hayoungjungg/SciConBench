#!/usr/bin/env bash
# Key group 3/4: SALAME_AZURE_OPENAI_KEY / SALAME_OPENAI_BASE_URL
# Runs its 3 assigned models sequentially (each internally shards 4-way in parallel).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_qwen_qwen3.5-9b.sh"
bash "${SCRIPT_DIR}/run_qwen_qwen3.5-9b_tools.sh"
bash "${SCRIPT_DIR}/run_qwen_qwen3.5-9b_tools_filter.sh"
