#!/usr/bin/env bash
# Key group 2/4: COCHRANE_DASHBOARD_OPENAI_KEY / COCHRANE_DASHBOARD_BASE_URL
# Runs its 3 assigned models sequentially (each internally shards 4-way in parallel).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_moonshotai_kimi-k3.sh"
bash "${SCRIPT_DIR}/run_moonshotai_kimi-k3_tools.sh"
bash "${SCRIPT_DIR}/run_moonshotai_kimi-k3_tools_filter.sh"
