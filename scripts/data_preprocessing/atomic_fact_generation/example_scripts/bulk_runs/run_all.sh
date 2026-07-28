#!/usr/bin/env bash
# Runs all 4 key groups in parallel; each key group independently chains
# through its 3 assigned models sequentially (4 shards each), moving on to
# its next model as soon as its current one finishes — with NO waiting on
# the other 3 keys. E.g. if key 1's model finishes first, key 1 immediately
# starts its next model while keys 2-4 are still on their first.
#
#   Key 1 (AZURE_OPENAI_KEY):            DeepSeek-V4-Pro -> DeepSeek-V4-Pro_tools -> DeepSeek-V4-Pro_tools_filter
#   Key 2 (COCHRANE_DASHBOARD_*):        moonshotai_kimi-k3 -> ..._tools -> ..._tools_filter
#   Key 3 (SALAME_*):                    qwen_qwen3.5-9b -> ..._tools -> ..._tools_filter
#   Key 4 (SKYLOR_*):                    z-ai_glm-5.2 -> ..._tools -> ..._tools_filter
#
# At any moment, at most 4 shards are in flight per key (one model at a
# time), and up to 16 total across all 4 keys.
#
# Logs for each key group are written under ../output/logs/ so you can tail
# progress even though this script backgrounds all 4 groups.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../output/logs"
mkdir -p "${LOG_DIR}"

echo "Starting all 4 key groups in parallel (each chains its 3 models independently). Logs: ${LOG_DIR}"

bash "${SCRIPT_DIR}/run_group_key1_deepseek.sh" > "${LOG_DIR}/group_key1_deepseek.log" 2>&1 &
pid1=$!
bash "${SCRIPT_DIR}/run_group_key2_kimi.sh"     > "${LOG_DIR}/group_key2_kimi.log"     2>&1 &
pid2=$!
bash "${SCRIPT_DIR}/run_group_key3_qwen.sh"     > "${LOG_DIR}/group_key3_qwen.log"     2>&1 &
pid3=$!
bash "${SCRIPT_DIR}/run_group_key4_glm.sh"      > "${LOG_DIR}/group_key4_glm.log"      2>&1 &
pid4=$!

status=0
for pid in "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -eq 0 ]]; then
  echo "All 4 key groups (12 models) finished successfully."
else
  echo "One or more key groups failed — check logs in ${LOG_DIR}" >&2
fi
exit "${status}"
