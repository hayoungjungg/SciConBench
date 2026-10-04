#!/bin/bash
# Stop bulk-opus-rest-cosaita when round 1 ends (later rounds would only retry
# the abstained DOIs) and bulk-gpt61 once its run finishes.
cd /n/fs/hamcore/hayoung/SciConBench/scicon-track/logs/bulk
opus_done=0; gpt_done=0
while [ $opus_done -eq 0 ] || [ $gpt_done -eq 0 ]; do
  if [ $opus_done -eq 0 ] && rg -q 'Query round 2/3|^Done |RuntimeError' opus_rest_xhigh8k_cosaita.log; then
    sleep 5; tmux kill-session -t bulk-opus-rest-cosaita 2>/dev/null
    echo "$(date '+%F %T') killed bulk-opus-rest-cosaita"; opus_done=1
  fi
  if [ $gpt_done -eq 0 ] && { rg -q '^Done |RuntimeError' gpt61_first100.log || ! pgrep -f 'bulk_query_new_models.py --providers openai --limit 100' >/dev/null; }; then
    sleep 5; tmux kill-session -t bulk-gpt61 2>/dev/null
    echo "$(date '+%F %T') killed bulk-gpt61"; gpt_done=1
  fi
  sleep 30
done
