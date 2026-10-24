#!/usr/bin/env bash
# Detached retry loop: every 30 minutes (for at most 24 hours) try to get a free T4 and resume the expert training.
# Never opens a second session: if an `expert` session exists, it stops. Log: outputs/agentic_ct_iqa/experts/progress.log
cd "$(dirname "$0")/.."
LOG=outputs/agentic_ct_iqa/experts/progress.log
for attempt in $(seq 1 48); do
  if bash colab/c.sh sessions 2>&1 | grep -q "\[expert\]"; then
    echo "$(date '+%F %T') retry: an expert session already exists, stopping retries" >> "$LOG"; exit 0
  fi
  echo "$(date '+%F %T') retry $attempt: trying to resume on a new T4" >> "$LOG"
  if bash colab/resume_experts.sh >> outputs/agentic_ct_iqa/experts/resume.log 2>&1; then
    echo "$(date '+%F %T') retry $attempt: resumed" >> "$LOG"; exit 0
  fi
  sleep 1800
done
echo "$(date '+%F %T') retry: gave up after 24 hours" >> "$LOG"
