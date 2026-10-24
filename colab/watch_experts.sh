#!/usr/bin/env bash
# Local watcher: every 5 minutes, download the job log, record progress, and download each finished fold's
# predictions. Uses only the file API (colab ls / download), so it works while a keepalive cell holds the kernel.
# Exits when all five folds are downloaded, the job reports a failure, or the session is gone.
cd "$(dirname "$0")/.."
OUT=outputs/agentic_ct_iqa/experts
WSL_OUT="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper/$OUT"
mkdir -p "$OUT"
while true; do
  if ! bash colab/c.sh download -s expert /content/logs/experts.log "$WSL_OUT/experts_vm.log" >/dev/null 2>&1; then
    if ! bash colab/c.sh sessions 2>&1 | grep -q expert; then echo "$(date '+%F %T') session gone" >> "$OUT/progress.log"; exit 1; fi
  fi
  last=$(tr '\r' '\n' < "$OUT/experts_vm.log" | grep -E "=== FOLD|Epoch [0-9]+/|FAILED|ALL_FOLDS_DONE" | tail -2 | tr '\n' ' ')
  echo "$(date '+%F %T') $last" >> "$OUT/progress.log"
  listing=$(bash colab/c.sh ls -s expert /content/out 2>&1 | tr -d '\r')
  for k in 1 2 3 4 5; do
    if echo "$listing" | grep -q "preds_fold$k.csv" && [ ! -s "$OUT/preds_fold$k.csv" ]; then
      bash colab/c.sh download -s expert "/content/out/preds_fold$k.csv" "$WSL_OUT/preds_fold$k.csv" >/dev/null 2>&1 \
        && echo "$(date '+%F %T') downloaded preds_fold$k.csv" >> "$OUT/progress.log"
    fi
  done
  if ls "$OUT"/preds_fold{1,2,3,4,5}.csv >/dev/null 2>&1; then echo "$(date '+%F %T') ALL DOWNLOADED" >> "$OUT/progress.log"; exit 0; fi
  if tr '\r' '\n' < "$OUT/experts_vm.log" | grep -q "FAILED"; then echo "$(date '+%F %T') job failed" >> "$OUT/progress.log"; exit 1; fi
  sleep 300
done
