#!/usr/bin/env bash
# Unattended finisher for the Empero Qwen3.8-4B-Distill sandbox job (detached Windows process). Assumes session e4 is
# set up and the vLLM server is running (colab/vm/e4_setup.py, e4_serve.py). Runs the single-request protocol (setups
# 20/21/24), ingests it, then the nine agent runs (setups 22/23/25), ingests them, rebuilds the results page, stops the
# session.
# Usage (Git Bash, project root): bash colab/e4_finish.sh
cd "$(dirname "$0")/.."
C="bash colab/c.sh"
S=e4
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
OUT=outputs/agentic_ct_iqa/sandbox
LOG=$OUT/e4_finish.log
RAW=$OUT/empero_raw.jsonl
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
finals() { grep -c '"final": true' "$RAW" 2>/dev/null || echo 0; }

log "finisher started"
for round in 1 2 3; do
  $C exec -s $S -f "$P/colab/vm/empero_single.py" --timeout 21600 >> $OUT/e4_single.log 2>&1
  $C download -s $S /content/e4/raw.jsonl "$P/$RAW" >/dev/null 2>&1
  log "single round $round: $(finals)/180"
  [ "$(finals)" -ge 180 ] && break
done
.venv/Scripts/python -m scripts.sandbox_empero ingest "$RAW" >> "$LOG" 2>&1
log "single ingested; agent runs start"
for round in 1 2; do
  $C exec -s $S -f "$P/colab/vm/e4_agent_all.py" --timeout 54000 >> $OUT/e4_agent.log 2>&1
  grep -q E4_AGENT_DONE $OUT/e4_agent.log && break
  log "agent round $round ended without DONE"
done
$C download -s $S /content/e4/agent.zip "$P/$OUT/empero_agent.zip" >/dev/null 2>&1
$C download -s $S /content/vllm.log "$P/$OUT/empero_vllm.log" >/dev/null 2>&1
.venv/Scripts/python -m scripts.sandbox_empero agent "$OUT/empero_agent.zip" >> "$LOG" 2>&1
.venv/Scripts/python -m scripts.make_sandbox_page >> "$LOG" 2>&1
$C stop -s $S >> "$LOG" 2>&1 && log "colab session stopped"
log "DONE"
