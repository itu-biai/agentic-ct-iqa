#!/usr/bin/env bash
# Unattended finisher for the Qwen3.8-27B sandbox job (detached Windows process). Assumes session q27 is set up and the
# vLLM server is running (colab/vm/q27_setup.py, q27_serve.py). Runs the single-request protocol (setups 16/17),
# ingests it, then the six agent runs (setups 18/19), ingests them, rebuilds the results page, stops the session.
# Usage (Git Bash, project root): bash colab/q27_finish.sh
cd "$(dirname "$0")/.."
C="bash colab/c.sh"
S=q27
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
OUT=outputs/agentic_ct_iqa/sandbox
LOG=$OUT/q27_finish.log
RAW=$OUT/qwen38_raw.jsonl
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
finals() { grep -c '"final": true' "$RAW" 2>/dev/null || echo 0; }

log "finisher started"
for round in 1 2 3; do
  $C exec -s $S -f "$P/colab/vm/qwen38_single.py" --timeout 14400 >> $OUT/q27_single.log 2>&1
  $C download -s $S /content/q27/raw.jsonl "$P/$RAW" >/dev/null 2>&1
  log "single round $round: $(finals)/120"
  [ "$(finals)" -ge 120 ] && break
done
.venv/Scripts/python -m scripts.sandbox_qwen38 ingest "$RAW" >> "$LOG" 2>&1
log "single ingested; agent runs start"
for round in 1 2; do
  $C exec -s $S -f "$P/colab/vm/q27_agent_all.py" --timeout 36000 >> $OUT/q27_agent.log 2>&1
  grep -q Q27_AGENT_DONE $OUT/q27_agent.log && break
  log "agent round $round ended without DONE"
done
$C download -s $S /content/q27/agent.zip "$P/$OUT/qwen38_agent.zip" >/dev/null 2>&1
$C download -s $S /content/vllm.log "$P/$OUT/qwen38_vllm.log" >/dev/null 2>&1
.venv/Scripts/python -m scripts.sandbox_qwen38 agent "$OUT/qwen38_agent.zip" >> "$LOG" 2>&1
.venv/Scripts/python -m scripts.make_sandbox_page >> "$LOG" 2>&1
$C stop -s $S >> "$LOG" 2>&1 && log "colab session stopped"
log "DONE"
