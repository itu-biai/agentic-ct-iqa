#!/usr/bin/env bash
# Unattended finisher for the MedGemma sandbox job (runs as a detached Windows process, so it outlives Claude's
# background-task limits). Keeps the job going until all 120 answers are in raw.jsonl: re-runs the resumable
# medgemma_run.py when an exec ends early (a queued exec waits for a running one, then skips finished work), and
# rebuilds the session from scratch, uploading the partial raw.jsonl, if Colab took it away. At the end: download,
# ingest into the standard result folders, rebuild the results page, stop the Colab session, write DONE.
# Usage (Git Bash, project root): bash colab/medgemma_finish.sh
cd "$(dirname "$0")/.."
C="bash colab/c.sh"
S=mg
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
OUT=outputs/agentic_ct_iqa/sandbox
LOG=$OUT/medgemma_finish.log
RAW=$OUT/medgemma_raw.jsonl
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
finals() { grep -c '"final": true' "$RAW" 2>/dev/null || echo 0; }
pull() { $C download -s $S /content/mg/raw.jsonl "$P/$RAW" >/dev/null 2>&1; }

rebuild() {
  log "session gone, rebuilding"
  $C new -s $S --gpu T4 >> "$LOG" 2>&1 || return 1
  echo "import os; os.makedirs('/content/mg/images', exist_ok=True); os.makedirs('/root/.cache/huggingface', exist_ok=True)" | $C exec -s $S >/dev/null
  for f in prompts.json images.zip; do $C upload -s $S "$P/colab/medgemma_job/$f" "/content/mg/$f" >/dev/null || return 1; done
  $C upload -s $S "$P/colab/vm/medgemma_run.py" /content/mg/medgemma_run.py >/dev/null || return 1
  [ -s "$RAW" ] && { $C upload -s $S "$P/$RAW" /content/mg/raw.jsonl >/dev/null || return 1; }
  timeout 120 wsl.exe -d Ubuntu -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; colab upload -s mg ~/.hf_token /content/.hf_token_upload' >/dev/null || return 1
  $C exec -s $S -f "$P/colab/vm/mg_setup.py" --timeout 300 >> "$LOG" 2>&1
}

log "finisher started, $(finals)/120 answers so far"
for round in 1 2 3 4 5 6; do
  pull
  [ "$(finals)" -ge 120 ] && break
  if ! $C sessions 2>&1 | grep -qw $S; then rebuild || { log "rebuild failed (no T4?), waiting 30 min"; sleep 1800; continue; }; fi
  log "round $round: exec (resumes from $(finals)/120)"
  $C exec -s $S -f "$P/colab/vm/medgemma_run.py" --timeout 14400 >> $OUT/medgemma_exec.log 2>&1
  pull
  log "round $round ended: $(finals)/120"
done

pull
log "final count $(finals)/120; ingesting"
.venv/Scripts/python -m scripts.sandbox_medgemma ingest "$RAW" >> "$LOG" 2>&1
.venv/Scripts/python -m scripts.make_sandbox_page >> "$LOG" 2>&1
$C stop -s $S >> "$LOG" 2>&1 && log "colab session stopped"
log "DONE"
