#!/usr/bin/env bash
# Raw Qwen3.8-27B without the expert on the 300-image test (scripts/make_noexpert.py) on a fresh G4 session: creates
# it (retries every 5 min while G4 is unavailable), install + weights (q27t_setup.py), sandbox zips and prompts under a
# keepalive cell, vLLM server, colab/vm/q27_noexp.py, downloads noexp_results.zip into colab/noexp_job and stops the
# session. Log: outputs/agentic_ct_iqa/noexp/noexp_vm.log. Then: python -m scripts.noexp_results
# Usage: bash colab/noexp_launch.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
mkdir -p outputs/agentic_ct_iqa/noexp
LOG=outputs/agentic_ct_iqa/noexp/noexp_vm.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
: > $LOG
for t in $(seq 1 12); do
  $C new -s "$S" --gpu G4 > /dev/null 2>&1 && { say "session $S up (try $t)"; break; }
  say "G4 unavailable (try $t)"; sleep 300
done
bash colab/vmsh.sh 'mkdir -p /content/q27 && rm -f /content/hold_stop && echo dirs-ok' "$S" 2>&1 | grep -q dirs-ok && say "dirs ok"
$C exec -s "$S" -f "$P/colab/vm/q27t_setup.py" --timeout 3600 2>&1 | grep -E "Version|opencode|gpu|seconds|Error" | grep -v -i "token\|proxy" >> $LOG
say "setup done"
(timeout 7200 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 7000 2>&1 | grep -E "HOLD_DONE|Error" >> $LOG &)
sleep 20
for f in q27_serve.py q27_agent.py q27_noexp.py; do up "$P/colab/vm/$f" "/content/q27/$f"; done
up "$P/colab/noexp_job/prompts300.json" /content/q27/prompts300.json
n=0
for f in $(ls colab/noexp_job | grep "^ne.._batch_.*zip$"); do up "$P/colab/noexp_job/$f" "/content/q27/$f"; n=$((n+1)); done
say "UPLOADS_DONE $n zips"
: > "$SCRB/hold_stop"; up "$SCR/hold_stop" /content/hold_stop
sleep 40
$C exec -s "$S" -f "$P/colab/vm/q27_serve.py" --timeout 3600 2>&1 | grep -E "READY|SERVER DIED|max_model_len|Error" | grep -v -i "token\|proxy" >> $LOG
say "noexp start"
$C exec -s "$S" -f "$P/colab/vm/q27_noexp.py" --timeout 21600 2>&1 | grep -E "images|scores|NO SCORES|done|NOEXP_DONE|Error|Traceback" | grep -v -i "token\|proxy" >> $LOG
for t in $(seq 1 72); do  # if the exec stream dropped early, the job is still running in the kernel: wait for its zip
  bash colab/vmsh.sh 'ls /content/q27/noexp_results.zip && echo zip-ok' "$S" 2>&1 | grep -q zip-ok && break
  sleep 300
done
sleep 60
for t in 1 2 3; do $C download -s "$S" /content/q27/noexp_results.zip "$P/colab/noexp_job/noexp_results.zip" >/dev/null 2>&1 && { say "noexp_results.zip downloaded"; break; }; sleep 10; done
$C stop -s "$S" 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
