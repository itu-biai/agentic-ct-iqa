#!/usr/bin/env bash
# E16b (issue #329) on a fresh G4 session: creates it (retries every 5 min), install + weights (q27t_setup.py), sandbox
# zips under a keepalive cell, vLLM server (q27_serve.py), colab/vm/q27_e16b.py with nohup (polled, not streamed),
# downloads e16b_results.zip into colab/e16b_job and stops the session.
# Log: outputs/agentic_ct_iqa/i329/e16b_vm.log. Usage: bash colab/e16b_launch.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
LOG=outputs/agentic_ct_iqa/i329/e16b_vm.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" < /dev/null 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
sh_vm() { bash colab/vmsh.sh "$1" "$S" < /dev/null 2>&1; }
: > $LOG
rm -f "$SCRB/e16b_stop"
for t in $(seq 1 12); do
  $C new -s "$S" --gpu G4 < /dev/null > /dev/null 2>&1 && { say "session $S up (try $t)"; break; }
  say "G4 unavailable (try $t)"; sleep 300
done
bash colab/c.sh sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || { say "no session, exit"; exit 1; }
sh_vm 'mkdir -p /content/q27 && rm -f /content/hold_stop && echo dirs-ok' | grep -q dirs-ok && say "dirs ok"
$C exec -s "$S" -f "$P/colab/vm/q27t_setup.py" --timeout 3600 < /dev/null 2>&1 | grep -E "Version|opencode|gpu|seconds|Error" | grep -v -i "token\|proxy" >> $LOG
say "setup done"
( while [ ! -f "$SCRB/e16b_stop" ]; do timeout 11000 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 10900 < /dev/null >/dev/null 2>&1; sleep 5; [ -f "$SCRB/e16b_serve" ] && break; done ) &
sleep 20
for f in q27_serve.py q27_agent.py q27_e16b.py; do up "$P/colab/vm/$f" "/content/q27/$f"; done
n=0
for f in $(ls colab/e16b_job | grep "zip$"); do up "$P/colab/e16b_job/$f" "/content/q27/$f"; n=$((n+1)); done
say "UPLOADS_DONE $n zips"
# the server runs in the kernel (q27_serve.py), so the keepalive must end first
: > "$SCRB/e16b_serve"; sh_vm "touch /content/hold_stop" > /dev/null
sleep 40
$C exec -s "$S" -f "$P/colab/vm/q27_serve.py" --timeout 3600 < /dev/null 2>&1 | grep -E "READY|SERVER DIED|max_model_len|Error" | grep -v -i "token\|proxy" >> $LOG
sh_vm 'cd /content/q27 && (setsid nohup python q27_e16b.py > e16b.log 2>&1 < /dev/null &); echo bg' > /dev/null
say "e16b start"
( while [ ! -f "$SCRB/e16b_stop" ]; do timeout 11000 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 10900 < /dev/null >/dev/null 2>&1; sleep 5; done ) &
sh_vm 'rm -f /content/hold_stop; echo ok' > /dev/null
for t in $(seq 1 60); do
  sleep 300
  r=$(sh_vm 'tail -3 /content/q27/e16b.log | tr "\n" " "')
  say "progress: $(echo "$r" | grep -o "e16[fn]_batch_0[0-9] [A-Z a-z]*" | tail -2 | tr "\n" ";")"
  echo "$r" | grep -q E16B_DONE && break
done
for t in 1 2 3; do $C download -s "$S" /content/q27/e16b_results.zip "$P/colab/e16b_job/e16b_results.zip" < /dev/null >/dev/null 2>&1 && { say "e16b_results.zip downloaded"; break; }; sleep 10; done
: > "$SCRB/e16b_stop"
$C stop -s "$S" < /dev/null 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
