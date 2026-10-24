#!/usr/bin/env bash
# Tool ablation on a fresh G4 session: creates it (retries every 5 min while G4 is unavailable), setup, then the 5
# fold weights, train/test and the code under a keepalive cell, md5 check, colab/vm/tool_ablation.py, downloads
# ablation.npz into colab/vft_job and stops the session (an idle session keeps billing).
# Log: outputs/agentic_ct_iqa/vision_ft/ablation_vm.log. Then: python -m scripts.tool_ablation_eval
# Usage: bash colab/ablation_launch.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
LOG=outputs/agentic_ct_iqa/vision_ft/ablation_vm.log
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
$C exec -s "$S" -f "$P/colab/vm/vftb_setup.py" --timeout 3000 2>&1 | grep -E "checkpoint|gpu|SETUP_DONE|Error" | grep -v -i "token\|proxy" >> $LOG
(timeout 7200 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 7000 2>&1 | grep -E "HOLD_DONE|Error" >> $LOG &)
sleep 20
bash colab/vmsh.sh 'mkdir -p /content/vft && echo dirs-ok' "$S" 2>&1 | grep -q dirs-ok && say "dirs ok"
for f in colab/vft_job/parts/*; do up "$P/$f" "/content/vft/$(basename $f)"; done
for f in config.json preprocessor_config.json; do up "$P/colab/vft_job/$f" "/content/vft/$f"; done
up "$P/colab/vm/vftb_train.py" /content/vft/vftb_train.py
up "$SCR/staging/manifest_ablation.txt" /content/manifest.txt
say "small files done"
n=0
for f in $(ls "$SCRB/staging/vft" | grep "^weights_fold_"); do up "$SCR/staging/vft/$f" "/content/vft/$f"; n=$((n+1)); done
say "UPLOADS_DONE $n parts"
: > "$SCRB/hold_stop"; up "$SCR/hold_stop" /content/hold_stop
sleep 40
$C exec -s "$S" -f "$P/colab/vm/vft_assemble.py" 2>&1 | grep -E "OK|MISMATCH" | grep -v -i "token\|proxy" >> $LOG
$C exec -s "$S" -f "$P/colab/vm/assemble_parts.py" --timeout 1200 2>&1 | grep -E "OK|MISMATCH|Error" | grep -v -i "token\|proxy" >> $LOG
say "ablation start"
$C exec -s "$S" -f "$P/colab/vm/tool_ablation.py" --timeout 10800 2>&1 | grep -E "fold . done|ABLATION_DONE|Error|Traceback|memory" | grep -v -i "token\|proxy" >> $LOG
for t in 1 2 3; do $C download -s "$S" /content/vft/ablation.npz "$P/colab/vft_job/ablation.npz" >/dev/null 2>&1 && { say "ablation.npz downloaded"; break; }; sleep 10; done
$C stop -s "$S" 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
