#!/usr/bin/env bash
# Teacher job on a ready G4 session (Git Bash, project root): setup, uploads under a keepalive cell (an idle kernel
# lets Colab reclaim the VM), md5-checked reassembly, then colab/vm/teacher.py. Log: outputs/agentic_ct_iqa/vision_ft/teacher_vm.log
# Usage: bash colab/teacher_launch.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
LOG=outputs/agentic_ct_iqa/vision_ft/teacher_vm.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {  # upload with up to 3 tries
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
: > $LOG
$C exec -s "$S" -f "$P/colab/vm/vftb_setup.py" --timeout 3000 2>&1 | grep -E "checkpoint|gpu|SETUP_DONE|Error" | grep -v -i "token\|proxy" >> $LOG
(timeout 14400 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 12000 2>&1 | grep -E "HOLD_DONE|Error" >> $LOG &)
sleep 20
bash colab/vmsh.sh 'mkdir -p /content/tolga/repo /content/vft && echo dirs-ok' "$S" 2>&1 | grep -q dirs-ok && say "dirs ok"
for f in colab/vft_job/parts/*; do up "$P/$f" "/content/vft/$(basename $f)"; done
for f in config.json preprocessor_config.json; do up "$P/colab/vft_job/$f" "/content/vft/$f"; done
up "$P/colab/vm/vftb_train.py" /content/vft/vftb_train.py
for f in config.py data.py model.py test.py predict.py train.py; do up "$SCR/tolga_repo/repo/$f" "/content/tolga/repo/$f"; done
up "$SCR/staging/manifest.txt" /content/manifest.txt
timeout 120 wsl.exe -d Ubuntu -e bash -lc "export PATH=\"\$HOME/.local/bin:\$PATH\"; colab upload -s $S ~/.hf_token /content/.hf_token_upload 2>&1 | grep -c Uploaded" >> $LOG
say "small files done"
n=0
for d in tolga vft; do for f in $(ls "$SCRB/staging/$d"); do up "$SCR/staging/$d/$f" "/content/$d/$f"; n=$((n+1)); [ $((n % 25)) = 0 ] && say "$n parts"; done; done
say "UPLOADS_DONE $n parts"
: > "$SCRB/hold_stop"; up "$SCR/hold_stop" /content/hold_stop
sleep 40
$C exec -s "$S" -f "$P/colab/vm/vft_assemble.py" 2>&1 | grep -E "OK|MISMATCH" | grep -v -i "token\|proxy" >> $LOG
$C exec -s "$S" -f "$P/colab/vm/assemble_parts.py" --timeout 1200 2>&1 | grep -E "OK|MISMATCH|Error" | grep -v -i "token\|proxy" >> $LOG
say "teacher start"
$C exec -s "$S" -f "$P/colab/vm/teacher.py" --timeout 14400 2>&1 | grep -E "teacher fold|TEACHER_DONE|Error|Traceback|memory" | grep -v -i "token\|proxy" >> $LOG
say "LAUNCH_END"
