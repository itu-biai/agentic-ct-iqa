#!/usr/bin/env bash
# Cross-dataset job on a ready G4 session (Git Bash, project root): setup, uploads under a keepalive cell, md5-checked
# reassembly, then Stage B (colab/vm/teacher.py) and MedSigLIP (colab/vm/medsig_pool.py) on cross.npz / cross.zip
# (scripts/cross_pool.py), both in cross mode; downloads cross_pred.npz and cross_fold<k>.csv into colab/vft_job.
# Log: outputs/agentic_ct_iqa/vision_ft/cross_vm.log. The session is left running; stop it after checking the files.
# Usage: bash colab/cross_launch.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
LOG=outputs/agentic_ct_iqa/vision_ft/cross_vm.log
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
timeout 120 wsl.exe -d Ubuntu -e bash -lc "export PATH=\"\$HOME/.local/bin:\$PATH\"; colab upload -s $S ~/.hf_token /content/.hf_token_upload 2>&1 | grep -c Uploaded" >> $LOG
: > "$SCRB/flags/cross"; up "$SCR/flags/cross" /content/vft/cross; up "$SCR/flags/cross" /content/tolga/cross_flag
say "small files done"
n=0
for f in $(ls "$SCRB/staging/vft" | grep "^weights_fold"); do up "$SCR/staging/vft/$f" "/content/vft/$f"; n=$((n+1)); [ $((n % 25)) = 0 ] && say "$n parts"; done
for f in $(ls "$SCRB/staging/tolga" | grep "^test.zip"); do up "$SCR/staging/tolga/$f" "/content/tolga/$f"; n=$((n+1)); done
while [ ! -f "$SCRB/staging/cross/ready" ]; do sleep 60; done  # scripts/cross_pool.py may still be running
up "$SCR/staging/manifest_cross.txt" /content/manifest.txt
for d in vft tolga; do for f in $(ls "$SCRB/staging/cross/$d"); do up "$SCR/staging/cross/$d/$f" "/content/$d/$f"; n=$((n+1)); [ $((n % 25)) = 0 ] && say "$n parts"; done; done
say "UPLOADS_DONE $n parts"
: > "$SCRB/hold_stop"; up "$SCR/hold_stop" /content/hold_stop
sleep 40
$C exec -s "$S" -f "$P/colab/vm/vft_assemble.py" 2>&1 | grep -E "OK|MISMATCH" | grep -v -i "token\|proxy" >> $LOG
$C exec -s "$S" -f "$P/colab/vm/assemble_parts.py" --timeout 1200 2>&1 | grep -E "OK|MISMATCH|Error" | grep -v -i "token\|proxy" >> $LOG
say "stage B start"
$C exec -s "$S" -f "$P/colab/vm/teacher.py" --timeout 14400 2>&1 | grep -E "teacher fold|TEACHER_DONE|Error|Traceback|memory" | grep -v -i "token\|proxy" >> $LOG
$C download -s "$S" /content/vft/cross_pred.npz "$P/colab/vft_job/cross_pred.npz" >/dev/null 2>&1 && say "cross_pred.npz downloaded"
say "medsiglip start"
$C exec -s "$S" -f "$P/colab/vm/medsig_pool.py" --timeout 7200 2>&1 | grep -E "^[1-5] |MEDSIG_DONE|setup s|Error|Traceback" | grep -v -i "token\|proxy" >> $LOG
mkdir -p colab/vft_job/medsig_cross
for k in 1 2 3 4 5; do $C download -s "$S" "/content/tolga/cross_fold$k.csv" "$P/colab/vft_job/medsig_cross/cross_fold$k.csv" >/dev/null 2>&1 && say "cross_fold$k.csv downloaded"; done
say "LAUNCH_END"
