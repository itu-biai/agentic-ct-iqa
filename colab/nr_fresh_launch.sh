#!/usr/bin/env bash
# No-reference PSNR / SSIM features on a fresh G4 session: setup, then fold 1 weights, train/test, cross.npz and
# nr.npz (scripts/nr_pool.py) under a keepalive cell, md5 check, colab/vm/nr_feat.py,
# downloads nr_feat.npz into colab/vft_job and
# stops the session (an idle session keeps billing). Log: outputs/agentic_ct_iqa/vision_ft/nr_vm.log.
# Usage: bash colab/nr_fresh_launch.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
LOG=outputs/agentic_ct_iqa/vision_ft/nr_vm.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
: > $LOG
$C exec -s "$S" -f "$P/colab/vm/vftb_setup.py" --timeout 3000 2>&1 | grep -E "checkpoint|gpu|SETUP_DONE|Error" | grep -v -i "token\|proxy" >> $LOG
(timeout 7200 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 7000 2>&1 | grep -E "HOLD_DONE|Error" >> $LOG &)
sleep 20
bash colab/vmsh.sh 'mkdir -p /content/vft && echo dirs-ok' "$S" 2>&1 | grep -q dirs-ok && say "dirs ok"
for f in colab/vft_job/parts/*; do up "$P/$f" "/content/vft/$(basename $f)"; done
for f in config.json preprocessor_config.json; do up "$P/colab/vft_job/$f" "/content/vft/$f"; done
up "$P/colab/vm/vftb_train.py" /content/vft/vftb_train.py
up "$SCR/staging/manifest_nrfresh.txt" /content/manifest.txt
say "small files done"
n=0
for f in $(ls "$SCRB/staging/vft" | grep "^weights_fold_1.pt"); do up "$SCR/staging/vft/$f" "/content/vft/$f"; n=$((n+1)); done
for f in $(ls "$SCRB/staging/cross/vft"); do up "$SCR/staging/cross/vft/$f" "/content/vft/$f"; n=$((n+1)); done
for f in $(ls "$SCRB/staging/nr"); do up "$SCR/staging/nr/$f" "/content/vft/$f"; n=$((n+1)); done
say "UPLOADS_DONE $n parts"
: > "$SCRB/hold_stop"; up "$SCR/hold_stop" /content/hold_stop
sleep 40
$C exec -s "$S" -f "$P/colab/vm/vft_assemble.py" 2>&1 | grep -E "OK|MISMATCH" | grep -v -i "token\|proxy" >> $LOG
$C exec -s "$S" -f "$P/colab/vm/assemble_parts.py" --timeout 1200 2>&1 | grep -E "OK|MISMATCH|Error" | grep -v -i "token\|proxy" >> $LOG
say "nr features start"
$C exec -s "$S" -f "$P/colab/vm/nr_feat.py" --timeout 10800 2>&1 | grep -E "features done|NR_DONE|Error|Traceback|memory" | grep -v -i "token\|proxy" >> $LOG
for t in 1 2 3; do $C download -s "$S" /content/vft/nr_feat.npz "$P/colab/vft_job/nr_feat.npz" >/dev/null 2>&1 && { say "nr_feat.npz downloaded"; break; }; sleep 10; done
$C stop -s "$S" 2>&1 | tr -d '' | tail -1 >> $LOG
say "LAUNCH_END"
