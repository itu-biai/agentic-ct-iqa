#!/usr/bin/env bash
# Teacher-student run on a ready G4 session (Git Bash, project root): setup, uploads (train/test, pool, pseudo-labels,
# the `pseudo` flag), md5 checks, then colab/vm/vftb_train.py -> /content/vft/out_pseudo. Log: outputs/agentic_ct_iqa/vision_ft/student_vm.log
# Usage: bash colab/student_launch.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
LOG=outputs/agentic_ct_iqa/vision_ft/student_vm.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
: > $LOG
$C exec -s "$S" -f "$P/colab/vm/vftb_setup.py" --timeout 3000 2>&1 | grep -E "checkpoint|gpu|SETUP_DONE|Error" | grep -v -i "token\|proxy" >> $LOG
for f in colab/vft_job/parts/*; do up "$P/$f" "/content/vft/$(basename $f)"; done
for f in config.json preprocessor_config.json pseudo_labels.npz; do up "$P/colab/vft_job/$f" "/content/vft/$f"; done
up "$P/colab/vm/vftb_train.py" /content/vft/vftb_train.py
for f in $(ls "$SCRB/staging/vft" | grep "^pseudo.npz.part"); do up "$SCR/staging/vft/$f" "/content/vft/$f"; done
echo "$(md5sum < colab/vft_job/pseudo.npz | cut -d' ' -f1)  vft/pseudo.npz" > "$SCRB/manifest_student.txt"
up "$SCR/manifest_student.txt" /content/manifest.txt
: > "$SCRB/pseudo"; up "$SCR/pseudo" /content/vft/pseudo
say "uploads done"
$C exec -s "$S" -f "$P/colab/vm/vft_assemble.py" 2>&1 | grep -E "OK|MISMATCH" | grep -v -i "token\|proxy" >> $LOG
$C exec -s "$S" -f "$P/colab/vm/assemble_parts.py" --timeout 1200 2>&1 | grep -E "OK|MISMATCH|Error" | grep -v -i "token\|proxy" >> $LOG
say "student start"
$C exec -s "$S" -f "$P/colab/vm/vftb_train.py" --timeout 36000 > outputs/agentic_ct_iqa/vision_ft/student_train_vm.log 2>&1
say "LAUNCH_END"
