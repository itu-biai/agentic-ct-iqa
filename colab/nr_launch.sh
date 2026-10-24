#!/usr/bin/env bash
# No-reference PSNR / SSIM features on the session that ran colab/cross_launch.sh (weights, cross.npz and the code are
# already on the VM): uploads nr.npz (scripts/nr_pool.py) under a keepalive cell, md5 check, colab/vm/nr_feat.py,
# downloads nr_feat.npz into colab/vft_job. Log: outputs/agentic_ct_iqa/vision_ft/nr_vm.log.
# Usage: bash colab/nr_launch.sh <session>
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
bash colab/vmsh.sh 'rm -f /content/hold_stop && echo cleared' "$S" 2>&1 | grep -q cleared && say "hold_stop cleared"
(timeout 7200 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 7000 2>&1 | grep -E "HOLD_DONE|Error" >> $LOG &)
sleep 20
up "$SCR/staging/manifest_nr.txt" /content/manifest.txt
n=0
for f in $(ls "$SCRB/staging/nr"); do up "$SCR/staging/nr/$f" "/content/vft/$f"; n=$((n+1)); done
say "UPLOADS_DONE $n parts"
: > "$SCRB/hold_stop"; up "$SCR/hold_stop" /content/hold_stop
sleep 40
$C exec -s "$S" -f "$P/colab/vm/assemble_parts.py" --timeout 1200 2>&1 | grep -E "OK|MISMATCH|Error" | grep -v -i "token\|proxy" >> $LOG
say "nr features start"
$C exec -s "$S" -f "$P/colab/vm/nr_feat.py" --timeout 10800 2>&1 | grep -E "features done|NR_DONE|Error|Traceback|memory" | grep -v -i "token\|proxy" >> $LOG
$C download -s "$S" /content/vft/nr_feat.npz "$P/colab/vft_job/nr_feat.npz" >/dev/null 2>&1 && say "nr_feat.npz downloaded"
say "LAUNCH_END"
