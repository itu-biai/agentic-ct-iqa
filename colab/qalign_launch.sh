#!/usr/bin/env bash
# E11 (issue #329) on a fresh G4 session: keepalive cell, uploads (Q-Align at fd0c3de, train/test parts, lab329 parts,
# colab/vm/qalign329.{py,sh}, the HF token without printing it), md5 checks, colab/vm/qalign329.sh with nohup (polled),
# downloads every fold's level logits into outputs/agentic_ct_iqa/i329/qalign/ and stops the session.
# Log: outputs/agentic_ct_iqa/i329/qalign_vm.log. Usage: bash colab/qalign_launch.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
OUT=outputs/agentic_ct_iqa/i329/qalign
mkdir -p $OUT
LOG=outputs/agentic_ct_iqa/i329/qalign_vm.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" < /dev/null 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
sh_vm() { bash colab/vmsh.sh "$1" "$S" < /dev/null 2>&1; }
: > $LOG
rm -f "$SCRB/qalign_stop"
for t in $(seq 1 12); do
  $C new -s "$S" --gpu G4 < /dev/null > /dev/null 2>&1 && { say "session $S up (try $t)"; break; }
  say "G4 unavailable (try $t)"; sleep 300
done
bash colab/c.sh sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || { say "no session, exit"; exit 1; }
( while [ ! -f "$SCRB/qalign_stop" ]; do timeout 11000 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 10900 < /dev/null >/dev/null 2>&1; sleep 5; done ) &
sleep 20
sh_vm 'mkdir -p /content/vft /content/qa && rm -f /content/hold_stop && echo ok' | grep -q ok && say "dirs ok"
up "$SCR/qalign.tgz" /content/qa/qalign.tgz
for f in qalign329.py qalign329.sh; do up "$P/colab/vm/$f" "/content/qa/$f"; done
up "$P/colab/vm/vft_assemble.py" /content/vft/vft_assemble.py
timeout 120 wsl.exe -d Ubuntu -e bash -lc "export PATH=\"\$HOME/.local/bin:\$PATH\"; colab upload -s $S ~/.hf_token /content/.hf_token_upload 2>&1 | grep -c Uploaded" < /dev/null >> $LOG
for f in colab/vft_job/parts/*; do up "$P/$f" "/content/vft/$(basename $f)"; done
for f in $(ls "$SCRB/staging329/vft" | grep "^lab329"); do up "$SCR/staging329/vft/$f" "/content/vft/$f"; done
LAB_MD5=$(grep lab329 "$SCRB/staging329/manifest329.txt" | cut -d' ' -f1)
say "uploads done"
sh_vm 'cd /content/vft && python vft_assemble.py 2>&1 | grep -c OK' | grep -q 2 && say "train/test OK"
sh_vm "cd /content/vft && cat lab329.npz.part* > lab329.npz && rm lab329.npz.part* && md5sum lab329.npz | cut -c1-32" | grep -q "$LAB_MD5" && say "lab329 OK"
sh_vm 'cd /content/qa && (setsid nohup bash qalign329.sh > qalign.log 2>&1 < /dev/null &); echo bg' > /dev/null
say "qalign start"
got=" "
while true; do
  sleep 300
  state=$(sh_vm 'cd /content/qa && ls fold_*.npz 2>/dev/null | tr "\n" " "; tail -2 qalign.log | tr "\n" " "')
  say "state: $(echo "$state" | grep -v -i token | tail -c 200)"
  for k in 1 2 3 4 5; do
    case "$state" in *"fold_$k.npz"*) case "$got" in *" $k "*) ;; *)
      $C download -s "$S" "/content/qa/fold_$k.npz" "$P/$OUT/fold_$k.npz" < /dev/null >/dev/null 2>&1 && { say "fold $k downloaded"; got="$got$k "; } ;; esac ;; esac
  done
  echo "$state" | grep -q QALIGN_DONE && break
  echo "$state" | grep -q -E "Traceback|Error|error:" && { say "FAILED"; break; }
done
$C download -s "$S" /content/qa/qalign.log "$P/$OUT/qalign.log" < /dev/null >/dev/null 2>&1
for k in 1 2 3 4 5; do $C download -s "$S" "/content/qa/train_$k.log" "$P/$OUT/train_$k.log" < /dev/null >/dev/null 2>&1; done
: > "$SCRB/qalign_stop"
$C stop -s "$S" < /dev/null 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
