#!/usr/bin/env bash
# E7 (issue #329) on a ready L4 session: keepalive cell, uploads (agaldran repository at 7f39d61, train/test parts,
# lab329.npz parts, colab/vm/agaldran329.py), md5 checks, then agaldran329.py with nohup; downloads every fold's
# probabilities into outputs/agentic_ct_iqa/i329/agaldran/ and stops the session.
# Log: outputs/agentic_ct_iqa/i329/agal_vm.log. Usage: bash colab/agal_launch.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
OUT=outputs/agentic_ct_iqa/i329/agaldran
mkdir -p $OUT
LOG=outputs/agentic_ct_iqa/i329/agal_vm.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" < /dev/null 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
sh_vm() { bash colab/vmsh.sh "$1" "$S" < /dev/null 2>&1; }
: > $LOG
rm -f "$SCRB/agal_stop"
( while [ ! -f "$SCRB/agal_stop" ]; do timeout 11000 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 10900 < /dev/null >/dev/null 2>&1; sleep 5; done ) &
sleep 20
sh_vm 'mkdir -p /content/vft /content/agal && rm -f /content/hold_stop && echo ok' | grep -q ok && say "dirs ok"
up "$SCR/agaldran_repo/repo.tgz" /content/agal/repo.tgz
for f in colab/vft_job/parts/*; do up "$P/$f" "/content/vft/$(basename $f)"; done
for f in $(ls "$SCRB/staging329/vft" | grep "^lab329"); do up "$SCR/staging329/vft/$f" "/content/vft/$f"; done
for f in vft_assemble.py agaldran329.py; do up "$P/colab/vm/$f" "/content/vft/$f"; done
LAB_MD5=$(grep lab329 "$SCRB/staging329/manifest329.txt" | cut -d' ' -f1)
say "uploads done"
sh_vm 'cd /content/vft && python vft_assemble.py 2>&1 | grep -c OK' | grep -q 2 && say "train/test OK"
sh_vm "cd /content/vft && cat lab329.npz.part* > lab329.npz && rm lab329.npz.part* && md5sum lab329.npz | cut -c1-32" | grep -q "$LAB_MD5" && say "lab329 OK"
sh_vm 'cd /content/agal && tar xzf repo.tgz && mv agaldran-ldct_iqa-* repo && ls repo/utils | head -2' | grep -q bit_models && say "repo ok"
sh_vm 'cd /content/agal && (setsid nohup python /content/vft/agaldran329.py > agal.log 2>&1 < /dev/null &); echo bg' > /dev/null
say "agaldran start"
got=" "
while true; do
  sleep 300
  state=$(sh_vm 'cd /content/agal && ls fold_*.npz 2>/dev/null | tr "\n" " "; tail -1 agal.log')
  for k in 1 2 3 4 5; do
    case "$state" in *"fold_$k.npz"*) case "$got" in *" $k "*) ;; *)
      $C download -s "$S" "/content/agal/fold_$k.npz" "$P/$OUT/fold_$k.npz" < /dev/null >/dev/null 2>&1 && { say "fold $k downloaded"; got="$got$k "; } ;; esac ;; esac
  done
  echo "$state" | grep -q AGAL_DONE && break
  echo "$state" | grep -q -E "Traceback|Error" && { say "FAILED: $(echo "$state" | tail -c 300)"; break; }
done
$C download -s "$S" /content/agal/agal.log "$P/$OUT/agal.log" < /dev/null >/dev/null 2>&1
for k in 1 2 3 4 5; do $C download -s "$S" "/content/agal/train_f$k.log" "$P/$OUT/train_f$k.log" < /dev/null >/dev/null 2>&1; done
: > "$SCRB/agal_stop"
$C stop -s "$S" < /dev/null 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
