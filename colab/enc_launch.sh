#!/usr/bin/env bash
# Encoder-only tasks of issue #329 (colab/vm/enc329.py: E19, T1 lab, ResNeXt lab) on a fresh L4 session; downloads
# i329/{enc,probe,cnn}.npz into outputs/agentic_ct_iqa/i329/infer and stops the session. Log: outputs/agentic_ct_iqa/i329/enc_vm.log
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
OUT=outputs/agentic_ct_iqa/i329
LOG=$OUT/enc_vm.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" < /dev/null 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
sh_vm() { bash colab/vmsh.sh "$1" "$S" < /dev/null 2>&1; }
: > $LOG
rm -f "$SCRB/enc_stop"
$C new -s "$S" --gpu L4 < /dev/null > /dev/null 2>&1 && say "session $S up" || { say "no session"; exit 1; }
$C exec -s "$S" -f "$P/colab/vm/enc_setup.py" --timeout 1800 < /dev/null 2>&1 | grep -E "outside|SETUP_DONE|Error" | grep -v -i "token\|proxy" >> $LOG
( while [ ! -f "$SCRB/enc_stop" ]; do timeout 11000 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 10900 < /dev/null >/dev/null 2>&1; sleep 5; done ) &
sleep 20
sh_vm 'rm -f /content/hold_stop; echo ok' > /dev/null
for f in colab/vft_job/parts/*; do up "$P/$f" "/content/vft/$(basename $f)"; done
for f in config.json preprocessor_config.json; do up "$P/colab/vft_job/$f" "/content/vft/$f"; done
for f in vftb_train.py cnn_train.py enc329.py vft_assemble.py assemble_parts.py; do up "$P/colab/vm/$f" "/content/vft/$f"; done
up "$SCR/staging329/manifest_enc.txt" /content/manifest.txt
pids=""
( for f in $(ls "$SCRB/staging/vft" | grep "^weights_fold_1\."); do up "$SCR/staging/vft/$f" "/content/vft/$f"; done ) & pids="$pids $!"
( for f in $(ls "$SCRB/staging329/vft" | grep "^lab329"); do up "$SCR/staging329/vft/$f" "/content/vft/$f"; done ) & pids="$pids $!"
( for f in $(ls "$SCRB/staging329/vft/cnn"); do up "$SCR/staging329/vft/cnn/$f" "/content/vft/cnn/$f"; done ) & pids="$pids $!"
wait $pids
say "uploads done"
sh_vm 'cd /content/vft && python vft_assemble.py 2>&1 | grep -c OK' | grep -q 2 && say "train/test OK"
sh_vm 'cd /content && (nohup python /content/vft/assemble_parts.py > /content/assemble.log 2>&1 < /dev/null &); echo bg' > /dev/null
for t in $(seq 1 20); do sleep 30; r=$(sh_vm 'tail -1 /content/assemble.log'); echo "$r" | grep -q -E "ASSEMBLE_OK|MISMATCH|Error" && break; done
say "assemble: $(echo $r | tail -c 60)"
sh_vm 'cd /content/vft && (setsid nohup python enc329.py > enc.log 2>&1 < /dev/null &); echo bg' > /dev/null
say "enc start"
for t in $(seq 1 60); do
  sleep 120
  r=$(sh_vm 'tail -2 /content/vft/enc.log | tr "\n" " "')
  echo "$r" | grep -q ENC_DONE && break
  echo "$r" | grep -q -E "Traceback|Error" && { say "FAILED: $(echo "$r" | tail -c 300)"; break; }
done
mkdir -p $OUT/infer
for f in enc probe cnn; do $C download -s "$S" "/content/vft/i329/$f.npz" "$P/$OUT/infer/$f.npz" < /dev/null >/dev/null 2>&1 && say "$f.npz downloaded"; done
$C download -s "$S" /content/vft/enc.log "$P/$OUT/enc.log" < /dev/null >/dev/null 2>&1
: > "$SCRB/enc_stop"
$C stop -s "$S" < /dev/null 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
