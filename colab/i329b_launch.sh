#!/usr/bin/env bash
# Issue #329, after the 2026-10-22 23:00 checkpoint: the rest of the GPU queue on two fresh G4 sessions (same setup and
# queue runner as colab/i329_launch.sh).
#   role a: queue "seed2 infer seed3". infer finishes E3/E17/E18 on fold models 4-5: the checkpoint's partial files
#           (folds 1-3) and pd/enc/cnn outputs go to /content/vft/i329 so infer329.py resumes and skips them (probe.npz is
#           an empty placeholder); uploads Stage B weights folds 4-5 + lab329 (manifest_a.txt) while seed2 trains.
#   role b: queue "notools phase2only infer". The retrained T5/T2 and T4 weights stay on the VM; infer runs only the
#           lab_notools / lab_p2only tasks (every other task has an empty placeholder); uploads lab329 (manifest_b.txt).
# Outputs (no weights) go to outputs/agentic_ct_iqa/i329/ (role b's retrained runs to out_notools_rerun /
# out_p2only_rerun, next to the original runs); the session is stopped when its queue is done.
# Uploads skip files already on the VM and retry missing ones (3 passes). Log: outputs/agentic_ct_iqa/i329/vm_<session>.log. Usage: bash colab/i329b_launch.sh <session> <a|b>
S="$1"; ROLE="$2"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
OUT=outputs/agentic_ct_iqa/i329
LOG=$OUT/vm_$S.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" < /dev/null 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
sh_vm() { bash colab/vmsh.sh "$1" "$S" < /dev/null 2>&1; }
get() { for t in 1 2 3; do $C download -s "$S" "$1" "$P/$2" < /dev/null >/dev/null 2>&1 && return 0; sleep 10; done; say "download FAIL $1"; }
push() {  # push <list file of "src|dst">: only files missing on the VM, 2 streams, up to 3 passes
  local list="$1" pass part pids
  for pass in 1 2 3; do
    present=$( { $C ls -s "$S" /content/vft; $C ls -s "$S" /content/vft/i329; } < /dev/null 2>/dev/null | grep -v "^\[colab\]" | tr "\n" " ")
    while IFS='|' read -r src dst; do case " $present " in *" $(basename "$dst") "*) ;; *) echo "$src|$dst" ;; esac; done < "$list" > "$list.todo"
    [ -s "$list.todo" ] || return 0
    say "upload pass $pass: $(wc -l < "$list.todo") files"
    rm -f "$list.todo_"*
    split -n l/2 -d "$list.todo" "$list.todo_"
    pids=""
    for part in "$list.todo_"0*; do
      ( while IFS='|' read -r src dst; do up "$src" "$dst"; done < "$part" ) &
      pids="$pids $!"
    done
    wait $pids
  done
  return 1
}
fetch_job() {  # outputs of one finished job, without weights
  local job="$1" dir=""
  get "/content/vft/q_$job.log" "$OUT/q_${job}_$S.log"
  case $job in
    notools) dir=out_notools ;; phase2only) dir=out_p2only ;; seed2) dir=out_seed2 ;; seed3) dir=out_seed3 ;;
  esac
  if [ -n "$dir" ]; then
    local dst=$dir; [ "$ROLE" = b ] && dst=${dir}_rerun
    mkdir -p "$OUT/$dst"
    for k in 1 2 3 4 5; do get "/content/vft/$dir/fold_$k.npz" "$OUT/$dst/fold_$k.npz"; done
    get "/content/vft/$dir/log.txt" "$OUT/$dst/log.txt"
  else
    mkdir -p "$OUT/infer"
    if [ "$ROLE" = a ]; then files="shuffle labshuf occl"; else files="lab_notools lab_p2only"; fi
    for f in $files; do get "/content/vft/i329/$f.npz" "$OUT/infer/$f.npz"; done
  fi
  say "fetched $job"
}
: > $LOG
rm -f "$SCRB/${S}_stop"
for t in $(seq 1 12); do
  $C new -s "$S" --gpu G4 < /dev/null > /dev/null 2>&1 && { say "session $S up (try $t)"; break; }
  say "G4 unavailable (try $t)"; sleep 300
done
bash colab/c.sh sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || { say "no session, exit"; exit 1; }
$C exec -s "$S" -f "$P/colab/vm/vftb_setup.py" --timeout 3000 < /dev/null 2>&1 | grep -E "checkpoint|gpu|SETUP_DONE|Error" | grep -v -i "token\|proxy" >> $LOG
( while [ ! -f "$SCRB/${S}_stop" ]; do timeout 11000 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 10900 < /dev/null >/dev/null 2>&1; sleep 5; done ) &
sleep 20
sh_vm 'mkdir -p /content/vft/i329 && rm -f /content/hold_stop && echo dirs-ok' | grep -q dirs-ok && say "dirs ok"
{
  for f in colab/vft_job/parts/*; do echo "$P/$f|/content/vft/$(basename $f)"; done
  for f in config.json preprocessor_config.json patient_folds.npy; do echo "$P/colab/vft_job/$f|/content/vft/$f"; done
  for f in vftb_train.py infer329.py queue329.py cnn_train.py vft_assemble.py assemble_parts.py; do echo "$P/colab/vm/$f|/content/vft/$f"; done
} > "$SCRB/${S}_small.txt"
push "$SCRB/${S}_small.txt" || say "small uploads incomplete"
if [ "$ROLE" = a ]; then
  for f in shuffle.partial labshuf.partial occl.partial pd enc cnn; do echo "$P/$OUT/infer/$f.npz|/content/vft/i329/$f.npz"; done > "$SCRB/${S}_i329.txt"
  push "$SCRB/${S}_i329.txt" || say "i329 uploads incomplete"
  sh_vm 'touch /content/vft/i329/probe.npz && echo ok' > /dev/null
  printf 'seed2 infer seed3\n' > "$SCRB/${S}_queue.txt"
else
  sh_vm 'cd /content/vft/i329 && touch shuffle.npz labshuf.npz occl.npz pd.npz enc.npz probe.npz cnn.npz && echo ok' > /dev/null
  printf 'notools phase2only infer\n' > "$SCRB/${S}_queue.txt"
fi
up "$SCR/${S}_queue.txt" /content/vft/queue.txt
up "$SCR/staging329/manifest_$ROLE.txt" /content/manifest.txt
sh_vm 'cd /content/vft && python vft_assemble.py 2>&1 | tail -1' | grep -q ASSEMBLED_OK && say "train/test assembled" || { say "ASSEMBLY FAILED, session kept"; : > "$SCRB/${S}_stop"; exit 1; }
sh_vm 'cd /content/vft && (setsid nohup python queue329.py > queue.log 2>&1 < /dev/null &) ; sleep 2; pgrep -f queue32[9] >/dev/null && echo queue-up' | grep -q queue-up && say "queue started"
# big uploads, three streams in parallel
{
  if [ "$ROLE" = a ]; then
    for f in $(ls "$SCRB/staging/vft" | grep "^weights_fold_[45]\."); do echo "$SCR/staging/vft/$f|/content/vft/$f"; done
  fi
  for f in $(ls "$SCRB/staging329/vft" | grep "^lab329"); do echo "$SCR/staging329/vft/$f|/content/vft/$f"; done
} > "$SCRB/${S}_uploads.txt"
push "$SCRB/${S}_uploads.txt" || say "big uploads incomplete"
say "UPLOADS_DONE $(wc -l < "$SCRB/${S}_uploads.txt") files"
sh_vm 'cd /content && (nohup python /content/vft/assemble_parts.py > /content/assemble.log 2>&1 < /dev/null &); echo bg' > /dev/null
for t in $(seq 1 30); do
  sleep 60
  r=$(sh_vm 'tail -1 /content/assemble.log')
  echo "$r" | grep -q ASSEMBLE_OK && { sh_vm 'touch /content/vft/uploads_ok && echo ok' | grep -q ok && say "weights assembled, uploads_ok"; break; }
  echo "$r" | grep -q -E "MISMATCH|Error" && { say "ASSEMBLE FAILED: $r"; break; }
done
# follow the queue: fetch each job's outputs when it finishes
fetched=" "
while true; do
  state=$(sh_vm 'cd /content/vft && ls done_* fail_* queue_done 2>/dev/null | tr "\n" " "')
  for m in $state; do
    case $m in
      done_*|fail_*) job=${m#*_}; case "$fetched" in *" $job "*) ;; *) say "$m"; fetch_job "$job"; fetched="$fetched$job " ;; esac ;;
    esac
  done
  echo "$state" | grep -q queue_done && break
  sh_vm 'tail -1 /content/vft/queue.log; ls -t /content/vft/q_*.log | head -1 | xargs tail -c 300 | tr "\r" "\n" | tail -1' | grep -v -i "token\|proxy" | tail -c 400 > "$OUT/now_$S.txt"
  sleep 300
done
get /content/vft/queue.log "$OUT/queue_$S.log"
: > "$SCRB/${S}_stop"
$C stop -s "$S" < /dev/null 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
