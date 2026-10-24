#!/usr/bin/env bash
# E12 Stage B seed3 (vftb_train.py flag seed3) on a fresh G4: setup, small files, md5 check (nohup), one queue runner,
# out_seed3 fetched into outputs/agentic_ct_iqa/i329/out_seed3, session stopped (also below 3 CU).
# Log: outputs/agentic_ct_iqa/i329/vm_<session>.log. Usage: bash colab/seed3_launch.sh <session> a
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
for t in 1 2 3; do bash colab/c.sh sessions < /dev/null 2>&1 | grep -q "^\[$S\]" && break; sleep 20; done
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
printf 'seed3\n' > "$SCRB/${S}_queue.txt"
up "$SCR/${S}_queue.txt" /content/vft/queue.txt
sh_vm 'cd /content/vft && (setsid nohup python vft_assemble.py > vft_assemble.log 2>&1 < /dev/null &); echo bg' > /dev/null
r=""
for t in $(seq 1 30); do
  sleep 20; r=$(sh_vm 'tail -1 /content/vft/vft_assemble.log')
  echo "$r" | grep -q -E "ASSEMBLED_OK|Error|MISMATCH" && break
done
echo "$r" | grep -q ASSEMBLED_OK && say "train/test md5 OK" || { say "ASSEMBLY FAILED: $r (session kept)"; exit 1; }
# start the queue exactly once (vmsh never re-runs a command that started), then check there is one runner
sh_vm 'cd /content/vft && (setsid nohup python queue329.py > queue.log 2>&1 < /dev/null &); echo bg' > /dev/null
sleep 15
n=$(sh_vm 'pgrep -fc queue32[9]')
say "queue runners: $n"
fetched=" "
while true; do
  state=$(sh_vm 'cd /content/vft && ls done_* fail_* queue_done 2>/dev/null | tr "\n" " "')
  for m in $state; do
    case $m in
      done_*|fail_*) job=${m#*_}; case "$fetched" in *" $job "*) ;; *) say "$m"; fetch_job "$job"; fetched="$fetched$job " ;; esac ;;
    esac
  done
  echo "$state" | grep -q queue_done && break
  sh_vm 'tail -n 1 /content/vft/out_seed3/log.txt' | grep -v -i "token\|proxy" | tail -c 300 > "$OUT/now_$S.txt"
  bal=$($C usage < /dev/null 2>&1 | grep -i balance | grep -o "[0-9.]*" | head -1)
  [ -n "$bal" ] && awk "BEGIN {exit !($bal < 3)}" && { say "balance $bal CU: stopping"; break; }
  sleep 300
done
get /content/vft/queue.log "$OUT/queue_$S.log"
: > "$SCRB/${S}_stop"
$C stop -s "$S" < /dev/null 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
