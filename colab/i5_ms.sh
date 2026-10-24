#!/usr/bin/env bash
# i5 after the 2026-10-23 restarts: follows its Stage B queue (notools, phase2only, infer), uploads the MedSigLIP data
# meanwhile, then runs E12 MedSigLIP seeds 43 and 44 side by side on the G4 if the balance allows, fetches and stops.
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
# ---- i5 after its Stage B queue: E12 MedSigLIP seeds 43 and 44 side by side on the G4 (they do not fit an L4)
say "ms on $S: uploading MedSigLIP data while the queue runs"
( while [ ! -f "$SCRB/${S}_stop" ]; do timeout 11000 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 10900 < /dev/null >/dev/null 2>&1; sleep 5; done ) &
sh_vm 'mkdir -p /content/ms/repo && echo ok' > /dev/null
for f in config.py data.py model.py train.py test.py; do up "$SCR/tolga_repo/repo/$f" "/content/ms/repo/$f"; done
up "$P/colab/vm/medsig_seed.py" /content/ms/medsig_seed.py
timeout 120 wsl.exe -d Ubuntu -e bash -lc "export PATH=\"\$HOME/.local/bin:\$PATH\"; colab upload -s $S ~/.hf_token /content/.hf_token_upload 2>&1 | grep -c Uploaded" < /dev/null >> $LOG
for pass in 1 2 3; do
  present=$($C ls -s "$S" /content/ms < /dev/null 2>/dev/null | grep -v "^\[colab\]" | tr "\n" " ")
  ls "$SCRB/staging329/ms" | while read -r f; do case " $present " in *" $f "*) ;; *) echo "$f" ;; esac; done > "$SCRB/${S}_ms.txt"
  [ -s "$SCRB/${S}_ms.txt" ] || break
  say "ms upload pass $pass: $(wc -l < "$SCRB/${S}_ms.txt") parts"
  rm -f "${SCRB:?}/${S:?}_ms_"*
  split -n l/2 -d "$SCRB/${S}_ms.txt" "$SCRB/${S}_ms_"
  pids=""
  for part in "$SCRB/${S}_ms_"0*; do ( while read -r f; do up "$SCR/staging329/ms/$f" "/content/ms/$f"; done < "$part" ) & pids="$pids $!"; done
  wait $pids
done
up "$SCR/staging329/manifest_ms.txt" /content/manifest.txt
sh_vm 'cd /content && (nohup python /content/vft/assemble_parts.py > /content/assemble_ms.log 2>&1 < /dev/null &); echo bg' > /dev/null
r=""
for t in $(seq 1 30); do sleep 30; r=$(sh_vm 'tail -1 /content/assemble_ms.log'); echo "$r" | grep -q -E "ASSEMBLE_OK|MISMATCH|Error" && break; done
say "ms assemble: $(echo $r | tail -c 40)"
sh_vm 'cd /content/ms && mkdir -p data && (nohup tar -xf ldct.tar -C data > untar.log 2>&1 < /dev/null &); echo bg' > /dev/null
# follow the Stage B queue
fetched=" "
while true; do
  state=$(sh_vm 'cd /content/vft && ls done_* fail_* queue_done 2>/dev/null | tr "\n" " "')
  for m in $state; do
    case $m in
      done_*) job=${m#*_}; case "$fetched" in *" $job "*) ;; *) say "$m"; fetch_job "$job"; fetched="$fetched$job " ;; esac ;;
      fail_*) job=${m#*_}; case "$fetched" in *" $job "*) ;; *) say "$m"; fetched="$fetched$job " ;; esac ;;
    esac
  done
  echo "$state" | grep -q queue_done && break
  sh_vm 'tail -1 /content/vft/queue.log' | grep -v -i "token\|proxy" | tail -c 300 > "$OUT/now_$S.txt"
  sleep 300
done
get /content/vft/queue.log "$OUT/queue_$S.log"
bal=$($C usage < /dev/null 2>&1 | grep -i balance | grep -o "[0-9.]*" | head -1)
say "queue done; balance $bal CU"
if awk "BEGIN {exit !($bal >= 16)}"; then
  seeds="43"; awk "BEGIN {exit !($bal >= 30)}" && seeds="43 44"
  sh_vm "cd /content/ms && for s in $seeds; do (setsid nohup python medsig_seed.py \$s > seed\$s.log 2>&1 < /dev/null &); done; echo bg" > /dev/null
  say "MedSigLIP seeds $seeds started"
  for t in $(seq 1 60); do
    sleep 300
    r=$(sh_vm 'pgrep -fc medsig_see[d]')
    sh_vm 'cd /content/ms; for s in 43 44; do tail -c 300 seed$s.log | tr "\r" "\n" | tail -1; done' | grep -v -i "token\|proxy" > "$OUT/now_ms_$S.txt"
    [ "$r" = "0" ] && break
    bal=$($C usage < /dev/null 2>&1 | grep -i balance | grep -o "[0-9.]*" | head -1)
    awk "BEGIN {exit !($bal < 4)}" && { say "balance $bal CU: stopping MedSigLIP early"; break; }
  done
  for n in 2 3; do
    s=$((41 + n)); mkdir -p "$OUT/medsig_seed$n"
    for f in preds.csv test_results.json medsiglip_iqa_cv_summary.json medsiglip_iqa_config.json; do get "/content/ms/seed$s/$f" "$OUT/medsig_seed$n/$f"; done
    get "/content/ms/seed$s.log" "$OUT/medsig_seed$n/train.log"
  done
  say "MedSigLIP outputs fetched"
else
  say "MedSigLIP seeds skipped: balance $bal CU < 16"
fi
: > "$SCRB/${S}_stop"
$C stop -s "$S" < /dev/null 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
