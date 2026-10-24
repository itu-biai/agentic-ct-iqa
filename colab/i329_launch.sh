#!/usr/bin/env bash
# Issue #329 GPU queue on a fresh G4 session (Git Bash, project root): creates it (retries every 5 min while G4 is
# unavailable), setup, a keepalive cell for the whole run, small files, then the job queue (colab/vm/queue329.py, nohup,
# jobs from colab/vft_job/queue329.txt) starts training while the Stage B weights, lab329.npz and the CNN weights upload
# in parallel; md5-checked reassembly writes uploads_ok, which the infer jobs wait for. Every finished job's outputs
# (no weights) are downloaded into outputs/agentic_ct_iqa/i329/; when the queue is done the session is stopped.
# Log: outputs/agentic_ct_iqa/i329/vm.log. Usage: bash colab/i329_launch.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
OUT=outputs/agentic_ct_iqa/i329
mkdir -p $OUT
LOG=$OUT/vm.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
sh_vm() { bash colab/vmsh.sh "$1" "$S" 2>&1; }
get() { for t in 1 2 3; do $C download -s "$S" "$1" "$P/$2" >/dev/null 2>&1 && return 0; sleep 10; done; say "download FAIL $1"; }
fetch_job() {  # outputs of one finished job, without weights
  local job="$1" dir
  get "/content/vft/q_$job.log" "$OUT/q_$job.log"
  case $job in
    notools) dir=out_notools ;; phase2only) dir=out_p2only ;; patientfolds) dir=out_patient ;;
    seed2) dir=out_seed2 ;; seed3) dir=out_seed3 ;; *) dir="" ;;
  esac
  if [ -n "$dir" ]; then
    mkdir -p "$OUT/$dir"
    for k in 1 2 3 4 5; do get "/content/vft/$dir/fold_$k.npz" "$OUT/$dir/fold_$k.npz"; done
    get "/content/vft/$dir/log.txt" "$OUT/$dir/log.txt"
  else
    mkdir -p "$OUT/infer"
    for f in shuffle labshuf pd occl enc probe lab_notools lab_p2only cnn; do
      [ -f "$OUT/infer/$f.npz" ] || get "/content/vft/i329/$f.npz" "$OUT/infer/$f.npz" 2>/dev/null
    done
  fi
  say "fetched $job"
}
: > $LOG
rm -f "$SCRB/i329_stop"
for t in $(seq 1 12); do
  $C new -s "$S" --gpu G4 > /dev/null 2>&1 && { say "session $S up (try $t)"; break; }
  say "G4 unavailable (try $t)"; sleep 300
done
bash colab/c.sh sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || { say "no session, exit"; exit 1; }
$C exec -s "$S" -f "$P/colab/vm/vftb_setup.py" --timeout 3000 2>&1 | grep -E "checkpoint|gpu|SETUP_DONE|Error" | grep -v -i "token\|proxy" >> $LOG
( while [ ! -f "$SCRB/i329_stop" ]; do timeout 11000 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 10900 >/dev/null 2>&1; sleep 5; done ) &
sleep 20
sh_vm 'mkdir -p /content/vft/cnn && rm -f /content/hold_stop && echo dirs-ok' | grep -q dirs-ok && say "dirs ok"
for f in colab/vft_job/parts/*; do up "$P/$f" "/content/vft/$(basename $f)"; done
for f in config.json preprocessor_config.json patient_folds.npy; do up "$P/colab/vft_job/$f" "/content/vft/$f"; done
up "$P/colab/vft_job/queue329.txt" /content/vft/queue.txt
for f in vftb_train.py infer329.py queue329.py cnn_train.py vft_assemble.py assemble_parts.py; do up "$P/colab/vm/$f" "/content/vft/$f"; done
up "$SCR/staging329/manifest329.txt" /content/manifest.txt
sh_vm 'cd /content/vft && python vft_assemble.py 2>&1 | tail -1' | grep -q ASSEMBLED_OK && say "train/test assembled" || say "ASSEMBLY FAILED"
sh_vm 'cd /content/vft && (setsid nohup python queue329.py > queue.log 2>&1 < /dev/null &) ; sleep 2; pgrep -f queue32[9] >/dev/null && echo queue-up' | grep -q queue-up && say "queue started"
# big uploads, three streams in parallel
upload_list() { while read -r src dst; do up "$src" "$dst" < /dev/null; done; }
{
  for f in $(ls "$SCRB/staging/vft" | grep "^weights_fold_"); do echo "$SCR/staging/vft/$f /content/vft/$f"; done
  for f in $(ls "$SCRB/staging329/vft" | grep "^lab329"); do echo "$SCR/staging329/vft/$f /content/vft/$f"; done
  for f in $(ls "$SCRB/staging329/vft/cnn"); do echo "$SCR/staging329/vft/cnn/$f /content/vft/cnn/$f"; done
} > "$SCRB/i329_uploads.txt"
split -n l/3 -d "$SCRB/i329_uploads.txt" "$SCRB/i329_up_"
for part in "$SCRB"/i329_up_0*; do upload_list < "$part" & done
wait $(jobs -p | tail -3) 2>/dev/null
say "UPLOADS_DONE $(wc -l < "$SCRB/i329_uploads.txt") files"
sh_vm 'cd /content && (nohup python /content/vft/assemble_parts.py > /content/assemble.log 2>&1 < /dev/null &); echo bg'
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
  sh_vm 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' | head -1 > "$OUT/gpu_now.txt"
  sleep 300
done
get /content/vft/queue.log "$OUT/queue.log"
: > "$SCRB/i329_stop"
$C stop -s "$S" 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
