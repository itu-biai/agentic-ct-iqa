#!/usr/bin/env bash
# Checkpoint for issue #329 (user: stop at 23:00 sharp): waits until 23:00, downloads every finished and partial infer output, the queue logs and
# the trained runs' fold files (no weights), then stops the session. Log: outputs/agentic_ct_iqa/i329/vm.log
S=i329; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
OUT=outputs/agentic_ct_iqa/i329
say() { echo "$(date +%H:%M) $*" >> $OUT/vm.log; }
while true; do
  now=$(date +%H%M)
  [ "$now" -ge 2300 ] && break
  sleep 30
done
say "checkpoint: downloading"
mkdir -p $OUT/infer
files=$(bash colab/vmsh.sh 'ls /content/vft/i329' $S < /dev/null 2>&1 | grep "\.npz$")
for f in $files; do $C download -s $S "/content/vft/i329/$f" "$P/$OUT/infer/$f" < /dev/null >/dev/null 2>&1 && say "got $f"; done
for f in queue.log q_notools.log q_phase2only.log q_patientfolds.log q_infer.log; do $C download -s $S "/content/vft/$f" "$P/$OUT/$f" < /dev/null >/dev/null 2>&1; done
for d in out_notools out_p2only out_patient; do $C download -s $S "/content/vft/$d/log.txt" "$P/$OUT/$d/log.txt" < /dev/null >/dev/null 2>&1; done
$C stop -s $S < /dev/null 2>&1 | tr -d '\r' | tail -1 >> $OUT/vm.log
: > "/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad/i329_stop"
say "CHECKPOINT_END"
