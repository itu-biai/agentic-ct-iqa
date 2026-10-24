#!/usr/bin/env bash
# E12 MedSigLIP seeds (colab/vm/medsig_seed.py) on a fresh L4 session: Tolga's repo, the challenge data tar (30 MB parts,
# md5 manifest_ms.txt), the HF token without printing it; runs one seed with nohup, downloads its preds.csv and
# summaries into outputs/agentic_ct_iqa/i329/medsig_seed<n>/ and stops the session.
# Parts already on the VM are not sent again; up to 3 passes until every part is there. A 4th argument "resume" reuses
# a running session. Log: outputs/agentic_ct_iqa/i329/ms_<session>.log. Usage: bash colab/ms_launch.sh <session> <seed> <n> [resume]
S="$1"; SEED="$2"; N="$3"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
OUT=outputs/agentic_ct_iqa/i329
LOG=$OUT/ms_$S.log
DST=$OUT/medsig_seed$N
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" < /dev/null 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
sh_vm() { bash colab/vmsh.sh "$1" "$S" < /dev/null 2>&1; }
: > $LOG
rm -f "$SCRB/${S}_stop"
if [ "$4" = resume ]; then say "resume on $S"; else
$C new -s "$S" --gpu L4 < /dev/null > /dev/null 2>&1 && say "session $S up" || { say "no session"; exit 1; }
fi
( while [ ! -f "$SCRB/${S}_stop" ]; do timeout 11000 $C exec -s "$S" -f "$P/colab/vm/hold.py" --timeout 10900 < /dev/null >/dev/null 2>&1; sleep 5; done ) &
sleep 20
sh_vm 'mkdir -p /content/ms/repo && rm -f /content/hold_stop && echo ok' | grep -q ok && say "dirs ok"
for f in config.py data.py model.py train.py test.py; do up "$SCR/tolga_repo/repo/$f" "/content/ms/repo/$f"; done
up "$P/colab/vm/medsig_seed.py" /content/ms/medsig_seed.py
up "$P/colab/vm/assemble_parts.py" /content/ms/assemble_parts.py
up "$SCR/staging329/manifest_ms.txt" /content/manifest.txt
timeout 120 wsl.exe -d Ubuntu -e bash -lc "export PATH=\"\$HOME/.local/bin:\$PATH\"; colab upload -s $S ~/.hf_token /content/.hf_token_upload 2>&1 | grep -c Uploaded" < /dev/null >> $LOG
for pass in 1 2 3; do
  present=$(sh_vm 'ls /content/ms | tr "\n" " "')
  ls "$SCRB/staging329/ms" | while read -r f; do case " $present " in *" $f "*) ;; *) echo "$f" ;; esac; done > "$SCRB/${S}_parts.txt"
  [ -s "$SCRB/${S}_parts.txt" ] || break
  say "upload pass $pass: $(wc -l < "$SCRB/${S}_parts.txt") parts"
  rm -f "$SCRB/${S}_up_"*
  split -n l/3 -d "$SCRB/${S}_parts.txt" "$SCRB/${S}_up_"
  pids=""
  for part in "$SCRB/${S}_up_"0*; do
    ( while read -r f; do up "$SCR/staging329/ms/$f" "/content/ms/$f"; done < "$part" ) &
    pids="$pids $!"
  done
  wait $pids
done
say "uploads done"
sh_vm 'cd /content && (nohup python /content/ms/assemble_parts.py > /content/assemble.log 2>&1 < /dev/null &); echo bg' > /dev/null
for t in $(seq 1 20); do sleep 30; r=$(sh_vm 'tail -1 /content/assemble.log'); echo "$r" | grep -q -E "ASSEMBLE_OK|MISMATCH|Error" && break; done
say "assemble: $(echo $r | tail -c 60)"
echo "$r" | grep -q ASSEMBLE_OK || { : > "$SCRB/${S}_stop"; say "assemble not OK, session kept for a resume"; exit 1; }
sh_vm "cd /content/ms && (setsid nohup python medsig_seed.py $SEED > seed$SEED.log 2>&1 < /dev/null &); echo bg" > /dev/null
say "seed $SEED start"
for t in $(seq 1 240); do
  sleep 180
  r=$(sh_vm "tail -c 400 /content/ms/seed$SEED.log | tr '\r' '\n' | tail -2 | tr '\n' ' '; pgrep -f medsig_see[d] > /dev/null || echo STOPPED")
  echo "$r" | grep -v -i "token" | tail -c 160 > "$OUT/ms_${S}_now.txt"
  echo "$r" | grep -q MS_DONE && break
  echo "$r" | grep -q STOPPED && { say "FAILED: $(echo "$r" | grep -v -i token | tail -c 300)"; break; }
done
mkdir -p $DST
for f in preds.csv test_results.json medsiglip_iqa_cv_summary.json medsiglip_iqa_config.json; do
  $C download -s "$S" "/content/ms/seed$SEED/$f" "$P/$DST/$f" < /dev/null >/dev/null 2>&1 && say "$f downloaded"
done
$C download -s "$S" "/content/ms/seed$SEED.log" "$P/$DST/train.log" < /dev/null >/dev/null 2>&1
: > "$SCRB/${S}_stop"
$C stop -s "$S" < /dev/null 2>&1 | tr -d '\r' | tail -1 >> $LOG
say "LAUNCH_END"
