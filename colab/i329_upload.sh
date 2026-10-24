#!/usr/bin/env bash
# Second half of colab/i329_launch.sh for a session whose job queue is already running (the launcher's first upload
# pass lost its file lists to wsl.exe reading stdin): uploads the Stage B weights, lab329.npz, the CNN weights, nr.npz
# and the MedSigLIP feature job (Tolga's repo, the HF token without printing it) in three parallel streams,
# reassembles them (md5, manifest329.txt), writes uploads_ok for the infer jobs and starts colab/vm/medsig_feat.py (E5)
# next to the running queue. Log: outputs/agentic_ct_iqa/i329/vm.log. Usage: bash colab/i329_upload.sh <session>
S="$1"; C="bash colab/c.sh"
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SCR="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
SCRB="/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad"
LOG=outputs/agentic_ct_iqa/i329/vm.log
say() { echo "$(date +%H:%M) $*" >> $LOG; }
up() {
  for t in 1 2 3; do $C upload -s "$S" "$1" "$2" < /dev/null 2>&1 | grep -q Uploaded && return 0; sleep 5; done
  say "FAIL $2"; return 1
}
sh_vm() { bash colab/vmsh.sh "$1" "$S" < /dev/null 2>&1; }
sh_vm 'mkdir -p /content/tolga/repo /content/vft/cnn && echo ok' | grep -q ok && say "upload pass 2: dirs ok"
for f in config.py data.py model.py predict.py; do up "$SCR/tolga_repo/repo/$f" "/content/tolga/repo/$f"; done
up "$P/colab/vm/medsig_feat.py" /content/tolga/medsig_feat.py
up "$SCR/staging329/manifest329.txt" /content/manifest.txt
timeout 120 wsl.exe -d Ubuntu -e bash -lc "export PATH=\"\$HOME/.local/bin:\$PATH\"; colab upload -s $S ~/.hf_token /content/.hf_token_upload 2>&1 | grep -c Uploaded" < /dev/null >> $LOG
present=$(sh_vm 'ls /content/vft /content/vft/cnn | grep part | sort | tr "\n" " "')
{
  for f in $(ls "$SCRB/staging/vft" | grep "^weights_fold_"); do echo "$SCR/staging/vft/$f /content/vft/$f"; done
  for f in $(ls "$SCRB/staging329/vft" | grep "^lab329\|^nr.npz"); do echo "$SCR/staging329/vft/$f /content/vft/$f"; done
  for f in $(ls "$SCRB/staging329/vft/cnn"); do echo "$SCR/staging329/vft/cnn/$f /content/vft/cnn/$f"; done
} | while read -r src dst; do case " $present " in *" $(basename $dst) "*) ;; *) echo "$src $dst" ;; esac; done > "$SCRB/i329_uploads2.txt"
say "upload pass 2: $(wc -l < "$SCRB/i329_uploads2.txt") files to send"
rm -f "$SCRB"/i329_up2_*
split -n l/3 -d "$SCRB/i329_uploads2.txt" "$SCRB/i329_up2_"
pids=""
for part in "$SCRB"/i329_up2_0*; do
  ( while read -r src dst; do up "$src" "$dst"; done < "$part" ) &
  pids="$pids $!"
done
wait $pids
say "UPLOADS2_DONE"
sh_vm 'cd /content && (nohup python /content/vft/assemble_parts.py > /content/assemble.log 2>&1 < /dev/null &); echo bg' > /dev/null
for t in $(seq 1 40); do
  sleep 60
  r=$(sh_vm 'tail -1 /content/assemble.log')
  echo "$r" | grep -q ASSEMBLE_OK && { sh_vm 'touch /content/vft/uploads_ok && echo ok' | grep -q ok && say "weights assembled, uploads_ok"; break; }
  echo "$r" | grep -q -E "MISMATCH|Error" && { say "ASSEMBLE FAILED: $r"; break; }
done
sh_vm 'cd /content/tolga && (setsid nohup python medsig_feat.py > medsig_feat.log 2>&1 < /dev/null &); echo bg' > /dev/null
say "medsig_feat started"
for t in $(seq 1 30); do
  sleep 60
  r=$(sh_vm 'tail -1 /content/tolga/medsig_feat.log')
  echo "$r" | grep -q MEDSIG_FEAT_DONE && { $C download -s "$S" /content/tolga/medsig_feat.npz "$P/outputs/agentic_ct_iqa/i329/medsig_feat.npz" < /dev/null > /dev/null 2>&1 && say "medsig_feat.npz downloaded"; break; }
  echo "$r" | grep -q -E "Error|Traceback" && { say "medsig_feat FAILED: $(echo $r | grep -v -i 'token' | head -c 300)"; break; }
done
say "UPLOAD_PASS2_END"
