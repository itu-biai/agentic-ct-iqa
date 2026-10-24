#!/usr/bin/env bash
# Resume the expert training on a fresh VM: rebuild it, upload the folds already finished (so run_experts.sh skips
# them), start the job, the keepalive cell and a detached local watcher. Usage (Git Bash, project root):
#   bash colab/resume_experts.sh
set -u
cd "$(dirname "$0")/.."
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
OUT=outputs/agentic_ct_iqa/experts
bash colab/bootstrap.sh || exit 1
echo "import os; os.makedirs('/content/out', exist_ok=True)" | bash colab/c.sh exec -s expert
for f in "$OUT"/preds_fold*.csv; do
  [ -e "$f" ] || continue
  bash colab/c.sh upload -s expert "$P/$f" "/content/out/$(basename "$f")" >/dev/null && echo "uploaded $(basename "$f")"
done
bash colab/c.sh upload -s expert "$P/colab/vm/run_experts.sh" /content/run_experts.sh >/dev/null
# wait for the data download started by bootstrap.sh
for i in $(seq 1 40); do
  bash colab/vmsh.sh 'grep -c SETUP_DONE /content/logs/setup_data.log' | grep -q 1 && break
  sleep 20
done
bash colab/vmsh.sh 'nohup bash /content/run_experts.sh >> /content/logs/experts.log 2>&1 & sleep 5; pgrep -af run_experts | head -1'
# keepalive cell: blocks the kernel until ALL_FOLDS_DONE; keeps running on the VM after this client exits
(bash colab/c.sh exec -s expert --timeout 86400 -f "$P/colab/vm/keepalive.py" >/dev/null 2>&1 &)
powershell.exe -NoProfile -Command "Start-Process -FilePath 'C:\Program Files\Git\bin\bash.exe' -ArgumentList '-c', '\"cd ''/c/Users/kerem/Desktop/KB TD Ortak Paper'' && bash colab/watch_experts.sh\"' -WindowStyle Hidden"
echo "resumed; progress in $OUT/progress.log"
