"""Smoke job (background): one epoch of fold 1 with Tolga's train.py, then expert_predict.py on that checkpoint.
Gives the epoch time and the prediction time, and catches bugs before the multi-hour run. Log: /content/logs/smoke.log"""
import subprocess

DATA = "/content/data"
script = f"""
set -u
cd /content/medsiglip
echo "SMOKE TRAIN START $(date -u +%s)"
python -u train.py --json {DATA}/LDCTIQAG2023_train/train.json --data-dir {DATA}/LDCTIQAG2023_train/image \\
  --checkpoint-dir /content/ckpt_smoke --exp-name medsiglip_iqa --seed 42 --folds 1 --epochs 1 || {{ echo SMOKE_FAILED; exit 1; }}
echo "SMOKE PREDICT START $(date -u +%s)"
python -u expert_predict.py --fold 1 --ckpt-dir /content/ckpt_smoke --exp-name medsiglip_iqa --seed 42 \\
  --train-json {DATA}/LDCTIQAG2023_train/train.json --train-dir {DATA}/LDCTIQAG2023_train/image \\
  --test-json {DATA}/LDCTIQAC_test/test.json --test-dir {DATA}/LDCTIQAC_test/images \\
  --inputs /content/expert_inputs.npz --out /content/preds_smoke.csv || {{ echo SMOKE_FAILED; exit 1; }}
echo "SMOKE END $(date -u +%s)"
echo SMOKE_DONE
"""
open("/content/smoke_job.sh", "w").write(script)
subprocess.Popen("nohup bash /content/smoke_job.sh > /content/logs/smoke.log 2>&1 &", shell=True)
print("smoke job started")
