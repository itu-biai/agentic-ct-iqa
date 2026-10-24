"""Timing test: Tolga's train.py unchanged, fold 1, one epoch, seed 42. Background job; log /content/logs/time_test.log"""
import subprocess

cmd = ("cd /content/medsiglip && python -u train.py "
       "--json /content/data/LDCTIQAG2023_train/train.json --data-dir /content/data/LDCTIQAG2023_train/image "
       "--checkpoint-dir /content/ckpt_time_test --exp-name medsiglip_iqa --seed 42 --folds 1 --epochs 1")
subprocess.Popen(f"nohup bash -c '/usr/bin/time -v {cmd}' > /content/logs/time_test.log 2>&1 &", shell=True)
print("time test started")
