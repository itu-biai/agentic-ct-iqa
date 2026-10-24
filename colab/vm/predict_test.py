"""Smoke test of expert_predict.py on the 1-epoch timing-test checkpoint. Background job; log /content/logs/predict_test.log"""
import subprocess

cmd = ("cd /content/medsiglip && python -u expert_predict.py --fold 1 --ckpt-dir /content/ckpt_time_test "
       "--exp-name medsiglip_iqa --seed 42 "
       "--train-json /content/data/LDCTIQAG2023_train/train.json --train-dir /content/data/LDCTIQAG2023_train/image "
       "--test-json /content/data/LDCTIQAC_test/test.json --test-dir /content/data/LDCTIQAC_test/images "
       "--inputs /content/expert_inputs.npz --out /content/preds_time_test.csv")
subprocess.Popen(f"nohup bash -c '/usr/bin/time -v {cmd}' > /content/logs/predict_test.log 2>&1 &", shell=True)
print("predict test started")
