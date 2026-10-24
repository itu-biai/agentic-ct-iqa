"""VM status: data folders, running jobs, GPU, and the tail of every job log in /content/logs."""
import glob
import os
import subprocess

def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

for d in ["/content/data/LDCTIQAG2023_train/image", "/content/data/LDCTIQAC_test/images"]:
    print(d, len(os.listdir(d)) if os.path.isdir(d) else "missing")
print("disk:", sh("du -sh /content/data 2>/dev/null"))
print("jobs:", sh("ps -eo pid,etime,cmd | grep -E 'train.py|expert_predict|curl|gdown|unzip' | grep -v grep") or "none")
print("gpu:", sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"))
for log in sorted(glob.glob("/content/logs/*.log")):
    print(f"--- {log}")
    print(sh(f"tail -n 6 {log}"))
