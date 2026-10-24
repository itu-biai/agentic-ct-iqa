"""Keep the Colab runtime alive while background jobs run: Colab keeps a VM only while its kernel is active.
Run with a long --timeout; the cell blocks (printing a heartbeat) until the watched log reports an end marker or
no watched process is left. It keeps running on the VM even if the local client disconnects."""
import os
import subprocess
import time

WATCH = ("train.py", "expert_predict.py", "setup_data.sh")
END_MARKERS = ("ALL_FOLDS_DONE", "TRAIN FAILED", "PREDICT FAILED")
LOG = "/content/logs/experts.log"  # the smoke job ends by the idle rule
idle_checks = 0
while True:
    running = subprocess.run("ps -eo cmd", shell=True, capture_output=True, text=True).stdout
    active = any(name in running for name in WATCH)
    tail = open(LOG).read()[-2000:] if os.path.exists(LOG) else ""
    if any(marker in tail for marker in END_MARKERS):
        print("keepalive: end marker seen, exiting", flush=True)
        break
    idle_checks = 0 if active else idle_checks + 1
    if idle_checks >= 10:  # ten minutes with nothing running
        print("keepalive: no job running for 10 minutes, exiting", flush=True)
        break
    print(f"keepalive {time.strftime('%H:%M:%S')} active={active}", flush=True)
    time.sleep(60)
