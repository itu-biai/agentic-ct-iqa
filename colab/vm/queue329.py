"""Issue #329 job queue on the VM, started with nohup outside the Jupyter kernel (colab/i329_launch.sh), so no `colab
exec` stream has to stay open for hours. Runs the jobs in ROOT/queue.txt one at a time, each in its own process (fresh
GPU memory), re-reading the file after every job, so jobs can be appended or removed while it runs. A job is a
vftb_train.py flag (notools, phase2only, patientfolds, seed2, seed3; only that flag file is set while it runs) or
infer* (infer329.py with no flag; waits for ROOT/uploads_ok, written after the weights are reassembled). Log
ROOT/q_<job>.log, marker ROOT/done_<job> or ROOT/fail_<job>; ROOT/queue_done when nothing is left.
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/content/vft")
FLAGS = ("notools", "phase2only", "patientfolds", "seed2", "seed3")


def clear_flags() -> None:
    for flag in FLAGS:
        (ROOT / flag).unlink(missing_ok=True)


while True:
    jobs = (ROOT / "queue.txt").read_text().split()
    todo = [j for j in jobs if not (ROOT / f"done_{j}").exists() and not (ROOT / f"fail_{j}").exists()]
    if not todo:
        break
    job = todo[0]
    clear_flags()
    if job.startswith("infer"):
        while not (ROOT / "uploads_ok").exists():
            time.sleep(30)
        script = "infer329.py"
    else:
        assert job in FLAGS, job
        (ROOT / job).touch()
        script = "vftb_train.py"
    start = time.time()
    with open(ROOT / f"q_{job}.log", "w") as log:
        code = subprocess.run(["python", script], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "VFT_ROOT": str(ROOT)}).returncode
    (ROOT / f"{'done' if code == 0 else 'fail'}_{job}").touch()
    print(f"{job} exit {code}, {(time.time() - start) / 60:.1f} min", flush=True)
    clear_flags()
(ROOT / "queue_done").touch()
print("QUEUE_DONE", flush=True)
