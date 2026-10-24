"""E16b (issue #329, scripts/make_e16b.py) on the VM, against the vLLM server of q27_serve.py: the ne19 agent protocol
(q27_agent.py, one run, up to 3 attempts) on 5 folders of 20 test images, with the tool files unchanged (e16f) and with
the noise tools removed (e16n); 4 OpenCode sessions at a time, conditions interleaved so both see the same server load.
Zips the results to /content/q27/e16b_results.zip and prints E16B_DONE. Run as a plain process (nohup).
"""
import subprocess, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/content/q27")
RESULTS = ROOT / "agent"
WORKERS = 4


def agent(job: tuple) -> None:
    i, setup = job
    if (RESULTS / setup / "run_1" / "scores.json").exists():
        return
    if i < WORKERS:
        time.sleep(i * 30)  # staggered first wave
    with open(ROOT / f"log_{setup}.txt", "a") as log:
        subprocess.run([sys.executable, str(ROOT / "q27_agent.py"), setup, "1"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print(setup, "scores" if (RESULTS / setup / "run_1" / "scores.json").exists() else "NO SCORES", flush=True)


setups = [f"{c}_batch_{b:02d}" for b in range(1, 6) for c in ("e16f", "e16n")]
with ThreadPoolExecutor(WORKERS) as pool:
    list(pool.map(agent, enumerate(setups)))
with zipfile.ZipFile(ROOT / "e16b_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in RESULTS.glob("e16*_batch_*/**/*"):
        if p.is_file():
            z.write(p, p.relative_to(RESULTS).as_posix())
print("E16B_DONE", flush=True)
