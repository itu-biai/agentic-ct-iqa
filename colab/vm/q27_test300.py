"""Runs setup 26's protocol (Qwen3.8-27B agent + tools + expert + 10 attached examples) on the 300-image test: one run
per batch, 15 batches, four OpenCode sessions at a time against the same vLLM server. Skips finished batches, so it can
be restarted. Zips /content/q27/agent/test300_batch_* to /content/q27/test300_results.zip and prints T300_DONE.
    python q27_test300.py
"""
import shutil, subprocess, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
RESULTS = Path("/content/q27/agent")
WORKERS = 4  # constant: `colab exec` runs this inside the Jupyter kernel, whose sys.argv is not ours


def batch(b: int) -> None:
    setup = f"test300_batch_{b:02d}"
    out = RESULTS / setup / "run_1"
    if (out / "scores.json").exists():
        return
    if out.exists():
        failed = RESULTS / setup / "failed" / "run_1_interrupted"
        failed.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(out, failed)
    if b <= WORKERS:
        time.sleep((b - 1) * 30)  # staggered first wave
    leftover = Path(f"/content/q27/sb/test300/batch_{b:02d}/scores.json")
    leftover.unlink(missing_ok=True)
    with open(f"/content/q27/log_{setup}.txt", "a") as log:
        subprocess.run([sys.executable, "/content/q27/q27_agent.py", setup, "1"], cwd="/content/q27", stdout=log,
                       stderr=subprocess.STDOUT)
    print(setup, "scores" if (out / "scores.json").exists() else "NO SCORES", flush=True)


with ThreadPoolExecutor(WORKERS) as pool:
    list(pool.map(batch, range(1, 16)))
with zipfile.ZipFile("/content/q27/test300_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in RESULTS.glob("test300_batch_*/**/*"):
        if p.is_file():
            z.write(p, p.relative_to(RESULTS).as_posix())
print("T300_DONE", flush=True)
