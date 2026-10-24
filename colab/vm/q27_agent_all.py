"""Runs the six Qwen3.8-27B agent runs (setups 18/19 x runs 1-3) one after another, skipping finished ones, then
zips /content/q27/agent to /content/q27/agent.zip. Restartable: a run folder left without scores.json by an
interruption is moved to failed/ first. Prints Q27_AGENT_DONE at the end."""
import shutil, subprocess, sys, zipfile
from pathlib import Path
RESULTS = Path("/content/q27/agent")
for setup in ("setup18_qwen38_27b_agent_vision_only", "setup19_qwen38_27b_agent_vision_tools"):
    for run in (1, 2, 3):
        out = RESULTS / setup / f"run_{run}"
        if (out / "scores.json").exists():
            continue
        if out.exists():
            failed = RESULTS / setup / "failed" / f"run_{run}_interrupted"
            failed.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(out, failed)
        for leftover in Path("/content/q27/sb").glob("*/scores.json"):
            leftover.unlink()
        subprocess.run([sys.executable, "/content/q27/q27_agent.py", setup, str(run)], cwd="/content/q27")
with zipfile.ZipFile("/content/q27/agent.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in RESULTS.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(RESULTS).as_posix())
print("Q27_AGENT_DONE", flush=True)
