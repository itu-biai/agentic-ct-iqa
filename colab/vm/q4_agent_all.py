"""Runs the six Qwen3.5-4B agent runs (setups 14/15 x runs 1-3) one after another, skipping finished ones, then
zips /content/q4/agent to /content/q4/agent.zip. Restartable: a run folder left without scores.json by an
interruption is moved to failed/ first. Prints Q4_AGENT_DONE at the end."""
import shutil, subprocess, sys, zipfile
from pathlib import Path
RESULTS = Path("/content/q4/agent")
for setup in ("setup14_qwen35_4b_agent_vision_only", "setup15_qwen35_4b_agent_vision_tools"):
    for run in (1, 2, 3):
        out = RESULTS / setup / f"run_{run}"
        if (out / "scores.json").exists():
            continue
        if out.exists():
            failed = RESULTS / setup / "failed" / f"run_{run}_interrupted"
            failed.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(out, failed)
        for leftover in Path("/content/q4/sb").glob("*/scores.json"):
            leftover.unlink()
        subprocess.run([sys.executable, "/content/q4/q4_agent.py", setup, str(run)], cwd="/content/q4")
with zipfile.ZipFile("/content/q4/agent.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in RESULTS.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(RESULTS).as_posix())
print("Q4_AGENT_DONE", flush=True)
