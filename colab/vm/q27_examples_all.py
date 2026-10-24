"""Runs the three Qwen3.8-27B agent runs of setup 26 (setup 19 plus the 10 rated examples, attached to the first
message) one after another, skipping finished ones, then zips /content/q27/agent to /content/q27/agent_examples.zip.
Prints Q27E_DONE at the end."""
import shutil, subprocess, sys, zipfile
from pathlib import Path
RESULTS = Path("/content/q27/agent")
SETUP = "setup26_qwen38_27b_agent_vision_tools_examples"
for run in (1, 2, 3):
    out = RESULTS / SETUP / f"run_{run}"
    if (out / "scores.json").exists():
        continue
    if out.exists():
        failed = RESULTS / SETUP / "failed" / f"run_{run}_interrupted"
        failed.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(out, failed)
    for leftover in Path("/content/q27/sb").glob("*/scores.json"):
        leftover.unlink()
    subprocess.run([sys.executable, "/content/q27/q27_agent.py", SETUP, str(run)], cwd="/content/q27")
with zipfile.ZipFile("/content/q27/agent_examples.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in RESULTS.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(RESULTS).as_posix())
print("Q27E_DONE", flush=True)
