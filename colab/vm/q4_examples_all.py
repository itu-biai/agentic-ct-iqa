"""Runs the three Qwen3.5-4B agent runs of setup 28 (setup 15 plus the 10 rated examples, attached to the first
message) one after another, skipping finished ones, then zips /content/q4/agent to /content/q4/agent_examples.zip.
Prints Q4E_DONE at the end."""
import shutil, subprocess, sys, zipfile
from pathlib import Path
RESULTS = Path("/content/q4/agent")
SETUP = "setup28_qwen35_4b_agent_vision_tools_examples"
for run in (1, 2, 3):
    out = RESULTS / SETUP / f"run_{run}"
    if (out / "scores.json").exists():
        continue
    if out.exists():
        failed = RESULTS / SETUP / "failed" / f"run_{run}_interrupted"
        failed.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(out, failed)
    for leftover in Path("/content/q4/sb").glob("*/scores.json"):
        leftover.unlink()
    subprocess.run([sys.executable, "/content/q4/q4_agent.py", SETUP, str(run)], cwd="/content/q4")
with zipfile.ZipFile("/content/q4/agent_examples.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in RESULTS.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(RESULTS).as_posix())
print("Q4E_DONE", flush=True)
