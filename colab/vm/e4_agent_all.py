"""Runs the nine Empero 4B agent runs (setups 22/23/25 x runs 1-3) one after another, skipping finished ones, then
zips /content/e4/agent to /content/e4/agent.zip. Restartable: a run folder left without scores.json by an
interruption is moved to failed/ first. Prints E4_AGENT_DONE at the end."""
import shutil, subprocess, sys, zipfile
from pathlib import Path
RESULTS = Path("/content/e4/agent")
for setup in ("setup22_empero_4b_agent_vision_only", "setup23_empero_4b_agent_vision_tools",
              "setup25_empero_4b_agent_vision_tools_examples"):
    for run in (1, 2, 3):
        out = RESULTS / setup / f"run_{run}"
        if (out / "scores.json").exists():
            continue
        if out.exists():
            failed = RESULTS / setup / "failed" / f"run_{run}_interrupted"
            failed.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(out, failed)
        for leftover in Path("/content/e4/sb").glob("*/scores.json"):
            leftover.unlink()
        subprocess.run([sys.executable, "/content/e4/e4_agent.py", setup, str(run)], cwd="/content/e4")
with zipfile.ZipFile("/content/e4/agent.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in RESULTS.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(RESULTS).as_posix())
print("E4_AGENT_DONE", flush=True)
