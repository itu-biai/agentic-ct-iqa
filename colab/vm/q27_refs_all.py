"""Setup 29 / 30 (Qwen3.8-27B agent, setup 26 without the expert score, plus reference_match and references/;
scripts/make_references.py; 30 adds the rule's blind spots): three runs at once against the same vLLM server, each in its own copy of the sandbox
(/content/q27/sb_run<n>, see q27_agent.sandbox_base). Skips finished runs. Zips /content/q27/agent/<SETUP> to
/content/q27/refs_results.zip and prints REFS_DONE.
    python q27_refs_all.py
"""
import shutil, subprocess, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
SETUP = "setup30_qwen38_27b_agent_tools_references_guided"  # constant: no sys.argv under `colab exec`
FOLDER = "setup4b_tools_references_guided"
RESULTS = Path("/content/q27/agent")


def one(run: int) -> None:
    out = RESULTS / SETUP / f"run_{run}"
    if (out / "scores.json").exists():
        return
    if out.exists():
        failed = RESULTS / SETUP / "failed" / f"run_{run}_interrupted"
        failed.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(out, failed)
    time.sleep((run - 1) * 30)
    Path(f"/content/q27/sb_run{run}/{FOLDER}/scores.json").unlink(missing_ok=True)
    with open(f"/content/q27/log_{SETUP}_{run}.txt", "a") as log:
        subprocess.run([sys.executable, "/content/q27/q27_agent.py", SETUP, str(run)], cwd="/content/q27",
                       stdout=log, stderr=subprocess.STDOUT)
    print(SETUP, run, "scores" if (out / "scores.json").exists() else "NO SCORES", flush=True)


with ThreadPoolExecutor(3) as pool:
    list(pool.map(one, (1, 2, 3)))
with zipfile.ZipFile("/content/q27/refs_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in RESULTS.glob(f"{SETUP}/**/*"):
        if p.is_file():
            z.write(p, p.relative_to(RESULTS).as_posix())
print("REFS_DONE", flush=True)
