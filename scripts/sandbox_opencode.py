"""Runs an OpenCode model (OpenCode Go subscription) in the same sandboxes as Opus, with the same limits.

Setups 6 and 7 are setups 1 and 2 with another model: the session starts in the same folder
(experiment_sandboxes/setup1_vision_only or setup2_vision_tools), gets the same one-line prompt and the same four
file tools (read, glob, edit, write). Everything else is denied and hidden from the model: shell, web fetch and
search, subagents, skills, questions, LSP, MCP servers, and any path outside the sandbox. The OpenCode config lives
outside the sandbox; Claude Code rules and skills (~/.claude) are switched off.

    python -m scripts.sandbox_opencode setup6_qwen_vision_only 1
    python -m scripts.sandbox_opencode setup7_qwen_vision_tools 1

Writes experiment_sandboxes/results/<setup>/run_<n>/ in the standard format: scores.json (moved out of the
sandbox), transcript.jsonl (OpenCode json events), session.json (OpenCode session export with per-step tokens and
cost, images stripped), run_info.json, extra/ (any other file the session wrote).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT

SANDBOXES = PROJECT_ROOT / "experiment_sandboxes"
RESULTS = SANDBOXES / "results"
CONFIG = SANDBOXES / "opencode_sandbox.json"
SETUPS = {
    "setup6_qwen_vision_only": ("setup1_vision_only", "opencode-go/qwen3.8-max"),
    "setup7_qwen_vision_tools": ("setup2_vision_tools", "opencode-go/qwen3.8-max"),
    "setup8_qwenflash_vision_only": ("setup1_vision_only", "opencode-go/qwen3.8-flash"),
    "setup9_qwenflash_vision_tools": ("setup2_vision_tools", "opencode-go/qwen3.8-flash"),
    "setup27_qwenflash_vision_tools_examples": ("setup2b_vision_tools_examples", "opencode-go/qwen3.8-flash"),
}
PROMPT = "Read TASK.md in this folder and complete the task it describes."
PERMISSION = {"*": "deny", "read": "allow", "glob": "allow", "edit": "allow", "external_directory": "deny"}


def write_config() -> None:
    CONFIG.write_text(json.dumps({"$schema": "https://opencode.ai/config.json", "autoupdate": False,
                                  "share": "disabled", "mcp": {}, "permission": PERMISSION}, indent=1),
                      encoding="utf-8")


def opencode(*args: str, cwd: Path, env: dict, timeout: int, stdout=subprocess.PIPE) -> subprocess.CompletedProcess:
    exe = shutil.which("opencode")
    return subprocess.run([exe, *args], cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                          stderr=subprocess.PIPE, timeout=timeout)


def strip_images(text: str) -> str:
    return re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "data:image/png;base64,<removed>", text)


def examples_message(sandbox: Path) -> tuple[str, list[str]]:
    """For the setup2b sandbox: the 10 rated examples go into the first message as attachments, so the model sees them
    before it starts. Attached images arrive without file names, so the prompt lists their order and scores (as the
    single-request setup 24 labels each example). Other sandboxes: the plain prompt, no attachments."""
    labels = sandbox / "examples" / "labels.json"
    if not labels.exists():
        return PROMPT, []
    examples = json.loads(labels.read_text(encoding="utf-8"))["scores"]
    order = ", ".join(f"{e['image_id']} (score {e['score']})" for e in examples)
    files = [arg for e in examples for arg in ("-f", f"examples/{e['image_id']}.png")]
    return (PROMPT + " The 10 reference examples from examples/ are attached to this message, in this order: "
            + order + "."), files


def run(setup: str, run_id: int) -> None:
    folder, model = SETUPS[setup]
    sandbox = SANDBOXES / folder
    out = RESULTS / setup / f"run_{run_id}"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    if (sandbox / "scores.json").exists():
        raise SystemExit(f"scores.json is still in {sandbox}; move it out first")
    out.mkdir(parents=True)
    prompt, attached = examples_message(sandbox)
    write_config()
    # OpenCode takes the project folder from PWD, not from the process's working directory; without this the
    # session opens in whatever folder the calling shell was in.
    env = {**os.environ, "OPENCODE_CONFIG": str(CONFIG), "OPENCODE_DISABLE_CLAUDE_CODE": "1", "PWD": str(sandbox)}

    start = datetime.now()
    with open(out / "transcript.jsonl", "wb") as transcript:
        done = opencode("run", "--standalone", "--format", "json", "-m", model, prompt, *attached, cwd=sandbox, env=env,
                        timeout=3600, stdout=transcript)
    end = datetime.now()
    if done.returncode:
        print(done.stderr.decode(errors="replace")[-2000:])

    events = [json.loads(line) for line in (out / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip().startswith("{")]
    (out / "transcript.jsonl").write_text("\n".join(strip_images(json.dumps(e)) for e in events), encoding="utf-8")
    session = next((e["sessionID"] for e in events if e.get("sessionID")), None)
    info, steps = {}, []
    if session:
        exported = opencode("session", "export", session, cwd=sandbox, env=env, timeout=120).stdout.decode("utf-8")
        exported = strip_images(exported[exported.index("{"):])
        (out / "session.json").write_text(exported, encoding="utf-8")
        record = json.loads(exported)
        info, steps = record["info"], [m for m in record["messages"] if m["type"] == "assistant"]
    directory = info.get("location", {}).get("directory")
    if directory is None or Path(directory).resolve() != sandbox.resolve():
        (out / "INVALID.txt").write_text(f"session opened in {directory}, not in {sandbox}\n", encoding="utf-8")
        raise SystemExit(f"INVALID RUN: session opened in {directory}, not in the sandbox")

    scores = sandbox / "scores.json"
    if scores.exists():
        shutil.move(scores, out / "scores.json")
    else:
        print("WARNING: the session wrote no scores.json")
    extra = [p for p in sandbox.rglob("*") if p.is_file() and ".git" not in p.parts
             and datetime.fromtimestamp(p.stat().st_mtime) >= start]
    if extra:
        (out / "extra").mkdir()
        for path in extra:
            shutil.move(path, out / "extra" / path.name)
        print(f"WARNING: the session wrote other files; moved to {out / 'extra'}")

    calls = [e["part"] for e in events if e.get("type") == "tool_use"]
    version = opencode("--version", cwd=sandbox, env=env, timeout=60).stdout.decode().strip()
    (out / "run_info.json").write_text(json.dumps({
        "setup": setup, "run": run_id, "agent": "opencode", "model": model, "variant": "default",
        "sandbox": folder, "session_directory": directory, "prompt": prompt, "attached": attached[1::2], "permission": PERMISSION,
        "env": "OPENCODE_DISABLE_CLAUDE_CODE=1", "start": start.isoformat(), "end": end.isoformat(),
        "exit_code": done.returncode, "session": session, "steps": len(steps),
        "cost_usd": round(info.get("cost") or 0, 4), "tokens": info.get("tokens"),
        "tools_used": sorted({c["tool"] for c in calls}), "opencode_version": version}, indent=1), encoding="utf-8")

    failed = [c for c in calls if c["state"].get("status") != "completed"]
    print(f"steps {len(steps)}, tool calls {len(calls)} ({len(failed)} failed), "
          f"cost ${info.get('cost') or 0:.3f}, {(end - start).seconds} s")
    print("done:", out)


def run_with_retries(setup: str, run_id: int, attempts: int = 3) -> None:
    """A provider error (the session stops without writing scores.json) restarts the run from scratch; the broken
    attempt is kept in results/<setup>/failed/ so it can still be inspected."""
    out = RESULTS / setup / f"run_{run_id}"
    for attempt in range(1, attempts + 1):
        run(setup, run_id)
        if (out / "scores.json").exists():
            return
        failed = RESULTS / setup / "failed" / f"run_{run_id}_attempt_{attempt}"
        failed.parent.mkdir(exist_ok=True)
        shutil.move(out, failed)
        print(f"attempt {attempt} wrote no scores; moved to {failed}")
    raise SystemExit(f"{setup} run {run_id}: no scores after {attempts} attempts")


if __name__ == "__main__":
    run_with_retries(sys.argv[1], int(sys.argv[2]))
