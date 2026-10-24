"""Empero's Qwen3.8-4B-Distill as an OpenCode agent in the setup 1/2/2b sandboxes, on the Colab VM: the protocol of
setups 14/15 (q4_agent.py), unchanged except for the model and the setup 2b sandbox (setup 2 plus 10 rated examples).

Same as scripts/sandbox_opencode.py, with the model served by the local vLLM server (e4_serve.py) instead of
OpenCode Go: same OpenCode version, same one-line prompt, same permission set (read, glob, edit; everything else
denied and hidden), config outside the sandbox, PWD = sandbox and a check that the session opened there. A run that
ends without scores.json is moved to <setup>/failed/ and restarted, up to three attempts.

Expects /content/e4/sb/{setup1_vision_only,setup2_vision_tools,setup2b_vision_tools_examples} (copies of the local
sandboxes, with .git). Writes /content/e4/agent/<setup>/run_<n>/: transcript.jsonl, session.json, scores.json,
run_info.json, extra/.
    python e4_agent.py <setup> <run>
"""
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/content/e4")
RESULTS = ROOT / "agent"
CONFIG = ROOT / "opencode_sandbox.json"
SETUPS = {"setup22_empero_4b_agent_vision_only": "setup1_vision_only",
          "setup23_empero_4b_agent_vision_tools": "setup2_vision_tools",
          "setup25_empero_4b_agent_vision_tools_examples": "setup2b_vision_tools_examples"}
MODEL = "local/empero-4b"
PROMPT = "Read TASK.md in this folder and complete the task it describes."
PERMISSION = {"*": "deny", "read": "allow", "glob": "allow", "edit": "allow", "external_directory": "deny"}
PROVIDER = {"local": {"npm": "@ai-sdk/openai-compatible", "name": "vLLM on this VM",
                      "options": {"baseURL": "http://localhost:8000/v1"},
                      "models": {"empero-4b": {"name": "Qwen3.8-4B-Distill", "attachment": True, "reasoning": True,
                                                "tool_call": True, "temperature": True,
                                                "modalities": {"input": ["text", "image"], "output": ["text"]},
                                                "limit": {"context": 55296, "output": 16384}}}}}


def write_config() -> None:
    CONFIG.write_text(json.dumps({"$schema": "https://opencode.ai/config.json", "autoupdate": False,
                                  "share": "disabled", "mcp": {}, "permission": PERMISSION, "provider": PROVIDER},
                                 indent=1), encoding="utf-8")


def opencode(*args: str, cwd: Path, env: dict, timeout: int, stdout=subprocess.PIPE) -> subprocess.CompletedProcess:
    return subprocess.run([shutil.which("opencode"), *args], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          stdout=stdout, stderr=subprocess.PIPE, timeout=timeout)


def strip_images(text: str) -> str:
    return re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "data:image/png;base64,<removed>", text)


def run(setup: str, run_id: int) -> None:
    folder = SETUPS[setup]
    sandbox = ROOT / "sb" / folder
    out = RESULTS / setup / f"run_{run_id}"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    if (sandbox / "scores.json").exists():
        raise SystemExit(f"scores.json is still in {sandbox}; move it out first")
    out.mkdir(parents=True)
    write_config()
    env = {**os.environ, "OPENCODE_CONFIG": str(CONFIG), "OPENCODE_DISABLE_CLAUDE_CODE": "1", "PWD": str(sandbox)}

    start = datetime.now()
    with open(out / "transcript.jsonl", "wb") as transcript:
        try:
            done = opencode("run", "--standalone", "--format", "json", "-m", MODEL, PROMPT, cwd=sandbox, env=env,
                            timeout=7200, stdout=transcript)
            code, err = done.returncode, done.stderr.decode(errors="replace")
        except subprocess.TimeoutExpired:
            code, err = "timeout", "timeout after 7200 s"
    end = datetime.now()
    if code:
        print(err[-2000:])

    events = [json.loads(line) for line in (out / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip().startswith("{")]
    (out / "transcript.jsonl").write_text("\n".join(strip_images(json.dumps(e)) for e in events), encoding="utf-8")
    session = next((e["sessionID"] for e in events if e.get("sessionID")), None)
    info, steps = {}, []
    if session:
        # Written to a file, not a pipe: on Linux the export through a pipe stops at 64 KB.
        raw = ROOT / "session_export.tmp"
        with open(raw, "wb") as handle:
            opencode("session", "export", session, cwd=sandbox, env=env, timeout=300, stdout=handle)
        exported = raw.read_text(encoding="utf-8")
        exported = strip_images(exported[exported.index("{"):])
        (out / "session.json").write_text(exported, encoding="utf-8")
        record = json.loads(exported)
        info, steps = record["info"], [m for m in record["messages"] if m["type"] == "assistant"]
    directory = info.get("location", {}).get("directory")
    if directory is None or Path(directory).resolve() != sandbox.resolve():
        (out / "INVALID.txt").write_text(f"session opened in {directory}, not in {sandbox}\n", encoding="utf-8")
        print(f"INVALID RUN: session opened in {directory}, not in the sandbox")

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
        "setup": setup, "run": run_id, "agent": "opencode", "model": "empero-ai/Qwen3.8-4B-Distill", "provider": PROVIDER,
        "server": "vLLM on Colab T4, fp16", "variant": "default", "sandbox": folder, "session_directory": directory,
        "prompt": PROMPT, "permission": PERMISSION, "env": "OPENCODE_DISABLE_CLAUDE_CODE=1",
        "start": start.isoformat(), "end": end.isoformat(), "exit_code": code, "session": session,
        "steps": len(steps), "cost_usd": 0, "tokens": info.get("tokens"),
        "tools_used": sorted({c["tool"] for c in calls}), "opencode_version": version}, indent=1), encoding="utf-8")
    failed = [c for c in calls if c["state"].get("status") != "completed"]
    print(f"{setup} run {run_id}: steps {len(steps)}, tool calls {len(calls)} ({len(failed)} failed), "
          f"{(end - start).seconds} s, scores {'yes' if (out / 'scores.json').exists() else 'NO'}", flush=True)


def run_with_retries(setup: str, run_id: int, attempts: int = 3) -> None:
    out = RESULTS / setup / f"run_{run_id}"
    for attempt in range(1, attempts + 1):
        run(setup, run_id)
        if (out / "scores.json").exists() and not (out / "INVALID.txt").exists():
            return
        failed = RESULTS / setup / "failed" / f"run_{run_id}_attempt_{attempt}"
        failed.parent.mkdir(exist_ok=True)
        shutil.move(out, failed)
        print(f"attempt {attempt} gave no valid scores; moved to {failed}", flush=True)
    print(f"{setup} run {run_id}: no scores after {attempts} attempts", flush=True)


if __name__ == "__main__":
    run_with_retries(sys.argv[1], int(sys.argv[2]))
