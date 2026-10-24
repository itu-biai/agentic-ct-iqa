"""Qwen3.8-27B as an OpenCode agent in the setup 1/2 sandboxes, on the Colab VM: the protocol of setups 6-9.

Same as scripts/sandbox_opencode.py, with the model served by the local vLLM server (q27_serve.py) instead of
OpenCode Go: same OpenCode version, same one-line prompt, same permission set (read, glob, edit; everything else
denied and hidden), config outside the sandbox, PWD = sandbox and a check that the session opened there. A run that
ends without scores.json is moved to <setup>/failed/ and restarted, up to three attempts.

Context and output limits come from /content/q27/serve.json, written by q27_serve.py (the context the server holds).
Expects /content/q27/sb/{setup1_vision_only,setup2_vision_tools} (copies of the local sandboxes, with .git).
Writes /content/q27/agent/<setup>/run_<n>/: transcript.jsonl, session.json, scores.json, run_info.json, extra/.
    python q27_agent.py <setup> <run>

Setup test300_batch_XX: the 300-image test in 15 sandboxes of 20 (scripts/make_test300.py), setup 26's protocol, in
/content/q27/sb/test300/batch_XX. Several of these run at once against the same server (q27_test300.py), so each run
restores only its own sandbox and uses its own export file.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path("/content/q27")
RESULTS = ROOT / "agent"
CONFIG = ROOT / "opencode_sandbox.json"
SETUPS = {"setup18_qwen38_27b_agent_vision_only": "setup1_vision_only",
          "setup19_qwen38_27b_agent_vision_tools": "setup2_vision_tools",
          "setup26_qwen38_27b_agent_vision_tools_examples": "setup2b_vision_tools_examples",
          "setup29_qwen38_27b_agent_tools_references": "setup4_tools_references",
          "setup30_qwen38_27b_agent_tools_references_guided": "setup4b_tools_references_guided"}
PARALLEL = {"setup29_qwen38_27b_agent_tools_references", "setup30_qwen38_27b_agent_tools_references_guided"}
MODEL = "local/qwen3.8-27b"
SERVE = json.loads((ROOT / "serve.json").read_text())
PROMPT = "Read TASK.md in this folder and complete the task it describes."
PERMISSION = {"*": "deny", "read": "allow", "glob": "allow", "edit": "allow", "external_directory": "deny"}
PROVIDER = {"local": {"npm": "@ai-sdk/openai-compatible", "name": "vLLM on this VM",
                      "options": {"baseURL": "http://localhost:8000/v1"},
                      "models": {"qwen3.8-27b": {"name": "Qwen3.8-27B", "attachment": True, "reasoning": True,
                                                "tool_call": True, "temperature": True,
                                                "modalities": {"input": ["text", "image"], "output": ["text"]},
                                                "limit": {"context": SERVE["max_model_len"],
                                                          "output": 32768}}}}}


def write_config() -> None:
    CONFIG.write_text(json.dumps({"$schema": "https://opencode.ai/config.json", "autoupdate": False,
                                  "share": "disabled", "mcp": {}, "permission": PERMISSION, "provider": PROVIDER},
                                 indent=1), encoding="utf-8")


def opencode(*args: str, cwd: Path, env: dict, timeout: int, stdout=subprocess.PIPE) -> subprocess.CompletedProcess:
    return subprocess.run([shutil.which("opencode"), *args], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          stdout=stdout, stderr=subprocess.PIPE, timeout=timeout)


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


def sandbox_folder(setup: str) -> str:
    if setup.startswith("test300_batch_"):
        return f"test300/{setup.removeprefix('test300_')}"
    if setup.startswith("test300g_batch_"):  # setup 30 on the 300-image test (scripts/make_references.py test300)
        return f"test300_guided/{setup.removeprefix('test300g_')}"
    if setup[:4] in ("ne18", "ne19", "ne26"):  # no-expert setups on the 300-image test (scripts/make_noexpert.py)
        return f"test300_{setup[:4]}/{setup[5:]}"
    if setup[:4] in ("e16f", "e16n"):  # issue #329 E16b, ne19 with and without the noise tools (scripts/make_e16b.py)
        return f"e16b_{setup[:4]}/{setup[5:]}"
    return SETUPS[setup]


def sandbox_base(setup: str, run_id: int) -> Path:
    """Runs of a setup in PARALLEL go at the same time (q27_refs_all.py), each in its own copy of the sandbox."""
    return ROOT / f"sb_run{run_id}" if setup in PARALLEL else ROOT / "sb"


def restore(folder: str, base: Path) -> None:
    """Fresh sandbox for an attempt: run() moves every file a session touched to extra/, so a session that edited an
    input (e.g. examples/labels.json) would leave the next attempt without it. Only this sandbox's files are rewritten:
    other runs may be working in their own sandboxes meanwhile."""
    for archive in sorted(ROOT.glob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.namelist() if m.startswith(folder + "/")]
            if members:
                z.extractall(base, members)
                return


def run(setup: str, run_id: int) -> None:
    folder = sandbox_folder(setup)
    sandbox = sandbox_base(setup, run_id) / folder
    out = RESULTS / setup / f"run_{run_id}"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    if (sandbox / "scores.json").exists():
        raise SystemExit(f"scores.json is still in {sandbox}; move it out first")
    out.mkdir(parents=True)
    prompt, attached = examples_message(sandbox)
    write_config()
    env = {**os.environ, "OPENCODE_CONFIG": str(CONFIG), "OPENCODE_DISABLE_CLAUDE_CODE": "1", "PWD": str(sandbox)}

    start = datetime.now()
    with open(out / "transcript.jsonl", "wb") as transcript:
        try:
            done = opencode("run", "--standalone", "--format", "json", "-m", MODEL, prompt, *attached, cwd=sandbox, env=env,
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
        raw = ROOT / f"session_export_{os.getpid()}.tmp"
        with open(raw, "wb") as handle:
            opencode("session", "export", session, cwd=sandbox, env=env, timeout=300, stdout=handle)
        exported = raw.read_text(encoding="utf-8")
        raw.unlink()
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
        "setup": setup, "run": run_id, "agent": "opencode", "model": "Qwen/Qwen3.8-27B", "provider": PROVIDER,
        "server": SERVE["server"], "variant": "default", "sandbox": folder, "session_directory": directory,
        "prompt": prompt, "attached": attached[1::2], "permission": PERMISSION, "env": "OPENCODE_DISABLE_CLAUDE_CODE=1",
        "start": start.isoformat(), "end": end.isoformat(), "exit_code": code, "session": session,
        "steps": len(steps), "cost_usd": 0, "tokens": info.get("tokens"),
        "tools_used": sorted({c["tool"] for c in calls}), "opencode_version": version}, indent=1), encoding="utf-8")
    failed = [c for c in calls if c["state"].get("status") != "completed"]
    print(f"{setup} run {run_id}: steps {len(steps)}, tool calls {len(calls)} ({len(failed)} failed), "
          f"{(end - start).seconds} s, scores {'yes' if (out / 'scores.json').exists() else 'NO'}", flush=True)


def run_with_retries(setup: str, run_id: int, attempts: int = 3) -> None:
    out = RESULTS / setup / f"run_{run_id}"
    for attempt in range(1, attempts + 1):
        restore(sandbox_folder(setup), sandbox_base(setup, run_id))
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
