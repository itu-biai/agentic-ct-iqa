"""Qwen3.5-4B (Qwen/Qwen3.5-4B) on the 20 sandbox images, served by vLLM on a Colab T4, in both protocols.

Setups 12 and 13 follow the MedGemma protocol (setups 10 and 11): one independent request per image, same prompts
(colab/medgemma_job/prompts.json), no tools or file system (colab/vm/qwen35_single.py). Setups 14 and 15 follow the
Qwen Max protocol (setups 6 and 7): an OpenCode agent in copies of the setup 1/2 sandboxes with the same prompt and
permissions (colab/vm/q4_agent.py).

    python -m scripts.sandbox_qwen35 sandboxes             # writes colab/qwen35_job/sandboxes.zip
    python -m scripts.sandbox_qwen35 ingest <raw.jsonl>    # setups 12/13 -> results/setup1x_qwen35_4b_*/run_<n>/
    python -m scripts.sandbox_qwen35 agent <agent.zip>     # setups 14/15 -> results/setup1x_qwen35_4b_agent_*/
"""
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from scripts.sandbox_llama import RESULTS, SANDBOXES, parse
from src.config import PROJECT_ROOT

MODEL = "Qwen/Qwen3.5-4B"
JOB = PROJECT_ROOT / "colab/qwen35_job"


def sandboxes() -> None:
    JOB.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(JOB / "sandboxes.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("setup1_vision_only", "setup2_vision_tools"):
            for path in (SANDBOXES / folder).rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(SANDBOXES).as_posix())
    print("wrote", JOB / "sandboxes.zip")


def ingest(raw: Path, model: str = MODEL, provider: str = "Colab Tesla T4, vLLM, fp16") -> None:
    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = next(r for r in records if r.get("type") == "meta")
    runs = defaultdict(list)
    for r in records:
        if r.get("type") == "generation":
            runs[(r["setup"], r["run"])].append(r)
    for (setup, run_id), rows in sorted(runs.items()):
        out = RESULTS / setup / f"run_{run_id}"
        if out.exists():
            print(f"{out} exists, skipped")
            continue
        out.mkdir(parents=True)
        scores = []
        for image_id in sorted({r["image_id"] for r in rows}):
            last = max((r for r in rows if r["image_id"] == image_id), key=lambda r: r["attempt"])
            # The server splits the thinking from the answer. If the thinking used up all tokens there is no answer,
            # and the score stated in the thinking is used (same rule as MedGemma).
            answer = parse(last["response"]) or parse(last["reasoning"])
            if answer is not None:
                scores.append({"image_id": image_id, **answer,
                               "thinking_complete": bool(last["response"]) and last["finish_reason"] != "length"})
        (out / "scores.json").write_text(json.dumps({"scores": scores}, indent=1), encoding="utf-8")
        (out / "transcript.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        (out / "run_info.json").write_text(json.dumps({
            "setup": setup, "run": run_id, "model": model, "provider": provider,
            "generation": meta["sampling"], "mode": "one independent request per image, no tools, no file system",
            "start": min(r["start"] for r in rows), "end": max(r["end"] for r in rows), "requests": len(rows)},
            indent=1), encoding="utf-8")
        print(f"{out}: {len(scores)}/20 scored, {sum(a['format'] == 'json' for a in scores)} as JSON, "
              f"{sum(not a['thinking_complete'] for a in scores)} without a finished answer")


def recover(setup: str, run: Path, model: str = MODEL, server: str = "vLLM on Colab T4, fp16") -> None:
    """Rebuilds scores.json and run_info.json of an agent run whose runner stopped after the session ended (the first
    runner version crashed on a truncated session export, before moving scores.json out of the sandbox). The file is
    replayed from the session's own write/edit tool calls on scores.json in transcript.jsonl, which hold its full
    content; nothing is added. run_info.json records that it was reconstructed."""
    events = [json.loads(line) for line in (run / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip().startswith("{")]
    content = None
    for e in events:
        if e.get("type") != "tool_use" or e["part"]["state"].get("status") != "completed":
            continue
        tool, args = e["part"]["tool"], e["part"]["state"].get("input", {})
        path = str(args.get("path") or args.get("filePath") or "")
        if not path.replace("\\", "/").endswith("scores.json"):
            continue
        if tool == "write":
            content = args["content"]
        elif tool == "edit" and content is not None:
            old, new = args.get("oldString", args.get("old_string")), args.get("newString", args.get("new_string"))
            content = content.replace(old, new, -1 if args.get("replaceAll") else 1)
    if content is not None:
        (run / "scores.json").write_text(content, encoding="utf-8")
    calls = [e["part"] for e in events if e.get("type") == "tool_use"]
    stamps = [e["timestamp"] for e in events if "timestamp" in e]
    tokens = {"input": 0, "output": 0, "reasoning": 0}
    for e in events:
        if e.get("type") == "step_finish":
            for k in tokens:
                tokens[k] += e["part"].get("tokens", {}).get(k, 0)
    from datetime import datetime
    import re
    head = (run / "session.json").read_text(encoding="utf-8")[:20000] if (run / "session.json").exists() else ""
    found = re.search(r'"directory":\s*"([^"]+)"', head)
    (run / "run_info.json").write_text(json.dumps({
        "setup": setup, "run": int(run.name.split("_")[1]), "agent": "opencode", "model": model,
        "server": server, "variant": "default",
        "sandbox": "setup1_vision_only" if "vision_only" in setup else
                   "setup2b_vision_tools_examples" if "examples" in setup else "setup2_vision_tools",
        "session_directory": found.group(1) if found else None, "start": datetime.fromtimestamp(min(stamps) / 1000).isoformat(),
        "end": datetime.fromtimestamp(max(stamps) / 1000).isoformat(), "exit_code": None,
        "session": next((e["sessionID"] for e in events if e.get("sessionID")), None),
        "steps": sum(e.get("type") == "step_finish" for e in events), "cost_usd": 0, "tokens": tokens,
        "tools_used": sorted({c["tool"] for c in calls}), "opencode_version": "opencode v2.0.26",
        "reconstructed": "scores.json replayed from the session's write/edit calls in transcript.jsonl; "
                         "runner stopped on a truncated session export"}, indent=1), encoding="utf-8")
    print(run, "reconstructed:", "scores.json written" if content is not None else "NO scores.json write in transcript")


def agent(archive: Path, setups: tuple = ("setup14_qwen35_4b_agent_vision_only", "setup15_qwen35_4b_agent_vision_tools"),
          **recovery) -> None:
    """Unpacks the VM's /content/q4/agent tree (one folder per setup) into results/, run folders and failed/."""
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target = RESULTS / name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    for setup in setups:
        for run in sorted((RESULTS / setup).glob("run_*")):
            if not (run / "run_info.json").exists():
                recover(setup, run, **recovery)
            scores = run / "scores.json"
            n = len(json.loads(scores.read_text(encoding="utf-8"))["scores"]) if scores.exists() else 0
            print(run, f"{n}/20 scored")


if __name__ == "__main__":
    {"sandboxes": lambda: sandboxes(), "ingest": lambda: ingest(Path(sys.argv[2])),
     "agent": lambda: agent(Path(sys.argv[2]))}[sys.argv[1]]()
