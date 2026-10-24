"""Builds the results page of the sandbox experiment (experiment_sandboxes/results/sonuclar.html) from every run in
experiment_sandboxes/results: metrics per run and per setup, per-image scores and rationales, run behaviour.

    python -m scripts.make_sandbox_page
"""
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

from scripts.sandbox_results import KEY, RESULTS, SETUPS, collect, load_run, metrics, run_agreement
from src.config import PROJECT_ROOT

TEMPLATE = Path(__file__).with_name("sandbox_page_template.html")
IMAGES = PROJECT_ROOT / "experiment_sandboxes/setup1_vision_only/images"


def thumbnail(path: Path, size: int = 260) -> str:
    image = Image.open(path).convert("L")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def behaviour(folder: Path) -> dict:
    """Duration, turns, API-equivalent cost and which files the session read, from its transcript. Llama runs are
    single requests per image: duration from run_info, requests and failed attempts from the transcript."""
    path = folder / "transcript.jsonl"
    if not path.exists():
        return {}
    info = json.loads((folder / "run_info.json").read_text(encoding="utf-8-sig")) if (folder / "run_info.json").exists() else {}
    if info.get("agent") == "opencode":
        reads = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("type") != "tool_use" or event["part"]["tool"] != "read":
                continue
            file = "/" + str(event["part"]["state"].get("input", {}).get("path", "")).replace("\\", "/")  # may be relative
            kind = ("images" if "/images/img_" in file else "tool_results" if "/tool_results/" in file else
                    "tool_views" if "/tool_views/" in file else "examples" if "/examples/" in file else "docs")
            reads[kind] = reads.get(kind, 0) + 1
        seconds = (datetime.fromisoformat(info["end"]) - datetime.fromisoformat(info["start"])).total_seconds()
        return {"duration_s": round(seconds), "turns": info["steps"], "cost_usd": info["cost_usd"], "reads": reads,
                "attached": len(info.get("attached", []))}
    if "mode" in info:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        seconds = (datetime.fromisoformat(info["end"]) - datetime.fromisoformat(info["start"])).total_seconds()
        return {"duration_s": round(seconds), "turns": len(lines), "cost_usd": 0.0, "reads": {"images": 20},
                "retries": sum(1 for line in lines if line["attempt"] > 1), "mode": info["mode"]}
    reads, result = {}, {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if message.get("type") == "result":
            result = message
        if message.get("type") != "assistant":
            continue
        for block in message.get("message", {}).get("content", []):
            if block.get("type") != "tool_use" or block["name"] != "Read":
                continue
            file = str(block["input"].get("file_path", "")).replace("\\", "/")
            kind = ("images" if "/images/" in file or file.startswith("images") else
                    "tool_results" if "tool_results" in file else "tool_views" if "tool_views" in file else "docs")
            reads[kind] = reads.get(kind, 0) + 1
    return {"duration_s": round(result.get("duration_ms", 0) / 1000), "turns": result.get("num_turns"),
            "cost_usd": round(result.get("total_cost_usd") or 0, 3), "reads": reads}


def main() -> None:
    collect()
    key = pd.read_csv(KEY).sort_values("image_id").set_index("image_id")
    ids = list(key.index)
    runs, per_setup = [], {}
    for setup in SETUPS:
        folders = sorted((RESULTS / setup).glob("run_*"))
        if not folders:
            continue
        table = {}
        for folder in folders:
            scores, problems = load_run(folder, ids)
            entries = json.loads((folder / "scores.json").read_text(encoding="utf-8-sig"))["scores"] \
                if (folder / "scores.json").exists() else []
            table[folder.name] = {"scores": scores, "rationale": {e["image_id"]: e.get("rationale", "") for e in entries},
                                  "confidence": {e["image_id"]: e.get("confidence") for e in entries}}
            runs.append({"setup": setup, "run": folder.name, **metrics(scores, key["mos"]), **behaviour(folder),
                         "problems": problems})
        per_setup[setup] = table

    setups = []
    for setup, table in per_setup.items():
        frame = pd.DataFrame({run: t["scores"] for run, t in table.items()})
        mean = frame.mean(axis=1)
        error = mean - key["mos"]
        setups.append({"setup": setup, "runs": len(frame.columns), "mean": metrics(mean, key["mos"]),
                       "agreement": run_agreement(frame),
                       "image_sd": round(float(frame.std(axis=1).mean()), 3) if len(frame.columns) > 1 else None,
                       "bias_by_bin": [round(float(error[key["mos_bin"] == b].mean()), 3) for b in range(5)],
                       "mae_by_bin": [round(float(error[key["mos_bin"] == b].abs().mean()), 3) for b in range(5)]})

    expert = pd.DataFrame({r: t["scores"] for r, t in per_setup["setup3_medsiglip"].items()}).mean(axis=1)
    images = []
    for image_id in ids:
        row = {"id": image_id, "mos": round(float(key.loc[image_id, "mos"]), 3), "bin": int(key.loc[image_id, "mos_bin"]),
               "thumb": thumbnail(IMAGES / f"{image_id}.png"), "scores": {}, "rationales": {}}
        for setup, table in per_setup.items():
            row["scores"][setup] = [None if pd.isna(t["scores"][image_id]) else round(float(t["scores"][image_id]), 3)
                                    for t in table.values()]
            if setup != "setup3_medsiglip":
                row["rationales"][setup] = [t["rationale"].get(image_id, "") for t in table.values()]
        images.append(row)
    reliance = {setup: {"mean_abs_diff": round(float((pd.DataFrame({r: t["scores"] for r, t in per_setup[setup].items()})
                                                      .mean(axis=1) - expert).abs().mean()), 3),
                        "srocc_with_expert": round(float(spearmanr(pd.DataFrame(
                            {r: t["scores"] for r, t in per_setup[setup].items()}).mean(axis=1), expert)[0]), 3)}
                for setup in SETUPS if setup != "setup3_medsiglip" and per_setup.get(setup)}
    data = {"runs": runs, "setups": setups, "images": images, "reliance": reliance,
            "key_note": f"{len(ids)} LDCT-IQA test slices, 4 per MOS bin"}
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    latest = TEMPLATE.with_name("sandbox_page_latest.html").read_text(encoding="utf-8")  # newest results, top of page
    examples = RESULTS / "examples_panel.html"  # scripts/interpret_examples.py
    latest = latest.replace("<!--EXAMPLES-->", examples.read_text(encoding="utf-8") if examples.exists() else "")
    page = page.replace('<div class="page">\n', '<div class="page">\n' + latest, 1)
    out = RESULTS / "sonuclar.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
