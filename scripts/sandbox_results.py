"""Setup 3 scores and the comparison of all sandbox runs (experiment_sandboxes/results/).

    python -m scripts.sandbox_results medsiglip   # writes results/setup3_medsiglip/run_1/scores.json
    python -m scripts.sandbox_results collect     # validates every run, writes per_image.csv, summary.csv, report.md

Result standard: results/<setup>/run_<n>/scores.json with {"scores": [{"image_id", "score", "confidence",
"rationale"}, ...]}, one entry per image (img_01 ... img_20), score in [0, 4]. Claude runs also keep
transcript.jsonl and run_info.json (written by run_claude.ps1).
"""
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.make_sandboxes import anonymise
from src.config import CONFIG, PROJECT_ROOT
from src.experts import expert_scores, load_fold_predictions, strip_ext
from src.metrics import challenge_scores, logistic4

RESULTS = PROJECT_ROOT / "experiment_sandboxes/results"
KEY = CONFIG.output_dir / "sandbox/key.csv"
SETUPS = ("setup1_vision_only", "setup2_vision_tools", "setup3_medsiglip", "setup4_llama_vision_only",
          "setup5_llama_vision_tools", "setup6_qwen_vision_only", "setup7_qwen_vision_tools",
          "setup8_qwenflash_vision_only", "setup9_qwenflash_vision_tools",
          "setup10_medgemma_vision_only", "setup11_medgemma_vision_tools",
          "setup12_qwen35_4b_vision_only", "setup13_qwen35_4b_vision_tools", "setup14_qwen35_4b_agent_vision_only",
          "setup15_qwen35_4b_agent_vision_tools", "setup16_qwen38_27b_vision_only", "setup17_qwen38_27b_vision_tools",
          "setup18_qwen38_27b_agent_vision_only", "setup19_qwen38_27b_agent_vision_tools",
          "setup20_empero_4b_vision_only", "setup21_empero_4b_vision_tools", "setup22_empero_4b_agent_vision_only",
          "setup23_empero_4b_agent_vision_tools", "setup24_empero_4b_vision_tools_examples",
          "setup25_empero_4b_agent_vision_tools_examples", "setup26_qwen38_27b_agent_vision_tools_examples",
          "setup27_qwenflash_vision_tools_examples", "setup28_qwen35_4b_agent_vision_tools_examples",
          "setup29_qwen38_27b_agent_tools_references", "setup30_qwen38_27b_agent_tools_references_guided")


def medsiglip() -> None:
    """Tolga's verified fold predictions for the 20 test slices (5-model mean), mapped to 0-4 with the logistic fitted
    on the 1000 out-of-fold training scores. Same checkpoints as MedIA Table 12; no new GPU run needed.
    The same scores then feed setup 2 as the output of its consult_experts tool."""
    key = pd.read_csv(KEY)
    test = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    test.index = strip_ext(test.index)
    params = np.load(CONFIG.expert_dir / "expert_logistic_params.npy")
    rows = [{"image_id": r.image_id, "score": round(float(logistic4(test[r.test_id], *params)), 3), "confidence": None,
             "rationale": "MedSigLIP-FiLM, mean of the 5 fold models, mapped to 0-4 with the training logistic",
             "raw_score": round(float(test[r.test_id]), 4)} for r in key.sort_values("image_id").itertuples()]
    out = RESULTS / "setup3_medsiglip/run_1"
    if not (out / "scores.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        (out / "scores.json").write_text(json.dumps({"scores": rows}, indent=1), encoding="utf-8")
        (out / "run_info.json").write_text(json.dumps({"setup": "setup3_medsiglip", "run": 1, "model": "MedSigLIP-FiLM",
                                                       "source": str(CONFIG.expert_dir),
                                                       "date": datetime.now().isoformat()}, indent=1), encoding="utf-8")
        print("wrote", out / "scores.json")
    feed_setup2(rows)


def feed_setup2(rows: list[dict]) -> None:
    """Writes the expert score into setup 2's tool results exactly as consult_experts returns it to the agent:
    the 0-4 score as the value and the profile card as text (dataset name removed, as in the other tool texts)."""
    card = anonymise((CONFIG.expert_dir / "profile_card.txt").read_text(encoding="utf-8").strip())
    folder = PROJECT_ROOT / "experiment_sandboxes/setup2_vision_tools/tool_results"
    for row in rows:
        path = folder / f"{row['image_id']}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["tools"]["consult_experts"] = {"values": {"medsiglip_film": row["score"]}, "text": card}
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"consult_experts written into {len(rows)} setup 2 tool results")


def load_run(folder: Path, ids: list[str]) -> tuple[pd.Series, list[str]]:
    """Scores of one run and a list of problems with the file (missing images, out-of-range scores, bad JSON)."""
    problems = []
    try:
        entries = json.loads((folder / "scores.json").read_text(encoding="utf-8-sig"))["scores"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return pd.Series(dtype=float), [f"unreadable scores.json: {error}"]
    scores = {}
    for e in entries:
        try:
            value = float(e["score"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"bad entry {e!r:.60}")
            continue
        if not 0 <= value <= 4:
            problems.append(f"{e.get('image_id')}: score {value} outside [0, 4]")
        scores[str(e.get("image_id"))] = float(np.clip(value, 0, 4))
    missing = sorted(set(ids) - set(scores))
    extra = sorted(set(scores) - set(ids))
    if missing:
        problems.append(f"missing {len(missing)}: {', '.join(missing)}")
    if extra:
        problems.append(f"unknown ids: {', '.join(extra)}")
    return pd.Series(scores).reindex(ids), problems


def audit(folder: Path) -> dict:
    """What the session did, from its transcript: tool calls by name and how many distinct images it opened."""
    path = folder / "transcript.jsonl"
    if not path.exists():
        return {}
    calls, opened = {}, set()
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if message.get("type") == "tool_use":  # OpenCode event
            part = message["part"]
            calls[part["tool"]] = calls.get(part["tool"], 0) + 1
            file = "/" + str(part["state"].get("input", {}).get("path", "")).replace("\\", "/")  # may be relative
            if "/images/img_" in file:
                opened.add(file.rsplit("/", 1)[-1])
            continue
        if message.get("type") != "assistant":
            continue
        for block in message.get("message", {}).get("content", []):
            if block.get("type") == "tool_use":
                calls[block["name"]] = calls.get(block["name"], 0) + 1
                file = str(block.get("input", {}).get("file_path", "")).replace("\\", "/")
                if "/images/img_" in file or file.startswith("images/img_"):
                    opened.add(file.rsplit("/", 1)[-1])
    return {"tool_calls": calls, "images_opened": len(opened)}


def run_agreement(frame: pd.DataFrame) -> float | None:
    """Mean SROCC over pairs of runs (columns), each pair on the images both runs scored; a pair sharing fewer than 3
    images (a run that scored almost nothing) is left out."""
    pairs = []
    for a, b in combinations(frame.columns, 2):
        both = frame[[a, b]].dropna()
        if len(both) >= 3:
            pairs.append(spearmanr(both[a], both[b])[0])
    return round(float(np.nanmean(pairs)), 3) if pairs else None


def metrics(pred: pd.Series, mos: pd.Series) -> dict:
    ok = pred.notna()
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    m = challenge_scores(pred[ok].to_numpy(), mos[ok].to_numpy())
    rng = np.random.default_rng(CONFIG.seed)
    idx = np.flatnonzero(ok)
    boot = [spearmanr(pred.iloc[s], mos.iloc[s])[0] for s in (rng.choice(idx, len(idx)) for _ in range(2000))]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    # The challenge metrics are absolute values; the signed SROCC shows a model that ranks the images backwards.
    return {"n": int(ok.sum()), **{k: round(v, 3) for k, v in m.items()}, "srocc_ci95": f"[{lo:.2f}, {hi:.2f}]",
            "srocc_signed": round(float(spearmanr(pred[ok], mos[ok])[0]), 3),
            "mae": round(float((pred[ok] - mos[ok]).abs().mean()), 3), "bias": round(float((pred[ok] - mos[ok]).mean()), 3)}


def collect() -> None:
    key = pd.read_csv(KEY).sort_values("image_id").set_index("image_id")
    ids = list(key.index)
    per_image = key[["mos", "mos_bin"]].copy()
    rows, runs_by_setup = [], {}
    for setup in SETUPS:
        for folder in sorted((RESULTS / setup).glob("run_*")):
            pred, problems = load_run(folder, ids)
            column = f"{setup}/{folder.name}"
            per_image[column] = pred
            runs_by_setup.setdefault(setup, []).append(column)
            info = audit(folder)
            rows.append({"setup": setup, "run": folder.name, **metrics(pred, key["mos"]),
                         "images_opened": info.get("images_opened"), "tool_calls": json.dumps(info.get("tool_calls", {})),
                         "problems": "; ".join(problems)})
    for setup, columns in runs_by_setup.items():
        if len(columns) > 1:
            mean = per_image[columns].mean(axis=1)
            per_image[f"{setup}/mean"] = mean
            rows.append({"setup": setup, "run": f"mean of {len(columns)}", **metrics(mean, key["mos"]),
                         "run_agreement_srocc": run_agreement(per_image[columns]),
                         "per_image_sd": round(float(per_image[columns].std(axis=1).mean()), 3)})
    summary = pd.DataFrame(rows)
    per_image.round(3).to_csv(RESULTS / "per_image.csv")
    summary.to_csv(RESULTS / "summary.csv", index=False)
    shown = [c for c in ("setup", "run", "n", "plcc", "srocc", "srocc_ci95", "krocc", "overall", "mae", "bias",
                         "run_agreement_srocc", "per_image_sd", "images_opened", "problems") if c in summary]
    report = ["# Sandbox results", "", f"{len(ids)} LDCT-IQA test slices, 4 per MOS bin. Correlations without logistic "
              "fitting (too few images to fit 4 parameters). SROCC 95% CI: bootstrap over images.", "",
              summary[shown].fillna("").to_markdown(index=False), "",
              "With 20 images, differences in SROCC below about 0.15 are within noise; read the CIs.", ""]
    (RESULTS / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    {"medsiglip": medsiglip, "collect": collect}[sys.argv[1]]()
