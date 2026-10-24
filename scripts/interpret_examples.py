"""Ten worked examples for interpretability: per test image, what Stage B was given and which inputs its score depends
on, next to what the Qwen3.8-27B agent (setup 26, test300 run) did on the same image: which tool files and views it
opened, the measurements and expert score it read, and its written rationale.

Stage B calls no tools and writes no text: its six tool percentiles are computed beforehand and put into the prompt as
numbers, and a linear head on the last token gives the score. Its "reasoning" is shown by sensitivity: the change of its
score when one input is set to the bank median (colab/vm/tool_ablation.py, 5-fold mean, llm head).

Selection fixed before looking at any example (2026-10-22): 2 test images per MOS bin (0-4), drawn with
numpy default_rng(0) from outputs/agentic_ct_iqa/test300/per_image.csv, sorted by MOS. No other filter.

    python -m scripts.interpret_examples   # -> outputs/agentic_ct_iqa/vision_ft/interpret_examples.md,
                                           #    experiment_sandboxes/results/examples_panel.html
"""
import base64
import html
import io
import json

import numpy as np
import pandas as pd
from PIL import Image

from scripts.qwen_vision_ft import JOB, OUT, RERUN, llm_predictions, on_grid
from src.config import CONFIG, PROJECT_ROOT

T300 = CONFIG.output_dir / "test300"
PANEL = PROJECT_ROOT / "experiment_sandboxes/results/examples_panel.html"
TOOL_NAMES = ("residual_std", "gradient_deviation_mean", "hf_residual_mean", "noise_score", "gradient_p95",
              "streak_anisotropy")
SHORT = {"residual_std": "gürültü artığı", "gradient_deviation_mean": "gradyan sapması", "hf_residual_mean": "yüksek frekans",
         "noise_score": "gürültü özeti", "gradient_p95": "yapı (kenar)", "streak_anisotropy": "çizgi artefaktı"}


def pick() -> pd.DataFrame:
    table = pd.read_csv(T300 / "per_image.csv")
    rng = np.random.default_rng(0)
    rows = [table.loc[rng.choice(table.index[table["mos_bin"] == b], 2, replace=False)] for b in range(5)]
    return pd.concat(rows).sort_values("mos")


def agent_trace(batch: int, image_id: str) -> dict:
    folder = T300 / "runs" / f"test300_batch_{batch:02d}" / "run_1"
    score = next(s for s in json.loads((folder / "scores.json").read_text(encoding="utf-8"))["scores"]
                 if s["image_id"] == image_id)
    opened, measures = [], {}
    for line in open(folder / "transcript.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["type"] != "tool_use" or r["part"]["tool"] != "read":
            continue
        path = r["part"]["state"]["input"]["path"]
        if f"/{image_id}" not in path:
            continue
        opened.append(path.split(f"batch_{batch:02d}/")[-1])
        if path.endswith(f"tool_results/{image_id}.json"):
            text = str(r["part"]["state"]["output"])
            body = json.loads("\n".join(l.split(": ", 1)[1] if ": " in l else "" for l in text.split("\n")[1:]
                                        if l[:1].isdigit()))
            measures = {"reference": body["tools"]["compare_to_reference"]["values"],
                        "expert": body["tools"]["consult_experts"]["values"]["medsiglip_film"]}
    return {"score": score["score"], "rationale": score["rationale"], "opened": opened, **measures}


def thumbnail(image: np.ndarray, size: int = 200) -> str:
    im = Image.fromarray(image)
    im.thumbnail((size, size))
    buffer = io.BytesIO()
    im.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    test = np.load(JOB / "test.npz")
    index = {k: i for i, k in enumerate(test["ids"])}
    stage_b = llm_predictions("llm", "test", RERUN)
    ab = np.load(JOB / "ablation.npz") if (JOB / "ablation.npz").exists() else None
    if ab is not None:
        full = np.mean([ab[f"llm_test_full_{k}"] for k in range(1, 6)], axis=0)
        sens = {t: full - np.mean([ab[f"llm_test_{t}_{k}"] for k in range(1, 6)], axis=0) for t in (*TOOL_NAMES, "all")}
    md = ["# Ten worked examples (2 per MOS bin, fixed draw)", ""]
    cards = []
    for _, row in pick().iterrows():
        i = index[row["test_id"]]
        tools = test["tools"][i]
        trace = agent_trace(int(row["batch"]), row["image_id"])
        given = ", ".join(f"{t} {min(99, int(round(v * 100))):02d}" for t, v in zip(TOOL_NAMES, tools))
        md += [f"## {row['test_id']}: MOS {row['mos']:.2f}", "",
               f"- Stage B: {stage_b[i]:.2f} (grid {on_grid(stage_b[i]):.1f}); MedSigLIP {row['expert']:.2f}; "
               f"agent {trace['score']:.1f}",
               f"- Stage B prompt numbers: {given}"]
        if ab is not None:
            md.append("- Stage B score change when the input is set to 50: "
                      + ", ".join(f"{t} {sens[t][i]:+.2f}" for t in (*TOOL_NAMES, "all")))
        md += [f"- Agent opened: {', '.join(trace['opened'])}",
               f"- Agent read: expert {trace.get('expert')}, reference percentiles {trace.get('reference')}",
               f"- Agent rationale: {trace['rationale']}", ""]
        bars = ""
        if ab is not None:
            scale = max(0.3, max(abs(sens[t][i]) for t in TOOL_NAMES))
            for t in TOOL_NAMES:
                v = sens[t][i]
                bars += (f'<div class="sbar"><span>{SHORT[t]} <b>{min(99, int(round(tools[TOOL_NAMES.index(t)] * 100))):02d}</b></span>'
                         f'<i style="--w:{abs(v) / scale * 100:.0f}%" class="{"pos" if v >= 0 else "neg"}"></i>'
                         f'<em>{v:+.2f}</em></div>')
            bars += f'<div class="sall">6 aracın hepsi 50 olunca: <b>{sens["all"][i]:+.2f}</b></div>'
        views = [p.split("/")[-1].replace(row["image_id"] + "_", "").replace(".png", "") for p in trace["opened"]
                 if "tool_views" in p]
        cards.append(f"""<article class="ex">
  <img src="{thumbnail(test['images'][i])}" alt="{row['test_id']}, MOS {row['mos']:.2f}" width="200" height="200">
  <div class="exbody">
    <div class="exhead"><b>{row['test_id']}</b> · MOS <b>{row['mos']:.2f}</b> · LabQwen <b>{stage_b[i]:.2f}</b> · MedSigLIP {row['expert']:.2f} · ajan {trace['score']:.1f}</div>
    <div class="excols">
      <div><span class="exlab">LabQwen: girdi yüzdelikleri ve skor duyarlılığı</span>{bars or '<span class="caption">ablasyon sonucu bekleniyor</span>'}</div>
      <div><span class="exlab">Ajan (düzen 26): açtığı araçlar ve gerekçesi</span>
        <div class="caption">Ölçüm dosyası{', görünümler: ' + ', '.join(views) if views else ', ek görünüm açmadı'} · MedSigLIP aracı {trace.get('expert', '–')}</div>
        <p class="why">{html.escape(trace['rationale'])}</p></div>
    </div>
  </div>
</article>""")
    (OUT / "interpret_examples.md").write_text("\n".join(md), encoding="utf-8")
    PANEL.write_text("\n".join(cards), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
