"""E16b (issue #329): counterfactual faithfulness of the zero-shot agent (Z3 = ne19: Qwen3.8-27B OpenCode agent, images +
tool files, no MedSigLIP). Fixed before running (2026-10-22):

- Subset: 100 test images, 20 per MOS bin, drawn with default_rng(42) before any run. The issue's "20 per MOS bin" needs
  5 bins; the integer bins (0-4) hold only 13 images at 0, so the bins are 5 equal-width MOS bins of 0.8:
  [0, 0.8), [0.8, 1.6), [1.6, 2.4), [2.4, 3.2), [3.2, 4]. The 100 are shuffled (same rng) into 5 folders of 20.
- Most-cited tool group (E15, scripts/agent_analysis.py): noise level (100% of rationales). Removed in condition e16n:
  the numeric noise tools of configs/tool_groups.AGENT_JSON (high_pass_energy, noise_sigma_estimate, flat_roi_sigma,
  noise_residual_map values) and the noise percentiles of compare_to_reference (high_pass_std_pct,
  immerkaer_sigma_pct, nlm_residual_std_pct), and their lines in TOOLS.md. The views stay (group visual).
- Control e16f: the same 5 folders with the ne19 files unchanged, run in the same session, so both conditions share the
  folder composition (the original ne19 scores, from other folders, are reported as a third column only).
- Report: mean |score_e16n - score_e16f|, share of images whose score changes by more than 0.25, Overall on the 100 in
  both conditions (and the original ne19 run), paired bootstrap CI over the 100 images (2000, rng 42).

    python -m scripts.make_e16b         # experiment_sandboxes/e16b_{e16f,e16n}/batch_0k, colab/e16b_job/*.zip, key.csv
"""
import json
import shutil
import zipfile

import numpy as np
import pandas as pd

from configs import tool_groups as tg
from scripts.make_sandboxes import ROOT, isolate
from src.config import CONFIG, PROJECT_ROOT

SOURCE = ROOT / "test300_ne19"
JOB = PROJECT_ROOT / "colab/e16b_job"
BINS = [0, 0.8, 1.6, 2.4, 3.2, 4.01]
NOISE_TOOLS = [t for t, g in tg.AGENT_JSON.items() if g == "noise" and t != "reference_match"]
NOISE_PCT = [k for k, g in tg.AGENT_REFERENCE_KEYS.items() if g == "noise"]


def without_noise(record: dict) -> dict:
    for tool in NOISE_TOOLS:
        record["tools"].pop(tool, None)
    ref = record["tools"]["compare_to_reference"]["values"]
    for key in NOISE_PCT:
        ref.pop(key, None)
    return record


def tools_md_without_noise(text: str) -> str:
    numeric, views = text.split("## Views")
    keep = [l for l in numeric.split("\n") if not any(l.startswith(f"- `{t}`") for t in NOISE_TOOLS)]
    return "\n".join(keep) + "## Views" + views


def main() -> None:
    if (ROOT / "e16b_e16f").exists():
        raise SystemExit("experiment_sandboxes/e16b_e16f exists; delete it first to rebuild")
    per = pd.read_csv(CONFIG.output_dir / "noexp/per_image.csv")
    per["bin"] = pd.cut(per["mos"], BINS, right=False, labels=False)
    rng = np.random.default_rng(42)
    picked = np.concatenate([rng.choice(np.sort(per.loc[per["bin"] == b, "test_id"].to_numpy()), 20, replace=False)
                             for b in range(5)])
    picked = rng.permutation(picked)
    src = per.set_index("test_id")
    rows = []
    for n, test_id in enumerate(picked):
        rows.append({"folder": f"batch_{n // 20 + 1:02d}", "image_id": f"img_{n % 20 + 1:02d}", "test_id": test_id,
                     "mos": src.loc[test_id, "mos"], "src_batch": int(src.loc[test_id, "batch"]),
                     "src_image": src.loc[test_id, "image_id"], "ne19": src.loc[test_id, "ne19"]})
    key = pd.DataFrame(rows)
    JOB.mkdir(parents=True, exist_ok=True)
    key.to_csv(JOB / "key.csv", index=False)
    for cond in ("e16f", "e16n"):
        for folder, items in key.groupby("folder"):
            out = ROOT / f"e16b_{cond}" / folder
            for sub in ("images", "tool_results", "tool_views"):
                (out / sub).mkdir(parents=True, exist_ok=True)
            first = SOURCE / f"batch_{items['src_batch'].iloc[0]:02d}"
            shutil.copy(first / "TASK.md", out / "TASK.md")
            text = (first / "TOOLS.md").read_text(encoding="utf-8")
            (out / "TOOLS.md").write_text(tools_md_without_noise(text) if cond == "e16n" else text, encoding="utf-8")
            for _, r in items.iterrows():
                b = SOURCE / f"batch_{r['src_batch']:02d}"
                shutil.copy(b / "images" / f"{r['src_image']}.png", out / "images" / f"{r['image_id']}.png")
                record = json.loads((b / "tool_results" / f"{r['src_image']}.json").read_text(encoding="utf-8"))
                record["image_id"] = r["image_id"]
                if cond == "e16n":
                    record = without_noise(record)
                (out / "tool_results" / f"{r['image_id']}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
                for view in b.glob(f"tool_views/{r['src_image']}_*.png"):
                    shutil.copy(view, out / "tool_views" / view.name.replace(r["src_image"], r["image_id"], 1))
            isolate(out)
            with zipfile.ZipFile(JOB / f"{cond}_{folder}.zip", "w", zipfile.ZIP_DEFLATED) as z:
                for path in sorted(out.rglob("*")):
                    if path.is_file():
                        z.write(path, path.relative_to(ROOT).as_posix())
    print(key.groupby(pd.cut(key["mos"], BINS, right=False), observed=True).size().to_dict())
    print(f"{len(list(JOB.glob('*.zip')))} zips -> {JOB}; noise tools removed: {NOISE_TOOLS} + {NOISE_PCT}")


if __name__ == "__main__":
    main()
