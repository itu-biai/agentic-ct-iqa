"""How much does each tool help Stage B? colab/vm/tool_ablation.py re-scores with the saved rerun weights while one tool
input (or a group) is set to the bank median, 0.5 ("as usual"); no retraining, so the drop measures how much the
trained model relies on that input, not what a model trained without it would reach.

Fixed before the run (2026-10-22):
- PRIMARY: out-of-fold on the 1000 training slices (each slice scored by the fold model that did not train on it; no
  test image), llm head as is: overall with real tools minus overall with the ablated input, 95% CI by a paired
  bootstrap over slices, for the six inputs one at a time, the four noise inputs together and all six.
- Secondary: the same on the 300 test images (5-fold mean; analysis only, nothing is selected on it), the vision head,
  and the mean absolute change of the prediction (MOS points). The noise inputs overlap (noise_score is the mean of the
  other three percentiles), so single noise drops understate the group; the "noise" row is the group's share.
- Check: the "full" predictions must reproduce the rerun's saved ones (outputs/.../llm2/out_llm).

    python -m scripts.tool_ablation_eval      # -> outputs/agentic_ct_iqa/vision_ft/tool_ablation.md
"""
import numpy as np
import pandas as pd

from scripts.qwen_probe import bootstrap
from scripts.qwen_vision_ft import JOB, OUT, RERUN, llm_predictions, overall
from src.config import CONFIG
from src.data import load_ldctiqa_split

ABLATIONS = ("residual_std", "gradient_deviation_mean", "hf_residual_mean", "noise_score", "gradient_p95",
             "streak_anisotropy", "noise", "all")
LABELS = {"noise": "4 noise inputs together", "all": "all 6 tools (image only)"}


def predictions(a, n_train: int, head: str, part: str, ab: str) -> np.ndarray:
    if part == "test":
        return np.mean([a[f"{head}_test_{ab}_{k}"] for k in range(1, 6)], axis=0)
    pred = np.full(n_train, np.nan)
    for k in range(1, 6):
        pred[a[f"va_{k}"]] = a[f"{head}_val_{ab}_{k}"]
    return pred


def main() -> None:
    a = np.load(JOB / "ablation.npz")
    z = np.load(JOB / "train.npz")
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    assert list(np.load(JOB / "test.npz")["ids"]) == [s.image_id for s in samples]
    mos = {"val": z["mos"], "test": np.array([s.mos for s in samples])}
    lines = ["# Tool ablation: how much each tool input helps Stage B (rerun weights, inference only)", "",
             "Ablated inputs set to 0.5, the bank median (\"as usual\"; the prompt reads 50). Drop = overall with real "
             "tools - overall ablated; 95% CI by a paired bootstrap over images. Out-of-fold: 1000 training slices, "
             "each scored by the fold model that did not see it. Test: 300 images, 5-fold mean.", ""]
    for head in ("llm", "vision"):
        for part in ("val", "test"):
            full = predictions(a, len(z["mos"]), head, part, "full")
            saved = llm_predictions(head, part, RERUN)
            rows = {"real tools": {"overall": overall(full, mos[part]), "drop": "", "95% CI": "", "mean |change|": ""}}
            for ab in ABLATIONS:
                p = predictions(a, len(z["mos"]), head, part, ab)
                rows[LABELS.get(ab, ab)] = {"overall": overall(p, mos[part]),
                                            "drop": round(overall(full, mos[part]) - overall(p, mos[part]), 4),
                                            "95% CI": "[{:+.3f}, {:+.3f}]".format(*bootstrap(full, p, mos[part])),
                                            "mean |change|": round(float(np.abs(full - p).mean()), 3)}
            table = pd.DataFrame(rows).T
            table["overall"] = table["overall"].astype(float).round(4)
            name = "out-of-fold, 1000 training slices" if part == "val" else "test, 300 images"
            tag = " (PRIMARY)" if (head, part) == ("llm", "val") else ""
            lines += [f"## {head} head, {name}{tag}", "", table.to_markdown(), "",
                      f"Check: max |real tools - saved rerun prediction| = {np.abs(full - saved).max():.4f}", ""]
    (OUT / "tool_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
