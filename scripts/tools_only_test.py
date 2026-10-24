"""Tools-only quality model, no expert score: the 26 screening measurements (outputs/agentic_ct_iqa/screening) of the
1000 training slices -> MOS. Train 5-fold CV first, then one look at the 300 test slices with everything fixed here.

    python -m scripts.tools_only_test cv     # training cross-validation only
    python -m scripts.tools_only_test test   # fit on all 1000 training slices, single test look -> probe/tools_only_test.md
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.qwen_probe import bootstrap
from src.config import CONFIG
from src.data import load_ldctiqa_split
from src.experts import expert_scores, load_fold_predictions, strip_ext
from src.metrics import challenge_scores

SCREENING = CONFIG.output_dir / "screening"
OUT = CONFIG.output_dir / "probe"
MODELS = {"tools ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
          "tools boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                                  random_state=CONFIG.seed)}


def load(split: str, columns: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """The test table carries no MOS column; its MOS comes from the label file."""
    frame = pd.read_csv(SCREENING / f"{split}.csv", dtype={"image_id": str}).set_index("image_id")
    if "mos" in frame:
        mos = frame.pop("mos").to_numpy()
    else:
        labels = {s.image_id: s.mos for s in load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)}
        mos = np.array([labels[k] for k in frame.index])
    frame = frame.drop(columns="__tools", errors="ignore")
    return (frame if columns is None else frame[columns]), mos


def expert(split: str, ids: pd.Index) -> np.ndarray:
    scores = expert_scores(load_fold_predictions(CONFIG.expert_dir))[split]
    scores.index = strip_ext(scores.index)
    return scores.reindex(ids).to_numpy()


def main(mode: str) -> None:
    train, mos = load("train")
    median = train.median()
    x = train.fillna(median).to_numpy()
    if mode == "cv":
        for name, model in MODELS.items():
            pred = np.zeros(len(mos))
            for a, b in KFold(5, shuffle=True, random_state=0).split(x):
                pred[b] = model().fit(x[a], mos[a]).predict(x[b])
            print(name, round(challenge_scores(pred, mos)["overall"], 4))
        return
    test, mos_t = load("test", list(train.columns))
    xt = test.fillna(median).to_numpy()
    expert_t = expert("test", test.index)
    preds = {"MedSigLIP-FiLM (expert)": expert_t}
    for name, model in MODELS.items():
        preds[name] = model().fit(x, mos).predict(xt)
    preds["expert/tools boosting 50/50"] = (expert_t + preds["tools boosting"]) / 2
    base = challenge_scores(expert_t, mos_t)["overall"]
    rows = []
    for name, pred in preds.items():
        s = challenge_scores(pred, mos_t)
        lo, hi = bootstrap(pred, expert_t, mos_t) if name != "MedSigLIP-FiLM (expert)" else (None, None)
        rows.append({"method": name, **{k: round(v, 4) for k, v in s.items()}, "minus expert": round(s["overall"] - base, 4),
                     "95% CI": "" if lo is None else f"[{lo:+.3f}, {hi:+.3f}]"})
    lines = ["# Tools-only model on the 300-image test (single look)", "",
             f"{x.shape[1]} screening measurements, no expert input; fit on all 1000 training slices (training CV: "
             "ridge 2.758, boosting 2.791, expert OOF 2.788). Raw overall, paired bootstrap over test images.", "",
             pd.DataFrame(rows).set_index("method").to_markdown(), ""]
    (OUT / "tools_only_test.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame({"test_id": test.index, "mos": mos_t, **preds}).round(4).to_csv(OUT / "tools_only_test.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
