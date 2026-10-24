"""T0 on the lab set for the denoiser tau (issue #329 E2): the 26 screening measurements (scripts/screen_tools.py TOOLS,
the columns of outputs/agentic_ct_iqa/screening/train.csv) on the 2496 lab IQA test images, converted to the
LDCT-IQA storage window as in scripts/cross_pool.py; then the tools-boosting model of scripts/tools_only_test.py
(same estimator, fit on all 1000 training slices with median fill) predicts them.

    python -m scripts.lab_tools26     # -> outputs/agentic_ct_iqa/i329/lab_tools26.csv, lab_t0.csv
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from scripts.pseudo_pool import to_storage
from scripts.screen_tools import TOOLS, measure
from scripts.tools_only_test import MODELS, load
from src.config import CONFIG, PROJECT_ROOT

OUT = CONFIG.output_dir / "i329"
JOB = PROJECT_ROOT / "colab/vft_job"


def one(image_id: str) -> dict:
    v = to_storage(np.asarray(Image.open(CONFIG.iqa2026_root / "test" / f"{image_id}.png")))
    return {"image_id": image_id, **measure(v, list(TOOLS))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "lab_tools26.csv"
    ids = [str(i) for i in np.load(JOB / "lab329.npz")["ids"]]
    if not cache.exists():
        rows = Parallel(n_jobs=-1, verbose=5)(delayed(one)(i) for i in ids)
        pd.DataFrame(rows).to_csv(cache, index=False)
    table = pd.read_csv(cache).set_index("image_id").loc[ids]
    train, mos = load("train")
    x = train.fillna(train.median()).to_numpy()
    xl = table[list(train.columns)].fillna(train.median()).to_numpy()
    pred = MODELS["tools boosting"]().fit(x, mos).predict(xl)
    pd.DataFrame({"image_id": ids, "T0": pred}).to_csv(OUT / "lab_t0.csv", index=False)
    print("T0 lab predictions", len(pred), "->", OUT / "lab_t0.csv")


if __name__ == "__main__":
    main()
