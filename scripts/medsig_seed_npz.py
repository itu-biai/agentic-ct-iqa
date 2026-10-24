"""E12: MedSigLIP seed runs (colab/vm/medsig_seed.py, Tolga's test.py preds.csv: path, mos, pred_fold1..5) as
outputs/agentic_ct_iqa/i329/medsig_seed<n>/fold_<k>.npz (pred_test in the order of colab/vft_job/test.npz ids), the
format scripts/issue329_eval.py reads. Usage: python -m scripts.medsig_seed_npz
"""
from pathlib import Path

import numpy as np
import pandas as pd

I329 = Path("outputs/agentic_ct_iqa/i329")
ids = np.load("colab/vft_job/test.npz")["ids"].astype(str)
for n in (2, 3):
    d = I329 / f"medsig_seed{n}"
    if not (d / "preds.csv").exists():
        continue
    p = pd.read_csv(d / "preds.csv")
    p.index = [Path(x).stem.lower() for x in p["path"]]
    p = p.loc[ids]
    for k in range(1, 6):
        np.savez(d / f"fold_{k}.npz", pred_test=p[f"pred_fold{k}"].to_numpy(), mos=p["mos"].to_numpy(), ids=ids)
    print(f"seed run {n}: {len(p)} test images, 5 folds")
