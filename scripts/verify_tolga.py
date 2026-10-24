"""Verify Tolga's shared fold checkpoints before using them as the expert.

1. Test Overall of each fold model vs his MedIA manuscript, Table 12 (seed 42).
2. Fold split: a fold model fits the images it was trained on much better than the ones it held out. If our
   reconstruction of train.py's split matches his, every model shows a clear gap between our "held-out" images and
   the rest (our own retrained models give the reference size of that gap). No gap means the split differs and the
   out-of-fold scores would be leaked.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.config import CONFIG
from src.data import load_ldctiqa_split
from src.metrics import challenge_scores

TOLGA_TABLE12 = {1: 2.6991, 2: 2.6801, 3: 2.7453, 4: 2.6996, 5: 2.6682}
SOURCES = {"tolga": CONFIG.output_dir / "experts_tolga", "ours": CONFIG.output_dir / "experts"}


def main() -> None:
    train_mos = {s.image_id: s.mos for s in load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels, load_images=False)}
    test_mos = {s.image_id: s.mos for s in load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)}
    strip = lambda keys: keys.str.replace(r"\.tiff?$", "", regex=True)
    rows = []
    for source, folder in SOURCES.items():
        for k in range(1, 6):
            path = Path(folder) / f"preds_fold{k}.csv"
            if not path.exists():
                continue
            p = pd.read_csv(path, dtype={"key": str})
            test = p[p.set == "test"]
            overall = challenge_scores(test.pred, strip(test.key).map(test_mos))["overall"]
            train = p[p.set == "train"].assign(mos=lambda f: strip(f.key).map(train_mos))
            held = set(p[p.set == "oof"].key)
            held_rho = spearmanr(train[train.key.isin(held)].pred, train[train.key.isin(held)].mos)[0]
            seen_rho = spearmanr(train[~train.key.isin(held)].pred, train[~train.key.isin(held)].mos)[0]
            rows.append({"source": source, "fold": k, "test_overall": overall,
                         "table12": TOLGA_TABLE12[k] if source == "tolga" else float("nan"),
                         "rho_held_out": held_rho, "rho_seen": seen_rho, "gap": seen_rho - held_rho})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    out = CONFIG.output_dir / "experts_tolga/verification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out, index=False)


if __name__ == "__main__":
    main()
