"""Step 1.4: classic NR-IQA baselines NIQE and BRISQUE (pyiqa, natural-image parameters, no training on our data).
Scores train (for the logistic map), test (Table 1) and the stress sets (Table 2, dev and test). Lower NIQE/BRISQUE is
better quality, so the quality score is the negated value. Output: outputs/agentic_ct_iqa/experts/classic_nriqa.csv
(columns set, key, metric, value) with keys in the same format as the expert predictions.
"""
import time

import numpy as np
import pandas as pd
import pyiqa
import torch
from scipy import ndimage

from src.config import CONFIG
from src.data import load_ldctiqa_split, read_id_list
from src.stress_sets import denoiser_group, dose_ladder

OUT = CONFIG.output_dir / "experts/classic_nriqa.csv"
METRICS = ("niqe", "brisque")


def items():
    """(set, key, image in [0, 1]) for every image the tables need."""
    train = {s.image_id: s for s in load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels)}
    test = {s.image_id: s for s in load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels)}
    for i, s in train.items():
        yield "train", i, s.image
    for i, s in test.items():
        yield "test", i, s.image
    for split, pool in (("s1_dev", train), ("s1_test", test)):
        for i in read_id_list(CONFIG.splits_dir / f"{split}.txt"):
            for sigma in (0.0, *CONFIG.blur_sigmas):
                yield split, f"{i}|{sigma}", pool[i].image if sigma == 0 else ndimage.gaussian_filter(pool[i].image, sigma)
    for split, source in (("s2_dev", "val"), ("s2_test", "test")):
        for g in read_id_list(CONFIG.splits_dir / f"{split}.txt"):
            for m, img in denoiser_group(CONFIG.iqa2026_root / source / g).items():
                yield split, f"{g}|{m}", img / 255.0
    for split, source in (("s3_dev", "val"), ("s3_test", "test")):
        for s in read_id_list(CONFIG.splits_dir / f"{split}.txt"):
            for r, img in dose_ladder(CONFIG.i2i_root / source / f"{s}.npz"):
                yield split, f"{s}|{r}", img


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    scorers = {m: pyiqa.create_metric(m, device="cpu") for m in METRICS}
    rows, start = [], time.time()
    with torch.no_grad():
        for n, (split, key, image) in enumerate(items(), 1):
            x = torch.from_numpy(np.clip(image, 0, 1).astype(np.float32))[None, None].repeat(1, 3, 1, 1)
            for metric, scorer in scorers.items():
                rows.append((split, key, metric, float(scorer(x).item())))
            if n % 200 == 0:
                print(f"{n} images, {time.time() - start:.0f}s", flush=True)
    pd.DataFrame(rows, columns=["set", "key", "metric", "value"]).to_csv(OUT, index=False)
    print(f"{len(rows)} scores -> {OUT}")


if __name__ == "__main__":
    main()
