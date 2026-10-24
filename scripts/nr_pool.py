"""Training images for no-reference PSNR / SSIM prediction on the biailab IQA 2026v1 set (scripts/nr_eval.py): 300
soft-tissue slices drawn (seed 0) from its train split, all 13 images each (gt, noisy 100k, 11 denoisers), in the
Stage B input format of scripts/pseudo_pool.py. The test images are the IQA test split already in cross.npz
(scripts/cross_pool.py). Targets: PSNR and SSIM of every non-gt image to its gt by the dataset's protocol (8-bit
PNGs, circular mask, data range 255), for train and test, in nr_targets.csv.

    python -m scripts.nr_pool      # -> colab/vft_job/nr.npz (ids, patient, kind, images, tools), nr_targets.csv
"""
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from scripts.cross_pool import METHODS
from scripts.pseudo_pool import TOOLS, to_storage
from scripts.qwen_vision_ft import CUES, NOISE, percentiles
from scripts.screen_tools import measure
from src.config import CONFIG, PROJECT_ROOT

JOB = PROJECT_ROOT / "colab/vft_job"
N_SLICES = 300


def mask() -> np.ndarray:
    yy, xx = np.ogrid[:512, :512]
    return np.sqrt((xx - 256) ** 2 + (yy - 256) ** 2) <= 256


def soft_tissue(split: str) -> list:
    return [g for g in sorted(p for p in (CONFIG.iqa2026_root / split).iterdir() if p.is_dir())
            if (np.asarray(Image.open(g / "gt.png")) == 0).mean() > 0.2]


def targets(groups, split: str, m: np.ndarray) -> list[dict]:
    rows = []
    for g in groups:
        gt = np.asarray(Image.open(g / "gt.png")) * m
        for method in METHODS[1:]:
            img = np.asarray(Image.open(g / f"{method}.png")) * m
            rows.append({"image_id": f"{g.name}/{method}", "split": split,
                         "psnr": peak_signal_noise_ratio(gt, img, data_range=255),
                         "ssim": structural_similarity(gt, img, data_range=255) * 100})
    return rows


def main() -> None:
    rng = np.random.default_rng(0)
    train = soft_tissue("train")
    train = [train[i] for i in sorted(rng.choice(len(train), N_SLICES, replace=False))]
    rows, images = [], []
    for n, g in enumerate(train):
        for method in METHODS:
            v = to_storage(np.asarray(Image.open(g / f"{method}.png")))
            rows.append({"image_id": f"{g.name}/{method}", "patient": g.name.split("_")[0], "kind": method, **measure(v, TOOLS)})
            images.append(np.round(v * 255).astype(np.uint8))
        if n % 50 == 0:
            print("train", n, len(train), flush=True)
    table = pd.DataFrame(rows).set_index("image_id")
    bank = pd.read_csv(CONFIG.output_dir / "screening/train.csv", dtype={"image_id": str}).set_index("image_id")
    p = percentiles(table[[*NOISE, *CUES]], bank)
    p.insert(3, "noise_score", p[list(NOISE)].mean(axis=1))
    np.savez_compressed(JOB / "nr.npz", ids=np.array(table.index), patient=table["patient"].to_numpy(),
                        kind=table["kind"].to_numpy(), images=np.stack(images), tools=p.to_numpy(np.float32))
    m = mask()
    tgt = pd.DataFrame(targets(train, "train", m) + targets(soft_tissue("test"), "test", m))
    tgt.to_csv(JOB / "nr_targets.csv", index=False)
    print(f"{len(table)} train images from {table['patient'].nunique()} patients; targets {tgt.groupby('split').size().to_dict()}")


if __name__ == "__main__":
    main()
