"""Unlabelled images for teacher-student training: the biailab IQA 2026v1 train + val splits (its test split is the S2
stress set and stays out), every soft-tissue-window slice at four noise levels, converted to the LDCT-IQA storage
window, with the six tool inputs of Stage B. No MOS anywhere. The levels are gt (full dose), noisy (low dose) and
two in between, gt + a * (noisy - gt) for a = 1/3, 2/3: the real low-dose noise of the slice, scaled (gt alone is cleaner
and noisy alone noisier than almost every LDCT-IQA training image, so the two ends leave out the range that matters).

The PNGs come in two windows: soft tissue (about W400 / L40: fat at ~38, soft tissue at ~127, air clipped to 0; 1100
slices) and lung (W1500 / L-600: air at 60, soft tissue clipped near 236-255; 201 slices, dropped). Soft tissue:
HU = g / 255 * 400 - 160, clipped pixels (g = 0) set to air (-1000), then v = (HU + 1000) / 1400 like LDCT-IQA.
No LDCT-IQA slice appears in the set (largest 64x64 correlation of a test image with a lab slice 0.929; variants of one
slice correlate >= 0.97).

    python -m scripts.pseudo_pool      # -> colab/vft_job/pseudo.npz (ids, patient, kind, images uint8, tools)
"""
import numpy as np
import pandas as pd
from PIL import Image

from scripts.qwen_vision_ft import CUES, NOISE, percentiles
from scripts.screen_tools import measure
from src.config import CONFIG, PROJECT_ROOT

OUT = PROJECT_ROOT / "colab/vft_job/pseudo.npz"
TOOLS = ["streak_anisotropy", "noise_residual_map", "iqat1:gradient_magnitude_map", "iqat1:high_frequency_residual_map",
         "iqat1:local_gradient_deviation_map"]
LEVELS = {"gt": 0.0, "a33": 1 / 3, "a67": 2 / 3, "noisy": 1.0}


def to_storage(png: np.ndarray) -> np.ndarray:
    hu = png.astype(np.float32) / 255 * 400 - 160
    hu[png == 0] = -1000
    return np.clip((hu + 1000) / 1400, 0, 1)


def main() -> None:
    rows, images = [], []
    for split in ("train", "val"):
        for group in sorted(p for p in (CONFIG.iqa2026_root / split).iterdir() if p.is_dir()):
            gt = np.asarray(Image.open(group / "gt.png"))
            if (gt == 0).mean() <= 0.2:  # lung window
                continue
            gt_v, noisy_v = to_storage(gt), to_storage(np.asarray(Image.open(group / "noisy.png")))
            for kind, a in LEVELS.items():
                v = np.clip(gt_v + a * (noisy_v - gt_v), 0, 1)
                rows.append({"image_id": f"{group.name}/{kind}", "patient": group.name.split("_")[0], "kind": kind,
                             "split": split, **measure(v, TOOLS)})
                images.append(np.round(v * 255).astype(np.uint8))
            if len(rows) % 400 == 0:
                print(len(rows), flush=True)
    table = pd.DataFrame(rows).set_index("image_id")
    bank = pd.read_csv(CONFIG.output_dir / "screening/train.csv", dtype={"image_id": str}).set_index("image_id")
    p = percentiles(table[[*NOISE, *CUES]], bank)
    p.insert(3, "noise_score", p[list(NOISE)].mean(axis=1))
    np.savez_compressed(OUT, ids=np.array(table.index), patient=table["patient"].to_numpy(), kind=table["kind"].to_numpy(),
                        images=np.stack(images), tools=p.to_numpy(np.float32))
    print(f"{len(table)} images from {table['patient'].nunique()} patients -> {OUT}")
    print(p.describe().loc[["min", "50%", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
