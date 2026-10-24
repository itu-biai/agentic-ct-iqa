"""Calibrate colab/vm/synth.py on the CPU against the 1000 training slices (no MOS, no test images).

    python -m scripts.synth_calibrate check      # noiseless 360-view FBP vs the source slice, orientation montage
    python -m scripts.synth_calibrate i0 3e5     # noise and streak measures of the 12 types vs the training slices

Source volumes: data/external/Task09_Spleen/imagesTr/*.nii.gz (Medical Segmentation Decathlon, CC-BY-SA 4.0).
"""
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch

from colab.vm.synth import TARGET_SPACING, TYPES, degrade, window01, zoom
from src.config import CONFIG, PROJECT_ROOT
from src.tools import air_artifact_energy, flat_roi_sigma, streak_anisotropy

SPLEEN = PROJECT_ROOT / "data/external/Task09_Spleen/imagesTr"
SCRATCH = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_ROOT / "outputs/agentic_ct_iqa/synth"
BODY_MIN, SATURATION = 0.2, 0.98


def load_slices(volumes: int = 4, per_volume: int = 2, seed: int = 0):
    """Axial slices in the dataset's orientation (anterior up, patient's right on the left), HU, with pixel spacing."""
    import nibabel as nib
    rng = np.random.default_rng(seed)
    paths = sorted(p for p in SPLEEN.glob("*.nii.gz") if not p.name.startswith("._"))
    out = []
    for path in rng.choice(paths, volumes, replace=False):
        img = nib.load(path)
        vol = np.asarray(img.dataobj, dtype=np.float32)
        spacing = float(img.header.get_zooms()[0])
        for z in rng.integers(int(0.2 * vol.shape[2]), int(0.8 * vol.shape[2]), per_volume):
            target = float(rng.uniform(*TARGET_SPACING))
            hu = zoom(torch.from_numpy(np.rot90(vol[:, :, z]).copy()), spacing, target).numpy()
            out.append((path.name, int(z), hu, target))
    return out


def measures(image01: np.ndarray) -> dict[str, float]:
    return {"flat_roi_sigma": flat_roi_sigma(image01, BODY_MIN, SATURATION).values["flat_roi_sigma"],
            "streak_anisotropy": streak_anisotropy(image01, BODY_MIN).values["streak_anisotropy"],
            "air_hp_std": air_artifact_energy(image01, BODY_MIN).values.get("air_hp_std", np.nan)}


def check() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    slices = load_slices(3, 1)
    tiles = []
    for name, z, hu, spacing in slices:
        rec = degrade(torch.from_numpy(hu), spacing, 0, 0, 1.0, None).numpy()
        body = hu > -500
        print(f"{name} z={z} spacing {spacing:.2f}: mean HU in body source {hu[body].mean():.1f}, "
              f"FBP {rec[body].mean():.1f}, rmse in body {np.sqrt(((rec - hu)[body] ** 2).mean()):.1f} HU")
        tiles.append(np.concatenate([window01(hu), window01(rec)], 1))
    train = iio.imread(sorted((CONFIG.train_images).glob("*.tif"))[0])
    iio.imwrite(SCRATCH / "check.png", (np.concatenate(tiles + [np.concatenate([train, train], 1)], 0)[::2, ::2] * 255).astype(np.uint8))
    print("montage ->", SCRATCH / "check.png")


def calibrate(i0: float) -> None:
    gen = torch.Generator().manual_seed(0)
    rows = []
    for name, z, hu, spacing in load_slices():
        for noise, streak in TYPES:
            rec = degrade(torch.from_numpy(hu), spacing, noise, streak, i0, gen).numpy()
            rows.append({"slice": f"{name}:{z}", "noise": noise, "streak": streak, **measures(window01(rec))})
    synth = pd.DataFrame(rows)
    train = pd.read_csv(CONFIG.output_dir / "screening/train.csv")
    cols = ["flat_roi_sigma", "streak_anisotropy", "air_hp_std"]
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    print("training slices, quantiles", q)
    print(train[cols].quantile(q).round(4).to_string())
    print(f"synthetic (I0 {i0:g}), quantiles")
    print(synth[cols].quantile(q).round(4).to_string())
    print("synthetic medians by type")
    print(synth.groupby(["noise", "streak"])[cols].median().round(4).to_string())


if __name__ == "__main__":
    {"check": check, "i0": lambda: calibrate(float(sys.argv[2]))}[sys.argv[1]]()
