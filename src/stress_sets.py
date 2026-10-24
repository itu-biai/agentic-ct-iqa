"""Label-free stress sets behind the robustness columns of Table 2 (S1 blur, S2 denoisers, S3 dose ladder)."""
from pathlib import Path

import numpy as np
from scipy import ndimage

from src.data import IqaSample

DENOISER_FILES: tuple[str, ...] = ("gt", "noisy", "bm3d", "tv", "dncnn", "redcnn", "unet", "nafnet",
                                   "ctformer", "wgan_vgg", "n2self", "nb2nb", "nr2n")
DOSE_KEYS: tuple[str, ...] = ("noisy_clipped_10k", "noisy_clipped_50k", "noisy_clipped_100k",
                              "noisy_clipped_500k", "noisy_clipped_1000k", "gt_clipped")


def blur_stress_set(samples: list[IqaSample], sigmas: tuple[float, ...], min_mos: float) -> list[tuple[str, float, np.ndarray]]:
    """S1: (image_id, sigma, blurred image) for every sample with MOS >= min_mos. Very noisy images are
    excluded on purpose, since mild smoothing can make them look better."""
    return [(s.image_id, sigma, ndimage.gaussian_filter(s.image, sigma))
            for s in samples if s.mos is not None and s.mos >= min_mos for sigma in sigmas]


def one_group_per_patient(group_paths: list[Path], count: int, seed: int) -> list[Path]:
    """Round-robin over patients: one group per patient before a second one from any patient."""
    rng = np.random.default_rng(seed)
    by_patient: dict[str, list[Path]] = {}
    for group in sorted(group_paths):
        by_patient.setdefault(group.name.split("_")[0], []).append(group)
    chosen: list[Path] = []
    while len(chosen) < count and any(by_patient.values()):
        for patient in sorted(by_patient):
            if by_patient[patient] and len(chosen) < count:
                chosen.append(by_patient[patient].pop(int(rng.integers(len(by_patient[patient])))))
    return sorted(chosen)


def denoiser_group(group_dir: Path) -> dict[str, np.ndarray]:
    """S2: the 13 images of one IQA 2026v1 group (8-bit DICOM-windowed PNG) as float32."""
    from PIL import Image
    return {name: np.asarray(Image.open(group_dir / f"{name}.png"), dtype=np.float32) for name in DENOISER_FILES}


def dose_ladder(npz_path: Path) -> list[tuple[int, np.ndarray]]:
    """S3: (dose rank, image) from 10k (rank 0) up to the NDCT reference (rank 5), values in [0, 1]."""
    with np.load(npz_path) as sample:
        return [(rank, sample[key].astype(np.float32)) for rank, key in enumerate(DOSE_KEYS)]
