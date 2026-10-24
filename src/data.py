import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from sklearn.model_selection import train_test_split

IMAGE_EXTS = (".tif", ".tiff", ".png")
MOS_BIN_EDGES = (0.5, 1.5, 2.5, 3.5)


@dataclass(frozen=True)
class IqaSample:
    image_id: str
    image: np.ndarray
    mos: float | None


def read_image(path: Path) -> np.ndarray:
    return iio.imread(path).astype(np.float32)


def index_images(image_dir: Path) -> dict[str, Path]:
    """Map lower-case file name and stem to path, so labels match regardless of case or extension."""
    index: dict[str, Path] = {}
    for path in image_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS:
            index[path.name.lower()] = path
            index[path.stem.lower()] = path
    return index


def load_labels(label_json: Path) -> dict[str, float]:
    raw = json.loads(label_json.read_text())
    return {str(name): float(mos) for name, mos in raw.items() if mos is not None}


def load_ldctiqa_split(image_dir: Path, label_json: Path | None, load_images: bool = True) -> list[IqaSample]:
    """One sample per labelled image (or per image if label_json is None), sorted by image ID."""
    index = index_images(image_dir)
    if label_json is None:
        paths = sorted({p for p in index.values()})
        return [IqaSample(p.stem, read_image(p) if load_images else np.empty(0), None) for p in paths]
    samples, missing = [], []
    for name, mos in load_labels(label_json).items():
        key = Path(name).name.lower()
        path = index.get(key) or index.get(Path(key).stem)
        if path is None:
            missing.append(name)
            continue
        samples.append(IqaSample(path.stem, read_image(path) if load_images else np.empty(0), mos))
    if missing:
        raise FileNotFoundError(f"{len(missing)} labelled images not found, e.g. {missing[:5]}")
    return sorted(samples, key=lambda s: s.image_id)


def mos_bins(mos: np.ndarray) -> np.ndarray:
    return np.digitize(mos, MOS_BIN_EDGES)


def dev_and_calibration_ids(samples: list[IqaSample], dev_size: int, calibration_size: int, seed: int) -> tuple[list[str], list[str]]:
    """Dev set (stratified by MOS) and a calibration set that contains it plus
    calibration_size - dev_size more stratified images from the rest."""
    ids = np.array([s.image_id for s in samples])
    mos = np.array([s.mos for s in samples], dtype=float)
    dev_idx, rest_idx = train_test_split(np.arange(len(ids)), train_size=dev_size,
                                         stratify=mos_bins(mos), random_state=seed)
    extra_idx, _ = train_test_split(rest_idx, train_size=calibration_size - dev_size,
                                    stratify=mos_bins(mos[rest_idx]), random_state=seed)
    dev = sorted(ids[dev_idx].tolist())
    calibration = sorted(dev + ids[extra_idx].tolist())
    return dev, calibration


def write_id_list(path: Path, ids: list[str]) -> None:
    """Split lists are written once and never regenerated."""
    if path.exists():
        existing = path.read_text().split()
        if existing != ids:
            raise RuntimeError(f"{path} exists with different content; split lists must not be regenerated")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")


def read_id_list(path: Path) -> list[str]:
    return path.read_text().split()


def patient_id_from_path(path: Path) -> str:
    return path.stem.split("_")[0]


def assert_no_patient_leakage(splits: dict[str, list[Path]]) -> None:
    patients = {name: {patient_id_from_path(p) for p in files} for name, files in splits.items()}
    names = list(patients)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            shared = patients[first] & patients[second]
            assert not shared, f"patients in both {first} and {second}: {sorted(shared)[:5]}"
