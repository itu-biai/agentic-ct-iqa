"""Reference banks for the stress datasets, from unprocessed images of their TRAINING splits only (no MOS, no
test patients): IQA 2026v1 train gt + noisy, I2I 2025v1 train dose ladder (all 5 doses + gt).
Images are sampled one per patient first (round-robin) to spread patients. Output CSVs in outputs/agentic_ct_iqa/screening/.
"""
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.skill_tools import DatasetProfile, reference_measurements
from src.stress_sets import DOSE_KEYS, one_group_per_patient

IQA2026 = DatasetProfile("IQA-2026v1", body_min=0.2, saturation=0.98, windows={})
I2I = DatasetProfile("I2I-2025v1", body_min=0.2, saturation=0.98, windows={})
N_GROUPS = 300  # groups / slices per bank (each contributes two or six images)


def iqa2026_bank() -> pd.DataFrame:
    from PIL import Image
    groups = one_group_per_patient(sorted(p for p in (CONFIG.iqa2026_root / "train").iterdir() if p.is_dir()), N_GROUPS, CONFIG.seed)
    rows = []
    for group in groups:
        for name in ("gt", "noisy"):
            image = np.asarray(Image.open(group / f"{name}.png"), dtype=np.float32) / 255.0
            rows.append({"group_id": group.name, "image": name, **reference_measurements(image, IQA2026)})
    return pd.DataFrame(rows)


def i2i_bank() -> pd.DataFrame:
    files = one_group_per_patient(sorted((CONFIG.i2i_root / "train").glob("*.npz")), N_GROUPS // 3, CONFIG.seed)
    rows = []
    for path in files:
        with np.load(path) as sample:
            for key in DOSE_KEYS:
                rows.append({"slice_id": path.stem, "image": key, **reference_measurements(sample[key].astype(np.float32), I2I)})
    return pd.DataFrame(rows)


def main() -> None:
    out = CONFIG.output_dir / "screening"
    for name, build in (("reference_iqa2026_train.csv", iqa2026_bank), ("reference_i2i_train.csv", i2i_bank)):
        frame = build()
        frame.to_csv(out / name, index=False)
        print(f"{name}: {len(frame)} images")
        print(frame.drop(columns=[c for c in frame.columns if frame[c].dtype == object]).describe().loc[["min", "50%", "max"]].round(4).to_string())


if __name__ == "__main__":
    main()
