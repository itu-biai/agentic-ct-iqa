"""Step 1.4 (first rows): high-pass noise scalar on LDCT-IQA test, raw and calibrated.
Also saves per-image tool measurements for train and test (reused by the stacker and profile cards)."""
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data import load_ldctiqa_split
from src.metrics import challenge_scores, fit_logistic
from src.tools import edge_width, flat_roi_sigma, high_pass_energy, radial_nps_summary, streak_anisotropy

BODY_MIN = 0.2  # LDCT-IQA images are windowed to [0, 1]; air is 0
SATURATION = 0.98  # bone and contrast clip at 1


def measure(image: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {}
    for result in (high_pass_energy(image), flat_roi_sigma(image, BODY_MIN, SATURATION),
                   radial_nps_summary(image, BODY_MIN, saturation=SATURATION),
                    edge_width(image), streak_anisotropy(image, BODY_MIN)):
        values.update(result.values)
    return values


def measurements(split: str) -> pd.DataFrame:
    path = CONFIG.output_dir / f"measurements_{split}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"image_id": str}).set_index("image_id")
    images, labels = (CONFIG.train_images, CONFIG.train_labels) if split == "train" else (CONFIG.test_images, CONFIG.test_labels)
    rows = [{"image_id": s.image_id, "mos": s.mos, **measure(s.image)} for s in load_ldctiqa_split(images, labels)]
    frame = pd.DataFrame(rows).set_index("image_id")
    frame.to_csv(path)
    return frame


def main() -> None:
    train, test = measurements("train"), measurements("test")
    # Higher noise means lower quality, so the scalar is negated before the (monotone increasing) logistic.
    params = fit_logistic(-train["high_pass_std"].to_numpy(), train["mos"].to_numpy())
    raw = challenge_scores(test["high_pass_std"].to_numpy(), test["mos"].to_numpy())
    calibrated = challenge_scores(-test["high_pass_std"].to_numpy(), test["mos"].to_numpy(), params)
    print("High-pass scalar on LDCT-IQA test (300)          PLCC    SROCC   KROCC   Overall")
    for name, row in (("raw (|corr|, no fit)", raw), ("calibrated (logistic fit on train)", calibrated)):
        print(f"  {name:40s} {row['plcc']:.4f}  {row['srocc']:.4f}  {row['krocc']:.4f}  {row['overall']:.4f}")
    print("\nSpearman correlation of each measurement with MOS (test):")
    print(test.drop(columns="mos").corrwith(test["mos"], method="spearman").round(3).to_string())


if __name__ == "__main__":
    main()
