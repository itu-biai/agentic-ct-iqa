"""Step 1.4: Table 2 robustness columns for score-based baselines (currently: the high-pass scalar).
A_sigma on S1 test (blur), R_up on S2 test (denoisers), tau_bar on S3 test (dose ladder, when available).
Per-image scores are saved so the statistics step can pair them with the agent."""
import numpy as np
import pandas as pd
from scipy import ndimage

from src.config import CONFIG
from src.data import load_ldctiqa_split, read_id_list
from src.metrics import dose_ladder_tau, ndct_above_rate, preference_accuracy, tie_rate
from src.stress_sets import denoiser_group, dose_ladder
from src.tools import high_pass_energy


def scalar_score(image: np.ndarray) -> float:
    """Quality score of the noise scalar: less high-pass energy reads as better quality."""
    return -high_pass_energy(image).values["high_pass_std"]


def s1_scores(score) -> pd.DataFrame:
    keep = set(read_id_list(CONFIG.splits_dir / "s1_test.txt"))
    rows = []
    for s in load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels):
        if s.image_id in keep:
            rows.append({"image_id": s.image_id, "sigma": 0.0, "score": score(s.image)})
            rows += [{"image_id": s.image_id, "sigma": sg, "score": score(ndimage.gaussian_filter(s.image, sg))}
                     for sg in CONFIG.blur_sigmas]
    return pd.DataFrame(rows)


def s2_scores(score) -> pd.DataFrame:
    rows = []
    for group_id in read_id_list(CONFIG.splits_dir / "s2_test.txt"):
        for method, image in denoiser_group(CONFIG.iqa2026_root / "test" / group_id).items():
            rows.append({"group_id": group_id, "method": method, "score": score(image)})
    return pd.DataFrame(rows)


def s3_scores(score) -> pd.DataFrame | None:
    path = CONFIG.splits_dir / "s3_test.txt"
    if not path.exists():
        return None
    rows = []
    for slice_id in read_id_list(path):
        for rank, image in dose_ladder(CONFIG.i2i_root / "test" / f"{slice_id}.npz"):
            rows.append({"slice_id": slice_id, "dose_rank": rank, "score": score(image)})
    return pd.DataFrame(rows)


def robustness_row(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame | None) -> dict[str, float]:
    original = s1[s1["sigma"] == 0].set_index("image_id")["score"]
    row = {}
    for sigma in CONFIG.blur_sigmas:
        blurred = s1[s1["sigma"] == sigma].set_index("image_id")["score"]
        row[f"A{int(sigma)}"] = preference_accuracy(original, blurred)
        row[f"A{int(sigma)}_ties"] = tie_rate(original.loc[blurred.index], blurred)
    row["R_up"] = ndct_above_rate(s2)
    row["tau_bar"] = dose_ladder_tau(s3) if s3 is not None else float("nan")
    return row


def main() -> None:
    s1, s2, s3 = s1_scores(scalar_score), s2_scores(scalar_score), s3_scores(scalar_score)
    out = CONFIG.output_dir / "scores"
    out.mkdir(parents=True, exist_ok=True)
    s1.assign(method="high_pass_scalar").to_csv(out / "s1_high_pass_scalar.csv", index=False)
    s2.assign(scorer="high_pass_scalar").to_csv(out / "s2_high_pass_scalar.csv", index=False)
    if s3 is not None:
        s3.assign(method="high_pass_scalar").to_csv(out / "s3_high_pass_scalar.csv", index=False)
    row = robustness_row(s1, s2, s3)
    print("Table 2, high-pass scalar:  " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
    reference = s2[s2["method"] == "gt"].set_index("group_id")["score"]
    per_method = (s2[~s2["method"].isin(["gt", "noisy"])]
                  .assign(above=lambda f: f["score"].to_numpy() > reference.loc[f["group_id"]].to_numpy())
                  .groupby("method")["above"].mean().sort_values(ascending=False))
    print("\nNDCT-above rate per denoiser (scalar):")
    print(per_method.round(3).to_string())


if __name__ == "__main__":
    main()
