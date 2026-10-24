"""Tool screening on training and dev data only (Step 1.3 check + pre-screen for Step 2.2 candidates).

For every scalar a tool returns:
  rho_mos      Spearman with MOS on the 1,000 training images
  partial_hp   Spearman partial correlation with MOS given the high-pass scalar (cubic control): information the
               tool adds beyond noise magnitude
  A1, A2       blur preference on dev images with MOS >= 3, treating the tool as a quality score in the direction
               of rho_mos (ties count as wrong)
  R_up         NDCT-above rate on the S2 dev groups (lower is better)
  tau5, tau6   mean per-slice Kendall tau with dose on the S3 dev slices, without and with the NDCT step
Writes outputs/agentic_ct_iqa/tool_screening.md and the per-image values as CSV.
"""
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import kendalltau, spearmanr

from src.config import CONFIG
from src.data import load_ldctiqa_split, read_id_list
from src.stress_sets import denoiser_group, dose_ladder
from src.tools import (air_artifact_energy, edge_width, flat_roi_sigma, fourier_high_band_fraction, high_pass_energy,
                       laplacian_variance, nlm_denoise, noise_residual_map, noise_sigma_estimate, radial_nps_summary,
                       streak_anisotropy, texture_coherence)
from src.tools_iqat1 import IQAT1_TOOLS

BODY_MIN, SATURATION = 0.2, 0.98  # all three datasets are display-windowed to [0, 1] here (PNGs divided by 255)

TOOLS: dict[str, Callable[[np.ndarray], dict[str, float]]] = {
    "high_pass_energy": lambda x: high_pass_energy(x).values,
    "flat_roi_sigma": lambda x: flat_roi_sigma(x, BODY_MIN, SATURATION).values,
    "radial_nps_summary": lambda x: radial_nps_summary(x, BODY_MIN, saturation=SATURATION).values,
    "edge_width": lambda x: edge_width(x).values,
    "streak_anisotropy": lambda x: streak_anisotropy(x, BODY_MIN).values,
    "fourier_high_band_fraction": lambda x: fourier_high_band_fraction(x).values,
    "laplacian_variance": lambda x: laplacian_variance(x).values,
    "air_artifact_energy": lambda x: air_artifact_energy(x, BODY_MIN).values,
    "texture_coherence": lambda x: texture_coherence(x, BODY_MIN).values,
    "noise_residual_map": lambda x: noise_residual_map(x, lambda y: ndimage.median_filter(y, 5)).values,
    **{f"iqat1:{name}": (lambda fn: lambda x: fn(x).values)(fn) for name, fn in IQAT1_TOOLS.items()},
    "noise_sigma_estimate": lambda x: noise_sigma_estimate(x, BODY_MIN).values,
    "nlm_residual": lambda x: {f"nlm_{k}": v for k, v in noise_residual_map(x, nlm_denoise).values.items()},
}
FIRST_RUN_TOOLS = [name for name in TOOLS if name not in ("noise_sigma_estimate", "nlm_residual")]


def measure(image: np.ndarray, names: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in names:
        try:
            values.update(TOOLS[name](image))
        except Exception:  # a tool that fails on an image reports NaN for it, the screen keeps going
            pass
    return values


def cached(path, items: Callable[[], list[tuple[dict, np.ndarray]]]) -> pd.DataFrame:
    """Per-image tool values, computed once per tool: a later run only adds the tools missing from the cache."""
    frame = pd.read_csv(path, dtype={"image_id": str, "group_id": str, "slice_id": str}) if path.exists() else None
    done = set(FIRST_RUN_TOOLS) if frame is not None else set()
    if frame is not None and "__tools" in frame.columns:
        done = set(str(frame["__tools"].iloc[0]).split("|"))
    missing = [name for name in TOOLS if name not in done]
    if missing:
        rows = [{**keys, **measure(image, missing)} for keys, image in items()]
        new = pd.DataFrame(rows)
        if frame is None:
            frame = new
        else:
            key_cols = [c for c in new.columns if c in ("image_id", "sigma", "group_id", "method", "slice_id", "dose_rank", "mos")]
            frame = frame.drop(columns="__tools", errors="ignore").merge(new, on=key_cols, how="left")
        frame["__tools"] = "|".join(sorted(done | set(missing)))
        frame.to_csv(path, index=False)
    return frame.drop(columns="__tools", errors="ignore")


def partial_spearman(a: pd.Series, b: pd.Series, control: pd.Series, degree: int = 3) -> float:
    rank = lambda s: s.rank().to_numpy() / len(s)
    ra, rb, rc = rank(a), rank(b), rank(control)
    resid = lambda r: r - np.polyval(np.polyfit(rc, r, degree), rc)
    return float(np.corrcoef(resid(ra), resid(rb))[0, 1])


def main() -> None:
    out = CONFIG.output_dir / "screening"
    out.mkdir(parents=True, exist_ok=True)
    train_samples = load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels)
    by_id = {s.image_id: s for s in train_samples}

    train = cached(out / "train.csv", lambda: [({"image_id": s.image_id, "mos": s.mos}, s.image) for s in train_samples])
    s1_ids = read_id_list(CONFIG.splits_dir / "s1_dev.txt")
    blur = cached(out / "s1_dev.csv", lambda: [
        ({"image_id": i, "sigma": sg}, by_id[i].image if sg == 0 else ndimage.gaussian_filter(by_id[i].image, sg))
        for i in s1_ids for sg in (0.0, *CONFIG.blur_sigmas)])
    s2 = cached(out / "s2_dev.csv", lambda: [
        ({"group_id": g, "method": m}, img / 255.0)
        for g in read_id_list(CONFIG.splits_dir / "s2_dev.txt")
        for m, img in denoiser_group(CONFIG.iqa2026_root / "val" / g).items()])
    s3 = cached(out / "s3_dev.csv", lambda: [
        ({"slice_id": s, "dose_rank": r}, img)
        for s in read_id_list(CONFIG.splits_dir / "s3_dev.txt")
        for r, img in dose_ladder(CONFIG.i2i_root / "val" / f"{s}.npz")])

    metrics = [c for c in train.columns if c not in ("image_id", "mos") and train[c].notna().mean() > 0.95]
    rows = []
    for metric in metrics:
        clean = train[list(dict.fromkeys(["mos", metric, "high_pass_std"]))].dropna()  # no duplicate column for hp itself
        rho = float(spearmanr(clean[metric], clean["mos"])[0])
        sign = 1.0 if rho >= 0 else -1.0  # orientation that makes the metric a quality score
        row = {"metric": metric, "rho_mos": rho,
               "partial_hp": float("nan") if metric == "high_pass_std" else
               partial_spearman(clean[metric], clean["mos"], clean["high_pass_std"])}
        original = blur[blur["sigma"] == 0].set_index("image_id")[metric] * sign
        for sigma in CONFIG.blur_sigmas:
            blurred = blur[blur["sigma"] == sigma].set_index("image_id")[metric] * sign
            row[f"A{int(sigma)}"] = float((original.loc[blurred.index] > blurred).mean())
        gt = s2[s2["method"] == "gt"].set_index("group_id")[metric] * sign
        outs = s2[~s2["method"].isin(["gt", "noisy"])]
        row["R_up"] = float((outs[metric].to_numpy() * sign > gt.loc[outs["group_id"]].to_numpy()).mean())
        for name, ladder in (("tau5", s3[s3["dose_rank"] < 5]), ("tau6", s3)):
            row[name] = float(np.nanmean([kendalltau(g["dose_rank"], g[metric] * sign)[0] for _, g in ladder.groupby("slice_id")]))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("rho_mos", key=np.abs, ascending=False)
    header = (f"# Tool screening (train n={len(train)}, S1 dev {len(s1_ids)} images x {len(CONFIG.blur_sigmas)} blurs, "
              f"S2 dev {s2['group_id'].nunique()} groups, S3 dev {s3['slice_id'].nunique()} slices)\n\n"
              "Higher A and tau are better; lower R_up is better. Direction of each metric is set by the sign of rho_mos.\n\n")
    (CONFIG.output_dir / "tool_screening.md").write_text(header + table.round(3).to_markdown(index=False), encoding="utf-8")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
