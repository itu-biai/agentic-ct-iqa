"""Screening of three new tool candidates for Stage B, on the 1000 training slices only (no test image is read).

Fixed before the run (2026-10-22). Stage B's out-of-fold predictions (llm head, vision_ft/llm2/out_llm, the 5 grouped
folds of train.npz) are the baseline; residual = MOS - prediction. Candidates, each a small family of scalars computed
on the original float slices (stored window 1400 / -300, so HU = 1400 v - 1000):
  lesion    NPW model observer: a 10 HU disk (radius 2, 4, 8 px) against up to 60 mean-subtracted 33 x 33 soft-tissue
            patches (smoothed 0-120 HU); d' = (t.s) / std(t.b), log; plus the small / large ratio. Signal inserted
            after reconstruction, so resolution loss is not modelled.
  anatomy   content of the body (smoothed slice): area, lung / fat / soft tissue / bone fractions, mean HU; also each
            times the prediction (region-dependent tolerance).
  streak    line detector for sparse-view streaks: median-filter residual in the body with bone and strong edges masked,
            Radon transform normalised by the ray's body path length and residual std (z ~ N(0, 1) for white noise);
            share of |z| > 4, max |z|, kurtosis of z, angular CV of per-angle energy, share of angular-harmonic energy
            above harmonic 8.
A candidate passes if both hold:
  R  max over its scalars of |Spearman(scalar, residual)| >= 0.10;
  G  OOF overall of ridge(prediction + its scalars) > OOF overall of ridge(prediction alone); same folds, each fold
     fitted on the other four (RidgeCV, alphas 1e-2..1e4). 95% CI of the gain by a paired bootstrap over slices (2000,
     not grouped by base slice); a pass whose CI includes 0 is reported as weak.
Nothing else is tuned on these numbers. The streak scalars are also checked on 20 training slices re-reconstructed with
64 and 128 views (src/simulate.py, the LDCT-IQA recipe): a sanity check that they respond to streaks, not a criterion.

    python -m scripts.new_tool_screen    # -> outputs/agentic_ct_iqa/tool_screen/{features.csv, report.md}
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.stats import kurtosis, spearmanr
from skimage.transform import radon
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from scripts.qwen_vision_ft import JOB
from src.config import CONFIG
from src.data import index_images, read_image
from src.metrics import challenge_scores
from src.simulate import sparse_view

warnings.filterwarnings("ignore", message="Radon transform")
OUT = CONFIG.output_dir / "tool_screen"
STAGE_B = CONFIG.output_dir / "vision_ft/llm2/out_llm"
FAMILIES = ("lesion", "anatomy", "streak")


def hu(v: np.ndarray) -> np.ndarray:
    return 1400.0 * v - 1000.0


def body_mask(image: np.ndarray) -> np.ndarray:
    body = ndimage.binary_fill_holes(ndimage.gaussian_filter(image, 4.0) > 0.2)
    labels, n = ndimage.label(body)
    return labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1) if n else body


def lesion(image: np.ndarray, seed: int) -> dict[str, float]:
    smooth = hu(ndimage.gaussian_filter(image, 3.0))
    soft = (smooth > 0) & (smooth < 120) & body_mask(image)
    centres = np.argwhere(ndimage.binary_erosion(soft, np.ones((33, 33))))
    if len(centres) < 10:
        return {f"lesion_logd_r{r}": np.nan for r in (2, 4, 8)} | {"lesion_ratio_2_8": np.nan, "lesion_patches": len(centres)}
    centres = centres[np.random.default_rng(seed).choice(len(centres), min(60, len(centres)), replace=False)]
    patches = np.stack([image[r - 16:r + 17, c - 16:c + 17] for r, c in centres]) * 1400.0
    patches -= patches.mean(axis=(1, 2), keepdims=True)
    yy, xx = np.mgrid[-16:17, -16:17]
    out = {}
    for r in (2, 4, 8):
        signal = 10.0 * (yy ** 2 + xx ** 2 <= r ** 2)
        out[f"lesion_logd_r{r}"] = float(np.log((signal * signal).sum() / (patches * signal).sum(axis=(1, 2)).std()))
    out["lesion_ratio_2_8"] = out["lesion_logd_r2"] - out["lesion_logd_r8"]
    out["lesion_patches"] = float(len(centres))
    return out


def anatomy(image: np.ndarray) -> dict[str, float]:
    body, smooth = body_mask(image), hu(ndimage.gaussian_filter(image, 2.0))
    inside = smooth[body]
    return {"anat_area": float(body.mean()), "anat_lung": float((inside < -500).mean()),
            "anat_fat": float(((inside >= -150) & (inside < -30)).mean()),
            "anat_soft": float(((inside >= 0) & (inside < 120)).mean()), "anat_bone": float((inside >= 300).mean()),
            "anat_mean_hu": float(inside.mean())}


def streak(image: np.ndarray) -> dict[str, float]:
    small = image[::2, ::2]  # 256 x 256 keeps the radon transform cheap
    body = ndimage.binary_erosion(body_mask(small), iterations=5)
    smooth = ndimage.gaussian_filter(small, 1.5)
    edges = ndimage.gaussian_gradient_magnitude(small, 2.0)
    keep = body & ~ndimage.binary_dilation(smooth > 0.93, iterations=3) & (edges < np.percentile(edges[body], 95))
    residual = (small - ndimage.median_filter(small, 5)) * keep
    sigma = residual[keep].std()
    theta = np.arange(180.0)
    sino = radon(residual, theta=theta, circle=False)
    path = radon(keep.astype(float), theta=theta, circle=False)
    valid = path > 20
    z = sino[valid] / (np.sqrt(path[valid]) * sigma)
    energy = np.array([(sino[:, a][valid[:, a]] ** 2).sum() / max(path[:, a][valid[:, a]].sum(), 1) for a in range(180)])
    harmonics = np.abs(np.fft.rfft(energy - energy.mean())) ** 2
    return {"streak_share_z4": float((np.abs(z) > 4).mean()), "streak_max_z": float(np.abs(z).max()),
            "streak_kurtosis": float(kurtosis(z)), "streak_angular_cv": float(energy.std() / energy.mean()),
            "streak_high_harmonics": float(harmonics[9:].sum() / harmonics[1:].sum())}


def measure(path, seed: int) -> dict[str, float]:
    image = read_image(path)
    return {**lesion(image, seed), **anatomy(image), **streak(image)}


def features(ids: np.ndarray) -> pd.DataFrame:
    cache = OUT / "features.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"image_id": str}).set_index("image_id").loc[ids]
    index = index_images(CONFIG.train_images)
    rows = Parallel(n_jobs=-1)(delayed(measure)(index[i], CONFIG.seed + n) for n, i in enumerate(ids))
    frame = pd.DataFrame(rows, index=pd.Index(ids, name="image_id"))
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache)
    return frame


def ridge_oof(x: np.ndarray, mos: np.ndarray, fold: np.ndarray) -> np.ndarray:
    out = np.empty(len(mos))
    for k in range(1, 6):
        tr, va = fold != k, fold == k
        scaler = StandardScaler().fit(x[tr])
        model = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(scaler.transform(x[tr]), mos[tr])
        out[va] = model.predict(scaler.transform(x[va]))
    return out


def overall(pred: np.ndarray, mos: np.ndarray) -> float:
    return challenge_scores(pred, mos)["overall"]


def gain_ci(a: np.ndarray, b: np.ndarray, mos: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(CONFIG.seed)
    diffs = [overall(a[i], mos[i]) - overall(b[i], mos[i]) for i in (rng.integers(0, len(mos), len(mos)) for _ in range(n))]
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def sanity(ids: np.ndarray) -> pd.DataFrame:
    """Streak scalars on 20 training slices as stored and re-reconstructed from 128 and 64 views."""
    index, rows = index_images(CONFIG.train_images), []
    for i in np.random.default_rng(CONFIG.seed).choice(ids, 20, replace=False):
        image = read_image(index[i])
        for views in (None, 128, 64):
            degraded = image if views is None else np.clip(sparse_view(image, views), 0, 1)
            rows.append({"views": views or "stored", **streak(degraded)})
    return pd.DataFrame(rows).groupby("views", sort=False).median()


def main() -> None:
    z = np.load(JOB / "train.npz")
    ids, mos, fold = z["ids"], z["mos"].astype(float), z["fold"]
    pred = np.full(len(ids), np.nan)
    for k in range(1, 6):
        f = np.load(STAGE_B / f"fold_{k}.npz")
        pred[f["va"]] = f["llm_val"]
    feats = features(ids)
    feats = feats.fillna(feats.median())
    residual = mos - pred
    for c in [c for c in feats if c.startswith("anat_")]:
        feats[f"{c}_x_pred"] = feats[c].to_numpy() * pred
    base = ridge_oof(pred[:, None], mos, fold)
    lines = ["# New tool candidates for Stage B: screening on the 1000 training slices (out-of-fold)", "",
             f"Stage B OOF (llm head) {overall(pred, mos):.3f}; ridge on the prediction alone {overall(base, mos):.3f} "
             "(the G baseline). Criteria fixed before the run: see scripts/new_tool_screen.py.", ""]
    table, verdicts = [], {}
    for family in (*FAMILIES, "all three"):
        cols = [c for c in feats if (family == "all three" or c.startswith(family[:4])) and c != "lesion_patches"]
        rhos = {c: spearmanr(feats[c], residual)[0] for c in cols}
        for c in cols:
            table.append({"family": family, "scalar": c, "rho_MOS": round(spearmanr(feats[c], mos)[0], 3),
                          "rho_residual": round(rhos[c], 3)} if family != "all three" else None)
        stacked = ridge_oof(np.column_stack([pred, feats[cols].to_numpy()]), mos, fold)
        gain, (lo, hi) = overall(stacked, mos) - overall(base, mos), gain_ci(stacked, base, mos)
        r_ok, g_ok = max(abs(v) for v in rhos.values()) >= 0.10, gain > 0
        verdict = ("PASS" + (" (weak: CI includes 0)" if lo <= 0 else "")) if r_ok and g_ok else "fail"
        verdicts[family] = {"max |rho_residual|": round(max(abs(v) for v in rhos.values()), 3), "R": r_ok,
                            "OOF ridge": round(overall(stacked, mos), 3), "gain": f"{gain:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                            "G": g_ok, "verdict": verdict if family != "all three" else verdict + " (info only)"}
    lines += ["## Verdicts", "", pd.DataFrame(verdicts).T.to_markdown(), "",
              "## Scalars", "", pd.DataFrame([t for t in table if t]).to_markdown(index=False), "",
              "## Sanity: streak scalars on 20 slices re-reconstructed with fewer views (median)", "",
              sanity(ids).round(4).to_markdown(), ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
