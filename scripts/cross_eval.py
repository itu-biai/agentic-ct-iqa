"""Cross-dataset test of Stage B (rerun weights) against MedSigLIP (Tolga's 5 checkpoints), both trained on LDCT-IQA only,
on lab images neither has seen (scripts/cross_pool.py; predictions from colab/vm/teacher.py and colab/vm/medsig_pool.py
with the cross flag, plus the teacher-pool predictions for gt and 100k). No MOS exists for these images.

Fixed before the run (2026-10-21); every score is the 5-fold mean prediction, CIs by a paired bootstrap over patients:
- PRIMARY dose: I2I 2025v1, 908 slices at 6 doses (gt, 1000k, 500k, 100k, 50k, 10k photons); per slice Kendall's tau
  between dose and predicted quality, averaged; Stage B llm - MedSigLIP. Secondary: slices in exactly the right order,
  accuracy of each adjacent pair, Kendall's tau over all images, Stage B vision; a training-free noise estimator
  (std of the 3x3 median-filter residual inside the soft-tissue range) as a reference.
- PRIMARY methods: IQA 2026v1 test split (31 patients, unseen by the 11 denoisers), per slice Kendall's tau between the
  predicted quality of the 12 non-gt images (noisy + 11 methods) and their SSIM to gt (the dataset's protocol: 8-bit
  PNGs, circular mask, data range 255), averaged; Stage B llm - MedSigLIP. Secondary: the same with PSNR; the order of
  the 12 methods by mean predicted score against their order by mean SSIM; share of slices where gt scores highest.
SSIM and PSNR are full-reference proxies, not radiologist judgement.

    python -m scripts.cross_eval      # -> outputs/agentic_ct_iqa/vision_ft/cross_eval.md
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import median_filter
from scipy.stats import kendalltau
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.config import CONFIG, PROJECT_ROOT

JOB = PROJECT_ROOT / "colab/vft_job"
OUT = CONFIG.output_dir / "vision_ft"
DOSE = {"gt": 5, "1000k": 4, "500k": 3, "noisy": 2, "50k": 1, "10k": 0}  # pool "noisy" is 100k
NON_GT = ("noisy", "bm3d", "ctformer", "dncnn", "n2self", "nafnet", "nb2nb", "nr2n", "redcnn", "tv", "unet", "wgan_vgg")


def medsig(prefix: str) -> pd.Series:
    folds = []
    for k in range(1, 6):
        f = pd.read_csv(JOB / f"medsig_{prefix}/{prefix}_fold{k}.csv")
        name = f["path"].str.split("/").str[-1]
        f = f[name.str.endswith(".png")]
        key = name[f.index].str.replace(".png", "", regex=False).str.replace("__", "/")
        folds.append(pd.Series(f["pred_mos"].to_numpy(), index=key.to_numpy()))
    return pd.concat(folds, axis=1).mean(axis=1)


def stage_b(path, ids, head: str) -> pd.Series:
    z = np.load(path, allow_pickle=True)
    return pd.Series(np.mean([z[f"{head}_{k}"] for k in range(1, 6)], axis=0), index=z["ids"]).loc[ids]


def noise_estimate(images: np.ndarray) -> np.ndarray:
    out = []
    for im in images:
        v = im.astype(np.float32) / 255
        r = v - median_filter(v, 3)
        mask = (v > 0.55) & (v < 0.85)  # roughly -230 to 190 HU in the storage window
        out.append(-r[mask].std())  # higher = less noise = better
    return np.array(out)


def boot(diff: pd.Series, patients: pd.Series, rng, n=2000):
    g = diff.groupby(patients.loc[diff.index]).agg(["sum", "count"])
    idx = rng.integers(0, len(g), (n, len(g)))
    m = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return f"[{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def per_slice_tau(wide: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    return pd.Series([kendalltau(t, p)[0] for p, t in zip(wide.to_numpy(), truth.to_numpy())], index=wide.index)


def dose_part(rng, lines):
    pool, cross = np.load(JOB / "pseudo.npz", allow_pickle=True), np.load(JOB / "cross.npz", allow_pickle=True)
    keep_pool = np.isin(pool["kind"], ["gt", "noisy"])
    keep_cross = cross["set"] == "dose"
    ids = np.concatenate([pool["ids"][keep_pool], cross["ids"][keep_cross]])
    images = np.concatenate([pool["images"][keep_pool], cross["images"][keep_cross]])
    preds = {
        "Stage B llm (PRIMARY)": pd.concat([stage_b(JOB / "teacher.npz", pool["ids"][keep_pool], "llm"),
                                            stage_b(JOB / "cross_pred.npz", cross["ids"][keep_cross], "llm")]),
        "Stage B vision": pd.concat([stage_b(JOB / "teacher.npz", pool["ids"][keep_pool], "vision"),
                                     stage_b(JOB / "cross_pred.npz", cross["ids"][keep_cross], "vision")]),
        "MedSigLIP": pd.concat([medsig("pool"), medsig("cross")]).loc[ids],
        "noise estimator (reference)": pd.Series(noise_estimate(images), index=ids)}
    slices = pd.Series([i.split("/")[0] for i in ids], index=ids)
    kind = pd.Series([i.split("/")[1] for i in ids], index=ids)
    patients = slices.groupby(slices).first().str.split("_").str[0]
    order = sorted(DOSE, key=DOSE.get, reverse=True)
    truth = pd.DataFrame({k: DOSE[k] for k in order}, index=sorted(set(slices)))
    rows, taus = {}, {}
    for name, p in preds.items():
        wide = pd.DataFrame({"s": slices, "k": kind, "p": p.loc[ids]}).pivot(index="s", columns="k", values="p")[order]
        taus[name] = per_slice_tau(wide, truth.loc[wide.index])
        row = {"tau": taus[name].mean(), "exact": (np.diff(wide.to_numpy(), axis=1) < 0).all(axis=1).mean()}
        row.update({f"{a}>{b}": (wide[a] > wide[b]).mean() for a, b in zip(order, order[1:])})
        row["tau all images"] = kendalltau(kind.map(DOSE).loc[ids], p.loc[ids])[0]
        rows[name] = row
    table = pd.DataFrame(rows).T.round(4).rename(columns={"noisy>50k": "100k>50k", "500k>noisy": "500k>100k"})
    lines += ["## Dose (I2I 2025v1): 908 slices x 6 doses, " f"{patients.nunique()} patients", "", table.to_markdown(), ""]
    for name in ("Stage B llm (PRIMARY)", "Stage B vision"):
        d = taus[name] - taus["MedSigLIP"]
        lines.append(f"- {name} - MedSigLIP, tau: {d.mean():+.4f}, 95% CI {boot(d, patients, rng)}")
    lines.append("")


def methods_part(rng, lines):
    cross = np.load(JOB / "cross.npz", allow_pickle=True)
    keep = cross["set"] == "methods"
    ids = cross["ids"][keep]
    slices = pd.Series([i.split("/")[0] for i in ids], index=ids)
    kind = pd.Series([i.split("/")[1] for i in ids], index=ids)
    ssim, psnr = {}, {}
    h = w = 512
    yy, xx = np.ogrid[:h, :w]
    mask = np.sqrt((xx - w // 2) ** 2 + (yy - h // 2) ** 2) <= min(w // 2, h // 2)
    for s in sorted(set(slices)):
        group = CONFIG.iqa2026_root / "test" / s
        gt = np.asarray(Image.open(group / "gt.png")) * mask
        for m in NON_GT:
            img = np.asarray(Image.open(group / f"{m}.png")) * mask
            ssim[f"{s}/{m}"] = structural_similarity(gt, img, data_range=255) * 100
            psnr[f"{s}/{m}"] = peak_signal_noise_ratio(gt, img, data_range=255)
    ssim, psnr = pd.Series(ssim), pd.Series(psnr)
    preds = {"Stage B llm (PRIMARY)": stage_b(JOB / "cross_pred.npz", ids, "llm"),
             "Stage B vision": stage_b(JOB / "cross_pred.npz", ids, "vision"),
             "MedSigLIP": medsig("cross").loc[ids],
             "noise estimator (reference)": pd.Series(noise_estimate(cross["images"][keep]), index=ids)}
    patients = slices.groupby(slices).first().str.split("_").str[0]
    non_gt = kind[kind != "gt"].index
    rows, taus = {}, {}
    for name, p in preds.items():
        frame = pd.DataFrame({"s": slices.loc[non_gt], "k": kind.loc[non_gt], "p": p.loc[non_gt],
                              "ssim": ssim.loc[non_gt], "psnr": psnr.loc[non_gt]})
        wide = frame.pivot(index="s", columns="k", values="p")[list(NON_GT)]
        taus[name] = per_slice_tau(wide, frame.pivot(index="s", columns="k", values="ssim")[list(NON_GT)])
        by_method = frame.groupby("k")[["p", "ssim"]].mean()
        top = pd.DataFrame({"s": slices, "k": kind, "p": p.loc[ids]}).pivot(index="s", columns="k", values="p")
        rows[name] = {"tau with SSIM": taus[name].mean(),
                      "tau with PSNR": per_slice_tau(wide, frame.pivot(index="s", columns="k", values="psnr")[list(NON_GT)]).mean(),
                      "method order vs SSIM": kendalltau(by_method["p"], by_method["ssim"])[0],
                      "gt highest": (top["gt"] > top[list(NON_GT)].max(axis=1)).mean()}
    table = pd.DataFrame(rows).T.round(4)
    lines += [f"## Denoising methods (IQA 2026v1 test): {slices.nunique()} slices x 13 images, {patients.nunique()} patients",
              "", table.to_markdown(), ""]
    for name in ("Stage B llm (PRIMARY)", "Stage B vision"):
        d = taus[name] - taus["MedSigLIP"]
        lines.append(f"- {name} - MedSigLIP, tau with SSIM: {d.mean():+.4f}, 95% CI {boot(d, patients, rng)}")
    means = pd.DataFrame({name: p.loc[ids].groupby(kind).mean() for name, p in preds.items() if "estimator" not in name})
    means["SSIM"] = ssim.groupby(ssim.index.str.split("/").str[1]).mean()
    lines += ["", "Mean predicted score per method (SSIM to gt for reference):", "",
              means.sort_values("SSIM", ascending=False).round(3).to_markdown(), ""]


def main() -> None:
    rng = np.random.default_rng(0)
    lines = ["# Cross-dataset: lab images unseen by both models (trained on LDCT-IQA only)", ""]
    dose_part(rng, lines)
    methods_part(rng, lines)
    (OUT / "cross_eval.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
