"""Cross-dataset check without MOS: the lab IQA set (biailab IQA 2026v1 train+val, 137 patients, never seen by either
model; scripts/pseudo_pool.py) gives each slice at 4 noise levels, gt + a (noisy - gt) for a = 0, 1/3, 2/3, 1, so the
correct order within a slice is known. Both models were trained on LDCT-IQA only; their predictions on these 3632
images were made for the teacher labels (colab/vm/teacher.py, colab/vm/medsig_pool.py).

Fixed before computing (2026-10-21): PRIMARY = mean over slices of Kendall's tau between the noise level and the
predicted score (sign: less noise = higher quality = +1), Stage B rerun llm 5-fold mean against MedSigLIP 5-fold mean,
95% CI by a paired bootstrap over patients. Secondary: share of slices in exactly the right order; pairwise accuracy for
each adjacent pair of levels; Stage B vision; the absolute score drop from gt to noisy.

    python -m scripts.cross_dataset      # -> outputs/agentic_ct_iqa/vision_ft/cross_dataset.md
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.config import CONFIG, PROJECT_ROOT

JOB = PROJECT_ROOT / "colab/vft_job"
OUT = CONFIG.output_dir / "vision_ft"
LEVELS = ("gt", "a33", "a67", "noisy")


def per_slice(pred: pd.DataFrame) -> pd.DataFrame:
    """pred: slices x LEVELS. Kendall tau with quality order, exact order, adjacent-pair accuracy, gt - noisy drop."""
    quality = np.array([3, 2, 1, 0])
    out = pd.DataFrame(index=pred.index)
    out["tau"] = [kendalltau(quality, row)[0] for row in pred.to_numpy()]
    out["exact"] = (np.diff(pred.to_numpy(), axis=1) < 0).all(axis=1).astype(float)
    for a, b in zip(LEVELS, LEVELS[1:]):
        out[f"{a}>{b}"] = (pred[a] > pred[b]).astype(float)
    out["drop"] = pred["gt"] - pred["noisy"]
    return out


def boot(diff: pd.Series, patients: pd.Series, rng, n=2000):
    groups = diff.groupby(patients).agg(["sum", "count"])
    idx = rng.integers(0, len(groups), (n, len(groups)))
    means = groups["sum"].to_numpy()[idx].sum(1) / groups["count"].to_numpy()[idx].sum(1)
    return round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)


def main() -> None:
    pool, teacher = np.load(JOB / "pseudo.npz", allow_pickle=True), np.load(JOB / "teacher.npz", allow_pickle=True)
    ids = pd.Index(pool["ids"])
    assert list(teacher["ids"]) == list(ids)
    med = []
    for k in range(1, 6):
        f = pd.read_csv(JOB / f"medsig_pool/pool_fold{k}.csv")
        name = f["path"].str.split("/").str[-1]
        f = f[name.str.endswith(".png")]
        key = name[f.index].str.replace(".png", "", regex=False).str.replace("__", "/")
        med.append(pd.Series(f["pred_mos"].to_numpy(), index=key.to_numpy()))
    preds = {"Stage B llm (PRIMARY)": np.mean([teacher[f"llm_{k}"] for k in range(1, 6)], axis=0),
             "Stage B vision": np.mean([teacher[f"vision_{k}"] for k in range(1, 6)], axis=0),
             "MedSigLIP": pd.concat(med, axis=1).loc[ids].mean(axis=1).to_numpy()}
    slices = pd.Series([i.split("/")[0] for i in ids], index=ids)
    kinds = pd.Series(pool["kind"], index=ids)
    patients = pd.Series(pool["patient"], index=ids).groupby(slices).first()
    tables = {}
    for label, p in preds.items():
        wide = pd.DataFrame({"slice": slices.to_numpy(), "kind": kinds.to_numpy(), "p": p}).pivot(index="slice", columns="kind", values="p")[list(LEVELS)]
        tables[label] = per_slice(wide)
    summary = pd.DataFrame({label: t.mean() for label, t in tables.items()}).T.round(4)
    rng = np.random.default_rng(0)
    lines = ["# Cross-dataset: noise-level ordering on the lab IQA set (137 patients, never seen in training)", "",
             f"{len(tables['MedSigLIP'])} slices x 4 levels (gt, 1/3, 2/3, full noise), {patients.nunique()} patients. "
             "tau: Kendall's tau between noise level and predicted quality per slice, averaged; exact: slices in "
             "exactly the right order; a>b: pairwise accuracy; drop: predicted score gt minus noisy.", "",
             summary.to_markdown(), ""]
    ref = tables["MedSigLIP"]
    for label in ("Stage B llm (PRIMARY)", "Stage B vision"):
        for col in ("tau", "exact"):
            diff = tables[label][col] - ref[col]
            lines.append(f"- {label} - MedSigLIP, {col}: {diff.mean():+.4f}, 95% CI {boot(diff, patients.loc[diff.index], rng)} (bootstrap over patients)")
    (OUT / "cross_dataset.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
