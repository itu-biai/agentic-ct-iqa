"""Pseudo-MOS for the unlabelled pool from two teachers, fixed before the student run (2026-10-20):
label_k = Stage B fold k llm (rerun weights, colab/vm/teacher.py) blended with the MedSigLIP 5-fold mean (Tolga's
checkpoints, colab/vm/medsig_pool.py) as 0.75 z(Stage B) + 0.25 z(MedSigLIP), z over the pool, mapped back to the
Stage B fold k scale (its pool mean and std). The weight 0.25 is the out-of-fold choice on the training set (Stage B x2
2.828 alone, 2.839 with 0.25 MedSigLIP). Fold k's student gets fold k's Stage B teacher, which never saw fold k's
validation slices; MedSigLIP's own folds are not grouped like ours, so its part is not fold-clean.

Checks first: the reloaded fold models must reproduce the rerun's saved test predictions, and MedSigLIP's 20 test
images must reproduce Tolga's preds.csv.

    python -m scripts.teacher_labels     # colab/vft_job/teacher.npz + medsig_pool/ -> colab/vft_job/pseudo_labels.npz
"""
import numpy as np
import pandas as pd

from src.config import CONFIG, PROJECT_ROOT

JOB = PROJECT_ROOT / "colab/vft_job"
RERUN = CONFIG.output_dir / "vision_ft/llm2/out_llm"
W_MED = 0.25


def zs(x):
    return (x - x.mean()) / x.std()


def main() -> None:
    pool, teacher = np.load(JOB / "pseudo.npz", allow_pickle=True), np.load(JOB / "teacher.npz", allow_pickle=True)
    ids = list(pool["ids"])
    assert list(teacher["ids"]) == ids
    for k in range(1, 6):
        saved = np.load(RERUN / f"fold_{k}.npz")["llm_test"]
        print(f"fold {k}: reloaded vs saved test llm max |diff| {np.abs(teacher[f'test_llm_{k}'] - saved).max():.4f}, "
              f"corr {np.corrcoef(teacher[f'test_llm_{k}'], saved)[0, 1]:.5f}")
    tolga = pd.read_csv(CONFIG.output_dir / "experts_tolga/official_test_py/preds.csv")
    tolga.index = tolga["path"].str.split("/").str[-1]
    med = []
    for k in range(1, 6):
        f = pd.read_csv(JOB / f"medsig_pool/pool_fold{k}.csv")  # path, pred_mos
        name = f["path"].str.split("/").str[-1]
        check = f[~name.str.endswith(".png")]
        diff = (check["pred_mos"].to_numpy() - tolga.loc[name[check.index], f"pred_fold{k}"].to_numpy())
        print(f"MedSigLIP fold {k}: {len(check)} test images vs Tolga's preds.csv max |diff| {np.abs(diff).max():.4f}")
        pool_rows = f[name.str.endswith(".png")]
        keys = name[pool_rows.index].str.replace(".png", "", regex=False).str.replace("__", "/")
        med.append(pd.Series(pool_rows["pred_mos"].to_numpy(), index=keys.to_numpy()))
    med = pd.concat(med, axis=1)
    pool_med = med.loc[ids].mean(axis=1).to_numpy()
    out = {"ids": np.array(ids)}
    for k in range(1, 6):
        b = teacher[f"llm_{k}"]
        out[f"label_{k}"] = b.mean() + b.std() * ((1 - W_MED) * zs(b) + W_MED * zs(pool_med))
        print(f"label_{k}: corr(Stage B, MedSigLIP) on pool {np.corrcoef(b, pool_med)[0, 1]:.3f}; "
              + ", ".join(f"{kind} {out[f'label_{k}'][pool['kind'] == kind].mean():.2f}" for kind in ("gt", "a33", "a67", "noisy")))
    np.savez(JOB / "pseudo_labels.npz", **out)
    print("->", JOB / "pseudo_labels.npz")


if __name__ == "__main__":
    main()
