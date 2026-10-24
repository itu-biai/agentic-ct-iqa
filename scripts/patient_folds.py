"""Patient-level folds for E8 (issue #329): do the grouped folds leak patients?

The 1000 training slices are ~100 base slices x 10 degradations from 7 patients (5 Mayo + 2 NCC, Lee et al. 2025), but
the patient of a slice is not given. The 90 base-slice groups (outputs/agentic_ct_iqa/probe/train_groups.csv) are
clustered into 7 pseudo-patients by body geometry, which stays nearly constant along one patient's scan and differs
between patients: lateral width, AP height, lowest body row (table height) and horizontal centre of the body mask,
measured on each group's highest-MOS slice; Ward linkage on the standardised features. The 7 clusters are then put
into 5 folds greedily (largest cluster first, into the fold with the fewest slices), so no pseudo-patient spans two
folds. Fixed before any training; no test image is read.

    python -m scripts.patient_folds   # -> colab/vft_job/patient_folds.npy, outputs/agentic_ct_iqa/vision_ft/patient_folds.{csv,png}
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage

from src.config import CONFIG, PROJECT_ROOT

JOB = PROJECT_ROOT / "colab/vft_job"
OUT = CONFIG.output_dir / "vision_ft"
PATIENTS = 7


def geometry(image: np.ndarray) -> list[float]:
    body = ndimage.binary_fill_holes(ndimage.gaussian_filter(image / 255.0, 4.0) > 0.2)
    labels, n = ndimage.label(body)
    body = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    rows, cols = np.nonzero(body)
    return [cols.max() - cols.min(), rows.max() - rows.min(), rows.max(), cols.mean()]


def main() -> None:
    z = np.load(JOB / "train.npz")
    groups = pd.read_csv(CONFIG.output_dir / "probe/train_groups.csv", dtype={"image_id": str}).set_index("image_id")
    group = groups.loc[z["ids"], "group"].to_numpy()
    reps = {g: np.nonzero(group == g)[0][np.argmax(z["mos"][group == g])] for g in np.unique(group)}
    feats = np.array([geometry(z["images"][i]) for i in reps.values()], float)
    std = (feats - feats.mean(0)) / feats.std(0)
    cluster = dict(zip(reps, fcluster(linkage(std, "ward"), PATIENTS, "maxclust")))
    patient = np.array([cluster[g] for g in group])
    sizes = pd.Series(patient).value_counts()
    fold_of, load = {}, np.zeros(5)
    for c in sizes.index:  # largest first
        f = int(np.argmin(load))
        fold_of[c], load[f] = f + 1, load[f] + sizes[c]
    fold = np.array([fold_of[c] for c in patient])
    np.save(JOB / "patient_folds.npy", fold)
    table = pd.DataFrame({"group": list(reps), "patient": [cluster[g] for g in reps],
                          **dict(zip(("width", "height", "bottom", "centre_x"), feats.T))})
    table.to_csv(OUT / "patient_folds.csv", index=False)
    print(table.groupby("patient")[["width", "height", "bottom", "centre_x"]].agg(["mean", "std"]).round(0))
    print("slices per pseudo-patient:", sizes.sort_index().to_dict())
    print("slices per fold:", pd.Series(fold).value_counts().sort_index().to_dict())
    print("old grouped folds per pseudo-patient:", {c: sorted(set(z["fold"][patient == c])) for c in sorted(sizes.index)})
    tiles = []  # montage: one row per pseudo-patient, its group representatives
    for c in sorted(sizes.index):
        row = [np.asarray(Image.fromarray(z["images"][reps[g]]).resize((96, 96))) for g in reps if cluster[g] == c][:16]
        tiles.append(np.hstack(row + [np.zeros((96, 96), np.uint8)] * (16 - len(row))))
    Image.fromarray(np.vstack(tiles)).save(OUT / "patient_folds.png")


if __name__ == "__main__":
    main()
