"""Step 1.4 with the expert folds available so far: Table 1 and Table 2 rows for the noise scalar, the MedSigLIP-FiLM
expert and the stacked regression; the expert profile card; coverage of dev/calibration by out-of-fold scores.
Selection and fitting use training images only (out-of-fold expert scores); test and stress sets are only scored.
Rerun when more folds arrive; everything is recomputed from outputs/agentic_ct_iqa/experts/preds_fold*.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import kendalltau

from scripts.screen_tools import TOOLS, cached
from src.config import CONFIG
from src.data import load_ldctiqa_split, read_id_list
from src.experts import (assert_out_of_fold, baseline_rows, expert_scores, fit_stacker, load_fold_predictions,
                         profile_card, strip_ext)
from src.metrics import challenge_scores, fit_logistic
from src.stress_sets import denoiser_group, dose_ladder

EXPERT_DIR = CONFIG.expert_dir
CLASSIC_CSV = CONFIG.output_dir / "experts/classic_nriqa.csv"
TABLES = CONFIG.output_dir / "tables"
STACK_FEATURES = ["high_pass_std", "air_hp_std", "body_coherence", "immerkaer_sigma", "nlm_residual_std",
                  "laplacian_variance", "nps_high_band_ratio", "edge_width_px"]


def available_folds() -> tuple[int, ...]:
    return tuple(k for k in range(1, 6) if (EXPERT_DIR / f"preds_fold{k}.csv").exists())


def stress_measurements() -> dict[str, pd.DataFrame]:
    """Tool measurements on the test images and the reported stress sets (cached; same tools as the screening)."""
    out = CONFIG.output_dir / "screening"
    test = {s.image_id: s for s in load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels)}
    return {
        "test": cached(out / "test.csv", lambda: [({"image_id": i}, s.image) for i, s in test.items()]),
        "s1_test": cached(out / "s1_test.csv", lambda: [
            ({"image_id": i, "sigma": sg}, test[i].image if sg == 0 else ndimage.gaussian_filter(test[i].image, sg))
            for i in read_id_list(CONFIG.splits_dir / "s1_test.txt") for sg in (0.0, *CONFIG.blur_sigmas)]),
        "s2_test": cached(out / "s2_test.csv", lambda: [
            ({"group_id": g, "method": m}, img / 255.0) for g in read_id_list(CONFIG.splits_dir / "s2_test.txt")
            for m, img in denoiser_group(CONFIG.iqa2026_root / "test" / g).items()]),
        "s3_test": cached(out / "s3_test.csv", lambda: [
            ({"slice_id": s, "dose_rank": r}, img) for s in read_id_list(CONFIG.splits_dir / "s3_test.txt")
            for r, img in dose_ladder(CONFIG.i2i_root / "test" / f"{s}.npz")]),
    }


def classic_scores() -> dict[tuple[str, str], pd.Series]:
    """NIQE / BRISQUE from scripts/classic_nriqa.py as quality scores (negated: lower NIQE/BRISQUE is better)."""
    frame = pd.read_csv(CLASSIC_CSV, dtype={"key": str})
    return {(s, m): -g.set_index("key")["value"] for (s, m), g in frame.groupby(["set", "metric"])}


def robustness(s1: pd.Series, s2: pd.Series, s3: pd.Series) -> dict[str, float]:
    """A1, A2 (ties count as wrong), R_up, tau over 5 doses and over 6 levels. Inputs are indexed by the stress keys."""
    s1 = s1.copy(); s1.index = pd.MultiIndex.from_tuples([(k.split("|")[0], float(k.split("|")[1])) for k in s1.index])
    s1 = s1.unstack()
    s2 = s2.copy(); s2.index = pd.MultiIndex.from_tuples([tuple(k.split("|")) for k in s2.index]); s2 = s2.unstack()
    s3 = s3.copy(); s3.index = pd.MultiIndex.from_tuples([(k.split("|")[0], int(k.split("|")[1])) for k in s3.index])
    s3 = s3.unstack()
    outputs = s2.drop(columns=["gt", "noisy"])
    tau = lambda frame: float(np.nanmean([kendalltau(frame.columns, row)[0] for _, row in frame.iterrows()]))
    return {"A1": float((s1[0.0] > s1[1.0]).mean()), "A2": float((s1[0.0] > s1[2.0]).mean()),
            "R_up": float((outputs.to_numpy() > s2[["gt"]].to_numpy()).mean()),
            "tau5": tau(s3[[0, 1, 2, 3, 4]]), "tau6": tau(s3)}


def keyed(frame: pd.DataFrame, set_name: str, column: str) -> pd.Series:
    """Tool measurement frame -> Series indexed like the expert prediction keys of that set."""
    if set_name == "s1_test":
        keys = frame["image_id"] + "|" + frame["sigma"].astype(float).astype(str)
    elif set_name == "s2_test":
        keys = frame["group_id"] + "|" + frame["method"]
    else:
        keys = frame["slice_id"] + "|" + frame["dose_rank"].astype(int).astype(str)
    return pd.Series(frame[column].to_numpy(), index=keys.to_numpy())


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    folds = available_folds()
    preds = load_fold_predictions(EXPERT_DIR, folds)
    oof_count = preds[preds["set"] == "oof"]["key"].nunique()
    assert_out_of_fold(preds, n_train=oof_count, n_folds=len(folds))
    scores = expert_scores(preds)
    print(f"Expert folds available: {folds}; out-of-fold training images: {oof_count}")

    train_meas = pd.read_csv(CONFIG.output_dir / "screening/train.csv", dtype={"image_id": str}).set_index("image_id")
    oof = scores["train"].copy(); oof.index = strip_ext(oof.index)
    train = train_meas.loc[oof.index].assign(expert=oof)
    dev, cal = read_id_list(CONFIG.splits_dir / "dev.txt"), read_id_list(CONFIG.splits_dir / "calibration.txt")
    print(f"Coverage by out-of-fold expert scores: dev {sum(i in oof.index for i in dev)}/{len(dev)}, "
          f"calibration {sum(i in oof.index for i in cal)}/{len(cal)}")

    meas = stress_measurements()
    test_meas = meas["test"].set_index("image_id")
    test_expert = scores["test"].copy(); test_expert.index = strip_ext(test_expert.index)
    test = test_meas.loc[test_expert.index].assign(expert=test_expert)
    mos_test = {s.image_id: s.mos for s in load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)}
    test_mos = test.index.map(mos_test).to_numpy(dtype=float)

    classic = classic_scores()  # (set, metric) -> quality score (negated NIQE / BRISQUE), indexed by key
    stacker = fit_stacker(train[["expert", *STACK_FEATURES]], train["mos"].to_numpy())
    train_cols = pd.DataFrame({"high_pass_scalar": -train["high_pass_std"],
                               **{m: classic[("train", m)].reindex(train.index) for m in ("niqe", "brisque")},
                               "medsiglip_film": train["expert"],
                               "stacked_regression": stacker.predict(train[["expert", *STACK_FEATURES]].to_numpy())})
    test_cols = pd.DataFrame({"high_pass_scalar": -test["high_pass_std"],
                              **{m: classic[("test", m)].reindex(test.index) for m in ("niqe", "brisque")},
                              "medsiglip_film": test["expert"],
                              "stacked_regression": stacker.predict(test[["expert", *STACK_FEATURES]].to_numpy())})
    # The stacker's training predictions are in-sample; its logistic is fitted on them only for the PLCC mapping.
    table1 = baseline_rows(train_cols, train["mos"].to_numpy(), test_cols, test_mos)
    table1.round(4).to_csv(TABLES / "table1_baselines.csv")
    print(f"\nTable 1 (LDCT-IQA test, 300; expert = mean of folds {folds}):")
    print(table1[["overall_raw", "plcc", "srocc", "krocc", "overall"]].round(4).to_string())

    rows = {}
    for name, column in (("high_pass_scalar", "high_pass_std"),):
        rows[name] = robustness(*(-keyed(meas[s], s, column) for s in ("s1_test", "s2_test", "s3_test")))
    for metric in ("niqe", "brisque"):
        rows[metric] = robustness(*(classic[(s, metric)] for s in ("s1_test", "s2_test", "s3_test")))
    rows["medsiglip_film"] = robustness(scores["s1_test"], scores["s2_test"], scores["s3_test"])
    stacked = []
    for s in ("s1_test", "s2_test", "s3_test"):
        features = pd.DataFrame({c: keyed(meas[s], s, c) for c in STACK_FEATURES})
        features.insert(0, "expert", scores[s].reindex(features.index))
        stacked.append(pd.Series(stacker.predict(features.to_numpy()), index=features.index))
    rows["stacked_regression"] = robustness(*stacked)
    table2 = pd.DataFrame(rows).T
    table2.round(3).to_csv(TABLES / "table2_baselines.csv")
    print("\nTable 2 (stress test sets; A higher is better, R_up lower is better, tau higher is better):")
    print(table2.round(3).to_string())

    params = fit_logistic(train["expert"].to_numpy(), train["mos"].to_numpy())
    s1_dev = scores["s1_dev"]
    original = s1_dev[s1_dev.index.str.endswith("|0.0")]; original.index = original.index.str.split("|").str[0]
    blurred = s1_dev[s1_dev.index.str.endswith("|2.0")]; blurred.index = blurred.index.str.split("|").str[0]
    card = profile_card(train["expert"], train["mos"], train["high_pass_std"], original, blurred, params)
    (EXPERT_DIR / "profile_card.txt").write_text(card, encoding="utf-8")
    np.save(EXPERT_DIR / "expert_logistic_params.npy", params)
    print("\nProfile card:\n" + card)


if __name__ == "__main__":
    main()
