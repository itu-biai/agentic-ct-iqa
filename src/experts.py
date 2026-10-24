"""Step 1.4: expert predictions (Tolga's MedSigLIP-FiLM, retrained with his unchanged code, seed 42, 5 folds),
baselines and profile cards.

preds_fold{k}.csv (from colab/expert_predict.py) holds, per fold model k: oof (held-out training images), train, test,
and the stress sets. Training images get their out-of-fold score; every other image gets the mean over the 5 fold
models, as in the plan ("test scores are the mean over the fold models").
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import challenge_scores, fit_logistic, logistic4

EXPERT = "medsiglip_film"


def load_fold_predictions(folder: Path, folds: tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    frames = [pd.read_csv(folder / f"preds_fold{k}.csv", dtype={"key": str}) for k in folds]
    return pd.concat(frames, ignore_index=True)


def assert_out_of_fold(preds: pd.DataFrame, n_train: int = 1000, n_folds: int = 5) -> None:
    """Every training image must be held out by exactly one fold model; an expert scored on its own training images
    looks almost perfect and poisons the stacker, the profile cards and the anchors."""
    oof = preds[preds["set"] == "oof"]
    counts = oof.groupby("key")["fold"].nunique()
    assert len(counts) == n_train, f"{len(counts)} training images have an out-of-fold score, expected {n_train}"
    assert (counts == 1).all(), f"{(counts != 1).sum()} images are held out by more than one fold"
    assert oof["fold"].nunique() == n_folds


def expert_scores(preds: pd.DataFrame) -> dict[str, pd.Series]:
    """Per set: one score per key. 'train' is out-of-fold; all others are the fold-model mean."""
    scores = {"train": preds[preds["set"] == "oof"].set_index("key")["pred"].rename(EXPERT)}
    for name, group in preds[~preds["set"].isin(["oof", "train"])].groupby("set"):
        scores[name] = group.groupby("key")["pred"].mean().rename(EXPERT)
    return scores


def strip_ext(keys: pd.Index) -> pd.Index:
    return keys.str.replace(r"\.tiff?$", "", regex=True)


def fit_stacker(train_features: pd.DataFrame, train_mos: np.ndarray) -> Pipeline:
    """Plan: ridge on out-of-fold expert scores plus skill measurements, alpha by internal cross-validation.
    A measurement that cannot be computed on an image (e.g. no air around the body) is filled with the training median."""
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return model.fit(train_features.to_numpy(), train_mos)


def baseline_rows(train_scores: pd.DataFrame, train_mos: np.ndarray,
                  test_scores: pd.DataFrame, test_mos: np.ndarray) -> pd.DataFrame:
    """One Table 1 row per column: raw (Tolga's protocol) and with g_beta fitted on training scores (plan protocol)."""
    rows = []
    for method in train_scores.columns:
        params = fit_logistic(train_scores[method].to_numpy(), train_mos)
        raw = challenge_scores(test_scores[method].to_numpy(), test_mos)
        fitted = challenge_scores(test_scores[method].to_numpy(), test_mos, params)
        rows.append({"method": method, **{f"{k}_raw": v for k, v in raw.items()}, **fitted})
    return pd.DataFrame(rows).set_index("method")


def profile_card(train_pred: pd.Series, train_mos: pd.Series, noise: pd.Series,
                 blur_original: pd.Series, blur_blurred: pd.Series, params: np.ndarray) -> str:
    """Text the agent reads before trusting the expert: calibrated error by MOS bin and by noise tercile on
    out-of-fold training scores, and how the expert reacts to blur on dev images."""
    calibrated = pd.Series(logistic4(train_pred.to_numpy(), *params), index=train_pred.index)
    error = calibrated - train_mos
    bins = pd.cut(train_mos, [-0.01, 0.5, 1.5, 2.5, 3.5, 4.01], labels=["0", "1", "2", "3", "4"])
    terciles = pd.qcut(noise, 3, labels=["low", "medium", "high"])
    by_bin = ", ".join(f"MOS {b}: {e:+.2f}" for b, e in error.groupby(bins, observed=True).mean().items())
    by_noise = ", ".join(f"{t} noise: {e:+.2f}" for t, e in error.groupby(terciles, observed=True).mean().items())
    shift = logistic4(blur_blurred.to_numpy(), *params) - logistic4(blur_original.loc[blur_blurred.index].to_numpy(), *params)
    prefers = float((blur_original.loc[blur_blurred.index].to_numpy() > blur_blurred.to_numpy()).mean())
    verdict = ("it rewards smoothing, so its score is not reliable for blurred or denoised images"
               if prefers < 0.5 else "it does not reward smoothing on these slices")
    return (f"Expert {EXPERT} (MedSigLIP-FiLM, trained on LDCT-IQA). Mean calibrated error on held-out training slices "
            f"(positive = over-scores): {by_bin}; {by_noise}. "
            f"On Gaussian-blurred dev slices the blurred version scores {np.mean(shift):+.2f} MOS relative to the "
            f"original on average, and the expert prefers the original in {prefers:.0%} of cases: {verdict}.")
