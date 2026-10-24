import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr


def logistic4(s: np.ndarray, b1: float, b2: float, b3: float, b4: float) -> np.ndarray:
    return (b1 - b2) / (1.0 + np.exp(-(np.asarray(s) - b3) / np.abs(b4))) + b2


def fit_logistic(raw: np.ndarray, mos: np.ndarray) -> np.ndarray:
    """Fit g_beta on training (or calibration) scores only; apply it to test scores with logistic4."""
    raw, mos = np.asarray(raw, float), np.asarray(mos, float)
    start = [float(mos.max()), float(mos.min()), float(np.median(raw)), float(raw.std()) + 1e-6]
    params, _ = curve_fit(logistic4, raw, mos, p0=start, maxfev=20000)
    return params


def challenge_scores(prediction: np.ndarray, mos: np.ndarray, logistic_params: np.ndarray | None = None) -> dict[str, float]:
    """LDCT-IQA challenge metrics. Correlations are absolute values and KROCC is scipy's tau-b,
    as in medsiglip-ldct-iqa/test.py. With logistic_params, PLCC is computed after g_beta
    (SROCC and KROCC are unchanged by a monotone map)."""
    prediction, mos = np.asarray(prediction, float), np.asarray(mos, float)
    mapped = logistic4(prediction, *logistic_params) if logistic_params is not None else prediction
    plcc = abs(float(pearsonr(mapped, mos)[0]))
    srocc = abs(float(spearmanr(prediction, mos)[0]))
    krocc = abs(float(kendalltau(prediction, mos)[0]))
    return {"plcc": plcc, "srocc": srocc, "krocc": krocc, "overall": plcc + srocc + krocc}


def preference_accuracy(original: pd.Series, degraded: pd.Series) -> float:
    """A_sigma: share of images scored strictly above their degraded version (ties count as wrong)."""
    return float((original.loc[degraded.index].to_numpy() > degraded.to_numpy()).mean())


def ndct_above_rate(scores: pd.DataFrame) -> float:
    """R_up: share of denoiser outputs scored strictly above the normal-dose image of their group.
    `scores` has columns group_id, method, score; method 'gt' is the reference, 'noisy' is excluded."""
    reference = scores[scores["method"] == "gt"].set_index("group_id")["score"]
    outputs = scores[~scores["method"].isin(["gt", "noisy"])]
    return float((outputs["score"].to_numpy() > reference.loc[outputs["group_id"]].to_numpy()).mean())


def dose_ladder_tau(scores: pd.DataFrame) -> float:
    """tau_bar: mean per-slice Kendall tau between score and dose rank. Columns slice_id, dose_rank, score."""
    taus = [kendalltau(group["dose_rank"], group["score"])[0] for _, group in scores.groupby("slice_id")]
    return float(np.nanmean(taus))


def tie_rate(first: np.ndarray, second: np.ndarray) -> float:
    """Share of exact ties in a paired comparison; report it when it exceeds 5%."""
    return float(np.mean(np.asarray(first) == np.asarray(second)))
