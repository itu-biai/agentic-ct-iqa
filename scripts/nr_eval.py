"""No-reference PSNR / SSIM prediction on the biailab IQA 2026v1 set: from one image alone (no gt), predict its PSNR and
SSIM to the gt. Trained on 300 lab train slices x 12 non-gt images (scripts/nr_pool.py), tested on the lab test split
(192 slices, 30 patients, unseen by the denoisers and by Stage B). Features: the frozen Stage B fold 1 model's
last-token hidden state (colab/vm/nr_feat.py).

Fixed before the run (2026-10-21):
- PRIMARY: SSIM, test overall (|PLCC| + |SROCC| + |KROCC|) of a ridge on the Stage B features (standardised; alpha from
  {1, 10, 100, 1000, 10^4, 10^5} by 5-fold CV grouped by patient on the train images), against a ridge on the 6 tool
  inputs (same alpha rule); 95% CI by a paired bootstrap over test patients.
- Secondary: the same for PSNR; per-slice Kendall's tau of the prediction with the target across the 12 images; zero-
  shot references that see no lab labels: Stage B's own MOS prediction, MedSigLIP's (scripts/cross_eval.py run), and
  the training-free noise estimator.

    python -m scripts.nr_eval      # -> outputs/agentic_ct_iqa/vision_ft/nr_eval.md
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.cross_eval import medsig, noise_estimate
from src.config import CONFIG, PROJECT_ROOT
from src.metrics import challenge_scores

JOB = PROJECT_ROOT / "colab/vft_job"
OUT = CONFIG.output_dir / "vision_ft"
ALPHAS = (1, 10, 100, 1000, 1e4, 1e5)


def fit_ridge(x_tr, y_tr, groups, x_te):
    def cv(alpha):
        pred = np.zeros(len(y_tr))
        for a, b in GroupKFold(5).split(x_tr, y_tr, groups):
            pred[b] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x_tr[a], y_tr[a]).predict(x_tr[b])
        return challenge_scores(pred, y_tr)["overall"]
    best = max(ALPHAS, key=cv)
    return make_pipeline(StandardScaler(), Ridge(alpha=best)).fit(x_tr, y_tr).predict(x_te), best


def slice_tau(pred: pd.Series, target: pd.Series) -> pd.Series:
    s = pred.index.str.split("/").str[0]
    return pd.Series({k: kendalltau(pred[s == k], target[s == k])[0] for k in np.unique(s)})


def boot(a: pd.Series, b: pd.Series, y: pd.Series, rng, n=2000):
    patients = a.index.str.split("_").str[0]
    groups = [np.nonzero(patients == p)[0] for p in np.unique(patients)]
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(challenge_scores(a.to_numpy()[idx], y.to_numpy()[idx])["overall"]
                     - challenge_scores(b.to_numpy()[idx], y.to_numpy()[idx])["overall"])
    return f"[{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}]"


def main() -> None:
    feats = np.load(JOB / "nr_feat.npz", allow_pickle=True)
    nr, cross = np.load(JOB / "nr.npz", allow_pickle=True), np.load(JOB / "cross.npz", allow_pickle=True)
    tgt = pd.read_csv(JOB / "nr_targets.csv").set_index("image_id")
    ids = pd.Index(feats["ids"])
    tools = pd.DataFrame(np.concatenate([nr["tools"], cross["tools"][cross["set"] == "methods"]]), index=ids)
    images = np.concatenate([nr["images"], cross["images"][cross["set"] == "methods"]])
    feat = pd.DataFrame(feats["feat"].astype(np.float32), index=ids)
    zero_shot = {"Stage B MOS prediction (zero-shot)": pd.Series(feats["llm"], index=ids),
                 "noise estimator (zero-shot)": pd.Series(noise_estimate(images), index=ids)}
    train = tgt.index[tgt["split"] == "train"].intersection(ids)
    test = tgt.index[tgt["split"] == "test"].intersection(ids)
    med = medsig("cross")
    rng = np.random.default_rng(0)
    lines = ["# No-reference PSNR / SSIM prediction on the lab IQA set", "",
             f"Train {len(train)} images ({train.str.split('_').str[0].nunique()} patients), test {len(test)} images "
             f"({test.str.split('_').str[0].nunique()} patients), 12 non-gt images per slice.", ""]
    groups = train.str.split("_").str[0]
    for target in ("ssim", "psnr"):
        y_tr, y_te = tgt.loc[train, target].to_numpy(), tgt.loc[test, target]
        preds, alphas = {}, {}
        for name, x in (("Stage B features + ridge (PRIMARY)" if target == "ssim" else "Stage B features + ridge", feat),
                        ("tool inputs + ridge", tools)):
            p, alphas[name] = fit_ridge(x.loc[train].to_numpy(), y_tr, groups, x.loc[test].to_numpy())
            preds[name] = pd.Series(p, index=test)
        preds.update({n: s.loc[test] for n, s in zero_shot.items()})
        preds["MedSigLIP MOS prediction (zero-shot)"] = med.loc[test]
        rows = {}
        for name, p in preds.items():
            s = challenge_scores(p.to_numpy(), y_te.to_numpy())
            rows[name] = {**s, "slice tau": slice_tau(p, y_te).mean(), "alpha": alphas.get(name, "")}
        table = pd.DataFrame(rows).T
        for c in ("plcc", "srocc", "krocc", "overall", "slice tau"):
            table[c] = table[c].astype(float).round(4)
        a, b = list(preds)[:2]
        lines += [f"## {target.upper()}", "", table.to_markdown(), "",
                  f"- {a} - {b}: {challenge_scores(preds[a].to_numpy(), y_te.to_numpy())['overall'] - challenge_scores(preds[b].to_numpy(), y_te.to_numpy())['overall']:+.4f}, "
                  f"95% CI {boot(preds[a], preds[b], y_te, rng)} (bootstrap over test patients)", ""]
    (OUT / "nr_eval.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
