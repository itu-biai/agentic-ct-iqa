"""Test-selected ("oracle") scores, for comparison with literature that picks its configuration on the 300-image test
(ClinReadNet, CAP-IQA). NOT an estimate of generalisation; the pre-registered numbers stay the primary results.

Search space, fixed before scoring (2026-10-21), the same kind for both methods:
- ours: every saved Stage B-family test prediction (5-fold mean), every single fold of those, and the mean of any 2 or
  3 of the 5-fold means; MedSigLIP: its 5 folds, the ensemble and the mean of any 2-4 folds;
- each passed as is, on the 0.1 grid, on the 1/6 grid (the test MOS is a mean of 6 ratings: known only from the
  test labels), or on the 0.2 grid with offset 0 / 0.05 / 0.1 / 0.15.
The best overall on the test is reported with the size of the search. Selection optimism is estimated by split halves:
pick the best configuration on a random half of the test, score it on the other half (100 splits).

    python -m scripts.oracle_search      # -> outputs/agentic_ct_iqa/vision_ft/oracle_test.md
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from src.config import CONFIG, PROJECT_ROOT

OUT = CONFIG.output_dir / "vision_ft"
JOB = PROJECT_ROOT / "colab/vft_job"
RUNS = {"original": ("llm/out_llm", ("vision", "llm")), "rerun": ("llm2/out_llm", ("vision", "llm")),
        "rank": ("rank/out_rank", ("vision", "vision_tta", "llm", "llm_tta")), "synth": ("syn/out_syn", ("vision", "llm")),
        "up2": ("up2/out_up2", ("vision", "llm")), "student": ("pseudo/out_pseudo", ("vision", "llm")), "cnn": ("cnn/out_cnn", ("cnn",))}
POSTS = {"as is": None, "grid 0.1": (0.1, 0), "grid 1/6": (1 / 6, 0), **{f"grid 0.2+{o}": (0.2, o) for o in (0, 0.05, 0.1, 0.15)}}
SPLITS = 100


def post(pred, how):
    if how is None:
        return pred
    step, off = how
    return np.clip(np.round((pred - off) / step) * step + off, 0, 4)


def scores(pred, mos):
    p, s, k = abs(pearsonr(pred, mos)[0]), abs(spearmanr(pred, mos)[0]), abs(kendalltau(pred, mos)[0])
    return p, s, k, p + s + k


def candidates_ours():
    means, out = {}, {}
    for run, (path, names) in RUNS.items():
        folds = [np.load(OUT / path / f"fold_{k}.npz") for k in range(1, 6)]
        for name in names:
            means[f"{run} {name}"] = np.mean([f[f"{name}_test"] for f in folds], axis=0)
            out.update({f"{run} {name} fold {k + 1}": f[f"{name}_test"] for k, f in enumerate(folds)})
    out.update(means)
    for r in (2, 3):
        for combo in combinations(means, r):
            if not all(c.startswith("cnn") for c in combo):
                out[" + ".join(combo)] = np.mean([means[c] for c in combo], axis=0)
    return out


def candidates_medsiglip(ids):
    tolga = pd.read_csv(CONFIG.output_dir / "experts_tolga/official_test_py/preds.csv")
    tolga.index = tolga["path"].str.split("/").str[-1].str.replace(".tiff", "", regex=False)
    folds = [tolga.loc[ids, f"pred_fold{k}"].to_numpy() for k in range(1, 6)]
    out = {f"MedSigLIP fold {k + 1}": f for k, f in enumerate(folds)}
    for r in (2, 3, 4, 5):
        for combo in combinations(range(5), r):
            out["MedSigLIP folds " + "".join(str(c + 1) for c in combo)] = np.mean([folds[c] for c in combo], axis=0)
    return out


def search(cands, mos, rng):
    table = {(name, how): post(pred, POSTS[how]) for name, pred in cands.items() for how in POSTS}
    full = {key: scores(pred, mos) for key, pred in table.items()}
    best = max(full, key=lambda key: full[key][3])
    gaps = []
    for _ in range(SPLITS):
        a = rng.permutation(len(mos))[:len(mos) // 2]
        b = np.setdiff1d(np.arange(len(mos)), a)
        on_a = {key: scores(pred[a], mos[a])[3] for key, pred in table.items()}
        pick = max(on_a, key=on_a.get)
        gaps.append(on_a[pick] - scores(table[pick][b], mos[b])[3])
    return best, full[best], len(table), float(np.mean(gaps))


def main() -> None:
    test = pd.read_csv(CONFIG.output_dir / "experts_tolga/official_test_py/preds.csv")
    test.index = test["path"].str.split("/").str[-1].str.replace(".tiff", "", regex=False)
    ids = list(np.load(JOB / "test.npz")["ids"])
    mos = test.loc[ids, "mos"].to_numpy()
    rng = np.random.default_rng(0)
    rows, lines = {}, []
    for label, cands in (("ours (Stage B family)", candidates_ours()), ("MedSigLIP", candidates_medsiglip(ids))):
        (name, how), (p, s, k, o), n, gap = search(cands, mos, rng)
        rows[label] = {"configuration": f"{name}, {how}", "plcc": p, "srocc": s, "krocc": k, "overall": o,
                       "configs searched": n, "split-half optimism": gap}
        print(label, rows[label], flush=True)
    table = pd.DataFrame(rows).T
    for c in ("plcc", "srocc", "krocc", "overall", "split-half optimism"):
        table[c] = table[c].astype(float).round(4)
    lines = ["# Test-selected (oracle) scores on the 300-image test", "",
             "Selected on the test itself, for comparison with test-selected literature only; not a generalisation estimate.", "",
             table.to_markdown(), "",
             "Reference: pre-registered Stage B rerun 2.7345 as is, 2.7581 on the 0.2 grid; ClinReadNet 2.7690 "
             "(about 60 configurations scored on the test); CAP-IQA 2.859 (ablations on the test).", ""]
    (OUT / "oracle_test.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
