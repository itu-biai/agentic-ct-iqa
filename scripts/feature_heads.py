"""New heads on the frozen features of a Stage B run (LPIPS-style: the network stays fixed, only a small layer on its
last-token features is trained, here with ranking or ordinal losses instead of MSE). CPU only, no GPU credits.

Per fold the head is fitted on the fold's 800 training slices (features of the fold's own model) and predicts its 200
validation slices (out-of-fold) and the 300 test slices (mean of the 5 fold heads). Candidates, hyper-parameters fixed
before any result:
  linear_mse   the original head refitted (ridge, alpha by GCV) - checks that refitting alone changes nothing
  rank         linear, soft RankNet over ALL training pairs (T 0.1 std) + MSE, full batch
  ordinal_rps  21 bins (0, 0.2, ..., 4), ranked probability score (agaldran's loss), prediction = expected value
  rank_rps     rank + ordinal_rps outputs averaged
Common: features standardised with the fold's training statistics, Adam lr 1e-3, weight decay 1e-2, 400 full-batch
steps, seed 0.

Rule, fixed 2026-10-20 before running: the winner is the candidate with the highest out-of-fold overall; the test set
is scored once, for the winner only, and only if its out-of-fold overall beats the run's own llm head (rank run 2.824).

    python -m scripts.feature_heads rank/out_rank        # out-of-fold table, then the test look if the rule allows
"""
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from torch import nn

from src.config import CONFIG, PROJECT_ROOT
from src.data import load_ldctiqa_split
from src.metrics import challenge_scores

JOB = PROJECT_ROOT / "colab/vft_job"
OUT = CONFIG.output_dir / "vision_ft"
BINS = np.round(np.arange(0, 4.01, 0.2), 1)
STEPS, LR, DECAY, TEMP = 400, 1e-3, 1e-2, 0.1


def scores(pred, mos) -> dict:
    s = challenge_scores(np.asarray(pred), np.asarray(mos))
    return {k: round(s[k], 4) for k in ("overall", "plcc", "srocc", "krocc")}


def fit_torch(x, y, kind):
    """x standardised features, y standardised MOS (rank) or raw MOS (rps). Returns a predict function -> MOS scale
    for rps, standardised scale for rank."""
    torch.manual_seed(0)
    x = torch.tensor(x, dtype=torch.float32)
    out = len(BINS) if kind == "rps" else 1
    layer = nn.Linear(x.shape[1], out)
    nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
    optim = torch.optim.Adam(layer.parameters(), lr=LR, weight_decay=DECAY)
    if kind == "rps":
        target = torch.tensor((y[:, None] <= BINS[None] + 1e-6).astype(np.float32))  # label CDF
    else:
        t = torch.tensor(y, dtype=torch.float32)
        i, j = torch.triu_indices(len(t), len(t), 1)
        q = torch.sigmoid((t[i] - t[j]) / TEMP)
    for _ in range(STEPS):
        p = layer(x)
        if kind == "rps":
            loss = ((torch.softmax(p, 1).cumsum(1) - target) ** 2).sum(1).mean()
        else:
            p = p.squeeze(1)
            loss = nn.functional.mse_loss(p, t) + nn.functional.binary_cross_entropy_with_logits((p[i] - p[j]) / TEMP, q)
        optim.zero_grad(); loss.backward(); optim.step()

    def predict(z):
        with torch.no_grad():
            p = layer(torch.tensor(z, dtype=torch.float32))
            return (torch.softmax(p, 1) @ torch.tensor(BINS, dtype=torch.float32)).numpy() if kind == "rps" else p.squeeze(1).numpy()
    return predict


def fold_heads(d, mos):
    tr, va = d["tr"], d["va"]
    f = {p: d[f"feat_{p}"].astype(np.float32) for p in ("tr", "val", "test")}
    mu, sd = f["tr"].mean(0), f["tr"].std(0) + 1e-6
    f = {p: (v - mu) / sd for p, v in f.items()}
    m, s = mos[tr].mean(), mos[tr].std()
    y = (mos[tr] - m) / s
    out = {}
    ridge = RidgeCV(alphas=np.logspace(0, 6, 13)).fit(f["tr"], y)
    out["linear_mse"] = {p: ridge.predict(f[p]) * s + m for p in ("val", "test")}
    rank = fit_torch(f["tr"], y, "rank")
    out["rank"] = {p: rank(f[p]) * s + m for p in ("val", "test")}
    rps = fit_torch(f["tr"], mos[tr], "rps")
    out["ordinal_rps"] = {p: rps(f[p]) for p in ("val", "test")}
    out["rank_rps"] = {p: (out["rank"][p] + out["ordinal_rps"][p]) / 2 for p in ("val", "test")}
    return out


def main(run: str) -> None:
    mos = np.load(JOB / "train.npz")["mos"]
    folds = [np.load(OUT / run / f"fold_{k}.npz") for k in range(1, 6)]
    names = ("linear_mse", "rank", "ordinal_rps", "rank_rps")
    oof = {n: np.full(len(mos), np.nan) for n in (*names, "original llm head")}
    test = {n: [] for n in names}
    for k, d in enumerate(folds, 1):
        heads = fold_heads(d, mos)
        for n in names:
            oof[n][d["va"]] = heads[n]["val"]
            test[n].append(heads[n]["test"])
        oof["original llm head"][d["va"]] = d["llm_val"]
        print(f"fold {k}:", {n: scores(heads[n]["val"], mos[d["va"]])["overall"] for n in names},
              "original", scores(d["llm_val"], mos[d["va"]])["overall"], flush=True)
    table = pd.DataFrame({n: scores(p, mos) for n, p in oof.items()}).T
    lines = [f"# Heads on frozen Stage B features ({run}), out-of-fold on the 1000 training slices", "", table.to_markdown(), ""]
    winner = max(names, key=lambda n: table.loc[n, "overall"])
    base = table.loc["original llm head", "overall"]
    lines.append(f"Winner by the fixed rule: {winner} ({table.loc[winner, 'overall']} vs original {base}).")
    if table.loc[winner, "overall"] > base:
        from scripts.qwen_probe import bootstrap
        from scripts.qwen_vision_ft import llm_predictions
        samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
        ids, tmos = [s.image_id for s in samples], np.array([s.mos for s in samples])
        assert list(np.load(JOB / "test.npz")["ids"]) == ids
        pred = np.mean(test[winner], axis=0)
        same = np.mean([d["llm_test"] for d in folds], axis=0)
        stage_b = llm_predictions("llm", "test")
        rows = {f"{winner} (TEST LOOK)": pred, "same run, original head (seen before)": same, "Stage B llm (seen before)": stage_b}
        lines += ["", "## Test (300), scored once", "", pd.DataFrame({n: scores(p, tmos) for n, p in rows.items()}).T.to_markdown(), "",
                  f"- {winner} - original head: {scores(pred, tmos)['overall'] - scores(same, tmos)['overall']:+.4f}, "
                  f"95% CI {bootstrap(pred, same, tmos)}",
                  f"- {winner} - Stage B llm: {scores(pred, tmos)['overall'] - scores(stage_b, tmos)['overall']:+.4f}, "
                  f"95% CI {bootstrap(pred, stage_b, tmos)}"]
    else:
        lines.append("No candidate beats the original head out-of-fold: no test look.")
    name = run.replace("/", "_")
    (OUT / f"heads_{name}.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
