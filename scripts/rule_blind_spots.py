"""Where the fixed noise rule of setup 29 (scripts/make_references.py) goes wrong, on training slices only.

Every bank slice gets the rule's neighbour MOS estimate with itself left out of its neighbours; the 100 calibration
slices are outside the bank already. The error (MOS - estimate) is compared with the other screening measurements,
each first freed of the noise score (rank residual), so that a measurement only counts if it explains error the noise
rule leaves. Writes outputs/agentic_ct_iqa/probe/rule_errors_train.csv and prints the strongest relations.

    python -m scripts.rule_blind_spots
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from scripts.make_references import NEIGHBOURS, noise_score, split, tables
from src.config import CONFIG
from src.metrics import challenge_scores


def main() -> None:
    bank, bank_scores, mos, held_out = split()
    values = tables()[0]
    scores = pd.concat([bank_scores, noise_score(values.loc[held_out], bank)])
    estimate = {}
    for image, score in scores.items():
        others = bank_scores.drop(index=image, errors="ignore")
        estimate[image] = mos[(others - score).abs().sort_values(kind="stable").index[:NEIGHBOURS]].mean()
    frame = pd.DataFrame({"mos": mos[scores.index], "noise_score": scores, "estimate": pd.Series(estimate)})
    frame["error"] = frame["mos"] - frame["estimate"]
    print("rule on all 1000 training slices (bank leave-one-out + calibration):",
          {k: round(v, 3) for k, v in challenge_scores(frame["estimate"].to_numpy(), frame["mos"].to_numpy()).items()})
    print("mean error by MOS bin:", frame.groupby(np.digitize(frame["mos"], [0.8, 1.6, 2.4, 3.2]))["error"].mean().round(2).tolist())
    screening = pd.read_csv(CONFIG.output_dir / "screening/train.csv", dtype={"image_id": str}).set_index("image_id")
    screening = screening.drop(columns=["mos", "__tools"], errors="ignore").reindex(frame.index)
    rank_noise = rankdata(frame["noise_score"])
    rows = []
    for column in screening:
        x = screening[column].fillna(screening[column].median())
        r = rankdata(x)
        fit = np.polyfit(rank_noise, r, 1)
        residual = r - np.polyval(fit, rank_noise)
        rows.append({"measurement": column, "rho_error": spearmanr(residual, frame["error"])[0],
                     "rho_with_noise_score": spearmanr(x, frame["noise_score"])[0]})
    table = pd.DataFrame(rows).set_index("measurement").sort_values("rho_error", key=abs, ascending=False).round(3)
    print(table.to_markdown())
    out = frame.join(screening)
    out.round(4).to_csv(CONFIG.output_dir / "probe/rule_errors_train.csv")
    worst = out.sort_values("error")
    print("most overestimated (MOS far below rule):", worst.head(8)[["mos", "estimate"]].round(2).to_dict("index"))
    print("most underestimated (MOS far above rule):", worst.tail(8)[["mos", "estimate"]].round(2).to_dict("index"))


if __name__ == "__main__":
    main()
