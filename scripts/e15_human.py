"""Issue #329 E15 human check: the frozen keyword rules (scripts.agent_analysis.code) against a human reader's codes.
The human coded the same 100 rationales as configs/rationale_hand_codes*.csv (Claude's reference codes), blind (text
only, no score, MOS or other codes), on a page with an English text and a Turkish translation, into
configs/rationale_human_codes.csv (set A, seed 42) and configs/rationale_human_codes_holdout.csv (set B, seed 43).
Set A is still optimistic for the rules: they were written while reading those 50 texts. Per category: Cohen's kappa
on presence (accepted if >= 0.7, as pre-registered) and direction agreement where both code the category, for
rules vs human (primary), rules vs Claude (the earlier numbers) and Claude vs human.

    python -m scripts.e15_human      # -> outputs/agentic_ct_iqa/i329/e15_human.md, e15_human.csv
"""
import numpy as np
import pandas as pd

from configs import tool_groups as tg
from scripts.agent_analysis import OUT, code, kappa
from src.config import PROJECT_ROOT

SETS = (("A", "seed 42, the 50 rationales read while the rules were written (optimistic)", ""),
        ("B", "seed 43, 50 rationales never shown while the rules were written", "_holdout"))


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    rows = {}
    for c in tg.CATEGORIES:
        x, y = (a[c] != "").to_numpy(), (b[c] != "").to_numpy()
        both = x & y
        rows[c] = {"% first": 100 * x.mean(), "% second": 100 * y.mean(), "presence agreement": np.mean(x == y),
                   "kappa": kappa(x, y), "accepted (kappa >= 0.7)": kappa(x, y) >= 0.7, "n both": int(both.sum()),
                   "direction agreement": np.mean(a[c][both].to_numpy() == b[c][both].to_numpy())
                   if both.any() else np.nan}
    return rows


def main() -> None:
    text = pd.read_csv(OUT / "rationales_ne19.csv").set_index("test_id")["rationale"]
    lines, out = ["# E15 human check: keyword rules vs a human reader (100 rationales)", ""], []
    for s, title, suffix in SETS:
        read = lambda name: pd.read_csv(PROJECT_ROOT / "configs" / f"{name}{suffix}.csv", dtype=str).fillna(
            "").set_index("test_id")
        human, claude = read("rationale_human_codes"), read("rationale_hand_codes")
        assert human.index.equals(claude.index)
        rules = pd.DataFrame([code(text[t]) for t in human.index], index=human.index)
        for pair, (a, b) in (("rules vs human", (rules, human)), ("rules vs Claude", (rules, claude)),
                             ("Claude vs human", (claude, human))):
            frame = pd.DataFrame(agreement(a, b)).T
            out.append(frame.assign(set=s, pair=pair).rename_axis("category").reset_index())
            lines += [f"## Set {s} ({title}): {pair}", "", frame.round(3).to_markdown(), ""]
    pd.concat(out).round(4).to_csv(OUT / "e15_human.csv", index=False)
    (OUT / "e15_human.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
