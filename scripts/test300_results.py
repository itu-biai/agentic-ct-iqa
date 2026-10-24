"""Setup 26's protocol on the whole LDCT-IQA test set (300 slices, 15 sandboxes of 20; scripts/make_test300.py):
unpacks the VM results and compares the agent with MedSigLIP-FiLM (the scores its consult_experts tool returned) and
the stacked regression of Table 1, on the same 300 images.

    python -m scripts.test300_results ingest <test300_results.zip>   # -> outputs/agentic_ct_iqa/test300/runs/
    python -m scripts.test300_results report                         # metrics, paired bootstrap, report.md
    python -m scripts.test300_results guided                         # setup 30 on the 300 (test300g_batch_*), report_guided.md

Setup 30 (scripts/make_references.py test300) is compared with the expert, setup 26's run, the noise rule alone
(neighbour_mos_mean) and the rule with a fixed structure correction. That correction was fixed on training slices
before the 300-image run (bank leave-one-out errors against structure_pct, scripts/rule_blind_spots.py): 0.368 MOS per
100 percentile points, centred at 50.
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.sandbox_results import load_run, metrics
from src.config import CONFIG, PROJECT_ROOT
from src.metrics import challenge_scores

OUT = CONFIG.output_dir / "test300"
RUNS = OUT / "runs"
KEY = OUT / "key.csv"
IDS = [f"img_{k:02d}" for k in range(1, 21)]


def ingest(archive: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target = RUNS / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    for folder in sorted(RUNS.glob("test300*_batch_*/run_1")):
        scores, problems = load_run(folder, IDS)
        print(folder.parent.name, f"{scores.notna().sum()}/20 scored", "; ".join(problems))


STRUCTURE_SLOPE = 0.368 / 100  # fixed on training slices before the setup 30 run on the 300
GUIDED = PROJECT_ROOT / "experiment_sandboxes/test300_guided"


def scores(prefix: str = "test300_batch") -> pd.DataFrame:
    """One row per test slice: MOS, expert, agent score (NaN where the batch has no valid run)."""
    key = pd.read_csv(KEY)
    agent = []
    for b, group in key.groupby("batch"):
        folder = RUNS / f"{prefix}_{b:02d}/run_1"
        run = load_run(folder, IDS)[0] if (folder / "scores.json").exists() else pd.Series(np.nan, index=IDS)
        agent.append(pd.Series(run.reindex(group["image_id"]).to_numpy(), index=group.index))
    key["agent"] = pd.concat(agent)
    return key


def paired_bootstrap(frame: pd.DataFrame, a: str, b: str, n: int = 2000) -> tuple[float, float, float]:
    """Overall(a) - Overall(b) on the same images, 95% interval over images resampled with replacement."""
    rng = np.random.default_rng(CONFIG.seed)
    diffs = []
    for _ in range(n):
        s = frame.iloc[rng.integers(0, len(frame), len(frame))]
        diffs.append(challenge_scores(s[a].to_numpy(), s["mos"].to_numpy())["overall"]
                     - challenge_scores(s[b].to_numpy(), s["mos"].to_numpy())["overall"])
    point = (challenge_scores(frame[a].to_numpy(), frame["mos"].to_numpy())["overall"]
             - challenge_scores(frame[b].to_numpy(), frame["mos"].to_numpy())["overall"])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(point, 3), round(lo, 3), round(hi, 3)


def report() -> None:
    frame = scores()
    ok = frame[frame["agent"].notna()]
    rows = {name: metrics(ok[name].reset_index(drop=True), ok["mos"].reset_index(drop=True))
            for name in ("agent", "expert")}
    error = ok["agent"] - ok["mos"]
    bins = {f"bias_bin{b}": round(float(error[ok["mos_bin"] == b].mean()), 3) for b in range(5)}
    expert_error = ok["expert"] - ok["mos"]
    expert_bins = {f"bias_bin{b}": round(float(expert_error[ok["mos_bin"] == b].mean()), 3) for b in range(5)}
    point, lo, hi = paired_bootstrap(ok, "agent", "expert")
    reliance = {"mean_abs_diff_to_expert": round(float((ok["agent"] - ok["expert"]).abs().mean()), 3),
                "srocc_with_expert": round(float(spearmanr(ok["agent"], ok["expert"])[0]), 3),
                "changed_by_more_than_0.5": int(((ok["agent"] - ok["expert"]).abs() > 0.5).sum())}
    # Where the agent moved away from the expert, did it move towards the MOS?
    moved = ok[(ok["agent"] - ok["expert"]).abs() > 0.25]
    better = int(((moved["agent"] - moved["mos"]).abs() < (moved["expert"] - moved["mos"]).abs()).sum())
    table1 = pd.read_csv(CONFIG.output_dir / "tables/table1_baselines.csv").set_index("method")
    lines = ["# Setup 26 protocol on the 300-image test", "",
             f"{len(ok)}/300 slices scored ({frame['agent'].isna().sum()} missing). Correlations without logistic "
             "fitting; Overall = |PLCC| + |SROCC| + |KROCC| as in the challenge.", "",
             pd.DataFrame({**rows}).T.join(pd.DataFrame({"agent": bins, "expert": expert_bins}).T)
             .to_markdown(), "",
             f"Agent - expert Overall: {point:+.3f}, paired bootstrap 95% interval [{lo:+.3f}, {hi:+.3f}].", "",
             f"Reliance on the expert: {reliance}.",
             f"Of the {len(moved)} slices where the agent moved more than 0.25 from the expert, it moved towards the "
             f"MOS on {better}.", "",
             f"Table 1 on the full test (same 300): MedSigLIP-FiLM overall {table1.loc['medsiglip_film', 'overall_raw']:.3f}"
             f" raw, stacked regression {table1.loc['stacked_regression', 'overall_raw']:.3f} raw.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    frame.round(3).to_csv(OUT / "per_image.csv", index=False)
    print("\n".join(lines))


def guided() -> None:
    frame = scores()
    frame = frame.rename(columns={"agent": "setup26"})
    frame["setup30"] = scores("test300g_batch")["agent"]
    values = pd.DataFrame([json.loads((GUIDED / f"batch_{r.batch:02d}/tool_results/{r.image_id}.json")
                                      .read_text(encoding="utf-8"))["tools"]["reference_match"]["values"]
                           for r in frame.itertuples()], index=frame.index)
    frame["rule"] = values["neighbour_mos_mean"]
    frame["rule_structure"] = values["neighbour_mos_mean"] + STRUCTURE_SLOPE * (values["structure_pct"] - 50)
    ok = frame[frame["setup30"].notna()]
    names = ("setup30", "expert", "setup26", "rule", "rule_structure")
    table = pd.DataFrame({n: metrics(ok[n].reset_index(drop=True), ok["mos"].reset_index(drop=True)) for n in names}).T
    error = {n: (ok[n] - ok["mos"]) for n in names}
    bins = pd.DataFrame({n: {f"bias_bin{b}": round(float(e[ok["mos_bin"] == b].mean()), 3) for b in range(5)}
                         for n, e in error.items()}).T
    pairs = [("setup30", "expert"), ("setup30", "rule"), ("setup30", "rule_structure"), ("setup30", "setup26"),
             ("rule_structure", "rule")]
    tests = [f"- {a} - {b}: {point:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]"
             for a, b in pairs for point, lo, hi in [paired_bootstrap(ok, a, b)]]
    moved = ok[(ok["setup30"] - ok["rule"]).abs() > 0.25]
    better = int(((moved["setup30"] - moved["mos"]).abs() < (moved["rule"] - moved["mos"]).abs()).sum())
    lines = ["# Setup 30 on the 300-image test", "",
             f"{len(ok)}/300 slices scored. No expert score in the sandbox. Raw overall (no logistic fit).", "",
             table.join(bins).to_markdown(), "", "Paired bootstrap over images (Overall difference):", *tests, "",
             f"Setup 30 moved more than 0.25 from the rule on {len(moved)} slices, towards the MOS on {better}.",
             f"SROCC setup 30 vs rule: {spearmanr(ok['setup30'], ok['rule'])[0]:.3f}.", ""]
    (OUT / "report_guided.md").write_text("\n".join(lines), encoding="utf-8")
    frame.round(3).to_csv(OUT / "per_image_guided.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    {"ingest": lambda: ingest(Path(sys.argv[2])), "report": report, "guided": guided}[sys.argv[1]]()
