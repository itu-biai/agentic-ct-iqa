"""Issue #329 E14-E16a: what the zero-shot agent reads, what its rationales cite, and whether the citations agree with
the measurements it saw. Agent = the Table 1 zero-shot row (E4: Z3, ne19 = Qwen3.8-27B agent + CT tools, no MedSigLIP,
300 test images in 15 folders of 20, one run). Tool groups: configs/tool_groups.py (fixed before parsing).

E14 reads: one row per file the agent opened (OpenCode `read` calls in session.json), mapped to the test image by
  folder + img_XX: image_id, tool, group, step. The numeric tools come in one JSON per image, so a JSON read counts as a
  read of all four numeric groups (they cannot be told apart); the 7 views are separate files (visual). Per image:
  number of reads (all files of that image), groups read; by MOS bin [0,1), [1,2), [2,3), [3,4] and by absolute-error
  bin [0, 0.25), [0.25, 0.5), [0.5, 1), [1, inf).
E15 citations: keyword rules (RULES below) code each rationale into the categories of configs/tool_groups.CATEGORIES with
  a direction (good / bad / neutral; per clause, the first clause that names a category sets its direction).
  Validation: the 50 rationales drawn with default_rng(42) (rationales_ne19.csv, column validation) were coded by
  reading them, before the rules were written, in configs/rationale_hand_codes.csv; Cohen's kappa per category on
  presence, accepted if >= 0.7, plus direction agreement where both code the category. These reference codes were made
  by Claude, not by a human reader. Also: agent absolute error by cited category (Fig 2c).
E16a faithfulness: for each cited category with a direction (noise, streaks, sharpness, hf) the measured percentile the
  agent saw (configs/tool_groups.FAITHFULNESS, above 50 = more of the property); agreement: noise / streaks / hf "bad"
  (= present, high) agrees if > 50, "good" if < 50; sharpness "good" (sharp) agrees if > 50, "bad" (blurred) if < 50;
  rate per group with a bootstrap CI over images (2000, rng 42).

    python -m scripts.agent_analysis      # -> outputs/agentic_ct_iqa/i329/agent_*.csv, agent_analysis.md
"""
import json
import re

import numpy as np
import pandas as pd

from configs import tool_groups as tg
from src.config import CONFIG, PROJECT_ROOT

OUT = CONFIG.output_dir / "i329"
RUNS = CONFIG.output_dir / "test300/runs"
SANDBOX = PROJECT_ROOT / "experiment_sandboxes/test300_ne19"
SETUP = "ne19"
NUMERIC = ("noise", "streaks", "sharpness", "hf")

CUES = {
    "noise": r"nois|grain|speckl|mottl|salt-and-pepper|quiet|\bclean|immerkaer|residual|high-pass|high_pass|graininess",
    "streaks": r"streak|spoke|cross-hatch|hatching|grid-like|grid pattern|radial pattern",
    "sharpness": r"sharp|crisp|blur|smear|\bsoft(?![- ]tissue)|softness|softest|smoothed|smoothing|over-smooth|edge width|"
                 r"wide edges|thin edges|painterly",
    "hf": r"high-frequency|high frequency|\bnps\b|fourier|spectrum|nyquist",
    "contrast": r"\bcontrast\b(?![- ](?:filled|enhanc|opacif))|window",
    "anatomy": r"delineat|identifi|recogni|discern|readab|depict|differentiat|well[- ]seen|clearly seen|rendered|distinct|"
               r"\banatomy\b|opacified|structures? (?:are|is|remain)|organs? (?:are|remain)",
}
GOOD = r"\blow|little|minimal|minor|\bmild|slight|faint|subtle|quiet|\bclean|\bno\b|without|very fine|\bonly\b"
STRONG_BAD = (r"heavy|heavi|strong|marked|severe|dense|coarse|considerable|substantial|moderate|prominent|noisiest|"
              r"\bnoisy|extreme|elevated|\bclearly\b|pronounced|obscur|degrad|compromis|disturb|more ")
WEAK_BAD = r"visibl|evident|apparent|noticeabl|present|grain"
SHARP_GOOD = r"sharp|crisp|thin edges|well-defined edges"
SHARP_BAD = r"blur|smear|\bsoft(?![- ]tissue)|softness|softest|smoothed|smoothing|over-smooth|wide|painterly|less sharp"
NEGATION = r"\bno\b|without|absent|free of|not "


def clauses(text: str) -> list[str]:
    """Top-level clauses first, then the parenthetical remarks (evidence, so they decide a direction only last)."""
    inner = re.findall(r"\(([^)]*)\)", text)
    outer = re.sub(r"\([^)]*\)", " ", text)
    split = r"[.;:,]|\bbut\b|\bthough\b|\balthough\b|\bwhile\b|\byet\b|\bwith\b|\bplus\b"
    return [c.strip().lower() for part in (outer, *inner) for c in re.split(split, part, flags=re.I) if c.strip()]


def direction(category: str, clause: str) -> str:
    if category == "sharpness":
        if re.search(SHARP_BAD, clause):
            return "bad"
        return "good" if re.search(SHARP_GOOD, clause) else "neutral"
    if category == "streaks":
        return "good" if re.search(NEGATION, clause) else "bad"
    if category == "hf":
        return "good" if re.search(r"\blow|little|minimal", clause) else "bad"
    if category == "contrast":
        return "bad" if re.search(r"poor|low contrast|reduced contrast", clause) else (
            "good" if re.search(r"good contrast|high contrast", clause) else "neutral")
    if category == "anatomy":
        return "bad" if re.search(r"obscur|hard to see|not (?:visible|seen|identifiable)|poorly", clause) else "good"
    if re.search(STRONG_BAD, clause):
        return "bad"
    if re.search(GOOD, clause):
        return "good"
    return "bad" if re.search(WEAK_BAD, clause) else "neutral"


def code(text: str) -> dict[str, str]:
    out = {}
    for clause in clauses(text):
        for category, cue in CUES.items():
            if category not in out and re.search(cue, clause):
                out[category] = direction(category, clause)
    return {c: out.get(c, "") for c in tg.CATEGORIES}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def reads() -> pd.DataFrame:
    key = pd.read_csv(CONFIG.output_dir / "noexp/per_image.csv")
    key = key.set_index(["batch", "image_id"])["test_id"]
    rows = []
    for b in range(1, 16):
        session = json.load(open(RUNS / f"{SETUP}_batch_{b:02d}/run_1/session.json", encoding="utf-8"))
        step = 0
        for message in session["messages"]:
            for part in message.get("content") or []:
                if part.get("type") != "tool":
                    continue
                step += 1
                path = str(part["state"].get("input", {}).get("path", ""))
                m = re.search(r"(images|tool_results|tool_views)/(img_\d\d)(?:_(\w+))?\.(png|json)", path)
                if part["name"] != "read" or not m:
                    continue
                folder, image, view = m.group(1), m.group(2), m.group(3)
                tool = {"images": "image", "tool_results": "tool_results"}.get(folder, view)
                group = {"image": "image", "tool_results": "numeric"}.get(tool, tg.AGENT_VIEWS.get(view, "other"))
                rows.append({"test_id": key[(b, image)], "batch": b, "image_id": image, "tool": tool, "group": group,
                             "step": step})
    return pd.DataFrame(rows)


def measured(test_ids) -> pd.DataFrame:
    """Per image, the oriented percentile of each numeric group from the tool JSON the agent saw."""
    key = pd.read_csv(CONFIG.output_dir / "noexp/per_image.csv").set_index("test_id")
    rows = {}
    for t in test_ids:
        b, image = key.loc[t, "batch"], key.loc[t, "image_id"]
        values = json.load(open(SANDBOX / f"batch_{b:02d}/tool_results/{image}.json", encoding="utf-8"))
        ref = values["tools"]["compare_to_reference"]["values"]
        rows[t] = {g: np.mean([ref[k] for k in plus] + [100 - ref[k] for k in minus])
                   for g, (plus, minus) in tg.FAITHFULNESS.items()}
    return pd.DataFrame(rows).T


def boot_mean(x: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return tuple(np.percentile(m, [2.5, 97.5]))


def main() -> None:
    frame = pd.read_csv(OUT / "rationales_ne19.csv")
    frame["abs_error"] = (frame["score"] - frame["mos"]).abs()
    frame["mos_bin"] = pd.cut(frame["mos"], [0, 1, 2, 3, 4.01], right=False, labels=["[0,1)", "[1,2)", "[2,3)", "[3,4]"])
    frame["error_bin"] = pd.cut(frame["abs_error"], [0, 0.25, 0.5, 1, np.inf], right=False,
                                labels=["[0,0.25)", "[0.25,0.5)", "[0.5,1)", "[1,inf)"])
    lines = [f"# Zero-shot agent (Z3 = {SETUP}): reads, citations, faithfulness (300 test images)", ""]

    # E14
    r = reads()
    r.to_csv(OUT / "agent_reads.csv", index=False)
    per = r.groupby("test_id").agg(reads=("tool", "size"), image_read=("group", lambda g: (g == "image").any()),
                                   numeric_read=("group", lambda g: (g == "numeric").any()),
                                   visual_read=("group", lambda g: (g == "visual").any()),
                                   views_read=("group", lambda g: (g == "visual").sum()))
    frame = frame.merge(per, left_on="test_id", right_index=True, how="left").fillna(
        {"reads": 0, "image_read": False, "numeric_read": False, "visual_read": False, "views_read": 0})
    share = lambda d: pd.Series({"images": len(d), "reads per image": d["reads"].mean(),
                                 "image opened %": 100 * d["image_read"].mean(),
                                 "numeric JSON read % (all 4 numeric groups)": 100 * d["numeric_read"].mean(),
                                 "any visual view read %": 100 * d["visual_read"].mean(),
                                 "views per image": d["views_read"].mean()})
    by_mos = frame.groupby("mos_bin", observed=True).apply(share)
    by_err = frame.groupby("error_bin", observed=True).apply(share)
    total = share(frame).to_frame("all").T
    per_view = r[r.group == "visual"].groupby("tool")["test_id"].nunique() / len(frame) * 100
    lines += ["## E14 reads", "", "A JSON read exposes all four numeric groups at once; views are separate files.", "",
              pd.concat([total, by_mos, by_err]).round(2).to_markdown(), "",
              "Share of images whose view was opened (%):", "", per_view.round(1).to_frame("%").to_markdown(), ""]
    pd.concat([total, by_mos]).round(3).to_csv(OUT / "fig2b_reads_by_mos.csv")

    # E15
    codes = pd.DataFrame([code(t) for t in frame["rationale"]], index=frame.index)
    codes.columns = [f"cite_{c}" for c in codes.columns]
    frame = pd.concat([frame, codes], axis=1)
    for name, title in (("rationale_hand_codes.csv", "the 50 drawn rationales (seed 42; read before the rules were "
                                                     "written, so optimistic)"),
                        ("rationale_hand_codes_holdout.csv", "50 further rationales (seed 43, from those never shown "
                                                             "while the rules were written; coded after the rules were "
                                                             "frozen, rules_frozen.md5)")):
        hand = pd.read_csv(PROJECT_ROOT / "configs" / name, dtype=str).fillna("").set_index("test_id")
        val = frame.set_index("test_id").loc[hand.index]
        rows = {}
        for c in tg.CATEGORIES:
            a, b = (hand[c] != "").to_numpy(), (val[f"cite_{c}"] != "").to_numpy()
            both = a & b
            rows[c] = {"hand %": 100 * a.mean(), "rules %": 100 * b.mean(), "presence agreement": np.mean(a == b),
                       "kappa": kappa(a, b),
                       "direction agreement": np.mean(hand[c][both].to_numpy() == val[f"cite_{c}"][both].to_numpy())
                       if both.any() else np.nan, "accepted (kappa >= 0.7)": kappa(a, b) >= 0.7}
        lines += [f"## E15 rationale coding: validation on {title}; reference codes by Claude, not a human", "",
                  pd.DataFrame(rows).T.round(3).to_markdown(), ""]
    cited = {c: 100 * (frame[f"cite_{c}"] != "").mean() for c in tg.CATEGORIES}
    by_dir = {c: frame[f"cite_{c}"].replace("", "not cited").value_counts().to_dict() for c in tg.CATEGORIES}
    err = {c: {"cited": frame.loc[frame[f"cite_{c}"] != "", "abs_error"].mean(),
               "not cited": frame.loc[frame[f"cite_{c}"] == "", "abs_error"].mean(),
               "n cited": int((frame[f"cite_{c}"] != "").sum())} for c in tg.CATEGORIES}
    lines += ["Cited (% of 300 rationales): " + ", ".join(f"{c} {v:.0f}" for c, v in cited.items()), "",
              "Directions: " + "; ".join(f"{c} {d}" for c, d in by_dir.items()), "",
              "Absolute error by cited category (Fig 2c):", "", pd.DataFrame(err).T.round(3).to_markdown(), ""]
    pd.DataFrame(err).T.drop(index="anatomy").round(4).to_csv(OUT / "fig2c_error_by_citation.csv")  # anatomy: kappa < 0.7 on the holdout

    # E16a
    m = measured(frame["test_id"])
    rows = {}
    for g in NUMERIC:
        d = frame[frame[f"cite_{g}"].isin(["good", "bad"])]
        pct = m.loc[d["test_id"], g].to_numpy()
        bad = (d[f"cite_{g}"] == "bad").to_numpy()
        if g == "sharpness":
            agree = np.where(bad, pct < 50, pct > 50)
        else:
            agree = np.where(bad, pct > 50, pct < 50)
        lo, hi = boot_mean(agree.astype(float)) if len(agree) else (np.nan, np.nan)
        rows[g] = {"citations with direction": len(agree), "agree %": 100 * agree.mean() if len(agree) else np.nan,
                   "CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]", "bad / good": f"{bad.sum()} / {(~bad).sum()}"}
    pd.DataFrame(rows).T.to_csv(OUT / "agent_faithfulness.csv")
    lines += ["## E16a citation faithfulness (measured percentile the agent saw; > 50 = more of the property)", "",
              pd.DataFrame(rows).T.to_markdown(), ""]
    frame.to_csv(OUT / "agent_per_image.csv", index=False)
    (OUT / "agent_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
