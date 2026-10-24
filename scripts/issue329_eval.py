"""Issue #329 (itu-biai/dlct-docs): Tables 1-3 and the analyses E1-E19, from saved predictions and the outputs of
colab/vm/infer329.py and the queue runs (colab/i329_launch.sh -> outputs/agentic_ct_iqa/i329/).

Fixed on 2026-10-22, before any new result was downloaded; the protocol of the issue applies unchanged:
- Overall = |PLCC| + |SROCC| + |KROCC| (KROCC = scipy kendalltau, tau-b). 0.2 grid = round(pred / 0.2) * 0.2, applied to
  every row we ran. Test predictions are 5-fold means.
- CIs: paired bootstrap over the 300 test images, B = 2000, numpy default_rng(42), percentile interval. p-value of a
  difference (for Holm): two-sided bootstrap p = min(1, 2 * min(share of resampled deltas <= 0, share >= 0)).
- Holm within the three families of the issue, nothing else corrected:
    Family 1 (Table 1): Stage B vs MedSigLIP as is; Stage B vs MedSigLIP on the 0.2 grid; Stage B vs agaldran retrained
      (E7, as is and grid); Stage B vs Q-Align (E11); Stage B vs ResNeXt-50. A comparison whose experiment is not done
      is dropped from the family (and said so).
    Family 2 (Table 2): Z2 vs Z1, Z3 vs Z2, Z4 vs Z3, T2 vs T1, T3 vs T2, T6 vs T4, T6 vs T5, T7 vs T6.
    Family 3 (Table 3): each of the 4 shuffled measurement groups vs Stage B (E17).
- Rows. Z1 ne16, Z2 ne17, Z3 ne19, Z4 setup 30 (test300g); Table 1's zero-shot row is Z3 (E4: setup 30 differs from
  card 5 by the reference_match tool, its neighbour MOS and the guidance text, not only by the 80 references). T0 tools
  boosting, T1 frozen encoder + ridge (scripts/qwen_probe.py test predictions), T2 out_notools vision head (Stage A
  recipe, no tool inputs), T3 the plain Stage A test run (vision_ft/test_fold_<k>.npy, 2.719), T4 out_p2only llm,
  T5 out_notools llm, T6 Stage B rerun llm (2.7345), T7 = E3. The Delta for T6 is reported vs T5 (row above) and vs T4
  and vs its own vision head, as in the issue.
- PRIMARY per new run: E6 T3 - T2 (tools in the vision head); E9 T6 - T5 (does the LLM use the measurements); E10 T6 -
  T4 (is encoder tuning needed); E3 T7 - T6; E8 OOF (vision head) with the pseudo-patient folds vs with the grouped
  folds (Stage A 2.836), and both against the test 2.719 (the gap).
- E3 / E17 shuffles: for each permutation seed s the 5-fold mean prediction; reported Overall = mean over the 5 seeds;
  CI: per seed 400 paired bootstrap resamples of (shuffled_s - Stage B), pooled (2000 values), rng 42.
- E17 partial dependence: fold 1, mean predicted MOS over the 300 test images at each set percentile.
- Denoiser tau (E2, and E3 on the lab set): lab IQA 2026v1 test split, 192 slices, 30 patients, per slice Kendall tau
  between the predicted quality of the 12 non-gt images and their SSIM to gt (as scripts/cross_eval.py); mean over
  slices; differences with a paired bootstrap over patients (2000, rng 42). T1 on the lab set = the same ridge as its
  test predictions, refit on the training features (scripts/qwen_probe.py), applied to the lab features; the first 50
  training features recomputed on the VM must match the local ones (else the cell is dropped).
- E18: occlusion drop = base - occluded prediction (5-fold mean) per 48-px patch; reported share = sum of positive
  drops whose patch centre is inside the body mask (stored value > 500 / 1400, i.e. HU > -500, largest component,
  holes filled) / sum of all positive drops.
- E19: ridge probes on the mean-pooled encoder features (pretrained vs fold-1 tuned), alpha by grouped 5-fold CV on the
  training slices (the 90 base-slice groups), targets = the 6 tool percentiles and MOS; test R^2 per target.
- E12 (added 2026-10-23, before the seed runs finished): 3 training seeds per model. Stage B: the rerun (seed 1) and
  vftb_train.py seed2 / seed3 (same folds, seed + 10000 / 20000). MedSigLIP: Tolga's checkpoints (seed 42) and his
  train.py at seed 43 / 44 (his folds kept: KFold random_state fixed at 42). Table 1 ± = std over the 3 seeds of the
  Overall of each seed's 5-fold mean prediction. PRIMARY: Stage B - MedSigLIP Overall at each seed pairing (1-42,
  2-43, 3-44), reported with the paired CI of each pairing and how many of the 3 are > 0; secondary: the same Δ for the
  3-seed mean predictions. Not added to Family 1 (the Family 1 comparison stays the original runs).

    python -m scripts.issue329_eval        # -> outputs/agentic_ct_iqa/i329/report.md
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import kendalltau, pearsonr, spearmanr
from skimage.metrics import structural_similarity
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from configs import tool_groups as tg
from src.config import CONFIG, PROJECT_ROOT

JOB = PROJECT_ROOT / "colab/vft_job"
VFT = CONFIG.output_dir / "vision_ft"
I329 = CONFIG.output_dir / "i329"
INFER = I329 / "infer"
B, SEED = 2000, 42
NON_GT = ("noisy", "bm3d", "ctformer", "dncnn", "n2self", "nafnet", "nb2nb", "nr2n", "redcnn", "tv", "unet", "wgan_vgg")


# ---------------------------------------------------------------- metrics

def metrics(pred, mos) -> dict:
    pred, mos = np.asarray(pred, float), np.asarray(mos, float)
    p, s, k = abs(pearsonr(pred, mos)[0]), abs(spearmanr(pred, mos)[0]), abs(kendalltau(pred, mos)[0])
    return {"plcc": p, "srocc": s, "krocc": k, "overall": p + s + k}


def overall(pred, mos) -> float:
    return metrics(pred, mos)["overall"]


def grid(pred) -> np.ndarray:
    return np.round(np.asarray(pred, float) / 0.2) * 0.2


def deltas(a, b, mos, n=B, seed=SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for r in range(n):
        i = rng.integers(0, len(mos), len(mos))
        out[r] = overall(a[i], mos[i]) - overall(b[i], mos[i])
    return out


def summary(point: float, d: np.ndarray) -> dict:
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    return {"delta": point, "lo": lo, "hi": hi, "p": p}


def compare(a, b, mos) -> dict:
    return summary(overall(a, mos) - overall(b, mos), deltas(a, b, mos))


def compare_pooled(shuffled: list, base, mos) -> dict:
    """Mean over permutation seeds; CI from 400 resamples per seed, pooled."""
    point = np.mean([overall(s, mos) for s in shuffled]) - overall(base, mos)
    d = np.concatenate([deltas(s, base, mos, n=B // len(shuffled), seed=SEED + j) for j, s in enumerate(shuffled)])
    return summary(point, d)


def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (len(ps) - i) * ps[name]))
        out[name] = running
    return out


def fmt(c: dict, holm_p: float | None = None) -> str:
    mark = "†" if holm_p is not None and holm_p < 0.05 else ""
    return f"{c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]{mark}"


# ---------------------------------------------------------------- predictions

def fold_mean(directory, key: str):
    files = [directory / f"fold_{k}.npz" for k in range(1, 6)]
    if not all(f.exists() for f in files):
        return None
    z = [np.load(f) for f in files]
    return np.mean([f[key] for f in z], 0) if all(key in f.files for f in z) else None


def test_predictions() -> tuple[pd.DataFrame, dict]:
    ids = np.load(JOB / "test.npz")["ids"]
    per = pd.read_csv(CONFIG.output_dir / "noexp/per_image.csv").set_index("test_id").loc[ids]
    guided = pd.read_csv(CONFIG.output_dir / "test300/per_image_guided.csv").set_index("test_id").loc[ids]
    probe = pd.read_csv(CONFIG.output_dir / "probe/test_predictions.csv").set_index("test_id").loc[ids]
    tools = pd.read_csv(CONFIG.output_dir / "probe/tools_only_test.csv").set_index("test_id").loc[ids]
    f = pd.DataFrame({"mos": per["mos"].to_numpy()}, index=ids)
    f["Z1"], f["Z2"], f["Z3"], f["Z4"] = per["ne16"], per["ne17"], per["ne19"], guided["setup30"]
    f["T0"] = tools["tools boosting"]
    f["T1"] = probe["Qwen encoder + ridge (primary)"]
    f["MedSigLIP"] = probe["MedSigLIP-FiLM (expert)"]
    f["T3"] = np.mean([np.load(VFT / f"test_fold_{k}.npy") for k in range(1, 6)], 0)
    f["T6"] = fold_mean(VFT / "llm2/out_llm", "llm_test")
    f["T6 vision head"] = fold_mean(VFT / "llm2/out_llm", "vision_test")
    f["ResNeXt-50"] = fold_mean(VFT / "cnn/out_cnn", "cnn_test")
    optional = {"T2": (I329 / "out_notools", "vision_test"), "T5": (I329 / "out_notools", "llm_test"),
                "T4": (I329 / "out_p2only", "llm_test"), "Stage B seed 2": (I329 / "out_seed2", "llm_test"),
                "Stage B seed 3": (I329 / "out_seed3", "llm_test"),
                "MedSigLIP seed 2": (I329 / "medsig_seed2", "pred_test"), "MedSigLIP seed 3": (I329 / "medsig_seed3", "pred_test")}
    for name, (d, key) in optional.items():
        v = fold_mean(d, key)
        if v is not None:
            f[name] = v
    for name, file, key in (("agaldran", I329 / "agaldran_test.csv", "expected"),
                            ("agaldran grid", I329 / "agaldran_test.csv", "argmax"),
                            ("Q-Align", I329 / "qalign_test.csv", "pred")):
        if file.exists():
            f[name] = pd.read_csv(file).set_index("test_id").loc[ids, key].to_numpy()
    shuffles = {}
    path = INFER / "shuffle.npz" if (INFER / "shuffle.npz").exists() else INFER / "shuffle.partial.npz"
    if path.exists():
        z = np.load(path)
        folds = [k for k in range(1, 6) if f"streaks_s4_{k}" in z.files]
        base = np.mean([z[f"base_{k}"] for k in folds], 0)
        saved = np.mean([np.load(VFT / f"llm2/out_llm/fold_{k}.npz")["llm_test"] for k in folds], 0)
        shuffles["check: base vs saved T6, max abs diff"] = float(np.abs(base - saved).max())
        shuffles["folds"] = folds
        for cond in ("all", *tg.STAGE_B_COLUMNS):
            shuffles[cond] = [np.mean([z[f"{cond}_s{s}_{k}"] for k in folds], 0) for s in range(5)]
        f["T7"] = np.mean(shuffles["all"], 0)
        f["T6 (shuffle folds)"] = base  # reference for E3 / E17: the same fold models, unshuffled
    return f, shuffles


# ---------------------------------------------------------------- lab set (denoiser tau)

def lab_frame() -> pd.DataFrame:
    """Per lab test image: slice, kind, patient, SSIM to gt (cached), and every model's prediction available."""
    lab = np.load(JOB / "lab329.npz")
    ids = pd.Index(lab["ids"].astype(str))
    f = pd.DataFrame({"s": ids.str.rsplit("/", n=1).str[0], "k": ids.str.rsplit("/", n=1).str[1]}, index=ids)
    f["patient"] = f["s"].str.split("_").str[0]
    cache = I329 / "lab_ssim.csv"
    if not cache.exists():
        h = w = 512
        yy, xx = np.ogrid[:h, :w]
        mask = np.sqrt((xx - w // 2) ** 2 + (yy - h // 2) ** 2) <= w // 2
        ssim = {}
        for s in sorted(set(f["s"])):
            group = CONFIG.iqa2026_root / "test" / s
            gt = np.asarray(Image.open(group / "gt.png")) * mask
            for m in NON_GT:
                ssim[f"{s}/{m}"] = structural_similarity(gt, np.asarray(Image.open(group / f"{m}.png")) * mask, data_range=255) * 100
        pd.Series(ssim, name="ssim").to_csv(cache)
    f["ssim"] = pd.read_csv(cache, index_col=0)["ssim"].reindex(ids)
    cross = np.load(JOB / "cross_pred.npz", allow_pickle=True)
    pred = lambda key: pd.Series(np.mean([cross[f"{key}_{k}"] for k in range(1, 6)], 0), index=cross["ids"]).loc[ids].to_numpy()
    f["T6"], f["T6 vision head"] = pred("llm"), pred("vision")
    f["T3"] = f["T6 vision head"]  # the plain Stage A run kept no weights; its recipe is the rerun's vision head
    from scripts.cross_eval import medsig
    f["MedSigLIP"] = medsig("cross").loc[ids].to_numpy()
    if (I329 / "lab_t0.csv").exists():
        f["T0"] = pd.read_csv(I329 / "lab_t0.csv").set_index("image_id").loc[ids, "T0"].to_numpy()
    part = INFER / "labshuf.npz" if (INFER / "labshuf.npz").exists() else INFER / "labshuf.partial.npz"
    if part.exists():
        z = np.load(part)
        folds = [k for k in range(1, 6) if f"llm_{k}" in z.files]
        f["T7"] = np.mean([z[f"llm_{k}"] for k in folds], 0)
        f["T6 (shuffle folds)"] = pd.Series(np.mean([cross[f"llm_{k}"] for k in folds], 0), index=cross["ids"]).loc[ids].to_numpy()
    for name, file, keys in ( ("T5", "lab_notools", ["llm"]), ("T2", "lab_notools", ["vision"]),
                             ("T4", "lab_p2only", ["llm"]), ("ResNeXt-50", "cnn", ["cnn"])):
        if (INFER / f"{file}.npz").exists():
            z = np.load(INFER / f"{file}.npz")
            f[name] = np.mean([z[f"{keys[0]}_{k}"] for k in range(1, 6)], 0)
    if (INFER / "probe.npz").exists():
        f["T1"] = t1_lab(np.load(INFER / "probe.npz")["lab"])
    for name, file, key in (("agaldran", "agaldran_lab.csv", "expected"), ("agaldran grid", "agaldran_lab.csv", "argmax"),
                            ("Q-Align", "qalign_lab.csv", "pred")):
        if (I329 / file).exists():
            f[name] = pd.read_csv(I329 / file).set_index("image_id").loc[ids, key].to_numpy()
    return f


def t1_lab(lab_features: np.ndarray) -> np.ndarray:
    """T1's ridge exactly as scripts/qwen_probe.py test(), refit on the training features, applied to the lab features."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train = np.load(CONFIG.output_dir / "probe/features.npz")
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 15))).fit(train["x"].astype(np.float64), train["mos"])
    return model.predict(lab_features.astype(np.float64))


def taus(f: pd.DataFrame, column: str) -> pd.Series:
    d = f[f["k"] != "gt"]
    wide = d.pivot(index="s", columns="k", values=column)[list(NON_GT)]
    truth = d.pivot(index="s", columns="k", values="ssim")[list(NON_GT)]
    return pd.Series([kendalltau(t, p)[0] for p, t in zip(wide.to_numpy(), truth.to_numpy())], index=wide.index)


def tau_diff(f: pd.DataFrame, a: str, b: str, n=B) -> str:
    d = taus(f, a) - taus(f, b)
    patients = f.groupby("s")["patient"].first().loc[d.index]
    g = d.groupby(patients).agg(["sum", "count"])
    idx = np.random.default_rng(SEED).integers(0, len(g), (n, len(g)))
    m = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return f"{d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]"


# ---------------------------------------------------------------- tables

def table1(f: pd.DataFrame, lab: pd.DataFrame, lines: list) -> None:
    mos = f["mos"].to_numpy()
    rows = {"Qwen3.8-27B agent + CT tools (Z3)": ("Z3", None), "Q-Align, retrained on our folds": ("Q-Align", None),
            "ResNeXt-50": ("ResNeXt-50", None), "agaldran, retrained on our folds": ("agaldran", "agaldran grid"),
            "MedSigLIP-FiLM": ("MedSigLIP", None), "Stage B (ours)": ("T6", None)}
    out = {}
    for label, (col, grid_col) in rows.items():
        if col not in f:
            continue
        m = metrics(f[col], mos)
        g = f[grid_col].to_numpy() if grid_col else grid(f[col])
        row = {"PLCC": m["plcc"], "SROCC": m["srocc"], "KROCC": m["krocc"], "Overall": m["overall"],
               "Overall, 0.2 grid": overall(g, mos)}
        row["Δ vs MedSigLIP [95% CI]"] = "reference" if col == "MedSigLIP" else fmt(compare(f[col].to_numpy(), f["MedSigLIP"].to_numpy(), mos))
        row["denoiser τ̄"] = taus(lab, col).mean() if col in lab else np.nan
        out[label] = row
    seeds = [c for c in ("T6", "Stage B seed 2", "Stage B seed 3") if c in f]
    lines += ["## Table 1: comparison with other methods (300 test images; rows we ran)", "",
              pd.DataFrame(out).T.round(4).to_markdown(), ""]
    if len(seeds) > 1:
        sd = pd.DataFrame([metrics(f[c], mos) for c in seeds]).std()
        lines += [f"Stage B over {len(seeds)} training seeds (std): " + ", ".join(f"{k} ±{v:.4f}" for k, v in sd.items()), ""]
    e12(f, mos, lines)
    # fold spread (the issue's fallback for ±): each fold model alone on the 300 test images
    stage_folds = [np.load(VFT / f"llm2/out_llm/fold_{k}.npz")["llm_test"] for k in range(1, 6)]
    tolga = pd.read_csv(CONFIG.output_dir / "experts_tolga/official_test_py/preds.csv")
    tolga.index = tolga["path"].str.split("/").str[-1].str.replace(".tiff", "", regex=False)
    med_folds = [tolga.loc[f.index, f"pred_fold{k}"].to_numpy() for k in range(1, 6)]
    spread = {name: pd.DataFrame([metrics(p, mos) for p in folds]).agg(["mean", "std"]).T
              for name, folds in (("Stage B", stage_folds), ("MedSigLIP-FiLM", med_folds))}
    lines += ["Single-fold models on the test (mean ± std over the 5 folds; the ± fallback without E12):", "",
              pd.concat(spread, axis=1).round(4).to_markdown(), ""]
    fam = {"Stage B vs MedSigLIP (as is)": (f["T6"].to_numpy(), f["MedSigLIP"].to_numpy()),
           "Stage B vs MedSigLIP (0.2 grid)": (grid(f["T6"]), grid(f["MedSigLIP"]))}
    if "agaldran" in f:
        fam["Stage B vs agaldran (as is)"] = (f["T6"].to_numpy(), f["agaldran"].to_numpy())
        fam["Stage B vs agaldran (grid: Stage B 0.2 grid vs its argmax class)"] = (grid(f["T6"]), f["agaldran grid"].to_numpy())
    if "Q-Align" in f:
        fam["Stage B vs Q-Align"] = (f["T6"].to_numpy(), f["Q-Align"].to_numpy())
    fam["Stage B vs ResNeXt-50"] = (f["T6"].to_numpy(), f["ResNeXt-50"].to_numpy())
    comps = {k: compare(a, b, mos) for k, (a, b) in fam.items()}
    adj = holm({k: c["p"] for k, c in comps.items()})
    lines += ["Family 1 (Holm over the comparisons run; † = Holm-adjusted p < 0.05):", "",
              pd.DataFrame({k: {"Δ [95% CI]": fmt(c, adj[k]), "p": c["p"], "Holm p": adj[k]} for k, c in comps.items()}).T.to_markdown(), ""]


FAMILY2 = (("Z2", "Z1"), ("Z3", "Z2"), ("Z4", "Z3"), ("T2", "T1"), ("T3", "T2"), ("T6", "T4"), ("T6", "T5"), ("T7", "T6"))
NAMES2 = {"Z1": "Image only", "Z2": "+ tools as text", "Z3": "+ agent loop", "Z4": "+ 80 references (setup 30)",
          "T0": "Tools only (boosting)", "T1": "Frozen encoder + ridge", "T2": "Tuned encoder", "T3": "+ tools in head (Stage A)",
          "T4": "Frozen encoder + LLM + tools", "T5": "Tuned encoder + LLM, no tools",
          "T6": "Tuned encoder + LLM + tools (Stage B)", "T7": "Stage B, tools shuffled at test"}


def e12(f: pd.DataFrame, mos, lines: list) -> None:
    stage = [c for c in ("T6", "Stage B seed 2", "Stage B seed 3") if c in f]
    med = [c for c in ("MedSigLIP", "MedSigLIP seed 2", "MedSigLIP seed 3") if c in f]
    if len(stage) < 2 and len(med) < 2:
        return
    lines += [f"## E12 training seeds (Stage B: {', '.join(stage)}; MedSigLIP: {', '.join(med)}; 5-fold mean each; MedSigLIP "
              "seeds 42 / 43 / 44 with Tolga's folds kept)", ""]
    if "Stage B seed 3" not in f:
        lines += ["Stage B seed3 was not run (Colab credits): the pairings below are rerun-42 and seed2-43.", ""]
    rows = {c: {"Overall": overall(f[c], mos), "Overall, 0.2 grid": overall(grid(f[c]), mos)} for c in stage + med}
    lines += [pd.DataFrame(rows).T.round(4).to_markdown(), ""]
    for name, cols in (("Stage B", stage), ("MedSigLIP", med)):
        if len(cols) > 1:
            o = [rows[c]["Overall"] for c in cols]
            lines.append(f"- {name}: Overall {np.mean(o):.4f} ± {np.std(o, ddof=1):.4f} over {len(cols)} seeds.")
    pairs = [(a, b) for a, b in zip(stage, med)]
    if pairs:
        comps = {f"{a} - {b}": compare(f[a].to_numpy(), f[b].to_numpy(), mos) for a, b in pairs}
        lines += ["", "PRIMARY, Stage B - MedSigLIP at each seed pairing:", "",
                  pd.DataFrame({k: {"Δ [95% CI]": fmt(c), "p": c["p"]} for k, c in comps.items()}).T.to_markdown(), "",
                  f"Δ > 0 in {sum(c['delta'] > 0 for c in comps.values())} of {len(comps)} pairings."]
    if len(stage) > 1 and len(med) > 1:
        a, b = np.mean([f[c] for c in stage], 0), np.mean([f[c] for c in med], 0)
        lines.append(f"Secondary, {len(stage)}-seed mean Stage B ({overall(a, mos):.4f}) - {len(med)}-seed mean MedSigLIP "
                     f"({overall(b, mos):.4f}): {fmt(compare(a, b, mos))}.")
    lines.append("")


def table2(f: pd.DataFrame, lab: pd.DataFrame, shuffles: dict, lines: list) -> dict:
    mos = f["mos"].to_numpy()
    comps = {}
    for a, b in FAMILY2:
        if a in f and b in f:
            comps[(a, b)] = compare_pooled(shuffles["all"], f["T6 (shuffle folds)"].to_numpy(), mos) if a == "T7" else \
                compare(f[a].to_numpy(), f[b].to_numpy(), mos)
    adj = holm({k: c["p"] for k, c in comps.items()})
    out = {}
    for row, name in NAMES2.items():
        if row not in f:
            continue
        r = {"Configuration": name, "Overall": np.mean([overall(s, mos) for s in shuffles["all"]]) if row == "T7" else overall(f[row], mos)}
        pair = next(((a, b) for a, b in comps if a == row), None)
        r["Δ vs row above [95% CI]"] = f"{fmt(comps[pair], adj[pair])} (vs {pair[1]})" if pair else "-"
        if row == "T6":
            r["Δ vs row above [95% CI]"] = " ; ".join(f"{fmt(comps[p], adj[p])} (vs {p[1]})" for p in comps if p[0] == "T6") + \
                f" ; {fmt(compare(f['T6'].to_numpy(), f['T6 vision head'].to_numpy(), mos))} (vs its vision head, not in Family 2)"
        r["denoiser τ̄"] = taus(lab, row).mean() if row in lab else np.nan
        out[row] = r
    lines += ["## Table 2: what made our model better (Family 2, Holm over the pairs run; † = Holm p < 0.05)", "",
              pd.DataFrame(out).T.round(4).to_markdown(), ""]
    if "T3" in lab:
        lines.append("T3's τ̄ uses the Stage B rerun's vision head (the same Stage A recipe; test 2.718 vs 2.719, prediction "
                     "corr 0.9996), because the plain Stage A run kept no weights.")
    if "T7" in lab:
        lines.append(f"T7 on the lab set (all tools permuted, seed 0): τ̄ {taus(lab, 'T7').mean():.3f}; T7 - T6 τ̄ "
                     f"{tau_diff(lab, 'T7', 'T6 (shuffle folds)')} (bootstrap over patients; against the same fold models unshuffled, "
                     f"τ̄ {taus(lab, 'T6 (shuffle folds)').mean():.3f}).")
    for a, b in (("T6", "T5"), ("T6", "T4"), ("T3", "T2")):
        if a in lab and b in lab:
            lines.append(f"τ̄ {a} - {b}: {tau_diff(lab, a, b)}")
    lines.append("")
    return comps


def table3(f: pd.DataFrame, shuffles: dict, lines: list) -> None:
    mos = f["mos"].to_numpy()
    rows, comps = {}, {}
    if shuffles:
        for g in tg.STAGE_B_COLUMNS:
            comps[g] = compare_pooled(shuffles[g], f["T6 (shuffle folds)"].to_numpy(), mos)
        adj = holm({g: c["p"] for g, c in comps.items()})
        for g, c in comps.items():
            rows[g] = {"Stage B: Δ shuffled [CI]": fmt(c, adj[g]), "Holm p": adj[g]}
        rows["all (T7)"] = {"Stage B: Δ shuffled [CI]": fmt(compare_pooled(shuffles["all"], f["T6 (shuffle folds)"].to_numpy(), mos)), "Holm p": np.nan}
    agent = I329 / "agent_per_image.csv"
    if agent.exists():
        a = pd.read_csv(agent).fillna({f"cite_{c}": "" for c in tg.CATEGORIES})
        faith = pd.read_csv(I329 / "agent_faithfulness.csv", index_col=0) if (I329 / "agent_faithfulness.csv").exists() else None
        for g in (*tg.STAGE_B_COLUMNS, "visual"):
            r = rows.setdefault(g, {})
            r["Agent: read (%)"] = 100 * (a["visual_read"].mean() if g == "visual" else a["numeric_read"].mean())
            if g != "visual":
                r["Agent: cited (%)"] = 100 * (a[f"cite_{g}"] != "").mean()
                if faith is not None:
                    r["Agent: citation agrees (%)"] = faith.loc[g, "agree %"]
        rows["reads per image"] = {"Agent: read (%)": a["reads"].mean()}
    lines += ["## Table 3: what the models rely on (Family 3: the 4 groups, Holm; † = Holm p < 0.05)", "",
              "Agent reads: one JSON holds all four numeric groups, so their read share is the same number; "
              "'reads per image' counts every file of the image (image, JSON, views).", "",
              pd.DataFrame(rows).T.round(2).to_markdown(), ""]
    if shuffles:
        lines += [f"Shuffles use fold models {shuffles['folds']} (all 5 when finished); every Δ is against the same fold "
                  f"models unshuffled (Overall {overall(f['T6 (shuffle folds)'], mos):.4f}). Check: unshuffled re-prediction "
                  f"vs saved predictions of those folds, max |diff| {shuffles['check: base vs saved T6, max abs diff']:.4f}.", ""]


# ---------------------------------------------------------------- analyses

def e8(f: pd.DataFrame, lines: list) -> None:
    d = I329 / "out_patient"
    if not all((d / f"fold_{k}.npz").exists() for k in range(1, 6)):
        return
    z = np.load(JOB / "train.npz")
    mos = z["mos"].astype(float)

    def oof(directory, key="vision_val"):
        p = np.full(len(mos), np.nan)
        for k in range(1, 6):
            r = np.load(directory / f"fold_{k}.npz")
            p[r["va"]] = r[key]
        return p
    patient, grouped = oof(d), oof(VFT / "llm2/out_llm")
    t_patient = fold_mean(d, "vision_test")
    rows = {"grouped folds (90 base-slice groups), Stage B rerun vision head": {"OOF": overall(grouped, mos), "test": overall(f["T6 vision head"], f["mos"])},
            "pseudo-patient folds (7 clusters), same recipe": {"OOF": overall(patient, mos), "test": overall(t_patient, f["mos"])}}
    for r in rows.values():
        r["OOF - test"] = r["OOF"] - r["test"]
    lines += ["## E8 patient-level grouping check (Stage A recipe = phase 1 only)", "",
              pd.DataFrame(rows).T.round(4).to_markdown(), "",
              "Plain Stage A run for reference: OOF 2.836, test 2.719. Pseudo-patients: scripts/patient_folds.py.", ""]


def e17_pd(lines: list) -> None:
    if not (INFER / "pd.npz").exists():
        return
    z = np.load(INFER / "pd.npz")
    table = pd.DataFrame({g: {p: z[f"{g}_{p}"].mean() for p in range(0, 101, 10)} for g in tg.STAGE_B_COLUMNS})
    table.index.name = "percentile set"
    table.round(4).to_csv(I329 / "fig2a_partial_dependence.csv")
    lines += ["## E17 partial dependence (fold 1; mean predicted MOS over the 300 test images; the prompt prints 100 as 99)", "",
              table.round(3).to_markdown(), "", "Range (max - min) per group: " +
              ", ".join(f"{g} {table[g].max() - table[g].min():.3f}" for g in table), ""]


def body_mask(image: np.ndarray) -> np.ndarray:
    body = ndimage.binary_fill_holes(image > 500 / 1400 * 255)
    labels, n = ndimage.label(body)
    return labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1) if n else body


def e18(lines: list) -> None:
    path = INFER / "occl.npz" if (INFER / "occl.npz").exists() else INFER / "occl.partial.npz"
    if not path.exists():
        return
    z, lab = np.load(path), np.load(JOB / "lab329.npz")
    folds = [k for k in range(1, 6) if f"map_{k}" in z.files]
    base = np.mean([z[f"base_{k}"] for k in folds], 0)
    maps = np.mean([z[f"map_{k}"] for k in folds], 0)
    lines.append(f"Occlusion from fold models {folds}.")
    rows = {}
    for n, name in enumerate(lab["occl_ids"]):
        drop = base[n] - maps[n]
        mask = body_mask(lab["occl_images"][n])
        centres = np.arange(20) * 24 + 24
        inside = mask[np.ix_(centres, centres)]
        pos = np.clip(drop, 0, None)
        rows[str(name)] = {"Stage B score": base[n], "max drop": drop.max(), "max rise": -drop.min(),
                           "share of positive drop inside body": pos[inside].sum() / pos.sum() if pos.sum() > 0 else np.nan,
                           "body share of patches": inside.mean()}
        np.save(I329 / f"occlusion_case{n + 1}.npy", drop)
    lines += ["## E18 occlusion (48 px patches, stride 24, filled with their mean; mean over the fold models listed; drop = base - occluded)", "",
              pd.DataFrame(rows).T.round(4).to_markdown(), ""]


def e19(lines: list) -> None:
    if not (INFER / "enc.npz").exists():
        return
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    z, t, e = np.load(JOB / "train.npz"), np.load(JOB / "test.npz"), np.load(INFER / "enc.npz")
    groups = pd.read_csv(CONFIG.output_dir / "probe/train_groups.csv", dtype={"image_id": str}).set_index("image_id").loc[z["ids"], "group"].to_numpy()
    names = ["residual_std", "gradient_deviation_mean", "hf_residual_mean", "noise_score", "gradient_p95", "streak_anisotropy"]
    test_mos = pd.read_csv(CONFIG.output_dir / "noexp/per_image.csv").set_index("test_id").loc[t["ids"], "mos"].to_numpy()
    targets = {**{n: (z["tools"][:, i], t["tools"][:, i]) for i, n in enumerate(names)}, "MOS": (z["mos"].astype(float), test_mos)}
    rows = {}
    for enc in ("pretrained", "tuned"):
        x, xt = e[f"{enc}_train"].astype(np.float64), e[f"{enc}_test"].astype(np.float64)
        for target, (y, yt) in targets.items():
            def cv(alpha):
                p = np.zeros(len(y))
                for a, b in GroupKFold(5).split(x, y, groups):
                    p[b] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[a], y[a]).predict(x[b])
                return r2_score(y, p)
            alpha = max(np.logspace(-2, 6, 17), key=cv)
            pred = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x, y).predict(xt)
            rows.setdefault(target, {})[f"{enc} test R2"] = r2_score(yt, pred)
    table = pd.DataFrame(rows).T
    table["tuned - pretrained"] = table["tuned test R2"] - table["pretrained test R2"]
    lines += ["## E19 what the tuned encoder learned (ridge probes on mean-pooled encoder tokens; fold-1 tuned encoder)", "",
              table.round(3).to_markdown(), ""]


def e5(lines: list) -> None:
    file = I329 / "medsig_feat.npz"
    if not file.exists():
        return
    from scripts.nr_eval import boot as patient_boot
    from scripts.nr_eval import fit_ridge, slice_tau
    stage = np.load(JOB / "nr_feat.npz", allow_pickle=True)
    med = np.load(file, allow_pickle=True)
    assert list(med["ids"]) == [str(i) for i in stage["ids"]], "id order differs"
    ids = pd.Index([str(i) for i in stage["ids"]])
    tgt = pd.read_csv(JOB / "nr_targets.csv").set_index("image_id")
    train = tgt.index[tgt["split"] == "train"].intersection(ids)
    test = tgt.index[tgt["split"] == "test"].intersection(ids)
    groups = train.str.split("_").str[0]
    feats = {"Stage B features + ridge": pd.DataFrame(stage["feat"].astype(np.float32), index=ids),
             "MedSigLIP features + ridge": pd.DataFrame(med["feat"].astype(np.float32), index=ids)}
    out = []
    for target in ("ssim", "psnr"):
        y_tr, y_te = tgt.loc[train, target].to_numpy(), tgt.loc[test, target]
        preds, rows = {}, {}
        for name, x in feats.items():
            p, alpha = fit_ridge(x.loc[train].to_numpy(), y_tr, groups, x.loc[test].to_numpy())
            preds[name] = pd.Series(p, index=test)
            m = metrics(p, y_te.to_numpy())
            rows[name] = {"Overall": m["overall"], "slice tau": slice_tau(preds[name], y_te).mean(),
                          "MAE": float(np.abs(p - y_te.to_numpy()).mean()), "alpha": alpha}
        a, b = list(preds)
        diff = rows[a]["Overall"] - rows[b]["Overall"]
        out += [f"### {target.upper()}", "", pd.DataFrame(rows).T.round(4).to_markdown(), "",
                f"- Stage B - MedSigLIP features: {diff:+.4f}, 95% CI "
                f"{patient_boot(preds[a], preds[b], y_te, np.random.default_rng(SEED))} (bootstrap over test patients)", ""]
    lines += ["## E5 MedSigLIP control for no-reference SSIM / PSNR (lab IQA; same splits and alpha rule as nr_eval.py)", "",
              f"MedSigLIP = Tolga's {med['ckpt']} (fold 1, as Stage B fold 1); features = FiLM-modulated tokens pooled "
              "as his heads read them (global mean, 4 local means, 2 texture maxima).", "", *out]


def e16b(lines: list) -> None:
    import json
    import zipfile
    archive = PROJECT_ROOT / "colab/e16b_job/e16b_results.zip"
    if not archive.exists():
        return
    key = pd.read_csv(PROJECT_ROOT / "colab/e16b_job/key.csv")
    scores = {}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith("run_1/scores.json"):
                setup = name.split("/")[0]
                cond, folder = setup[:4], setup[5:]
                for r in json.loads(z.read(name))["scores"]:
                    scores[(cond, folder, r["image_id"])] = float(r["score"])
    for cond in ("e16f", "e16n"):
        key[cond] = [scores.get((cond, fo, i), np.nan) for fo, i in zip(key["folder"], key["image_id"])]
    ok = key.dropna(subset=["e16f", "e16n"])
    change = (ok["e16n"] - ok["e16f"]).abs()
    mos = ok["mos"].to_numpy()
    c = compare(ok["e16n"].to_numpy(), ok["e16f"].to_numpy(), mos)
    rows = {cond: {"Overall": overall(ok[col], mos), "MAE": float((ok[col] - ok["mos"]).abs().mean())}
            for cond, col in (("noise tools removed (e16n)", "e16n"), ("all tools, same folders (e16f)", "e16f"),
                              ("original ne19 run (other folders)", "ne19"))}
    lines += ["## E16b counterfactual faithfulness (100 test images, 5 MOS bins of 0.8 x 20; noise group removed)", "",
              f"Scored in both conditions: {len(ok)} of 100. Mean |dscore| {change.mean():.3f}; share with |dscore| > 0.25: "
              f"{(change > 0.25).mean():.2f}; mean signed d (e16n - e16f) {(ok['e16n'] - ok['e16f']).mean():+.3f}.", "",
              pd.DataFrame(rows).T.round(4).to_markdown(), "",
              f"Overall e16n - e16f: {fmt(c)} (paired bootstrap over the images).", ""]
    key.to_csv(I329 / "e16b_per_image.csv", index=False)


def rerun_check(f: pd.DataFrame, lines: list) -> None:
    """The T2/T4/T5 lab cells use retrained weights (the first runs kept none): their test predictions vs the first runs'."""
    rows = {}
    for name, d, key in (("T5", "out_notools", "llm_test"), ("T2", "out_notools", "vision_test"), ("T4", "out_p2only", "llm_test")):
        v = fold_mean(I329 / f"{d}_rerun", key)
        if v is not None and name in f:
            rows[name] = {"first run": overall(f[name], f["mos"]), "rerun": overall(v, f["mos"]),
                          "prediction corr": np.corrcoef(v, f[name])[0, 1]}
    if rows:
        lines += ["Retrained weights behind the T2 / T4 / T5 denoiser τ̄ (same folds and seed; test Overall of the first run, "
                  "which the table reports, vs the rerun):", "", pd.DataFrame(rows).T.round(4).to_markdown(), ""]


def main() -> None:
    f, shuffles = test_predictions()
    lab = lab_frame()
    lines = ["# Issue #329: tables and analyses", "",
             "Protocol and primary comparisons fixed in the docstring of scripts/issue329_eval.py before the new runs; "
             "every Overall is on the 300 official test images, 5-fold mean predictions.", ""]
    table1(f, lab, lines)
    table2(f, lab, shuffles, lines)
    rerun_check(f, lines)
    table3(f, shuffles, lines)
    for part in (e17_pd, e18, e19, e5, e16b):
        part(lines)
    e8(f, lines)
    agent = I329 / "agent_analysis.md"
    if agent.exists():
        lines += ["", agent.read_text(encoding="utf-8").replace("# Zero-shot", "## E14-E16a Zero-shot")]
    f.to_csv(I329 / "test_predictions.csv")
    (I329 / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
