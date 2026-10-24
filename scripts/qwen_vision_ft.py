"""Stage A of the fine-tune: Qwen3.8-27B's vision encoder (0.4B) fine-tuned with a small head that also reads tool
percentiles, on the 1000 training slices, 5 folds grouped by base slice. The VM side is colab/vm/vft_train.py.

The training set is ~100 base slices x 10 degradations; outputs/agentic_ct_iqa/probe/train_groups.csv holds 90 groups
(complete-linkage clusters of heavily smoothed slices; 10 groups join two adjacent base slices). Folds never split a
group. The expert's own folds do (every group spans 3-5 of them), so its out-of-fold score is slightly optimistic.

Gate, fixed before training: the 5-fold out-of-fold overall (final epoch, no epoch selection) must reach 2.85. The frozen
encoder + ridge probe reaches ~2.80 on grouped folds and scored 2.569 on test, so passing it is not enough on its own.
Only if the gate passes is the test set prepared and scored once, with the mean of the 5 fold models.

    python -m scripts.qwen_vision_ft prep     # colab/vft_job/: train.npz, configs; reference scores on the same folds
    python -m scripts.qwen_vision_ft report   # out-of-fold scores from the VM results, gate
"""
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG, PROJECT_ROOT
from src.data import load_ldctiqa_split
from src.experts import expert_scores, load_fold_predictions, strip_ext
from src.metrics import challenge_scores

PROBE = CONFIG.output_dir / "probe"
JOB = PROJECT_ROOT / "colab/vft_job"
OUT = CONFIG.output_dir / "vision_ft"
NOISE = ("residual_std", "gradient_deviation_mean", "hf_residual_mean")  # the noise rule's measures
CUES = ("gradient_p95", "streak_anisotropy")  # structure_pct and streak_pct of setup 30
GATE = 2.85


def percentiles(values: pd.DataFrame, bank: pd.DataFrame) -> pd.DataFrame:
    """Share (0-1) of bank slices with a higher value, per column; no MOS involved."""
    return pd.DataFrame({c: values[c].apply(lambda v, b=np.sort(bank[c].to_numpy()): (len(b) - np.searchsorted(b, v, "right"))
                                            / len(b)) for c in values}, index=values.index)


def tool_inputs(ids: list[str], split: str) -> np.ndarray:
    """Noise percentiles, their mean (the rule's noise score) and the two cue percentiles, relative to the 1000
    training slices."""
    bank = pd.read_csv(CONFIG.output_dir / "screening/train.csv", dtype={"image_id": str}).set_index("image_id")
    table = pd.read_csv(CONFIG.output_dir / f"screening/{split}.csv", dtype={"image_id": str}).set_index("image_id")
    p = percentiles(table.loc[ids, [*NOISE, *CUES]], bank)
    p.insert(3, "noise_score", p[list(NOISE)].mean(axis=1))
    return p.to_numpy(np.float32)


def folds(ids: list[str]) -> np.ndarray:
    groups = pd.read_csv(PROBE / "train_groups.csv", dtype={"image_id": str}).set_index("image_id")["group"]
    fold = np.empty(len(ids), int)
    for k, (_, held) in enumerate(GroupKFold(5, shuffle=True, random_state=CONFIG.seed).split(ids, groups=groups[ids])):
        fold[held] = k + 1
    return fold


def overall(pred, mos) -> float:
    return round(challenge_scores(np.asarray(pred), np.asarray(mos))["overall"], 3)


def references(ids: list[str], mos: np.ndarray, fold: np.ndarray) -> dict[str, float]:
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["train"]
    expert.index = strip_ext(expert.index)
    z = np.load(PROBE / "features.npz")
    assert list(z["ids"]) == ids
    probe = np.empty(len(ids))
    for k in range(1, 6):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 15)))
        probe[fold == k] = model.fit(z["x"][fold != k], mos[fold != k]).predict(z["x"][fold == k])
    return {"expert out-of-fold (its own, ungrouped folds)": overall(expert[ids], mos),
            "frozen encoder + ridge, same grouped folds": overall(probe, mos)}


def prep() -> None:
    samples = load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples], np.float32)
    images = np.stack([np.round(np.clip(s.image, 0, 1) * 255).astype(np.uint8) for s in samples])  # as the sandbox PNGs
    fold = folds(ids)
    JOB.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(JOB / "train.npz", ids=np.array(ids), mos=mos, images=images, tools=tool_inputs(ids, "train"),
                        fold=fold)
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy(PROBE / name, JOB / name)
    print("fold sizes", np.bincount(fold)[1:].tolist())
    for name, value in references(ids, mos, fold).items():
        print(f"{name}: {value}")


def report() -> None:
    oof = pd.read_csv(OUT / "oof.csv", dtype={"image_id": str})
    z = np.load(JOB / "train.npz")
    ids, mos, fold = list(z["ids"]), z["mos"], z["fold"]
    oof = oof.set_index("image_id").loc[ids]
    lines = ["# Stage A: Qwen vision encoder fine-tuned with tool inputs (5 grouped folds, training slices only)", ""]
    for name, value in {**references(ids, mos, fold), "fine-tuned encoder + tools (final epoch)": overall(oof["pred"], mos)}.items():
        lines.append(f"- {name}: {value}")
    per_fold = {k: overall(oof["pred"][fold == k], mos[fold == k]) for k in range(1, 6)}
    score = overall(oof["pred"], mos)
    lines += [f"- per fold: {per_fold}", "",
              f"Gate (fixed before training): {GATE}. {'PASSED' if score >= GATE else 'NOT PASSED'} ({score}).", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def test_prep() -> None:
    """Images and tool percentiles of the 300 test slices, no MOS (it stays local)."""
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels)
    ids = [s.image_id for s in samples]
    images = np.stack([np.round(np.clip(s.image, 0, 1) * 255).astype(np.uint8) for s in samples])
    np.savez_compressed(JOB / "test.npz", ids=np.array(ids), images=images, tools=tool_inputs(ids, "test"))
    print(len(ids), "test slices ->", JOB / "test.npz")


def test() -> None:
    """The test look: mean of the 5 fold models (as MedSigLIP's test score is the mean of its 5 fold models)."""
    from scripts.qwen_probe import bootstrap
    from scripts.make_references import tables, noise_score, neighbours, split
    pred = pd.read_csv(OUT / "test.csv", dtype={"image_id": str}).set_index("image_id")
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    pred = pred.loc[ids]
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    bank, bank_scores, train_mos, _ = split()
    scores = noise_score(tables()[2].loc[ids], bank)
    rule = scores.apply(lambda s: train_mos[neighbours(s, bank_scores)].mean()).to_numpy()
    ours = pred["pred"].to_numpy()
    rows = {"fine-tuned encoder + tools (5-fold mean)": ours, "MedSigLIP (5-fold mean)": expert, "noise rule": rule,
            "50/50 ours + MedSigLIP": (ours + expert) / 2}
    rows.update({f"single fold {k}": pred[f"fold{k}"].to_numpy() for k in range(1, 6)})
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(3)
    lines = ["# Stage A on the 300-image test (single look)", "",
             f"Out-of-fold on train (this run): {overall(pd.read_csv(OUT / 'oof.csv', dtype={'image_id': str}).set_index('image_id').loc[list(np.load(JOB / 'train.npz')['ids']), 'pred'], np.load(JOB / 'train.npz')['mos'])}. "
             f"Gate {GATE} was not passed in the first run (2.837); the test look was the user's decision.", "",
             table.to_markdown(), "",
             f"ours - MedSigLIP: {overall(ours, mos) - overall(expert, mos):+.3f}, 95% CI {bootstrap(ours, expert, mos)}",
             f"ours - rule: {overall(ours, mos) - overall(rule, mos):+.3f}, 95% CI {bootstrap(ours, rule, mos)}",
             f"50/50 - MedSigLIP: {overall((ours + expert) / 2, mos) - overall(expert, mos):+.3f}, 95% CI "
             f"{bootstrap((ours + expert) / 2, expert, mos)}", ""]
    (OUT / "test_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


VARIANTS = OUT / "variants"  # colab/vm/vft_variants.py results: out_<variant>/fold_<k>.npz
STAGE_A_OOF = 2.836  # this run's out-of-fold score; its test score (2.719) was already seen


def variant_predictions(name: str, tag: str, part: str) -> np.ndarray:
    """part "val": out-of-fold predictions on the 1000 training slices; "test": 5-fold mean on the 300."""
    z = np.load(JOB / "train.npz")
    folds_ = [np.load(VARIANTS / f"out_{name}/fold_{k}.npz") for k in range(1, 6)]
    if part == "test":
        return np.mean([f[f"test_{tag}"] for f in folds_], axis=0)
    pred = np.full(len(z["mos"]), np.nan)
    for f in folds_:
        pred[f["va"]] = f[f"val_{tag}"]
    return pred


def select() -> tuple[str, str | None]:
    """Fixed rule: highest out-of-fold overall among Stage A and the four variant scores. Opens no test score."""
    mos = np.load(JOB / "train.npz")["mos"]
    oof = {"stageA": STAGE_A_OOF}
    for name in ("rank_tools", "rank_vision"):
        for tag in ("centre", "tta"):
            oof[f"{name}/{tag}"] = overall(variant_predictions(name, tag, "val"), mos)
    winner = max(oof, key=oof.get)
    counterpart = None if winner == "stageA" else winner.replace("tools", "@").replace("vision", "tools").replace("@", "vision")
    lines = ["# Improvement round: out-of-fold (5 grouped folds, training slices)", "",
             *[f"- {k}: {v}" for k, v in oof.items()], "", f"Winner (fixed rule, highest out-of-fold): {winner}; "
             f"test scores opened only for it and {counterpart}.", ""]
    (OUT / "variants_select.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return winner, counterpart


def final() -> None:
    from scripts.qwen_probe import bootstrap
    winner, counterpart = select()
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    assert list(np.load(JOB / "test.npz")["ids"]) == ids
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    stage_a = pd.read_csv(OUT / "test.csv", dtype={"image_id": str}).set_index("image_id").loc[ids, "pred"].to_numpy()
    rows = {"MedSigLIP": expert, "stageA (seen before)": stage_a}
    for name in (winner, counterpart):
        if name and name != "stageA":
            rows[name] = variant_predictions(*name.split("/"), "test")
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(3)
    lines = ["# Improvement round on the 300-image test", "", table.to_markdown(), ""]
    for name in rows:
        if name != "MedSigLIP":
            lines.append(f"- {name} - MedSigLIP: {overall(rows[name], mos) - overall(expert, mos):+.3f}, "
                         f"95% CI {bootstrap(rows[name], expert, mos)}")
    if counterpart:
        a, b = (winner, counterpart) if "tools" in winner else (counterpart, winner)
        lines.append(f"- tools effect ({a} - {b}): {overall(rows[a], mos) - overall(rows[b], mos):+.3f}, "
                     f"95% CI {bootstrap(rows[a], rows[b], mos)}")
    (OUT / "variants_test.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


LLM = OUT / "llm"  # colab/vm/vftb_train.py results: out_llm/fold_<k>.npz
LLM_CANDIDATES = ("vision", "llm", "mean")  # mean = (vision + llm) / 2


def llm_predictions(name: str, part: str, run: str = "llm/out_llm") -> np.ndarray:
    """Stage B, part "val": out-of-fold on the 1000 training slices; "test": 5-fold mean on the 300."""
    folds_ = [np.load(OUT / run / f"fold_{k}.npz") for k in range(1, 6)]
    if name == "mean":
        return (llm_predictions("vision", part, run) + llm_predictions("llm", part, run)) / 2
    if part == "test":
        return np.mean([f[f"{name}_test"] for f in folds_], axis=0)
    pred = np.full(len(np.load(JOB / "train.npz")["mos"]), np.nan)
    for f in folds_:
        pred[f["va"]] = f[f"{name}_val"]
    return pred


def llm_select() -> str:
    """Fixed rule, set before training: highest out-of-fold overall among vision, llm and their mean. Opens no test."""
    mos = np.load(JOB / "train.npz")["mos"]
    fold = np.load(JOB / "train.npz")["fold"]
    oof = {name: overall(llm_predictions(name, "val"), mos) for name in LLM_CANDIDATES}
    winner = max(oof, key=oof.get)
    per_fold = {name: [overall(llm_predictions(name, "val")[fold == k], mos[fold == k]) for k in range(1, 6)]
                for name in LLM_CANDIDATES}
    lines = ["# Stage B: fine-tuned eye + Qwen3.8-27B language model (LoRA), out-of-fold (5 grouped folds)", "",
             f"Stage A out-of-fold for reference: {STAGE_A_OOF}.", "",
             *[f"- {k}: {v} (folds {per_fold[k]})" for k, v in oof.items()], "",
             f"Winner (fixed rule, highest out-of-fold): {winner}.", ""]
    (OUT / "llm_select.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return winner


def llm_final() -> None:
    """The test look: all three candidates are reported, the winner is the one chosen on out-of-fold."""
    from scripts.qwen_probe import bootstrap
    winner = llm_select()
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    assert list(np.load(JOB / "test.npz")["ids"]) == ids
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    stage_a = pd.read_csv(OUT / "test.csv", dtype={"image_id": str}).set_index("image_id").loc[ids, "pred"].to_numpy()
    rows = {"MedSigLIP": expert, "stageA (seen before)": stage_a}
    rows.update({f"{name}{' (WINNER)' if name == winner else ''}": llm_predictions(name, "test") for name in LLM_CANDIDATES})
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(3)
    lines = ["# Stage B on the 300-image test", "", table.to_markdown(), ""]
    for name in rows:
        if name != "MedSigLIP":
            lines.append(f"- {name} - MedSigLIP: {overall(rows[name], mos) - overall(expert, mos):+.3f}, "
                         f"95% CI {bootstrap(rows[name], expert, mos)}")
    llm, vision = llm_predictions("llm", "test"), llm_predictions("vision", "test")
    lines.append(f"- language model effect (llm - vision): {overall(llm, mos) - overall(vision, mos):+.3f}, "
                 f"95% CI {bootstrap(llm, vision, mos)}")
    (OUT / "llm_test.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


FULL = OUT / "full"  # colab/vm/vftb_train.py with full_data: out_full/seed_<s>.npz
SEEDS = 3


def full_final() -> None:
    """The Stage B recipe trained on all 1000 training slices, 3 seeds. Primary, fixed before the run: llm, mean over
    the seeds (chosen after Stage B's test look, where llm scored best; there is no out-of-fold score here)."""
    from scripts.qwen_probe import bootstrap
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    assert list(np.load(JOB / "test.npz")["ids"]) == ids
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    seeds = [np.load(FULL / f"out_full/seed_{s}.npz") for s in range(1, SEEDS + 1)]
    full = {name: np.mean([z[f"{name}_test"] for z in seeds], axis=0) for name in ("vision", "llm")}
    full["mean"] = (full["vision"] + full["llm"]) / 2
    stage_b = llm_predictions("llm", "test")
    rows = {"MedSigLIP (our 5-fold predictions)": expert, "Stage B llm, 5 folds (seen before)": stage_b,
            "full data llm, 3 seeds (PRIMARY)": full["llm"], "full data vision, 3 seeds": full["vision"],
            "full data mean": full["mean"], "full llm + Stage B llm": (full["llm"] + stage_b) / 2}
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(3)
    lines = ["# Full-data Stage B (1000 training slices, 3 seeds) on the 300-image test", "", table.to_markdown(), "",
             f"Single seeds, llm: {[overall(z['llm_test'], mos) for z in seeds]}; "
             f"vision: {[overall(z['vision_test'], mos) for z in seeds]}", ""]
    for name, pred in rows.items():
        if not name.startswith("MedSigLIP"):
            lines.append(f"- {name} - MedSigLIP: {overall(pred, mos) - overall(expert, mos):+.3f}, "
                         f"95% CI {bootstrap(pred, expert, mos)}")
    lines.append(f"- full llm - Stage B llm: {overall(full['llm'], mos) - overall(stage_b, mos):+.3f}, "
                 f"95% CI {bootstrap(full['llm'], stage_b, mos)}")
    lines.append("- Tolga's latest MedSigLIP test score: 2.7453 (per-image predictions not available here, no paired test).")
    (OUT / "full_test.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


UP2 = "up2/out_up2"  # colab/vm/vftb_train.py with upscale2: every crop enlarged 2x before the encoder


def up2_final() -> None:
    """Stage B with 2x-enlarged input. Primary, fixed before the run: llm, 5-fold mean, against Stage B llm (same
    recipe at 1x) and MedSigLIP. Also Tolga's protocol (mean of the 5 single-fold scores)."""
    from scripts.qwen_probe import bootstrap
    z = np.load(JOB / "train.npz")
    oof = {name: overall(llm_predictions(name, "val", UP2), z["mos"]) for name in LLM_CANDIDATES}
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    assert list(np.load(JOB / "test.npz")["ids"]) == ids
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    stage_b = llm_predictions("llm", "test")
    rows = {"MedSigLIP (5-fold ensemble)": expert, "Stage B llm 1x (seen before)": stage_b,
            **{f"2x {name}{' (PRIMARY)' if name == 'llm' else ''}": llm_predictions(name, "test", UP2) for name in LLM_CANDIDATES},
            "2x llm + 1x llm": (llm_predictions("llm", "test", UP2) + stage_b) / 2}
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(3)
    singles = {name: [overall(np.load(OUT / UP2 / f"fold_{k}.npz")[f"{name}_test"], mos) for k in range(1, 6)]
               for name in ("vision", "llm")}
    lines = ["# Stage B with 2x-enlarged input (5 grouped folds) on the 300-image test", "",
             f"Out-of-fold: {oof} (Stage B 1x: vision 2.835, llm 2.823, mean 2.832).", "", table.to_markdown(), "",
             f"Single folds 2x llm: {singles['llm']} (mean of folds {np.mean(singles['llm']):.4f}); 2x vision: "
             f"{singles['vision']}. Tolga's protocol for MedSigLIP: mean of folds 2.6986, best fold 2.7453; "
             f"Stage B 1x llm: mean of folds 2.7252, best 2.7327.", ""]
    for name, pred in rows.items():
        if not name.startswith("MedSigLIP"):
            lines.append(f"- {name} - MedSigLIP: {overall(pred, mos) - overall(expert, mos):+.3f}, "
                         f"95% CI {bootstrap(pred, expert, mos)}")
    up = llm_predictions("llm", "test", UP2)
    lines.append(f"- 2x llm - 1x llm: {overall(up, mos) - overall(stage_b, mos):+.3f}, 95% CI {bootstrap(up, stage_b, mos)}")
    (OUT / "up2_test.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


SYN = "syn/out_syn"  # colab/vm/vftb_train.py with synth + windows3: rank pre-training on MSD slices, 3-window input


def syn_final() -> None:
    """Stage B after rank pre-training on synthetic degradations of 61 other patients' slices, with the soft-tissue and
    narrow windows as extra input channels. Primary, fixed before the run (2026-10-20): llm, 5-fold mean, against
    MedSigLIP and Stage B 1x llm. Also Tolga's protocol (mean of the 5 single-fold scores)."""
    from scripts.qwen_probe import bootstrap
    z = np.load(JOB / "train.npz")
    oof = {name: overall(llm_predictions(name, "val", SYN), z["mos"]) for name in LLM_CANDIDATES}
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    assert list(np.load(JOB / "test.npz")["ids"]) == ids
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    stage_b = llm_predictions("llm", "test")
    syn = llm_predictions("llm", "test", SYN)
    rows = {"MedSigLIP (5-fold ensemble)": expert, "Stage B llm (seen before)": stage_b,
            **{f"synth {name}{' (PRIMARY)' if name == 'llm' else ''}": llm_predictions(name, "test", SYN)
               for name in LLM_CANDIDATES},
            "synth llm + Stage B llm": (syn + stage_b) / 2}
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(3)
    singles = {name: [overall(np.load(OUT / SYN / f"fold_{k}.npz")[f"{name}_test"], mos) for k in range(1, 6)]
               for name in ("vision", "llm")}
    lines = ["# Stage B + synthetic rank pre-training + 3 windows (5 grouped folds) on the 300-image test", "",
             f"Out-of-fold: {oof} (Stage B: vision 2.835, llm 2.823, mean 2.832).", "", table.to_markdown(), "",
             f"Single folds synth llm: {singles['llm']} (mean of folds {np.mean(singles['llm']):.4f}); synth vision: "
             f"{singles['vision']} (mean {np.mean(singles['vision']):.4f}). Tolga's protocol for MedSigLIP: mean of folds "
             f"2.6986, best fold 2.7453; Stage B llm: mean of folds 2.7252, best 2.7327.", ""]
    for name, pred in rows.items():
        if not name.startswith("MedSigLIP"):
            lines.append(f"- {name} - MedSigLIP: {overall(pred, mos) - overall(expert, mos):+.3f}, "
                         f"95% CI {bootstrap(pred, expert, mos)}")
    lines.append(f"- synth llm - Stage B llm: {overall(syn, mos) - overall(stage_b, mos):+.3f}, "
                 f"95% CI {bootstrap(syn, stage_b, mos)}")
    vision_b = llm_predictions("vision", "test")
    vision_s = llm_predictions("vision", "test", SYN)
    lines.append(f"- synth vision - Stage B vision: {overall(vision_s, mos) - overall(vision_b, mos):+.3f}, "
                 f"95% CI {bootstrap(vision_s, vision_b, mos)}")
    (OUT / "syn_test.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


RANKED = "rank/out_rank"  # colab/vm/vftb_train.py with rank: MSE + soft RankNet term, test-time augmentation
WINNER = {"overall": 2.7427, "plcc": 0.9491, "srocc": 0.9495, "krocc": 0.8440}  # agaldran, Lee et al. 2025, Table 3


def rank_final() -> None:
    """Stage B with a soft pairwise ranking term in both phases and 6-view test-time augmentation. Primary, fixed
    before the run (2026-10-20): (ranked llm_tta 5-fold mean + Stage B llm) / 2, against Stage B llm and MedSigLIP,
    and its KROCC against the challenge winner's 0.844. Secondary: ranked llm_tta alone - Stage B llm (the ranking
    term + TTA), and llm_tta - llm within the ranked run (TTA alone)."""
    from scripts.qwen_probe import bootstrap
    z = np.load(JOB / "train.npz")
    names = ("vision", "vision_tta", "llm", "llm_tta")
    oof = {name: overall(llm_predictions(name, "val", RANKED), z["mos"]) for name in names}
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    assert list(np.load(JOB / "test.npz")["ids"]) == ids
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    stage_b = llm_predictions("llm", "test")
    ranked = {name: llm_predictions(name, "test", RANKED) for name in names}
    primary = (ranked["llm_tta"] + stage_b) / 2
    rows = {"MedSigLIP (5-fold ensemble)": expert, "Stage B llm (seen before)": stage_b,
            **{f"ranked {name}": pred for name, pred in ranked.items()},
            "ranked llm_tta + Stage B llm (PRIMARY)": primary}
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(4)
    table.loc["challenge winner agaldran (reported)"] = [WINNER[c] for c in table.columns]
    singles = [overall(np.load(OUT / RANKED / f"fold_{k}.npz")["llm_tta_test"], mos) for k in range(1, 6)]
    lines = ["# Stage B + ranking loss + TTA (5 grouped folds) on the 300-image test", "",
             f"Out-of-fold: {oof} (Stage B: vision 2.835, llm 2.823).", "", table.to_markdown(), "",
             f"Single folds ranked llm_tta: {singles} (mean of folds {np.mean(singles):.4f}); Stage B llm: mean of folds "
             f"2.7252, best 2.7327. Tolga's protocol for MedSigLIP: mean of folds 2.6986, best fold 2.7453.", ""]
    for name, pred in rows.items():
        if not name.startswith("MedSigLIP"):
            lines.append(f"- {name} - MedSigLIP: {overall(pred, mos) - overall(expert, mos):+.3f}, "
                         f"95% CI {bootstrap(pred, expert, mos)}")
    for label, a, b in (("PRIMARY - Stage B llm", primary, stage_b),
                        ("ranked llm_tta - Stage B llm", ranked["llm_tta"], stage_b),
                        ("TTA alone: ranked llm_tta - ranked llm", ranked["llm_tta"], ranked["llm"])):
        lines.append(f"- {label}: {overall(a, mos) - overall(b, mos):+.3f}, 95% CI {bootstrap(a, b, mos)}")
    (OUT / "rank_test.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


RERUN = "llm2/out_llm"  # colab/vm/vftb_train.py, no flags: Stage B repeated with the same seeds, weights kept


def rerun_final() -> None:
    """Stage B repeated (same code path, seeds and GPU type) to obtain its weights. Primary, fixed before the run
    (2026-10-20): rerun llm 5-fold mean, against the original Stage B llm and MedSigLIP; also the two runs averaged."""
    from scripts.qwen_probe import bootstrap
    z = np.load(JOB / "train.npz")
    oof = {name: overall(llm_predictions(name, "val", RERUN), z["mos"]) for name in ("vision", "llm")}
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    assert list(np.load(JOB / "test.npz")["ids"]) == ids
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    stage_b, rerun = llm_predictions("llm", "test"), llm_predictions("llm", "test", RERUN)
    rows = {"MedSigLIP (5-fold ensemble)": expert, "Stage B llm (original, seen before)": stage_b,
            "rerun llm (PRIMARY)": rerun, "rerun vision": llm_predictions("vision", "test", RERUN),
            "rerun llm + original llm": (rerun + stage_b) / 2}
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(4)
    singles = [overall(np.load(OUT / RERUN / f"fold_{k}.npz")["llm_test"], mos) for k in range(1, 6)]
    lines = ["# Stage B repeated with the same seeds (weights kept) on the 300-image test", "",
             f"Out-of-fold: {oof} (original: vision 2.835, llm 2.823). Correlation of the test predictions with the "
             f"original llm: {np.corrcoef(rerun, stage_b)[0, 1]:.4f}.", "", table.to_markdown(), "",
             f"Single folds rerun llm: {singles} (mean {np.mean(singles):.4f}); original: mean of folds 2.7252.", ""]
    for name, pred in rows.items():
        if not name.startswith("MedSigLIP"):
            lines.append(f"- {name} - MedSigLIP: {overall(pred, mos) - overall(expert, mos):+.3f}, "
                         f"95% CI {bootstrap(pred, expert, mos)}")
    lines.append(f"- rerun llm - original llm: {overall(rerun, mos) - overall(stage_b, mos):+.3f}, "
                 f"95% CI {bootstrap(rerun, stage_b, mos)}")
    (OUT / "rerun_test.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


STUDENT = "pseudo/out_pseudo"  # colab/vm/vftb_train.py with pseudo: teacher-student on 3632 unlabelled lab images


def student_final() -> None:
    """Stage B student trained on its fold's 800 slices plus as many pool images per epoch (scripts/pseudo_pool.py,
    137 other patients) labelled by 0.75 Stage B fold k + 0.25 MedSigLIP (scripts/teacher_labels.py). Primary, fixed
    before the run (2026-10-20): student llm, 5-fold mean, against the Stage B rerun llm (its teacher, same seeds) and
    MedSigLIP. Secondary: student vision vs rerun vision; student llm + rerun llm averaged."""
    from scripts.qwen_probe import bootstrap
    z = np.load(JOB / "train.npz")
    oof = {name: overall(llm_predictions(name, "val", STUDENT), z["mos"]) for name in ("vision", "llm")}
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    assert list(np.load(JOB / "test.npz")["ids"]) == ids
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    rerun, student = llm_predictions("llm", "test", RERUN), llm_predictions("llm", "test", STUDENT)
    rows = {"MedSigLIP (5-fold ensemble)": expert, "Stage B llm rerun (teacher, seen before)": rerun,
            "student llm (PRIMARY)": student, "student vision": llm_predictions("vision", "test", STUDENT),
            "rerun vision (seen before)": llm_predictions("vision", "test", RERUN), "student llm + rerun llm": (student + rerun) / 2}
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(4)
    table.loc["challenge winner agaldran (reported)"] = [WINNER[c] for c in table.columns]
    singles = [overall(np.load(OUT / STUDENT / f"fold_{k}.npz")["llm_test"], mos) for k in range(1, 6)]
    lines = ["# Teacher-student Stage B (unlabelled lab images, 137 patients) on the 300-image test", "",
             f"Out-of-fold: {oof} (rerun: vision 2.835, llm 2.822).", "", table.to_markdown(), "",
             f"Single folds student llm: {singles} (mean {np.mean(singles):.4f}); rerun: 2.755/2.721/2.708/2.716/2.725.", ""]
    for name, pred in rows.items():
        if not name.startswith("MedSigLIP"):
            lines.append(f"- {name} - MedSigLIP: {overall(pred, mos) - overall(expert, mos):+.3f}, "
                         f"95% CI {bootstrap(pred, expert, mos)}")
    for label, a, b in (("PRIMARY student llm - rerun llm", student, rerun),
                        ("student vision - rerun vision", rows["student vision"], rows["rerun vision (seen before)"])):
        lines.append(f"- {label}: {overall(a, mos) - overall(b, mos):+.3f}, 95% CI {bootstrap(a, b, mos)}")
    (OUT / "student_test.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


CNN = "cnn/out_cnn"  # colab/vm/cnn_train.py: ResNeXt-50, full 512 slice, image only, same 5 grouped folds


def cnn_final() -> None:
    """Stage B (rerun llm) fused with a CNN view. Fixed before the run (2026-10-21): fused = (1 - w) z(Stage B) +
    w z(CNN), z within each set (no labels), w in {0, 0.25, 0.5} chosen by the highest out-of-fold overall on the 1000
    training slices; PRIMARY = the fused test prediction with that w, against Stage B rerun llm and MedSigLIP; the
    challenge winner's 2.7427 for reference. Secondary: the CNN alone, and every w (reported, not chosen on test)."""
    from scripts.qwen_probe import bootstrap
    z = np.load(JOB / "train.npz")
    zs = lambda x: (x - x.mean()) / x.std()
    fuse = lambda b, c, w: (1 - w) * zs(b) + w * zs(c)
    b_oof, c_oof = llm_predictions("llm", "val", RERUN), llm_predictions("cnn", "val", CNN)
    oof = {w: overall(fuse(b_oof, c_oof, w), z["mos"]) for w in (0.0, 0.25, 0.5)}
    w_star = max(oof, key=oof.get)
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    assert list(np.load(JOB / "test.npz")["ids"]) == ids
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    b_test, c_test = llm_predictions("llm", "test", RERUN), llm_predictions("cnn", "test", CNN)
    primary = fuse(b_test, c_test, w_star)
    rows = {"MedSigLIP (5-fold ensemble)": expert, "Stage B llm rerun (seen before)": b_test, "CNN alone": c_test,
            f"Stage B + CNN, w = {w_star} (PRIMARY)": primary,
            **{f"Stage B + CNN, w = {w} (secondary)": fuse(b_test, c_test, w) for w in (0.25, 0.5) if w != w_star}}
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(4)
    table.loc["challenge winner agaldran (reported)"] = [WINNER[c] for c in table.columns]
    singles = [overall(np.load(OUT / CNN / f"fold_{k}.npz")["cnn_test"], mos) for k in range(1, 6)]
    lines = ["# Stage B + CNN view on the 300-image test", "",
             f"Out-of-fold: CNN alone {overall(c_oof, z['mos'])}; fused by w {oof} -> w = {w_star} (fixed rule).", "",
             table.to_markdown(), "", f"Single folds CNN: {singles}. Test correlation CNN vs Stage B: {np.corrcoef(c_test, b_test)[0, 1]:.4f}.", ""]
    for name, pred in rows.items():
        if not name.startswith("MedSigLIP"):
            lines.append(f"- {name} - MedSigLIP: {overall(pred, mos) - overall(expert, mos):+.3f}, "
                         f"95% CI {bootstrap(pred, expert, mos)}")
    lines.append(f"- PRIMARY - Stage B llm rerun: {overall(primary, mos) - overall(b_test, mos):+.3f}, "
                 f"95% CI {bootstrap(primary, b_test, mos)}")
    (OUT / "cnn_test.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


GRID = 0.2  # MOS is the mean of 5 integer ratings (0-4): 21 levels, 0.2 apart


def on_grid(pred: np.ndarray, step: float = GRID) -> np.ndarray:
    return np.clip(np.round(pred / step) * step, 0, 4)


def grid_final() -> None:
    """Predictions put on the MOS grid. KROCC is tau-b: pairs tied in MOS earn nothing for a continuous prediction but
    still count in its denominator; a prediction on the same 0.2 grid ties them too. Out of fold (1000 training slices)
    the rerun llm goes 2.822 -> 2.838 (KROCC 0.875 -> 0.893), in every fold of three runs (+0.006 to +0.021). Fixed
    before the test look (2026-10-21), step chosen a priori as the label grid: PRIMARY = on_grid(rerun llm 5-fold
    mean), against the rerun llm as is and MedSigLIP (as is, and on the grid for a like-for-like comparison); the
    challenge winner's 2.7427 for reference. Secondary: step 0.1; rerun + original llm on the grid; single folds."""
    from scripts.qwen_probe import bootstrap
    z = np.load(JOB / "train.npz")
    b_oof = llm_predictions("llm", "val", RERUN)
    oof = {step: round(overall(on_grid(b_oof, step) if step else b_oof, z["mos"]), 4) for step in (0, 0.1, 0.2)}
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    ids, mos = [s.image_id for s in samples], np.array([s.mos for s in samples])
    assert list(np.load(JOB / "test.npz")["ids"]) == ids
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    expert.index = strip_ext(expert.index)
    expert = expert.reindex(ids).to_numpy()
    rerun, stage_b = llm_predictions("llm", "test", RERUN), llm_predictions("llm", "test")
    primary = on_grid(rerun)
    rows = {"MedSigLIP (5-fold ensemble)": expert, "MedSigLIP on the grid": on_grid(expert),
            "Stage B llm rerun (seen before)": rerun, "rerun llm on the 0.2 grid (PRIMARY)": primary,
            "rerun llm on a 0.1 grid": on_grid(rerun, 0.1), "rerun + original llm on the 0.2 grid": on_grid((rerun + stage_b) / 2)}
    table = pd.DataFrame({n: challenge_scores(p, mos) for n, p in rows.items()}).T.round(4)
    table.loc["challenge winner agaldran (reported)"] = [WINNER[c] for c in table.columns]
    folds = [np.load(OUT / RERUN / f"fold_{k}.npz")["llm_test"] for k in range(1, 6)]
    singles = [round(overall(on_grid(f), mos), 4) for f in folds]
    lines = ["# Stage B predictions on the MOS grid (0.2) on the 300-image test", "",
             f"Out-of-fold rerun llm by grid step (0 = as is): {oof}. Test MOS levels: {len(np.unique(mos))}.", "",
             table.to_markdown(), "",
             f"Single folds on the grid: {singles} (mean {np.mean(singles):.4f}); as is: 2.755/2.721/2.708/2.716/2.725 (mean 2.7250).", ""]
    for name, pred in rows.items():
        if not name.startswith("MedSigLIP"):
            lines.append(f"- {name} - MedSigLIP: {overall(pred, mos) - overall(expert, mos):+.3f}, "
                         f"95% CI {bootstrap(pred, expert, mos)}")
    for label, a, b in (("PRIMARY - rerun llm as is", primary, rerun),
                        ("PRIMARY - MedSigLIP on the grid", primary, rows["MedSigLIP on the grid"])):
        lines.append(f"- {label}: {overall(a, mos) - overall(b, mos):+.3f}, 95% CI {bootstrap(a, b, mos)}")
    (OUT / "grid_test.md").write_text("\n".join(lines + [""]), encoding="utf-8")
    print("\n".join(lines))


def prep16() -> None:
    """train16.npz / test16.npz: the same slices, ids, tools and folds as train.npz / test.npz, images as uint16 of the
    stored [0, 1] values (width 1400 / level -300), so the radiologists' soft-tissue window (350 / 40, a quarter of the
    range) keeps its grey levels. No test MOS."""
    for split, images_dir, labels in (("train", CONFIG.train_images, CONFIG.train_labels),
                                      ("test", CONFIG.test_images, CONFIG.test_labels)):
        old = np.load(JOB / f"{split}.npz")
        samples = {s.image_id: s for s in load_ldctiqa_split(images_dir, labels)}
        ids = list(old["ids"])
        images = np.stack([np.round(np.clip(samples[i].image, 0, 1) * 65535).astype(np.uint16) for i in ids])
        assert np.abs(images[:20] / 257.0 - old["images"][:20]).max() <= 0.51  # the same slices as the uint8 file
        np.savez_compressed(JOB / f"{split}16.npz", **{k: old[k] for k in old.files if k != "images"}, images=images)
        print(split, len(ids), "->", JOB / f"{split}16.npz")


if __name__ == "__main__":
    {"prep": prep, "report": report, "test-prep": test_prep, "test": test, "select": select, "final": final,
     "llm-select": llm_select, "llm-final": llm_final, "full-final": full_final,
     "up2-final": up2_final, "prep16": prep16, "syn-final": syn_final, "rank-final": rank_final, "rerun-final": rerun_final, "student-final": student_final, "cnn-final": cnn_final, "grid-final": grid_final}[sys.argv[1]]()
