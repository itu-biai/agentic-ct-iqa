"""Setup 29 sandbox: setup 26 (setup2b: images, tools, views, 10 rated examples) without the expert score, plus a
reference_match tool that places each image among rated training slices by a fixed, unlearned noise rule.

Noise score: the mean of three percentiles, one per noise measurement (residual_std, gradient_deviation_mean,
hf_residual_mean: the three measurements most correlated with MOS on the training set, tool_screening.md), each
the share of reference slices noisier than the image on that measurement; 0 = noisiest, 100 = cleanest. Reference
bank: the 900 training slices outside the dev/calibration split. No weights are fitted: the training set only fixes
which measurements are used and supplies the rated neighbours.

For each image the tool reports the noise score, the MOS range and mean of the 10 bank slices with the closest
noise score, and four of them as PNGs in references/ (the two closest noisier and the two closest cleaner ones).

    python -m scripts.make_references check   # the rule on the 100 held-out calibration slices (no test images)
    python -m scripts.make_references build   # experiment_sandboxes/setup4_tools_references + zip for the VM
    python -m scripts.make_references guided  # setup 30: setup4b_tools_references_guided (blind spots in TASK.md)
    python -m scripts.make_references test300 # setup 30 on the 300-image test: test300_guided/batch_XX + zips
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_sandboxes import EXAMPLES, KEY, ROOT, isolate, png
from src.config import CONFIG, PROJECT_ROOT
from src.data import load_ldctiqa_split
from src.experts import expert_scores, load_fold_predictions, strip_ext
from src.metrics import challenge_scores

MEASURES = ("residual_std", "gradient_deviation_mean", "hf_residual_mean")  # all: higher = noisier
NEIGHBOURS = 10
SANDBOX = ROOT / "setup4_tools_references"
ZIP = PROJECT_ROOT / "colab/refs_job/sandboxes.zip"
REF_KEY = CONFIG.output_dir / "sandbox/references_key.csv"

TOOL_DOC = ("- `reference_match`: Places the image among 900 rated training slices of this dataset by a fixed noise "
            "rule. `noise_score` (0 to 100, 100 = cleanest) is the mean, over three noise measurements (denoiser "
            "residual std, gradient deviation, high-frequency residual), of the share of training slices that are "
            "noisier than this image. `neighbour_mos_mean`, `neighbour_mos_min` and `neighbour_mos_max` describe the "
            f"radiologist scores (MOS) of the {NEIGHBOURS} training slices with the closest noise score. Four of those "
            "slices are in `references/` (`img_XX_ref1.png` ... `img_XX_ref4.png`); the `text` of this entry gives "
            "each one's MOS and noise score. The rule measures noise only: it does not see streaks, blur or anatomy.")
TASK_SECTION = """## Matched references

`reference_match` in each `tool_results/img_XX.json` places the image among rated training slices with similar
noise, and `references/` holds four of those slices for each image (`img_XX_ref1.png` ... `img_XX_ref4.png`), with
their scores in the entry's `text`. Compare the image with its references: they are rated slices of the same kind,
not among the 20 images you rate. The match uses noise only; judge everything else yourself.
"""


def tables() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train = pd.read_csv(CONFIG.output_dir / "screening/train.csv", dtype={"image_id": str}).set_index("image_id")
    test = pd.read_csv(CONFIG.output_dir / "screening/test.csv", dtype={"image_id": str}).set_index("image_id")
    return train[list(MEASURES)], train["mos"], test[list(MEASURES)]


def noise_score(values: pd.DataFrame, bank: pd.DataFrame) -> pd.Series:
    """Mean over MEASURES of the share (in %) of bank slices that are noisier than each row."""
    shares = [values[m].apply(lambda v, b=np.sort(bank[m].to_numpy()): 100 * (len(b) - np.searchsorted(b, v, "right"))
                              / len(b)) for m in MEASURES]
    return pd.concat(shares, axis=1).mean(axis=1).rename("noise_score")


def neighbours(score: float, bank_scores: pd.Series, k: int = NEIGHBOURS) -> pd.Index:
    return (bank_scores - score).abs().sort_values(kind="stable").index[:k]


def split() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    values, mos, test = tables()
    held_out = (CONFIG.splits_dir / "calibration.txt").read_text().split()
    bank = values.drop(index=held_out)
    bank_scores = noise_score(bank, bank)
    return bank, bank_scores, mos, held_out


def check() -> None:
    """Calibration slices are outside the bank: the rule and the neighbour MOS on 100 unseen rated slices, next to the
    expert's out-of-fold score on the same slices."""
    bank, bank_scores, mos, held_out = split()
    values = tables()[0].loc[held_out]
    scores = noise_score(values, bank)
    estimate = scores.apply(lambda s: mos[neighbours(s, bank_scores)].mean())
    expert = expert_scores(load_fold_predictions(CONFIG.expert_dir))["train"]
    expert.index = strip_ext(expert.index)
    target = mos[held_out].to_numpy()
    for name, pred in {"noise score": scores, "neighbour MOS mean": estimate, "expert (out-of-fold)": expert[held_out]}.items():
        s = challenge_scores(pred.to_numpy(), target)
        print(f"{name:22s}", {k: round(v, 3) for k, v in s.items()})


def add_references(folder: Path, key: pd.DataFrame, bank: pd.DataFrame, bank_scores: pd.Series, mos: pd.Series,
                   test: pd.DataFrame, train_images: dict) -> list[dict]:
    """references/ and the reference_match entry (in place of consult_experts) for each key row (image_id, test_id)
    of a sandbox copied from setup2b; returns one row per reference written."""
    (folder / "references").mkdir()
    rows = []
    for row in key.itertuples():
        score = float(noise_score(test.loc[[row.test_id]], bank).iloc[0])
        near = neighbours(score, bank_scores)
        noisier = [i for i in near if bank_scores[i] <= score][:2]
        cleaner = [i for i in near if bank_scores[i] > score][:2]
        picked = sorted(noisier + cleaner, key=lambda i: bank_scores[i])
        picked += [i for i in near if i not in picked][:4 - len(picked)]  # image at either end of the bank
        lines = []
        for r, ref in enumerate(picked, 1):
            name = f"{row.image_id}_ref{r}"
            (folder / "references" / f"{name}.png").write_bytes(png(train_images[ref]))
            side = "noisier" if bank_scores[ref] <= score else "cleaner"
            lines.append(f"{name}.png: MOS {mos[ref]:.2f}, noise score {bank_scores[ref]:.1f} ({side} than this image)")
            rows.append({"image_id": row.image_id, "reference": name, "train_id": ref, "mos": round(mos[ref], 3),
                         "noise_score": round(bank_scores[ref], 1)})
        path = folder / "tool_results" / f"{row.image_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["tools"].pop("consult_experts")
        record["tools"]["reference_match"] = {
            "values": {"noise_score": round(score, 1), "neighbour_mos_mean": round(float(mos[near].mean()), 2),
                       "neighbour_mos_min": round(float(mos[near].min()), 2),
                       "neighbour_mos_max": round(float(mos[near].max()), 2)},
            "text": "\n".join(lines)}
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return rows


def build() -> None:
    if SANDBOX.exists():
        raise SystemExit(f"{SANDBOX} already exists; delete it first to rebuild")
    bank, bank_scores, mos, _ = split()
    test = tables()[2]
    key = pd.read_csv(KEY)
    train_images = {s.image_id: s.image for s in load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels)}
    shutil.copytree(EXAMPLES, SANDBOX, ignore=shutil.ignore_patterns(".git", "scores.json"))
    rows = add_references(SANDBOX, key, bank, bank_scores, mos, test, train_images)
    tools = (SANDBOX / "TOOLS.md").read_text(encoding="utf-8").splitlines()
    expert_line = [i for i, line in enumerate(tools) if line.startswith("- `consult_experts`")]
    assert len(expert_line) == 1
    tools[expert_line[0]] = TOOL_DOC
    (SANDBOX / "TOOLS.md").write_text("\n".join(tools) + "\n", encoding="utf-8")
    task = (SANDBOX / "TASK.md").read_text(encoding="utf-8")
    assert task.count("\n## Rules") == 1
    (SANDBOX / "TASK.md").write_text(task.replace("\n## Rules", "\n" + TASK_SECTION + "\n## Rules"), encoding="utf-8")
    isolate(SANDBOX)
    pd.DataFrame(rows).to_csv(REF_KEY, index=False)
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SANDBOX.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{SANDBOX.name}: {len(key)} images, {len(rows)} references; key -> {REF_KEY}; zip -> {ZIP}")


GUIDED = ROOT / "setup4b_tools_references_guided"
GUIDED_ZIP = PROJECT_ROOT / "colab/refs_job/sandboxes_guided.zip"
CUES = {"structure_pct": "gradient_p95", "streak_pct": "streak_anisotropy"}  # percentiles among the bank
GUIDED_TOOL_DOC = (" `structure_pct` (0 to 100) is where the image's amount of strong edges (95th percentile of the "
                   "gradient magnitude) falls among the same training slices, and `streak_pct` where its streak "
                   "measure (angular non-uniformity of the residual) falls; high values mean much high-contrast "
                   "anatomy, respectively strong streaks.")
GUIDED_SECTION = """## Where the noise rule is wrong

On rated training slices the rule behind `reference_match` errs in known ways. Its noise measurements count every
sharp intensity change, so they partly measure anatomy, not only noise:

- Slices with much high-contrast anatomy (gas- or contrast-filled bowel, large bones, many organ edges; high
  `structure_pct`) look noisier to the rule than they are: their score is often higher than the neighbours' MOS.
- Slices with little structure (large homogeneous soft tissue), and slices with streaks (high `streak_pct`), look
  cleaner to the rule than they are: their score is often lower than the neighbours' MOS.
- At the ends of the scale the neighbours' MOS is pulled to the middle: on average it is about 0.2 too high for the
  worst slices and about 0.2 too low for the best ones.

So judge the noise yourself where there is no anatomy: in flat soft tissue (muscle, liver, spleen, subcutaneous fat),
away from edges; compare that texture with the same tissue in the references. Use `neighbour_mos_mean` as a starting
point and move away from it when what you see calls for it; say in the rationale what you adjusted and why.
"""


def add_cues(folder: Path, key: pd.DataFrame, bank: pd.Index, screening: pd.DataFrame, test: pd.DataFrame) -> None:
    """structure_pct and streak_pct (percentiles among the bank) into each image's reference_match values."""
    for row in key.itertuples():
        path = folder / "tool_results" / f"{row.image_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, column in CUES.items():
            reference = np.sort(screening.loc[bank, column].to_numpy())
            share = 100 * np.searchsorted(reference, test.loc[row.test_id, column], "right") / len(reference)
            record["tools"]["reference_match"]["values"][name] = round(float(share), 1)
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def build_guided() -> None:
    """Setup 30: the setup 29 sandbox plus what scripts/rule_blind_spots.py found about the rule's errors on
    training slices: a TASK.md section on its blind spots and two cue percentiles in reference_match."""
    if GUIDED.exists():
        raise SystemExit(f"{GUIDED} already exists; delete it first to rebuild")
    if not SANDBOX.exists():
        raise SystemExit(f"build {SANDBOX.name} first")
    bank = split()[0].index
    screening = pd.read_csv(CONFIG.output_dir / "screening/train.csv", dtype={"image_id": str}).set_index("image_id")
    test = pd.read_csv(CONFIG.output_dir / "screening/test.csv", dtype={"image_id": str}).set_index("image_id")
    key = pd.read_csv(KEY)
    shutil.copytree(SANDBOX, GUIDED, ignore=shutil.ignore_patterns(".git", "scores.json"))
    add_cues(GUIDED, key, bank, screening, test)
    tools = (GUIDED / "TOOLS.md").read_text(encoding="utf-8")
    assert tools.count(TOOL_DOC) == 1
    (GUIDED / "TOOLS.md").write_text(tools.replace(TOOL_DOC, TOOL_DOC + GUIDED_TOOL_DOC), encoding="utf-8")
    task = (GUIDED / "TASK.md").read_text(encoding="utf-8")
    assert task.count("\n## Rules") == 1
    (GUIDED / "TASK.md").write_text(task.replace("\n## Rules", "\n" + GUIDED_SECTION + "\n## Rules"), encoding="utf-8")
    isolate(GUIDED)
    with zipfile.ZipFile(GUIDED_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(GUIDED.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{GUIDED.name}: {len(key)} images; zip -> {GUIDED_ZIP}")


TEST300 = ROOT / "test300"
TEST300_GUIDED = ROOT / "test300_guided"
TEST300_KEY = CONFIG.output_dir / "test300/key.csv"
TEST300_ZIPS = PROJECT_ROOT / "colab/test300g_job"


def build_test300() -> None:
    """Setup 30 on the 300-image test: each test300/batch_XX (scripts/make_test300.py: setup 26's sandbox with its
    own 20 images) without the expert score, with references/, reference_match and the cues, and setup 30's TASK.md
    and TOOLS.md (the batches' documents are setup2b's, so setup 30's apply unchanged). One zip per batch."""
    if TEST300_GUIDED.exists():
        raise SystemExit(f"{TEST300_GUIDED} already exists; delete it first to rebuild")
    bank, bank_scores, mos, _ = split()
    test = tables()[2]
    screening = pd.read_csv(CONFIG.output_dir / "screening/train.csv", dtype={"image_id": str}).set_index("image_id")
    full_test = pd.read_csv(CONFIG.output_dir / "screening/test.csv", dtype={"image_id": str}).set_index("image_id")
    key = pd.read_csv(TEST300_KEY)
    train_images = {s.image_id: s.image for s in load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels)}
    TEST300_ZIPS.mkdir(parents=True, exist_ok=True)
    rows = []
    for b, group in key.groupby("batch"):
        source, folder = TEST300 / f"batch_{b:02d}", TEST300_GUIDED / f"batch_{b:02d}"
        for doc in ("TASK.md", "TOOLS.md"):
            assert (source / doc).read_bytes() == (EXAMPLES / doc).read_bytes(), f"{source / doc} differs from setup2b"
        shutil.copytree(source, folder, ignore=shutil.ignore_patterns(".git", "scores.json"))
        rows += [{"batch": b, **r} for r in add_references(folder, group, bank, bank_scores, mos, test, train_images)]
        add_cues(folder, group, bank.index, screening, full_test)
        for doc in ("TASK.md", "TOOLS.md"):
            shutil.copy(GUIDED / doc, folder / doc)
        isolate(folder)
        with zipfile.ZipFile(TEST300_ZIPS / f"batch_{b:02d}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(ROOT).as_posix())
        print("wrote", folder.name, flush=True)
    pd.DataFrame(rows).to_csv(CONFIG.output_dir / "test300/references_key.csv", index=False)
    print(f"{len(rows)} references in {key['batch'].nunique()} batches; zips -> {TEST300_ZIPS}")


def verify() -> None:
    """The helpers reproduce setup 30's tool results (a rebuild into a temporary folder, compared file by file)."""
    import tempfile
    bank, bank_scores, mos, _ = split()
    test = tables()[2]
    screening = pd.read_csv(CONFIG.output_dir / "screening/train.csv", dtype={"image_id": str}).set_index("image_id")
    full_test = pd.read_csv(CONFIG.output_dir / "screening/test.csv", dtype={"image_id": str}).set_index("image_id")
    key = pd.read_csv(KEY)
    train_images = {s.image_id: s.image for s in load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels)}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "sb"
        shutil.copytree(EXAMPLES / "tool_results", folder / "tool_results")
        add_references(folder, key, bank, bank_scores, mos, test, train_images)
        add_cues(folder, key, bank.index, screening, full_test)
        for path in sorted((GUIDED / "tool_results").glob("*.json")):
            assert path.read_bytes() == (folder / "tool_results" / path.name).read_bytes(), path.name
        for path in sorted((GUIDED / "references").glob("*.png")):
            assert path.read_bytes() == (folder / "references" / path.name).read_bytes(), path.name
    print("helpers reproduce setup 30's tool_results and references")


if __name__ == "__main__":
    {"check": check, "build": build, "guided": build_guided, "test300": build_test300, "verify": verify}[sys.argv[1]]()
