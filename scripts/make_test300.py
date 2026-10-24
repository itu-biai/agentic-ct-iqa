"""Builds experiment_sandboxes/test300/batch_01 ... batch_15: all 300 LDCT-IQA test slices in 15 sandboxes of 20, each
a copy of the setup 26 sandbox (setup2b_vision_tools_examples) with its own images. Same TASK.md, TOOLS.md and 10
reference examples; tool_results (with the consult_experts entry, as setup 2 gets it) and tool_views computed exactly
as for the 20-image sandboxes. Images get neutral names img_01 ... img_20 per batch; the order over the 300 is a seeded
random permutation. The answer key goes outside the sandboxes, to outputs/agentic_ct_iqa/test300/key.csv.

    python -m scripts.make_test300           # sandboxes + key
    python -m scripts.make_test300 zip       # colab/test300_job/batch_XX.zip for the VM
"""
import json
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd

from scripts.make_sandboxes import EXAMPLES, ROOT, VIEWS, anonymise, isolate, png, tool_outputs
from src.config import CONFIG, PROJECT_ROOT
from src.data import load_ldctiqa_split, mos_bins
from src.experts import expert_scores, load_fold_predictions, strip_ext
from src.metrics import logistic4

OUT = ROOT / "test300"
KEY = CONFIG.output_dir / "test300/key.csv"
BATCH = 20
ZIPS = PROJECT_ROOT / "colab/test300_job"


def expert() -> pd.Series:
    """MedSigLIP-FiLM on the test set as setup 2's consult_experts returns it: 5-fold mean, training logistic to 0-4."""
    test = expert_scores(load_fold_predictions(CONFIG.expert_dir))["test"]
    test.index = strip_ext(test.index)
    params = np.load(CONFIG.expert_dir / "expert_logistic_params.npy")
    return pd.Series(logistic4(test.to_numpy(), *params), index=test.index).round(3)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; delete it first to rebuild")
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels)
    order = np.random.default_rng(CONFIG.seed).permutation(len(samples))
    scores = expert()
    card = anonymise((CONFIG.expert_dir / "profile_card.txt").read_text(encoding="utf-8").strip())
    rows = []
    for b in range(len(samples) // BATCH):
        folder = OUT / f"batch_{b + 1:02d}"
        shutil.copytree(EXAMPLES, folder, ignore=shutil.ignore_patterns(".git", "scores.json", "images", "tool_results",
                                                                         "tool_views"))
        for sub in ("images", "tool_results", "tool_views"):
            (folder / sub).mkdir()
        for k, i in enumerate(order[b * BATCH:(b + 1) * BATCH]):
            image_id, sample = f"img_{k + 1:02d}", samples[i]
            (folder / "images" / f"{image_id}.png").write_bytes(png(sample.image))
            tools = tool_outputs(sample.image)
            tools["consult_experts"] = {"values": {"medsiglip_film": float(scores[sample.image_id])}, "text": card}
            (folder / "tool_results" / f"{image_id}.json").write_text(
                json.dumps({"image_id": image_id, "tools": tools}, indent=1), encoding="utf-8")
            for view, tool in VIEWS.items():
                (folder / "tool_views" / f"{image_id}_{view}.png").write_bytes(tool(sample.image).image_png)
            rows.append({"batch": b + 1, "image_id": image_id, "test_id": sample.image_id, "mos": sample.mos})
        isolate(folder)
        print("wrote", folder.name, flush=True)
    key = pd.DataFrame(rows)
    key["mos_bin"] = mos_bins(key["mos"].to_numpy())
    key["expert"] = scores[key["test_id"]].to_numpy()
    KEY.parent.mkdir(parents=True, exist_ok=True)
    key.to_csv(KEY, index=False)
    print(f"{len(key)} images in {key['batch'].nunique()} batches; key -> {KEY}")


def make_zip() -> None:
    """One zip per batch (with .git, so each sandbox is its own repository on the VM too), paths under test300/. One
    zip of all 15 (366 MB) times out in `colab upload`; 24 MB each goes through."""
    ZIPS.mkdir(parents=True, exist_ok=True)
    for batch in sorted(OUT.glob("batch_*")):
        with zipfile.ZipFile(ZIPS / f"{batch.name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(batch.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(ROOT).as_posix())
    print(f"wrote {len(list(ZIPS.glob('batch_*.zip')))} zips to {ZIPS}")


if __name__ == "__main__":
    make_zip() if sys.argv[1:] == ["zip"] else main()
