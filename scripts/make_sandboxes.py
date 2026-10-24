"""Builds experiment_sandboxes/: 20 LDCT-IQA test slices rated in three isolated setups.

  setup1_vision_only   Claude Opus 5.5 looks at the images; nothing else.
  setup2_vision_tools  The same, plus our tool outputs for each image (no expert score).
  setup3_medsiglip     Tolga's MedSigLIP-FiLM (5 fold models, mean), no Claude.
  setup2b_vision_tools_examples  setup 2 plus 10 rated training slices as reference examples (`examples` command).

Images get neutral names (img_01 ... img_20) in shuffled order; the answer key (original test ID, MOS) is written
outside the sandboxes, to outputs/agentic_ct_iqa/sandbox/key.csv. Selection: 4 images per MOS bin, seed 42.

    python -m scripts.make_sandboxes
    python -m scripts.make_sandboxes examples   # setup2b, after setup 2 exists
"""
import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.agent import Case
from src.config import CONFIG, PROJECT_ROOT
from src.data import load_ldctiqa_split, mos_bins
from src.skill_tools import LDCTIQA, build_tools
from src.tools import nlm_denoise, noise_residual_map, window_view
from src.tools_iqat1 import (fourier_magnitude_spectrum, gradient_orientation_coherence_map,
                             local_gradient_deviation_map, mscn_map)

ROOT = PROJECT_ROOT / "experiment_sandboxes"
KEY = CONFIG.output_dir / "sandbox/key.csv"
PER_BIN, N_BINS = 4, 5
EXAMPLES = ROOT / "setup2b_vision_tools_examples"
EXAMPLES_KEY = CONFIG.output_dir / "sandbox/examples_key.csv"
EXAMPLES_PER_BIN = 2
CLAUDE_SETUPS = ("setup1_vision_only", "setup2_vision_tools")
VIEWS = {  # tool output images given to setup 2, as the agent would receive them
    "window_soft_tissue": lambda x: window_view(x, *LDCTIQA.windows["soft_tissue"]),
    "window_narrow": lambda x: window_view(x, *LDCTIQA.windows["narrow"]),
    "noise_residual_map": lambda x: noise_residual_map(x, nlm_denoise),
    "orientation_coherence_map": gradient_orientation_coherence_map,
    "fourier_spectrum": fourier_magnitude_spectrum,
    "mscn_map": mscn_map,
    "gradient_deviation_map": local_gradient_deviation_map,
}
EXCLUDED_SKILLS = ("expert_models", "ct_view", "evidence_maps")  # expert is setup 3; views are written as PNGs


def anonymise(text: str) -> str:
    """Tool texts name the dataset; the sandboxes do not (setup 1 does not get it either)."""
    return text.replace("LDCT-IQA training slices", "training slices of this dataset").replace(
        "LDCT-IQA slices", "slices of this dataset").replace("LDCT-IQA", "this dataset")

CONCEPT = """# Task: rate the diagnostic image quality of CT slices

## Your role

You are a radiologist taking part in an image quality reading study. You rate each image on your own,
independently of the others.

## What image quality assessment is

Image quality assessment (IQA) means judging how good an image is for its purpose. For medical images the purpose
is diagnosis, so quality means diagnostic quality: how well a reader can see the anatomy and could detect disease in
the image. This study is no-reference: there is no clean version of an image to compare with; each image is judged
on its own.

## The images

`images/` holds 20 axial slices of abdominal CT, 512 x 512 pixels, grayscale PNG. They come from low-dose CT
examinations: CT scans taken with a reduced radiation dose, which can make images noisier or introduce artifacts.
The slices are already shown in a fixed display window. The file names (img_01 ... img_20) carry no information
about quality, and their order is random.

## The rating scale

Rate each image on this five-level scale. Scores are continuous: use decimals (for example 2.6) when an image falls
between two levels.

- 0 nondiagnostic: noise or artifacts hide anatomy; no diagnosis is possible.
- 1 poor: major structures visible, but noise or streaks make interpretation unreliable.
- 2 fair: diagnosable with effort; noise or streaks clearly visible in soft tissue.
- 3 good: small noise or artifacts that do not interfere with diagnosis.
- 4 excellent: anatomy crisp, soft-tissue texture clean, no disturbing artifacts.
"""

RULES = """## Rules

- Open and look at every image yourself, with your file-reading tool. Rate what you see.
- Use only the files in this folder. Do not look anywhere else on the computer or on the internet, and do not try to
  find out where the images come from.
- Do not write or run code, and do not build your own measurement tools.
"""

OUTPUT = """## Output

When you have rated all 20 images, write `scores.json` in this folder, exactly in this form (one entry per image,
all 20 images, ordered by image_id):

```json
{
  "scores": [
    {"image_id": "img_XX", "score": <number from 0 to 4>, "confidence": <number from 0 to 1>, "rationale": "<one or two sentences>"},
    ...
  ]
}
```

- `image_id`: the file name without `.png`.
- `score`: a number from 0 to 4 (decimals allowed).
- `confidence`: a number from 0 to 1, how sure you are of this score.
- `rationale`: one or two sentences.

Write the file once, at the end. Do not create any other files.
"""

TOOLS_SECTION = """## Measurements

For every image, an automatic measurement pipeline has already run a set of image analysis tools. Their outputs are
in this folder:

- `tool_results/img_XX.json`: the numeric outputs of each tool for that image.
- `tool_views/img_XX_<view>.png`: images produced by the tools for that image.

`TOOLS.md` describes what each tool measures and each view shows. Use the measurements as support for your own
reading; the final score is your judgment.
"""


EXAMPLES_SECTION = """## Reference examples

`examples/` holds 10 other slices of the same kind, not among the 20 images you rate, that radiologists rated on the
same scale. `examples/labels.json` gives the score of each example: the mean of the radiologists' ratings. Look at
the examples and their scores before you start, to see how the scale is used.
"""


def select(samples: list, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mos = np.array([s.mos for s in samples])
    bins = mos_bins(mos)
    picked = [int(i) for b in range(N_BINS) for i in rng.choice(np.flatnonzero(bins == b), PER_BIN, replace=False)]
    order = rng.permutation(picked)
    return pd.DataFrame({"image_id": [f"img_{k + 1:02d}" for k in range(len(order))],
                         "test_id": [samples[i].image_id for i in order], "mos": [samples[i].mos for i in order],
                         "mos_bin": [int(bins[i]) for i in order], "index": order})


def png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def tool_outputs(image: np.ndarray) -> dict:
    case = Case("", image)
    out = {}
    for skill, specs in build_tools(LDCTIQA).items():
        if skill in EXCLUDED_SKILLS:
            continue
        for spec in specs:
            if spec.name == "noise_residual_map":
                continue  # its numbers are kept below, its image goes to tool_views
            try:
                result = spec.run(case)
                entry = {"values": {k: float(f"{v:.5g}") for k, v in result.values.items()}}
                if result.text:
                    entry["text"] = anonymise(result.text)
            except ValueError as error:
                entry = {"error": str(error)}
            out[spec.name] = entry
    residual = noise_residual_map(image, nlm_denoise)
    out["noise_residual_map"] = {"values": {k: float(f"{v:.5g}") for k, v in residual.values.items()}}
    return out


def tools_glossary() -> str:
    tools = build_tools(LDCTIQA)
    lines = ["# Tools", "", "Descriptions of the tools whose outputs are in `tool_results/` and `tool_views/`.", "",
             "## Numeric outputs (`tool_results/img_XX.json`)", ""]
    for skill, specs in tools.items():
        if skill in EXCLUDED_SKILLS:
            continue
        for spec in specs:
            lines.append(f"- `{spec.name}`: {anonymise(spec.description)}")
    expert = tools["expert_models"][0]
    lines.append(f"- `{expert.name}`: {anonymise(expert.description)} The model is a MedSigLIP-FiLM network; its "
                 "profile card is the `text` of this entry.")
    lines += ["", "Values are in the image's own units (pixel values from 0 to 1). `_pct` values are percentiles "
              "(0 to 100) among unprocessed CT slices of the same kind.", "",
              "## Views (`tool_views/img_XX_<view>.png`)", ""]
    views = {spec.name: spec.description for specs in tools.values() for spec in specs}
    lines += ["- `window_soft_tissue`, `window_narrow`: " + views["window_view"].split(";")[0] +
              "; soft-tissue and narrow presets.",
              f"- `noise_residual_map`: {views['noise_residual_map']}",
              f"- `orientation_coherence_map`: {views['orientation_coherence_map']}",
              f"- `fourier_spectrum`: {views['fourier_spectrum']}",
              f"- `mscn_map`: {views['mscn_map']}",
              f"- `gradient_deviation_map`: {views['gradient_deviation_map']}", ""]
    return "\n".join(lines)


def isolate(folder: Path) -> None:
    """Make the sandbox its own git repository, so a Claude session started there sees neither the project's git
    history nor any memory keyed to the project repository."""
    if not (folder / ".git").exists():
        subprocess.run(["git", "init", "-q", str(folder)], check=True)


def main() -> None:
    if ROOT.exists() and any(ROOT.glob("setup*/images/*.png")):
        raise SystemExit(f"{ROOT} already has images; delete the setup folders first to rebuild")
    samples = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels)
    key = select(samples, CONFIG.seed)
    KEY.parent.mkdir(parents=True, exist_ok=True)
    key.drop(columns="index").to_csv(KEY, index=False)

    setups = (*CLAUDE_SETUPS, "setup3_medsiglip")
    for setup in setups:
        (ROOT / setup / "images").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    for row in key.itertuples():
        image = samples[row.index].image
        data = png(image)
        for setup in setups:
            (ROOT / setup / "images" / f"{row.image_id}.png").write_bytes(data)
        tools_dir, views_dir = ROOT / "setup2_vision_tools/tool_results", ROOT / "setup2_vision_tools/tool_views"
        tools_dir.mkdir(exist_ok=True)
        views_dir.mkdir(exist_ok=True)
        (tools_dir / f"{row.image_id}.json").write_text(
            json.dumps({"image_id": row.image_id, "tools": tool_outputs(image)}, indent=1), encoding="utf-8")
        for view, tool in VIEWS.items():
            (views_dir / f"{row.image_id}_{view}.png").write_bytes(tool(image).image_png)
        print("wrote", row.image_id, flush=True)

    (ROOT / "setup1_vision_only/TASK.md").write_text(CONCEPT + "\n" + RULES + "\n" + OUTPUT, encoding="utf-8")
    (ROOT / "setup2_vision_tools/TASK.md").write_text(CONCEPT + "\n" + TOOLS_SECTION + "\n" + RULES + "\n" + OUTPUT,
                                                       encoding="utf-8")
    (ROOT / "setup2_vision_tools/TOOLS.md").write_text(tools_glossary(), encoding="utf-8")
    for setup in CLAUDE_SETUPS:
        isolate(ROOT / setup)
    # The launcher stays outside the sandboxes: a session can read every file in its own folder.
    shutil.copy(Path(__file__).with_name("sandbox_run.ps1"), ROOT / "run_claude.ps1")
    print(f"{len(key)} images; key -> {KEY}")


def examples() -> None:
    """setup2b: a copy of setup 2 (same images, tool outputs, views, TOOLS.md) plus examples/ with 10 LDCT-IQA training
    slices (2 per MOS bin, seed 42, outside the dev/calibration split) and their MOS. TASK.md is setup 2's with one
    added section. The training-set IDs go to examples_key.csv, outside the sandbox."""
    if EXAMPLES.exists():
        raise SystemExit(f"{EXAMPLES} already exists; delete it first to rebuild")
    held_out = set((CONFIG.splits_dir / "calibration.txt").read_text().split())
    samples = [s for s in load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels) if s.image_id not in held_out]
    rng = np.random.default_rng(CONFIG.seed)
    bins = mos_bins(np.array([s.mos for s in samples]))
    picked = rng.permutation([int(i) for b in range(N_BINS)
                              for i in rng.choice(np.flatnonzero(bins == b), EXAMPLES_PER_BIN, replace=False)])
    key = pd.DataFrame({"example_id": [f"ex_{k + 1:02d}" for k in range(len(picked))],
                        "train_id": [samples[i].image_id for i in picked],
                        "mos": [round(samples[i].mos, 2) for i in picked], "mos_bin": [int(bins[i]) for i in picked]})
    shutil.copytree(ROOT / "setup2_vision_tools", EXAMPLES, ignore=shutil.ignore_patterns(".git", "scores.json"))
    (EXAMPLES / "examples").mkdir()
    for row, i in zip(key.itertuples(), picked):
        (EXAMPLES / "examples" / f"{row.example_id}.png").write_bytes(png(samples[i].image))
    labels = {"scores": [{"image_id": r.example_id, "score": r.mos} for r in key.sort_values("example_id").itertuples()]}
    (EXAMPLES / "examples/labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    task = (ROOT / "setup2_vision_tools/TASK.md").read_text(encoding="utf-8")
    assert task.count("\n## Rules") == 1
    (EXAMPLES / "TASK.md").write_text(task.replace("\n## Rules", "\n" + EXAMPLES_SECTION + "\n## Rules"), encoding="utf-8")
    isolate(EXAMPLES)
    key.to_csv(EXAMPLES_KEY, index=False)
    print(key.to_string(index=False))


if __name__ == "__main__":
    import sys
    examples() if sys.argv[1:] == ["examples"] else main()
