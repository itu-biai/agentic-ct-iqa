"""Inputs of colab/vm/infer329.py (issue #329): the lab IQA test images and the E18 cases -> colab/vft_job/lab329.npz.

- images, tools, ids: the 2496 lab IQA 2026v1 test images (192 slices x 13: gt, noisy, 11 denoisers) of cross.npz
  (scripts/cross_pool.py), the set scored in scripts/cross_eval.py.
- occl_images, occl_tools, occl_ids: the 4 E18 cases by the rule fixed in the issue before anything is generated (ties
  by lowest image ID); Stage B = rerun llm 5-fold mean, MedSigLIP = the predictions setup 26 saw, agent = the zero-shot
  row chosen in E4 (Z3: ne19, agent + tools, no MedSigLIP):
    1. test image with the largest |agent - MOS|
    2. test image with the largest |Stage B - MedSigLIP|
    3. lab test slice where Stage B scores a denoiser (one of the 11 methods, not noisy) above gt by the largest
       margin; the case is that denoiser image
    4. test image whose Stage B absolute error is closest to the median absolute error

    python -m scripts.make_lab329
"""
import numpy as np
import pandas as pd

from src.config import CONFIG, PROJECT_ROOT

JOB = PROJECT_ROOT / "colab/vft_job"
RERUN = CONFIG.output_dir / "vision_ft/llm2/out_llm"
DENOISERS = ("bm3d", "ctformer", "dncnn", "n2self", "nafnet", "nb2nb", "nr2n", "redcnn", "tv", "unet", "wgan_vgg")


def first(values: pd.Series, best) -> str:
    """ID of the extreme value (best = max / min of the series), ties by lowest ID."""
    hit = values[values == best(values)]
    return sorted(hit.index)[0]


def main() -> None:
    c = np.load(JOB / "cross.npz", allow_pickle=True)
    lab = np.nonzero(c["set"] == "methods")[0]
    t = np.load(JOB / "test.npz")
    frame = pd.read_csv(CONFIG.output_dir / "noexp/per_image.csv").set_index("test_id")
    stage_b = pd.Series(np.mean([np.load(RERUN / f"fold_{k}.npz")["llm_test"] for k in range(1, 6)], 0), index=t["ids"])
    frame["stage_b"] = stage_b.loc[frame.index]
    cases = {"1 agent error": first((frame["ne19"] - frame["mos"]).abs(), max),
             "2 Stage B vs MedSigLIP": first((frame["stage_b"] - frame["expert"]).abs(), max)}
    err = (frame["stage_b"] - frame["mos"]).abs()
    cases["4 typical"] = first((err - err.median()).abs(), min)
    p = np.load(JOB / "cross_pred.npz", allow_pickle=True)
    pred = pd.Series(np.mean([p[f"llm_{k}"] for k in range(1, 6)], 0), index=p["ids"])
    ids = pd.Series(c["ids"][lab])
    slice_of = ids.str.rsplit("/", n=1).str[0]
    kind = ids.str.rsplit("/", n=1).str[1]
    margin = {}
    for s in sorted(slice_of.unique()):
        gt = pred[f"{s}/gt"]
        margin.update({f"{s}/{d}": pred[f"{s}/{d}"] - gt for d in DENOISERS})
    margin = pd.Series(margin)
    lab_case = first(margin, max)
    test_index = {i: n for n, i in enumerate(t["ids"])}
    images, tools, names = [], [], []
    for name in ("1 agent error", "2 Stage B vs MedSigLIP"):
        images.append(t["images"][test_index[cases[name]]]); tools.append(t["tools"][test_index[cases[name]]])
        names.append(f"{name}: {cases[name]}")
    n = int(np.nonzero(c["ids"] == lab_case)[0][0])
    images.append(c["images"][n]); tools.append(c["tools"][n]); names.append(f"3 denoiser above gt: {lab_case}")
    images.append(t["images"][test_index[cases["4 typical"]]]); tools.append(t["tools"][test_index[cases["4 typical"]]])
    names.append(f"4 typical: {cases['4 typical']}")
    np.savez_compressed(JOB / "lab329.npz", images=c["images"][lab], tools=c["tools"][lab], ids=c["ids"][lab].astype(str),
             occl_images=np.stack(images), occl_tools=np.stack(tools).astype(np.float32), occl_ids=np.array(names))
    for name in names:
        key = name.split(": ")[1]
        row = frame.loc[key] if key in frame.index else None
        extra = (f"MOS {row['mos']:.2f}, Stage B {row['stage_b']:.2f}, MedSigLIP {row['expert']:.2f}, agent {row['ne19']:.2f}"
                 if row is not None else f"margin over gt {margin[key]:+.2f}")
        print(name, "|", extra)
    print("lab images", len(lab), kind.value_counts().to_dict())


if __name__ == "__main__":
    main()
