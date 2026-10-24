"""Images for the cross-dataset test (scripts/cross_eval.py), in the Stage B input format of scripts/pseudo_pool.py
(soft-tissue window only, LDCT-IQA storage window, six tool inputs):

- dose: the 908 slices of the teacher pool at the four I2I 2025v1 dose levels the pool lacks (1000k, 500k, 50k, 10k
  photons). The pool's gt and 100k images are the same slices' gt_clipped and noisy_clipped_100k, so their predictions
  are reused; as a check, rebuilding them from I2I must give the pool images exactly. I2I floats go through the same
  8-bit step as the IQA PNGs (floor(255 x)).
- methods: the biailab IQA 2026v1 test split (31 patients, unseen by the 11 denoisers), every soft-tissue slice with all
  13 images (gt, noisy 100k, 11 methods).

    python -m scripts.cross_pool      # -> colab/vft_job/cross.npz (ids, patient, kind, set, images, tools)
                                      #    and cross.zip (PNG per image, name id with / -> __) for MedSigLIP
"""
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from scripts.pseudo_pool import TOOLS, to_storage
from scripts.qwen_vision_ft import CUES, NOISE, percentiles
from scripts.screen_tools import measure
from src.config import CONFIG, PROJECT_ROOT

JOB = PROJECT_ROOT / "colab/vft_job"
I2I = PROJECT_ROOT / "data/dataset_biailab_I2I_2025v1"
DOSES = ("1000k", "500k", "50k", "10k")
METHODS = ("gt", "noisy", "bm3d", "ctformer", "dncnn", "n2self", "nafnet", "nb2nb", "nr2n", "redcnn", "tv", "unet", "wgan_vgg")


def as_png(x: np.ndarray) -> np.ndarray:
    return np.floor(255 * np.clip(x, 0, 1)).astype(np.uint8)


def main() -> None:
    pool = np.load(JOB / "pseudo.npz", allow_pickle=True)
    pool_images = dict(zip(pool["ids"], pool["images"]))
    slices = sorted({i.split("/")[0] for i in pool["ids"]})
    splits = {p.name.replace(".png.npz", ""): p for split in ("train", "val") for p in (I2I / split).glob("*.png.npz")}
    rows, images = [], []

    def add(image_id, patient, kind, subset, v):
        rows.append({"image_id": image_id, "patient": patient, "kind": kind, "set": subset, **measure(v, TOOLS)})
        images.append(np.round(v * 255).astype(np.uint8))

    worst = 0
    for n, s in enumerate(slices):
        z = np.load(splits[s])
        for key, kind in (("gt_clipped", "gt"), ("noisy_clipped_100k", "noisy")):
            check = np.round(to_storage(as_png(z[key])) * 255).astype(np.uint8)
            worst = max(worst, int(np.abs(check.astype(int) - pool_images[f"{s}/{kind}"]).max()))
        for dose in DOSES:
            add(f"{s}/{dose}", s.split("_")[0], dose, "dose", to_storage(as_png(z[f"noisy_clipped_{dose}"])))
        if n % 100 == 0:
            print("dose", n, len(slices), flush=True)
    print(f"pool gt/100k rebuilt from I2I: max |diff| {worst} grey levels", flush=True)
    assert worst <= 1, "I2I does not reproduce the pool images"
    groups = sorted(p for p in (CONFIG.iqa2026_root / "test").iterdir() if p.is_dir())
    for group in groups:
        gt = np.asarray(Image.open(group / "gt.png"))
        if (gt == 0).mean() <= 0.2:  # lung window, as in the pool
            continue
        for method in METHODS:
            add(f"{group.name}/{method}", group.name.split("_")[0], method, "methods",
                to_storage(np.asarray(Image.open(group / f"{method}.png"))))
    table = pd.DataFrame(rows).set_index("image_id")
    bank = pd.read_csv(CONFIG.output_dir / "screening/train.csv", dtype={"image_id": str}).set_index("image_id")
    p = percentiles(table[[*NOISE, *CUES]], bank)
    p.insert(3, "noise_score", p[list(NOISE)].mean(axis=1))
    stacked = np.stack(images)
    np.savez_compressed(JOB / "cross.npz", ids=np.array(table.index), patient=table["patient"].to_numpy(),
                        kind=table["kind"].to_numpy(), set=table["set"].to_numpy(), images=stacked, tools=p.to_numpy(np.float32))
    with zipfile.ZipFile(JOB / "cross.zip", "w", zipfile.ZIP_STORED) as out:
        for image_id, im in zip(table.index, stacked):
            buf = __import__("io").BytesIO()
            Image.fromarray(im).save(buf, format="PNG")
            out.writestr(image_id.replace("/", "__") + ".png", buf.getvalue())
    print(table.groupby("set")["patient"].agg(["count", "nunique"]).to_string())
    print("->", JOB / "cross.npz", JOB / "cross.zip")


if __name__ == "__main__":
    main()
