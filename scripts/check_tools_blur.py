"""Step 1.3 check on real data: Gaussian blur (sigma 1 and 2) on 20 dev images with MOS >= 3.
Measurements that detect smoothing must move in the expected direction on (almost) every image."""
import numpy as np
import pandas as pd
from scipy import ndimage

from scripts.scalar_baseline import measure
from src.config import CONFIG
from src.data import load_ldctiqa_split, read_id_list

EXPECTED = {"high_pass_std": "lower", "flat_roi_sigma": "lower", "nps_f_mean": "lower",
            "nps_high_band_ratio": "lower", "edge_width_px": "higher"}


def main() -> None:
    dev = set(read_id_list(CONFIG.splits_dir / "dev.txt"))
    samples = [s for s in load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels)
               if s.image_id in dev and s.mos >= CONFIG.blur_min_mos][:20]
    rows = []
    for sample in samples:
        base = measure(sample.image)
        for sigma in CONFIG.blur_sigmas:
            blurred = measure(ndimage.gaussian_filter(sample.image, sigma))
            rows += [{"sigma": sigma, "metric": k, "original": base[k], "blurred": blurred[k]} for k in EXPECTED]
    frame = pd.DataFrame(rows)
    frame["correct"] = np.where(frame["metric"].map(EXPECTED) == "lower",
                                frame["blurred"] < frame["original"], frame["blurred"] > frame["original"])
    summary = frame.groupby(["metric", "sigma"]).agg(original=("original", "median"), blurred=("blurred", "median"),
                                                     correct_share=("correct", "mean"))
    print(f"{len(samples)} dev images with MOS >= {CONFIG.blur_min_mos}")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
