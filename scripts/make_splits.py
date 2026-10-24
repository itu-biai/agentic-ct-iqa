"""Step 1.1: dev (50) and calibration (100, contains dev) sets from the LDCT-IQA training images,
stratified by MOS. Written once to outputs/agentic_ct_iqa/splits/ and never regenerated."""
import numpy as np

from src.config import CONFIG
from src.data import dev_and_calibration_ids, load_ldctiqa_split, mos_bins, write_id_list
from src.metrics import challenge_scores


def main() -> None:
    train = load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels, load_images=False)
    test = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    dev, calibration = dev_and_calibration_ids(train, CONFIG.dev_size, CONFIG.calibration_size, CONFIG.seed)
    write_id_list(CONFIG.splits_dir / "dev.txt", dev)
    write_id_list(CONFIG.splits_dir / "calibration.txt", calibration)
    write_id_list(CONFIG.splits_dir / "train_all.txt", sorted(s.image_id for s in train))
    write_id_list(CONFIG.splits_dir / "test_all.txt", sorted(s.image_id for s in test))

    mos = {s.image_id: s.mos for s in train}
    all_mos = np.array(list(mos.values()))
    print(f"train / test images: {len(train)} / {len(test)}")
    print(f"dev / calibration: {len(dev)} / {len(calibration)}, dev inside calibration: {set(dev) <= set(calibration)}")
    for name, ids in (("train", list(mos)), ("dev", dev), ("calibration", calibration)):
        values = np.array([mos[i] for i in ids])
        print(f"  {name:12s} MOS bins {np.bincount(mos_bins(values), minlength=5)}  mean {values.mean():.3f}")
    print(f"MOS range {all_mos.min()} to {all_mos.max()}")
    print(f"challenge_scores(mos, mos)['overall'] = {challenge_scores(all_mos, all_mos)['overall']:.4f}")


if __name__ == "__main__":
    main()
