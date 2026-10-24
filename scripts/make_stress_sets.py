"""Step 1.2: fixed image lists for the stress sets. Dev versions come from training/val data,
reported versions from test data. Lists are written once; every method reads the same lists.

S1 blur:      LDCT-IQA images with MOS >= 3 (dev: from the dev set; test: from the official test set)
S2 denoisers: IQA 2026v1 groups, one per patient before a second (dev: val split; test: test split)
S3 dose:      I2I 2025v1 slices, same rule (dev: val split; test: test split); skipped until downloaded
"""
from pathlib import Path

from src.config import CONFIG
from src.data import assert_no_patient_leakage, load_ldctiqa_split, read_id_list, write_id_list
from src.stress_sets import DENOISER_FILES, one_group_per_patient


def s1_lists() -> None:
    dev_ids = set(read_id_list(CONFIG.splits_dir / "dev.txt"))
    train = load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels, load_images=False)
    test = load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels, load_images=False)
    s1_dev = sorted(s.image_id for s in train if s.image_id in dev_ids and s.mos >= CONFIG.blur_min_mos)
    s1_test = sorted(s.image_id for s in test if s.mos >= CONFIG.blur_min_mos)
    write_id_list(CONFIG.splits_dir / "s1_dev.txt", s1_dev)
    write_id_list(CONFIG.splits_dir / "s1_test.txt", s1_test)
    print(f"S1 blur: dev {len(s1_dev)} x {len(CONFIG.blur_sigmas)} levels, test {len(s1_test)} x {len(CONFIG.blur_sigmas)} levels")


def s2_lists() -> None:
    root = CONFIG.iqa2026_root
    groups = {split: sorted(p for p in (root / split).iterdir() if p.is_dir()) for split in ("val", "test")}
    for split, paths in groups.items():
        incomplete = [p.name for p in paths if any(not (p / f"{m}.png").exists() for m in DENOISER_FILES)]
        assert not incomplete, f"{split}: groups missing images: {incomplete[:5]}"
    s2_dev = one_group_per_patient(groups["val"], CONFIG.denoiser_groups_dev, CONFIG.seed)
    s2_test = one_group_per_patient(groups["test"], CONFIG.denoiser_groups, CONFIG.seed)
    assert_no_patient_leakage({"s2_dev": s2_dev, "s2_test": s2_test})
    write_id_list(CONFIG.splits_dir / "s2_dev.txt", [p.name for p in s2_dev])
    write_id_list(CONFIG.splits_dir / "s2_test.txt", [p.name for p in s2_test])
    patients = {name: len({p.name.split("_")[0] for p in paths}) for name, paths in (("dev", s2_dev), ("test", s2_test))}
    all_test_patients = len({p.name.split("_")[0] for p in groups["test"]})
    print(f"S2 denoisers: dev {len(s2_dev)} groups ({patients['dev']} patients), "
          f"test {len(s2_test)} groups x {len(DENOISER_FILES)} images ({patients['test']} of {all_test_patients} test patients)")


def s3_lists() -> None:
    root = CONFIG.i2i_root
    if not (root / "test").exists():
        print("S3 dose ladder: I2I 2025v1 not extracted yet, skipped")
        return
    files = {split: sorted((root / split).glob("*.npz")) for split in ("val", "test")}
    s3_dev = one_group_per_patient(files["val"], CONFIG.dose_slices_dev, CONFIG.seed)
    s3_test = one_group_per_patient(files["test"], CONFIG.dose_slices, CONFIG.seed)
    assert_no_patient_leakage({"s3_dev": s3_dev, "s3_test": s3_test})
    write_id_list(CONFIG.splits_dir / "s3_dev.txt", [p.stem for p in s3_dev])
    write_id_list(CONFIG.splits_dir / "s3_test.txt", [p.stem for p in s3_test])
    print(f"S3 dose ladder: dev {len(s3_dev)} slices, test {len(s3_test)} slices x 6 levels "
          f"({len({p.stem.split('_')[0] for p in s3_test})} test patients)")


if __name__ == "__main__":
    s1_lists()
    s2_lists()
    s3_lists()
