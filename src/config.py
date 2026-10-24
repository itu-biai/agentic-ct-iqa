from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class IqaConfig:
    ldctiqa_root: Path = PROJECT_ROOT / "data/ldctiqa2023"
    iqa2026_root: Path = PROJECT_ROOT / "data/dataset_biailab_IQA_2026v1"
    i2i_root: Path = PROJECT_ROOT / "data/dataset_biailab_I2I_2025v1"
    output_dir: Path = PROJECT_ROOT / "outputs/agentic_ct_iqa"
    dev_size: int = 50
    calibration_size: int = 100
    blur_sigmas: tuple[float, ...] = (1.0, 2.0)
    blur_min_mos: float = 3.0
    denoiser_groups: int = 40
    denoiser_groups_dev: int = 8  # not fixed in the plan; kept small because selection re-runs it many times
    dose_slices: int = 50
    dose_slices_dev: int = 10
    seed: int = 42

    @property
    def splits_dir(self) -> Path:
        return self.output_dir / "splits"

    @property
    def expert_dir(self) -> Path:
        """Expert predictions in use: Tolga's own fold checkpoints (shared 2026-10-06, verified against MedIA
        Table 12). Our retrained folds 1-3 stay in outputs/agentic_ct_iqa/experts as a backup."""
        return self.output_dir / "experts_tolga"

    @property
    def train_images(self) -> Path:
        return self.ldctiqa_root / "train/LDCTIQAG2023_train/image"

    @property
    def train_labels(self) -> Path:
        return self.ldctiqa_root / "train/LDCTIQAG2023_train/train.json"

    @property
    def test_images(self) -> Path:
        return self.ldctiqa_root / "test/LDCTIQAC_test/images"

    @property
    def test_labels(self) -> Path:
        return self.ldctiqa_root / "test/LDCTIQAC_test/test.json"


CONFIG = IqaConfig()
