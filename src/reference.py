"""Reference bank: where a measurement falls among unprocessed slices of the same dataset.

Every distortion in LDCT-IQA adds high-frequency content and none smooths (Tolga's MedIA manuscript, Sec. 4.1),
so a slice whose noise measures fall below the range of every unprocessed slice has been smoothed. On the dev blur
set, every Gaussian-blurred slice falls below the 1st percentile of the training slices for the high-pass,
Laplacian, Fourier, Immerkaer and NLM residual measures (outputs/agentic_ct_iqa/screening/smoothing_detection.csv).
The bank uses only unprocessed training-split images and no MOS. Banks are per dataset: units differ between datasets.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools import ToolResult

DEFAULT_MEASURES = ("high_pass_std", "immerkaer_sigma", "nlm_residual_std", "laplacian_variance",
                    "fourier_high_band_fraction", "nps_high_band_ratio", "edge_width_px", "air_hp_std", "body_coherence")


@dataclass(frozen=True)
class ReferenceBank:
    dataset: str
    values: dict[str, np.ndarray]  # measure -> sorted values over unprocessed reference slices

    @classmethod
    def from_csv(cls, dataset: str, path: Path, measures: tuple[str, ...] = DEFAULT_MEASURES) -> "ReferenceBank":
        frame = pd.read_csv(path)
        return cls(dataset, {m: np.sort(frame[m].dropna().to_numpy()) for m in measures if m in frame.columns})

    def percentile(self, measure: str, value: float) -> float:
        ref = self.values[measure]
        return float(100.0 * np.searchsorted(ref, value, side="right") / len(ref))

    def compare(self, measurements: dict[str, float]) -> ToolResult:
        """Percentile of each measurement among the unprocessed reference slices, with a plain-language flag."""
        values, notes = {}, []
        for measure, value in measurements.items():
            if measure not in self.values or not np.isfinite(value):
                continue
            pct = self.percentile(measure, value)
            values[f"{measure}_pct"] = round(pct, 1)
            ref = self.values[measure]
            if value < ref[0] or value > ref[-1]:
                notes.append(f"{measure}={value:.4g} is outside the range of all {len(ref)} unprocessed "
                             f"{self.dataset} slices [{ref[0]:.4g}, {ref[-1]:.4g}]")
            elif pct < 1 or pct > 99:
                notes.append(f"{measure}={value:.4g} is at the {pct:.1f}th percentile of unprocessed {self.dataset} slices")
        text = "; ".join(notes) if notes else f"All measures lie within the range of unprocessed {self.dataset} slices."
        return ToolResult(values, text=text)
