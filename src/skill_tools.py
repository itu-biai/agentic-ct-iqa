"""ToolSpecs behind each skill, for one dataset profile (value units, body threshold, window presets, reference bank).
Tool outputs from different datasets are not comparable in absolute units; each dataset gets its own profile."""
from dataclasses import dataclass
from pathlib import Path

from src.agent import Case, Skill, ToolSpec, load_skill
from src.config import CONFIG
from src.reference import DEFAULT_MEASURES, ReferenceBank
from src.tools import (ToolResult, air_artifact_energy, edge_width, flat_roi_sigma, fourier_high_band_fraction,
                       high_pass_energy, laplacian_variance, nlm_denoise, noise_residual_map, noise_sigma_estimate,
                       radial_nps_summary, streak_anisotropy, texture_coherence, window_view, zoom_view)
from src.tools_iqat1 import (fourier_magnitude_spectrum, gradient_orientation_coherence_map,
                             local_gradient_deviation_map, mscn_map)

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    body_min: float
    saturation: float
    windows: dict[str, tuple[float, float]]  # preset -> (level, width) in the dataset's value units
    reference_csv: Path | None = None  # per-image measurements of unprocessed training-split slices


# LDCT-IQA 2023 slices are already display-windowed to [0, 1] (air 0, bone clipped at 1);
# soft tissue sits around 0.64-0.75 (histogram peaks of MOS-4 training images).
LDCTIQA = DatasetProfile("LDCT-IQA", body_min=0.2, saturation=0.98,
                         windows={"full": (0.5, 1.0), "soft_tissue": (0.70, 0.30), "narrow": (0.70, 0.15)},
                         reference_csv=CONFIG.output_dir / "screening/train.csv")
# Dev and calibration runs query training images, so their bank must exclude the dev set.
LDCTIQA_DEV = DatasetProfile(**{**LDCTIQA.__dict__,
                                "reference_csv": CONFIG.output_dir / "screening/reference_ldctiqa_nodev.csv"})
# Stress datasets: per-image DICOM-windowed (PNG / 255, npz in [0, 1]); banks from their training splits only
# (IQA 2026v1: gt + noisy; I2I: the five doses + gt). Window presets are generic until checked on these images.
IQA2026 = DatasetProfile("IQA-2026v1", body_min=0.2, saturation=0.98, windows={"full": (0.5, 1.0), "narrow": (0.5, 0.5)},
                         reference_csv=CONFIG.output_dir / "screening/reference_iqa2026_train.csv")
I2I = DatasetProfile("I2I-2025v1", body_min=0.2, saturation=0.98, windows={"full": (0.5, 1.0), "narrow": (0.5, 0.5)},
                     reference_csv=CONFIG.output_dir / "screening/reference_i2i_train.csv")


def reference_measurements(image, profile: DatasetProfile) -> dict[str, float]:
    """The reference-bank measures for one image. A measure that cannot be computed on this image (for example no
    flat patch inside a small body) is left out instead of failing the whole comparison."""
    p = profile
    tools = (lambda: high_pass_energy(image), lambda: noise_sigma_estimate(image, p.body_min),
             lambda: laplacian_variance(image), lambda: fourier_high_band_fraction(image),
             lambda: radial_nps_summary(image, p.body_min, saturation=p.saturation), lambda: edge_width(image),
             lambda: air_artifact_energy(image, p.body_min), lambda: texture_coherence(image, p.body_min),
             lambda: ToolResult({"nlm_residual_std": noise_residual_map(image, nlm_denoise).values["residual_std"]}))
    values: dict[str, float] = {}
    for tool in tools:
        try:
            values.update(tool().values)
        except ValueError:
            continue
    return {k: v for k, v in values.items() if k in DEFAULT_MEASURES}


def _window_tool(profile: DatasetProfile) -> ToolSpec:
    def run(case: Case, preset: str) -> ToolResult:
        level, width = profile.windows[preset]
        return window_view(case.image, level, width)
    return ToolSpec("window_view", "Show the whole slice in a display window preset; narrower windows make noise and "
                    "texture more visible.",
                    {"type": "object", "properties": {"preset": {"type": "string", "enum": sorted(profile.windows)}},
                     "required": ["preset"], "additionalProperties": False}, run)


def _zoom_tool(profile: DatasetProfile) -> ToolSpec:
    # Stress-set profiles have no soft-tissue preset yet; their narrow window is the closest.
    preset = "soft_tissue" if "soft_tissue" in profile.windows else "narrow"
    level, width = profile.windows[preset]

    def run(case: Case, row: int, col: int) -> ToolResult:
        return zoom_view(case.image, row, col, level, width)
    return ToolSpec("zoom_view", f"Native-pixel 128x128 crop (shown 4x) at the given top-left row/col of the 512x512 "
                    f"slice, {preset.replace('_', '-')} window.",
                    {"type": "object", "properties": {"row": {"type": "integer"}, "col": {"type": "integer"}},
                     "required": ["row", "col"], "additionalProperties": False}, run)


def _no_input(name: str, description: str, fn) -> ToolSpec:
    return ToolSpec(name, description, {"type": "object", "properties": {}, "additionalProperties": False},
                    lambda case: fn(case.image))


def _reference_tool(profile: DatasetProfile) -> ToolSpec:
    bank = ReferenceBank.from_csv(profile.name, profile.reference_csv)
    return _no_input("compare_to_reference", f"Measure noise, texture and sharpness and report where each value falls "
                     f"among unprocessed {profile.name} training slices (percentiles). Values below every unprocessed "
                     f"slice mean the image was smoothed or blurred.",
                     lambda x: bank.compare(reference_measurements(x, profile)))


def expert_tool(card: str, logistic_params) -> ToolSpec:
    """The trained expert as a tool: its score for this slice mapped to the MOS scale, plus its profile card.
    Scores come with the Case (out-of-fold for training images, fold-model mean otherwise), never computed here."""
    from src.metrics import logistic4

    def run(case: Case) -> ToolResult:
        if not case.expert_scores:
            return ToolResult({}, text="No expert score is available for this slice.")
        values = {name: round(float(logistic4(score, *logistic_params)), 3) for name, score in case.expert_scores.items()}
        return ToolResult(values, text=card)
    return ToolSpec("consult_experts", "Scores of trained CT quality models for this slice on the 0-4 MOS scale, with "
                    "each model's profile card (where it errs and how it reacts to smoothing).",
                    {"type": "object", "properties": {}, "additionalProperties": False}, run)


def build_tools(profile: DatasetProfile) -> dict[str, tuple[ToolSpec, ...]]:
    p = profile
    tools = {
        "ct_view": (_window_tool(p), _zoom_tool(p)),
        "noise_measure": (
            _no_input("high_pass_energy", "Std of the high-pass residual (x - Gaussian(x), sigma 2). Higher means "
                      "more noise, but also more fine detail.", high_pass_energy),
            _no_input("noise_sigma_estimate", "Immerkaer noise std estimate inside the body.",
                      lambda x: noise_sigma_estimate(x, p.body_min)),
            _no_input("flat_roi_sigma", "Median std of the 8 flattest 64x64 patches inside the body (noise magnitude).",
                      lambda x: flat_roi_sigma(x, p.body_min, p.saturation)),
            _no_input("noise_residual_map", "Residual of a non-local-means denoiser as an image, with its std and its "
                      "correlation with edges (high correlation: structure, not only noise, is present in the residual).",
                      lambda x: noise_residual_map(x, nlm_denoise)),
        ),
        "texture_sharpness": (
            _no_input("radial_nps_summary", "Noise power spectrum of flat patches: mean frequency and share of power "
                      "above half Nyquist.", lambda x: radial_nps_summary(x, p.body_min, saturation=p.saturation)),
            _no_input("edge_width", "Median 10-90% rise distance (pixels) across the strongest edges. Wider means blur.",
                      edge_width),
            _no_input("fourier_high_band_fraction", "Share of Fourier power above a quarter of Nyquist.",
                      fourier_high_band_fraction),
            _no_input("texture_coherence", "Mean local gradient-orientation coherence inside the body: low for random "
                      "noise, high for structured tissue (also high after smoothing).",
                      lambda x: texture_coherence(x, p.body_min)),
        ),
        "artifact_check": (
            _no_input("air_artifact_energy", "High-pass energy in the air just outside the body, where a clean slice "
                      "is flat: streaks and noise measured without anatomy.", lambda x: air_artifact_energy(x, p.body_min)),
            _no_input("streak_anisotropy", "Angular non-uniformity of mid-frequency residual energy inside the body.",
                      lambda x: streak_anisotropy(x, p.body_min)),
        ),
        "evidence_maps": (
            _no_input("orientation_coherence_map", "IQA-T1 map: local gradient-orientation coherence (bright = "
                      "consistent edges or streaks, dark = random noise).", gradient_orientation_coherence_map),
            _no_input("fourier_spectrum", "IQA-T1 map: centred log Fourier magnitude; streak aliasing shows as spokes.",
                      fourier_magnitude_spectrum),
            _no_input("mscn_map", "IQA-T1 map: mean-subtracted contrast-normalised coefficients, with variance and "
                      "kurtosis.", mscn_map),
            _no_input("gradient_deviation_map", "IQA-T1 map: deviation of gradient magnitude from its local mean.",
                      local_gradient_deviation_map),
        ),
    }
    if p.reference_csv is not None and p.reference_csv.exists():
        tools["reference_check"] = (_reference_tool(p),)
    card_path = CONFIG.expert_dir / "profile_card.txt"
    params_path = CONFIG.expert_dir / "expert_logistic_params.npy"
    if card_path.exists() and params_path.exists():
        import numpy as np
        tools["expert_models"] = (expert_tool(card_path.read_text(encoding="utf-8"), np.load(params_path)),)
    return tools


def build_skills(profile: DatasetProfile, names: list[str]) -> list[Skill]:
    tools = build_tools(profile)
    return [load_skill(SKILLS_DIR / name, tools.get(name, ())) for name in names]
