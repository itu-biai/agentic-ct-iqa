import numpy as np
import pytest
from scipy import ndimage

from src.tools import (air_artifact_energy, edge_width, flat_roi_sigma, fourier_high_band_fraction, high_pass_energy,
                       immerkaer_sigma, laplacian_variance, noise_residual_map, radial_nps_summary, streak_anisotropy,
                       texture_coherence, window_view, zoom_view)
from src.tools_iqat1 import IQAT1_TOOLS

BODY_MIN = -500.0


def phantom(noise_std: float = 20.0, seed: int = 0) -> np.ndarray:
    """HU-like phantom: air background, a water body with two inserts, white noise."""
    yy, xx = np.mgrid[:512, :512]
    image = np.full((512, 512), -1000.0)
    image[(yy - 256) ** 2 + (xx - 256) ** 2 < 220 ** 2] = 0.0
    image[(yy - 200) ** 2 + (xx - 180) ** 2 < 30 ** 2] = 200.0
    image[(yy - 320) ** 2 + (xx - 330) ** 2 < 25 ** 2] = -100.0
    return image + np.random.default_rng(seed).normal(0, noise_std, image.shape)


def test_flat_roi_sigma_recovers_noise_std():
    assert flat_roi_sigma(phantom(20.0), BODY_MIN).values["flat_roi_sigma"] == pytest.approx(20.0, rel=0.05)


def test_blur_widens_edges_and_lowers_nps_frequency():
    sharp = phantom(5.0)
    blurred = ndimage.gaussian_filter(sharp, 2.0)
    assert edge_width(blurred).values["edge_width_px"] > edge_width(sharp).values["edge_width_px"]
    assert radial_nps_summary(blurred, BODY_MIN).values["nps_f_mean"] < radial_nps_summary(sharp, BODY_MIN).values["nps_f_mean"]
    assert (radial_nps_summary(blurred, BODY_MIN).values["nps_high_band_ratio"]
            < radial_nps_summary(sharp, BODY_MIN).values["nps_high_band_ratio"])


def test_high_pass_energy_grows_with_noise():
    assert high_pass_energy(phantom(40.0)).values["high_pass_std"] > high_pass_energy(phantom(10.0)).values["high_pass_std"]


def test_streaks_raise_anisotropy():
    clean = phantom(10.0)
    yy, xx = np.mgrid[:512, :512]
    streaked = clean + 60.0 * np.sin(2 * np.pi * (xx * np.cos(0.3) + yy * np.sin(0.3)) / 6.0) * (clean > BODY_MIN)
    assert streak_anisotropy(streaked, BODY_MIN).values["streak_anisotropy"] > streak_anisotropy(clean, BODY_MIN).values["streak_anisotropy"]


def test_views_return_png():
    image = phantom()
    assert window_view(image, 40, 400).image_png[:4] == b"\x89PNG"
    zoom = zoom_view(image, 500, 500, 40, 400)
    assert zoom.values["row"] == 384 and zoom.image_png[:4] == b"\x89PNG"


def test_residual_of_structure_removing_denoiser_correlates_with_edges():
    image = phantom(5.0)
    oversmooth = noise_residual_map(image, lambda x: ndimage.gaussian_filter(x, 3.0)).values["residual_edge_corr"]
    gentle = noise_residual_map(image, lambda x: ndimage.median_filter(x, 3)).values["residual_edge_corr"]
    assert oversmooth > gentle


def windowed_phantom(noise_std: float, seed: int = 0) -> np.ndarray:
    """Phantom in display units [0, 1] like LDCT-IQA: air 0, body ~0.65, insert ~0.8."""
    return np.clip((phantom(noise_std * 1000, seed) + 1000) / 1600, 0, 1)


def test_immerkaer_recovers_white_noise_std():
    flat = np.full((256, 256), 0.6) + np.random.default_rng(1).normal(0, 0.03, (256, 256))
    assert immerkaer_sigma(flat) == pytest.approx(0.03, rel=0.05)


def test_high_frequency_descriptors_grow_with_noise():
    clean, noisy = windowed_phantom(0.005), windowed_phantom(0.04)
    assert fourier_high_band_fraction(noisy).values["fourier_high_band_fraction"] > fourier_high_band_fraction(clean).values["fourier_high_band_fraction"]
    assert laplacian_variance(noisy).values["laplacian_variance"] > laplacian_variance(clean).values["laplacian_variance"]


def test_air_artifact_energy_sees_streaks_outside_body():
    clean = windowed_phantom(0.01)
    yy, xx = np.mgrid[:512, :512]
    lines = 0.15 * (np.sin(2 * np.pi * (xx * np.cos(0.4) + yy * np.sin(0.4)) / 5.0) > 0.9)
    streaked = np.clip(clean + lines, 0, 1)
    assert air_artifact_energy(streaked, 0.2).values["air_hp_std"] > 3 * air_artifact_energy(clean, 0.2).values["air_hp_std"]


def test_texture_coherence_drops_with_noise():
    assert texture_coherence(windowed_phantom(0.04), 0.2).values["body_coherence"] < texture_coherence(windowed_phantom(0.005), 0.2).values["body_coherence"]


def test_iqat1_ports_return_maps_and_finite_values():
    image = windowed_phantom(0.02)
    for name, tool in IQAT1_TOOLS.items():
        result = tool(image)
        assert result.image_png[:4] == b"\x89PNG", name
        assert all(np.isfinite(v) for v in result.values.values()), name
