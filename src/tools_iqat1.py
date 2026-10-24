"""Pure-Python ports of the grayscale evidence tools of IQA-T1 (Wu et al., ECCV 2026; github.com/zibuyu-02/IQA-T1,
scripts/tools.py). Plan Step 1.3: try the public tool code and keep any tool that adds information.

The originals use OpenCV on 8-bit images and min-max normalise each map for display; the ports keep that rendering
(so the agent sees the same maps) and add summary scalars so the tools can also be screened numerically.
Colour tools (colourfulness, chromatic deviation, saturation clipping) do not apply to grayscale CT and are omitted.
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage

from src.tools import ToolResult


def _gray255(image: np.ndarray) -> np.ndarray:
    """IQA-T1 works on 8-bit images; our slices are in [0, 1]."""
    return np.clip(image, 0.0, 1.0).astype(np.float32) * 255.0


def _sobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cv2.Sobel(ksize=3) equivalents: gx along columns, gy along rows."""
    return ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)


def _gauss(gray: np.ndarray, sigma: float, ksize: int | None = None) -> np.ndarray:
    """cv2.GaussianBlur: with ksize given, truncate the kernel to that size; with (0, 0), cv2 uses about 3 sigma."""
    truncate = ((ksize - 1) / 2) / sigma if ksize else 3.0
    return ndimage.gaussian_filter(gray, sigma, truncate=truncate)


def _minmax_png(values: np.ndarray) -> bytes:
    low, high = float(values.min()), float(values.max())
    scaled = (values - low) / (high - low) if high > low else np.zeros_like(values)
    buffer = io.BytesIO()
    Image.fromarray((scaled * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def gradient_magnitude_map(image: np.ndarray) -> ToolResult:
    gx, gy = _sobel(_gray255(image))
    magnitude = np.hypot(gx, gy)
    shown = np.clip(magnitude, 0, 255)  # normalize_grad_fixed(clip_val=255)
    return ToolResult({"gradient_mean": float(magnitude.mean()), "gradient_p95": float(np.percentile(magnitude, 95))},
                      _minmax_png(shown))


def gradient_orientation_coherence_map(image: np.ndarray, ksize: int = 15) -> ToolResult:
    """Length of the box-averaged unit gradient vector: 1 = consistent orientation (edges/streaks), 0 = random (noise)."""
    gx, gy = _sobel(_gray255(image))
    magnitude = np.hypot(gx, gy)
    nonzero = magnitude >= 1e-6
    ux, uy = np.zeros_like(gx), np.zeros_like(gy)
    ux[nonzero], uy[nonzero] = gx[nonzero] / magnitude[nonzero], gy[nonzero] / magnitude[nonzero]
    coherence = np.hypot(ndimage.uniform_filter(ux, ksize), ndimage.uniform_filter(uy, ksize))
    return ToolResult({"orientation_coherence_mean": float(coherence.mean())}, _minmax_png(coherence))


def high_frequency_residual_map(image: np.ndarray, sigma: float = 3.0) -> ToolResult:
    gray = _gray255(image)
    residual = np.abs(gray - _gauss(gray, sigma))
    return ToolResult({"hf_residual_mean": float(residual.mean())}, _minmax_png(residual))


def dog_sharpness_map(image: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.5) -> ToolResult:
    gray = _gray255(image)
    dog = np.abs(_gauss(gray, sigma1) - _gauss(gray, sigma2))
    return ToolResult({"dog_mean": float(dog.mean())}, _minmax_png(dog))


def structure_masked_noise_map(image: np.ndarray) -> ToolResult:
    """IQA-T1 NoiseResidualMap: |x - G_1(x)| down-weighted where the smoothed gradient (structure) is strong."""
    gray = _gray255(image)
    high = gray - _gauss(gray, 1.0)
    gx, gy = _sobel(gray)
    smooth = _gauss(np.hypot(gx, gy), 1.0, ksize=5)
    mask = 1.0 - (smooth / (smooth.max() + 1e-6)) ** 0.6
    residual = np.abs(high) * mask
    return ToolResult({"masked_noise_mean": float(residual.mean())}, _minmax_png(residual))


def fourier_magnitude_spectrum(image: np.ndarray) -> ToolResult:
    """log(1 + |F|), centred. Periodic patterns and sparse-view aliasing show up as spokes or rings."""
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(_gray255(image)))))
    return ToolResult({"log_spectrum_mean": float(spectrum.mean())}, _minmax_png(spectrum))


def mscn_map(image: np.ndarray, ksize: int = 7, sigma: float = 1.166) -> ToolResult:
    """Mean-subtracted contrast-normalised coefficients (BRISQUE/NIQE building block) and their distribution shape."""
    gray = _gray255(image)
    mu = _gauss(gray, sigma, ksize)
    local_sigma = np.sqrt(np.abs(_gauss(gray * gray, sigma, ksize) - mu * mu))
    mscn = (gray - mu) / (local_sigma + 1e-8)
    data = mscn[np.abs(mscn) < 10]
    centered = data - data.mean()
    kurtosis = float((centered ** 4).mean() / (centered.var() ** 2 + 1e-12))
    return ToolResult({"mscn_variance": float(data.var()), "mscn_kurtosis": kurtosis}, _minmax_png(mscn))


def local_gradient_deviation_map(image: np.ndarray) -> ToolResult:
    gx, gy = _sobel(_gray255(image))
    magnitude = np.hypot(gx, gy)
    deviation = np.abs(magnitude - _gauss(magnitude, 1.5, ksize=7))
    return ToolResult({"gradient_deviation_mean": float(deviation.mean())}, _minmax_png(deviation))


IQAT1_TOOLS = {
    "gradient_magnitude_map": gradient_magnitude_map,
    "gradient_orientation_coherence_map": gradient_orientation_coherence_map,
    "high_frequency_residual_map": high_frequency_residual_map,
    "dog_sharpness_map": dog_sharpness_map,
    "structure_masked_noise_map": structure_masked_noise_map,
    "fourier_magnitude_spectrum": fourier_magnitude_spectrum,
    "mscn_map": mscn_map,
    "local_gradient_deviation_map": local_gradient_deviation_map,
}
