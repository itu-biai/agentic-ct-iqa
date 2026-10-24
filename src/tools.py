"""Measurement and view tools behind the agent skills (paper Section 2.2). No tool ever sees the MOS."""
import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class ToolResult:
    values: dict[str, float]
    image_png: bytes | None = None
    text: str | None = None


def to_png(image: np.ndarray, low: float, high: float) -> bytes:
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    buffer = io.BytesIO()
    Image.fromarray((scaled * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def window_view(image: np.ndarray, level: float, width: float) -> ToolResult:
    return ToolResult({"level": level, "width": width}, to_png(image, level - width / 2, level + width / 2))


def zoom_view(image: np.ndarray, row: int, col: int, level: float, width: float, size: int = 128) -> ToolResult:
    """Native-pixel crop upsampled 4x with np.kron, so the model sees pixels and not a resampling blur.
    Keep `size` fixed so zooms are comparable across calls."""
    row = int(np.clip(row, 0, image.shape[0] - size))
    col = int(np.clip(col, 0, image.shape[1] - size))
    crop = np.kron(image[row:row + size, col:col + size], np.ones((4, 4)))
    return ToolResult({"row": float(row), "col": float(col), "size": float(size)}, to_png(crop, level - width / 2, level + width / 2))


def high_pass_energy(image: np.ndarray, sigma: float = 2.0) -> ToolResult:
    """Tolga's scalar: std of the high-pass residual x - G_sigma * x."""
    return ToolResult({"high_pass_std": float((image - ndimage.gaussian_filter(image, sigma)).std())})


def fourier_high_band_fraction(image: np.ndarray, cutoff: float = 0.25) -> ToolResult:
    """Share of Fourier power at radial frequency >= cutoff x Nyquist (no mean removal).
    Tolga's MedIA manuscript, Table 15: reproduces Overall 2.425 on LDCT-IQA test."""
    power = np.abs(np.fft.fft2(image)) ** 2
    freq = np.fft.fftfreq(image.shape[0])
    radius = np.hypot(*np.meshgrid(freq, np.fft.fftfreq(image.shape[1]), indexing="ij")) / 0.5
    return ToolResult({"fourier_high_band_fraction": float(power[radius >= cutoff].sum() / power.sum())})


def laplacian_variance(image: np.ndarray) -> ToolResult:
    """Variance of the Laplacian (autofocus sharpness operator). Tolga's Table 15: Overall 2.156."""
    return ToolResult({"laplacian_variance": float(ndimage.laplace(image).var())})


def immerkaer_sigma(image: np.ndarray) -> float:
    """Fast noise std estimate (Immerkaer 1996): mean |x * M| with a 3x3 Laplacian-difference mask, scaled."""
    kernel = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    response = ndimage.convolve(image.astype(np.float64), kernel, mode="reflect")[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(response).mean() / 6.0)


def noise_sigma_estimate(image: np.ndarray, body_min: float) -> ToolResult:
    """Immerkaer noise std inside the body bounding box (air outside the body is clipped and would bias it low)."""
    rows, cols = np.nonzero(ndimage.gaussian_filter(image, 4.0) > body_min)
    if rows.size == 0:
        return ToolResult({"immerkaer_sigma": float("nan")})
    box = image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return ToolResult({"immerkaer_sigma": immerkaer_sigma(box)})


def nlm_denoise(image: np.ndarray) -> np.ndarray:
    """Non-local means (scikit-image), the plan's fallback when BM3D is not used; h is set from the Immerkaer sigma."""
    from skimage.restoration import denoise_nl_means
    sigma = immerkaer_sigma(image)
    return denoise_nl_means(image, h=0.8 * sigma, sigma=sigma, patch_size=5, patch_distance=6, fast_mode=True)


def flattest_patches(image: np.ndarray, body_min: float, size: int = 64, count: int = 8,
                     saturation: float = np.inf) -> list[np.ndarray]:
    """The `count` patches inside the body with the lowest mean gradient (half-overlapping grid).
    Patches with pixels at or above `saturation` are skipped: in display-windowed images, clipped
    regions (bone, contrast) look perfectly flat and would read as zero noise."""
    gradient = ndimage.gaussian_gradient_magnitude(image, 2.0)
    smooth = ndimage.gaussian_filter(image, 3.0)  # mask on the smoothed image, so noise alone does not break it
    body = (smooth > body_min) & (smooth < saturation)
    candidates = []
    for row in range(0, image.shape[0] - size + 1, size // 2):
        for col in range(0, image.shape[1] - size + 1, size // 2):
            if body[row:row + size, col:col + size].all():
                candidates.append((gradient[row:row + size, col:col + size].mean(), row, col))
    if not candidates:
        raise ValueError("no patch lies fully inside the body; check body_min for this dataset")
    candidates.sort()
    return [image[r:r + size, c:c + size] for _, r, c in candidates[:count]]


def flat_roi_sigma(image: np.ndarray, body_min: float, saturation: float = np.inf) -> ToolResult:
    patches = flattest_patches(image, body_min, saturation=saturation)
    return ToolResult({"flat_roi_sigma": float(np.median([p.std() for p in patches]))})


def radial_nps_summary(image: np.ndarray, body_min: float, pixel_mm: float = 1.0, saturation: float = np.inf) -> ToolResult:
    """Radially averaged NPS of the flattest patches; smoothing lowers f_mean and the high-band ratio."""
    patches = flattest_patches(image, body_min, saturation=saturation)
    size = patches[0].shape[0]
    nps = np.mean([np.abs(np.fft.fftshift(np.fft.fft2(p - p.mean()))) ** 2 for p in patches], axis=0) * pixel_mm**2 / size**2
    freq = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_mm))
    radius = np.hypot(*np.meshgrid(freq, freq))
    edges = np.linspace(0.0, radius.max(), size // 2 + 1)
    index = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(index, minlength=len(edges) - 1)
    radial = np.bincount(index, weights=nps.ravel(), minlength=len(edges) - 1) / np.maximum(counts, 1)
    centers = (edges[:-1] + edges[1:]) / 2
    nyquist = 0.5 / pixel_mm
    return ToolResult({
        "nps_f_mean": float((centers * radial).sum() / radial.sum()),
        "nps_high_band_ratio": float(radial[centers >= nyquist / 2].sum() / radial.sum()),
    })


def _crossing(profile: np.ndarray, offsets: np.ndarray, level: float) -> float:
    """Sub-sample position of the first upward crossing of `level` (linear interpolation)."""
    index = int(np.argmax(profile >= level))
    if index == 0:
        return float(offsets[0])
    before, after = profile[index - 1], profile[index]
    return float(offsets[index - 1] + (level - before) / (after - before) * (offsets[index] - offsets[index - 1]))


def edge_width(image: np.ndarray, count: int = 200, seed: int = 0) -> ToolResult:
    """Median 10%-90% rise distance of intensity profiles across the strongest edges."""
    gy, gx = np.gradient(ndimage.gaussian_filter(image, 1.0))
    magnitude = np.hypot(gx, gy)
    rows, cols = np.nonzero(magnitude >= np.quantile(magnitude, 0.995))
    pick = np.random.default_rng(seed).choice(len(rows), size=min(count, len(rows)), replace=False)
    offsets = np.linspace(-8.0, 8.0, 33)
    widths = []
    for r, c in zip(rows[pick], cols[pick]):
        direction = np.array([gy[r, c], gx[r, c]]) / magnitude[r, c]
        coords = np.stack([r + offsets * direction[0], c + offsets * direction[1]])
        profile = ndimage.map_coordinates(image, coords, order=1)
        low, high = profile[:4].mean(), profile[-4:].mean()
        if abs(high - low) < 1e-6:
            continue
        normalized = (profile - low) / (high - low)
        if not (normalized >= 0.9).any():
            continue
        t10, t90 = _crossing(normalized, offsets, 0.1), _crossing(normalized, offsets, 0.9)
        if t90 > t10:
            widths.append(t90 - t10)
    return ToolResult({"edge_width_px": float(np.median(widths)) if widths else float("nan")})


def streak_anisotropy(image: np.ndarray, body_min: float, angles: int = 180) -> ToolResult:
    """Angular non-uniformity of mid-frequency residual energy inside the eroded body mask."""
    body = ndimage.binary_erosion(image > body_min, iterations=10)
    residual = (image - ndimage.gaussian_filter(image, 2.0)) * body
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual))) ** 2
    yy, xx = np.indices(spectrum.shape) - np.array(spectrum.shape)[:, None, None] // 2
    radius, theta = np.hypot(yy, xx), np.mod(np.arctan2(yy, xx), np.pi)
    ring = (radius > 0.15 * image.shape[0]) & (radius < 0.45 * image.shape[0])
    bins = np.minimum((theta[ring] / np.pi * angles).astype(int), angles - 1)
    energy = np.bincount(bins, weights=spectrum[ring], minlength=angles)
    return ToolResult({"streak_anisotropy": float(energy.std() / energy.mean())})


def body_and_air_masks(image: np.ndarray, body_min: float, air_max: float = 0.08,
                       ring: tuple[int, int] = (4, 40)) -> tuple[np.ndarray, np.ndarray]:
    """Body = largest filled connected region of the smoothed slice above body_min. Air = a ring of `ring` pixels
    around the body, inside the field of view, where the smoothed slice is near zero (excludes the table)."""
    smooth = ndimage.gaussian_filter(image, 4.0)
    body = ndimage.binary_fill_holes(smooth > body_min)
    labels, count = ndimage.label(body)
    if count:
        body = labels == (int(np.argmax(ndimage.sum(body, labels, range(1, count + 1)))) + 1)
    yy, xx = np.indices(image.shape)
    centre, radius = (np.array(image.shape) - 1) / 2, 0.49 * min(image.shape)
    fov = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 < radius ** 2
    near = ndimage.binary_dilation(body, iterations=ring[1]) & ~ndimage.binary_dilation(body, iterations=ring[0])
    return body, near & fov & (smooth < air_max)


def air_artifact_energy(image: np.ndarray, body_min: float) -> ToolResult:
    """High-pass energy in the air just outside the body. A clean slice has flat air, so anything there is streaks
    or noise, measured without anatomy. On LDCT-IQA training data it adds information beyond the high-pass scalar
    (partial Spearman with MOS given the scalar: -0.38)."""
    _, air = body_and_air_masks(image, body_min)
    residual = image - ndimage.gaussian_filter(image, 2.0)
    if air.sum() < 500:
        return ToolResult({"air_hp_std": float("nan")}, text="Too little air around the body to measure.")
    return ToolResult({"air_hp_std": float(residual[air].std()), "air_std": float(image[air].std())})


def texture_coherence(image: np.ndarray, body_min: float, ksize: int = 15) -> ToolResult:
    """Mean local orientation coherence of Sobel gradients inside the body (IQA-T1's coherence map, restricted to
    tissue). Random noise gives low coherence; anatomy and edges give high coherence. Partial Spearman with MOS given
    the high-pass scalar on LDCT-IQA training data: +0.54. Smoothing also raises it, so it does not detect blur."""
    body, _ = body_and_air_masks(image, body_min)
    inner = ndimage.binary_erosion(body, iterations=10)
    gx, gy = ndimage.sobel(image, axis=1), ndimage.sobel(image, axis=0)
    magnitude = np.hypot(gx, gy)
    safe = np.where(magnitude > 1e-6, magnitude, 1.0)
    ux, uy = np.where(magnitude > 1e-6, gx / safe, 0.0), np.where(magnitude > 1e-6, gy / safe, 0.0)
    coherence = np.hypot(ndimage.uniform_filter(ux, ksize), ndimage.uniform_filter(uy, ksize))
    return ToolResult({"body_coherence": float(coherence[inner].mean())})


def noise_residual_map(image: np.ndarray, denoiser: Callable[[np.ndarray], np.ndarray]) -> ToolResult:
    """Residual of a classical denoiser; a residual that correlates with edges means structure is removed."""
    denoised = denoiser(image)
    residual = image - denoised
    edges = np.abs(ndimage.sobel(denoised))
    edge_corr = float(np.corrcoef(np.abs(residual).ravel(), edges.ravel())[0, 1])
    scale = 3.0 * residual.std()
    return ToolResult({"residual_std": float(residual.std()), "residual_edge_corr": edge_corr}, to_png(residual, -scale, scale))
