"""Controlled degradations used only to validate tools (never to score the benchmark).
Sparse-view streaks follow the LDCT-IQA recipe: forward projection, fewer views, FBP with a Shepp-Logan filter."""
import numpy as np
from skimage.transform import iradon, radon


def sparse_view(image: np.ndarray, n_views: int) -> np.ndarray:
    """Reconstruct `image` from n_views parallel projections over 180 degrees (FBP, Shepp-Logan)."""
    angles = np.linspace(0.0, 180.0, n_views, endpoint=False)
    sinogram = radon(image, theta=angles, circle=True)
    recon = iradon(sinogram, theta=angles, filter_name="shepp-logan", circle=True, output_size=image.shape[0])
    return recon.astype(np.float32)
