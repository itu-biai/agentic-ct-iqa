"""Synthetic LDCT-IQA-style degradations of full-dose abdominal CT slices from other patients (MSD Task09 Spleen,
CC-BY-SA 4.0), for rank pre-training of the vision encoder (colab/vm/vftb_train.py, ROOT/synth present).

The challenge pipeline (Lee et al., MedIA 2025, Sec. 2.2): forward projection of the full-dose slice with 720, 360 or
180 views per rotation, Poisson noise for about 100%, 50%, 25% and 10% of the full dose, FBP with a Shepp-Logan ramp,
then HU windowed to [0, 1] with width 1400 / level -300. Here: parallel beam over 180 degrees (360, 180, 90 views, the
same angular step as 720/360/180 over 360 degrees), detector pitch = pixel, torch so it runs on the GPU. The full-dose
count I0 is a calibration constant matched to the training slices' noise (see scripts/synth_calibrate.py).

No MOS anywhere: within one source slice, a variant with less noise and at least as many views (or vice versa) is
better; that partial order is the training signal.
"""
import numpy as np
import torch
import torch.nn.functional as F

MU_WATER = 0.0192  # 1/mm at ~70 keV
VIEWS = (360, 180, 90)  # streak levels 0, 1, 2 (= 720, 360, 180 per full rotation)
DOSES = (1.0, 0.5, 0.25, 0.1)  # noise levels 0..3
TYPES = [(n, s) for n in range(4) for s in range(3)]  # (noise level, streak level), 12 types
TARGET_SPACING = (0.6, 0.85)  # mm per pixel after zoom()
# Calibrated on the training slices (scripts/synth_calibrate.py, 2026-10-20): flat-patch noise 5/50/95% 0.025/0.045/0.086
# (training 0.027/0.053/0.082), NPS mean frequency 0.017/0.031/0.090 (0.021/0.035/0.109). Count-dependent noise
# (alpha 1) put far more noise in the body centre than the training slices show, so the projection noise is uniform.
I0, ALPHA, BLUR = 3e3, 0.0, 0.7


def build_bank(paths: list, per_volume: int, variants: int, seed: int, device, log=print) -> dict[str, np.ndarray]:
    """`per_volume` slices from the middle 70% of each volume (body at least 15% of the frame), `variants` of the 12
    types each (no repeats), random zoom, full-dose count jittered x0.6-1.6 and detector blur 0.5-0.9 per slice.
    Returns images (uint16 of the [0, 1] window), base (slice index), noise and streak levels."""
    import nibabel as nib
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    images, base, noise, streak = [], [], [], []
    for v, path in enumerate(paths):
        img = nib.load(path)
        vol = np.asarray(img.dataobj, dtype=np.float32)
        spacing = float(img.header.get_zooms()[0])
        if vol.shape[0] != vol.shape[1]:
            continue
        zs = np.linspace(0.15 * vol.shape[2], 0.85 * vol.shape[2], per_volume).astype(int)
        for z in zs:
            hu = torch.from_numpy(np.rot90(vol[:, :, z]).copy()).to(device)
            if hu.shape[-1] != 512 or (hu > -500).float().mean() < 0.15:
                continue
            target = float(rng.uniform(*TARGET_SPACING))
            hu = zoom(hu, spacing, target)
            i0, blur = I0 * float(np.exp(rng.uniform(np.log(0.6), np.log(1.6)))), float(rng.uniform(0.5, 0.9))
            slice_id = base[-1] + 1 if base else 0
            for t in rng.choice(len(TYPES), variants, replace=False):
                n, s = TYPES[t]
                out = window01(degrade(hu, target, n, s, i0, generator, ALPHA, blur))
                images.append((out * 65535).round().cpu().numpy().astype(np.uint16))
                base.append(slice_id); noise.append(n); streak.append(s)
        if v % 10 == 0:
            log(f"synthetic bank: {v + 1}/{len(paths)} volumes, {len(images)} images")
    return {"images": np.stack(images), "base": np.array(base), "noise": np.array(noise), "streak": np.array(streak)}


def window01(hu: np.ndarray | torch.Tensor):
    """The dataset's storage window: width 1400, level -300 -> [0, 1]."""
    return ((hu + 1000.0) / 1400.0).clip(0, 1)


def angles(views: int, device) -> torch.Tensor:
    return torch.arange(views, device=device, dtype=torch.float32) * (np.pi / views)


def radon(mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """mu [H, W] (attenuation per pixel, zero outside the inscribed circle) -> sinogram [A, W] of line integrals."""
    size = mu.shape[-1]
    t = torch.linspace(-1, 1, size, device=mu.device)  # detector and ray positions in grid_sample units
    s = t
    cos, sin = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
    tt, ss = t[None, :, None], s[None, None, :]
    x = tt * cos - ss * sin
    y = tt * sin + ss * cos
    grid = torch.stack([x, y], -1)  # [A, W(t), W(s), 2]
    out = []
    for chunk in torch.split(torch.arange(len(theta), device=mu.device), 45):
        sampled = F.grid_sample(mu[None, None].expand(len(chunk), 1, -1, -1), grid[chunk], align_corners=True)
        out.append(sampled[:, 0].sum(-1))  # sum along the ray, step = one pixel
    return torch.cat(out)


def shepp_logan(size: int, device) -> torch.Tensor:
    """Ramp built in the spatial domain (Kak and Slaney) times the Shepp-Logan sinc, as in skimage.iradon."""
    n = torch.cat([torch.arange(1, size // 2 + 1, 2), torch.arange(size // 2 - 1, 0, -2)]).to(device)
    f = torch.zeros(size, device=device)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n.float()) ** 2
    ramp = 2 * torch.fft.fft(f).real
    omega = np.pi * torch.fft.fftfreq(size, device=device)
    sinc = torch.ones_like(omega)
    sinc[1:] = torch.sin(omega[1:] / 2) / (omega[1:] / 2)
    return ramp * sinc


def fbp(sino: torch.Tensor, theta: torch.Tensor, size: int) -> torch.Tensor:
    """Filtered backprojection of a parallel-beam sinogram [A, W] -> [H, W]."""
    pad = max(64, int(2 ** np.ceil(np.log2(2 * size))))
    filtered = torch.fft.ifft(torch.fft.fft(F.pad(sino, (0, pad - size)), dim=-1) * shepp_logan(pad, sino.device),
                              dim=-1).real[:, :size]
    t = torch.linspace(-1, 1, size, device=sino.device)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    image = torch.zeros(size, size, device=sino.device)
    for chunk in torch.split(torch.arange(len(theta), device=sino.device), 45):
        pos = xx[None] * torch.cos(theta[chunk])[:, None, None] + yy[None] * torch.sin(theta[chunk])[:, None, None]
        grid = torch.stack([pos, torch.zeros_like(pos)], -1)  # [a, H, W, 2], y ignored (height 1)
        image += F.grid_sample(filtered[chunk][:, None, None, :], grid, align_corners=True)[:, 0].sum(0)
    return image * np.pi / (2 * len(theta))


def to_mu(hu: torch.Tensor, spacing: float) -> torch.Tensor:
    """HU slice (square, any air value) -> attenuation per pixel, air and everything outside the circle set to 0."""
    size = hu.shape[-1]
    t = torch.linspace(-1, 1, size, device=hu.device)
    inside = (t[None] ** 2 + t[:, None] ** 2) <= 1
    return (MU_WATER * spacing * (1 + hu.clamp(min=-1000) / 1000)).clamp(min=0) * inside


def zoom(hu: torch.Tensor, spacing: float, target: float) -> torch.Tensor:
    """Resample a 512 slice about its centre to `target` mm per pixel (training slices fill more of the frame than the
    MSD scans); outside the source frame is air."""
    size = hu.shape[-1]
    t = torch.linspace(-1, 1, size, device=hu.device) * (target / spacing)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    grid = torch.stack([xx, yy], -1)[None]
    return F.grid_sample((hu + 1000)[None, None], grid, align_corners=True, padding_mode="zeros")[0, 0] - 1000


def degrade(hu: torch.Tensor, spacing: float, noise: int, streak: int, i0: float, generator,
            alpha: float = ALPHA, blur: float = BLUR) -> torch.Tensor:
    """One distortion type of one slice -> HU image. noise 0 adds nothing (the source is the full dose); higher levels add
    Poisson noise so the total variance is about 1/dose times the full dose's: counts I0 * d / (1 - d)."""
    mu = to_mu(hu, spacing)
    theta = angles(VIEWS[streak], hu.device)
    sino = radon(mu, theta)
    dose = DOSES[noise]
    if dose < 1:  # alpha < 1 flattens the noise across the body, as a bowtie filter and tube-current modulation do
        counts = i0 * dose / (1 - dose) * torch.exp(-alpha * sino)
        noise_term = -torch.log(torch.poisson(counts, generator=generator).clamp(min=1) / counts)
        if blur > 0:  # detector aperture wider than a pixel: the training slices' noise is coarser than white
            x = torch.arange(-3 * int(np.ceil(blur)), 3 * int(np.ceil(blur)) + 1, device=hu.device, dtype=torch.float32)
            kernel = torch.exp(-x ** 2 / (2 * blur ** 2))
            noise_term = F.conv1d(noise_term[:, None], (kernel / kernel.sum())[None, None], padding=len(x) // 2)[:, 0]
        sino = sino + noise_term
    image = fbp(sino, theta, hu.shape[-1]) / (MU_WATER * spacing) * 1000 - 1000
    size = hu.shape[-1]
    t = torch.linspace(-1, 1, size, device=hu.device)
    return torch.where((t[None] ** 2 + t[:, None] ** 2) <= 1, image, torch.full_like(image, -1000.0))
