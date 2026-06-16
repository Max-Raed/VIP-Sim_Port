#!/usr/bin/env python3
"""
VipSim image filters — Python reimplementation of VIP-Sim Unity shaders.

Simulates visual impairments on screenshots for VLM accessibility analysis.
Based on shader code from https://github.com/Max-Raed/VIP-Sim (CC BY 4.0).

Usage:
    python vipsim_filters.py input.png output.png --filter blur --severity 0.5
    python vipsim_filters.py input.png output.png --filter cvd --type deuteranomaly --severity 0.8
    python vipsim_filters.py input.png output.png --filter contrast --brightness 0.8 --contrast 0.7 --gamma 1.5
    python vipsim_filters.py input.png output.png --filter cataracts --severity 0.5
    python vipsim_filters.py input.png output.png --preset moderate_low_vision
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


# =============================================================================
# Color Vision Deficiency (from myRecolour.cs)
# =============================================================================
# Exact transformation matrices from VIP-Sim, indexed by severity 0.0-1.0 (11 steps)

CVD_MATRICES = {
    "protanomaly": np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0.856167, 0.182038, -0.038205], [0.029342, 0.955115, 0.015544], [-0.00288, -0.001563, 1.004443]],
        [[0.734766, 0.334872, -0.069637], [0.05184, 0.919198, 0.028963], [-0.004928, -0.004209, 1.009137]],
        [[0.630323, 0.465641, -0.095964], [0.069181, 0.890046, 0.040773], [-0.006308, -0.007724, 1.014032]],
        [[0.539009, 0.579343, -0.118352], [0.082546, 0.866121, 0.051332], [-0.007136, -0.011959, 1.019095]],
        [[0.458064, 0.679578, -0.137642], [0.092785, 0.846313, 0.060902], [-0.007494, -0.016807, 1.024301]],
        [[0.38545, 0.769005, -0.154455], [0.100526, 0.829802, 0.069673], [-0.007442, -0.02219, 1.029632]],
        [[0.319627, 0.849633, -0.169261], [0.106241, 0.815969, 0.07779], [-0.007025, -0.028051, 1.035076]],
        [[0.259411, 0.923008, -0.18242], [0.110296, 0.80434, 0.085364], [-0.006276, -0.034346, 1.040622]],
        [[0.203876, 0.990338, -0.194214], [0.112975, 0.794542, 0.092483], [-0.005222, -0.041043, 1.046265]],
        [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    ]),
    "deuteranomaly": np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0.866435, 0.177704, -0.044139], [0.049567, 0.939063, 0.01137], [-0.003453, 0.007233, 0.99622]],
        [[0.760729, 0.319078, -0.079807], [0.090568, 0.889315, 0.020117], [-0.006027, 0.013325, 0.992702]],
        [[0.675425, 0.43385, -0.109275], [0.125303, 0.847755, 0.026942], [-0.00795, 0.018572, 0.989378]],
        [[0.605511, 0.52856, -0.134071], [0.155318, 0.812366, 0.032316], [-0.009376, 0.023176, 0.9862]],
        [[0.547494, 0.607765, -0.155259], [0.181692, 0.781742, 0.036566], [-0.01041, 0.027275, 0.983136]],
        [[0.498864, 0.674741, -0.173604], [0.205199, 0.754872, 0.039929], [-0.011131, 0.030969, 0.980162]],
        [[0.457771, 0.731899, -0.18967], [0.226409, 0.731012, 0.042579], [-0.011595, 0.034333, 0.977261]],
        [[0.422823, 0.781057, -0.203881], [0.245752, 0.709602, 0.044646], [-0.011843, 0.037423, 0.974421]],
        [[0.392952, 0.82361, -0.216562], [0.263559, 0.69021, 0.046232], [-0.01191, 0.040281, 0.97163]],
        [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
    ]),
    "tritanomaly": np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0.92667, 0.092514, -0.019184], [0.021191, 0.964503, 0.014306], [0.008437, 0.054813, 0.93675]],
        [[0.89572, 0.13333, -0.02905], [0.029997, 0.9454, 0.024603], [0.013027, 0.104707, 0.882266]],
        [[0.905871, 0.127791, -0.033662], [0.026856, 0.941251, 0.031893], [0.01341, 0.148296, 0.838294]],
        [[0.948035, 0.08949, -0.037526], [0.014364, 0.946792, 0.038844], [0.010853, 0.193991, 0.795156]],
        [[1.017277, 0.027029, -0.044306], [-0.006113, 0.958479, 0.047634], [0.006379, 0.248708, 0.744913]],
        [[1.104996, -0.046633, -0.058363], [-0.032137, 0.971635, 0.060503], [0.001336, 0.317922, 0.680742]],
        [[1.193214, -0.109812, -0.083402], [-0.058496, 0.97941, 0.079086], [-0.002346, 0.403492, 0.598854]],
        [[1.257728, -0.139648, -0.118081], [-0.078003, 0.975409, 0.102594], [-0.003316, 0.501214, 0.502102]],
        [[1.278864, -0.125333, -0.153531], [-0.084748, 0.957674, 0.127074], [-0.000989, 0.601151, 0.399838]],
        [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.3039]],
    ]),
}


def apply_cvd(img: Image.Image, cvd_type: str = "deuteranomaly", severity: float = 1.0) -> Image.Image:
    """Apply color vision deficiency simulation.

    Args:
        img: Input PIL Image (RGB).
        cvd_type: One of 'protanomaly', 'deuteranomaly', 'tritanomaly', 'monochrome'.
        severity: 0.0 (no effect) to 1.0 (full effect).
    """
    severity = np.clip(severity, 0.0, 1.0)
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img, dtype=np.float32) / 255.0

    if cvd_type == "monochrome":
        gray_weights = np.array([0.299, 0.587, 0.114])
        gray = np.dot(arr[..., :3], gray_weights)
        gray_rgb = np.stack([gray] * 3, axis=-1)
        result = arr[..., :3] * (1 - severity) + gray_rgb * severity
    else:
        matrices = CVD_MATRICES[cvd_type]
        # Interpolate between matrix steps (same as VIP-Sim)
        n = len(matrices)
        idx = severity * (n - 1)
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, n - 1)
        w = idx - i0
        mat = matrices[i0] * (1 - w) + matrices[i1] * w  # 3x3
        result = np.dot(arr[..., :3], mat.T)

    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)
    return Image.fromarray(result)


# =============================================================================
# Gaussian Blur / Hyperopia (from myBlur.shader)
# =============================================================================

def apply_blur(img: Image.Image, severity: float = 0.5) -> Image.Image:
    """Apply Gaussian blur to simulate reduced visual acuity (hyperopia).

    Args:
        img: Input PIL Image.
        severity: 0.0 (no blur) to 1.0 (heavy blur). Maps to blur radius 0-15px.
    """
    severity = np.clip(severity, 0.0, 1.0)
    if severity == 0.0:
        return img.copy()
    radius = severity * 15.0
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# =============================================================================
# sRGB <-> linear helpers
# =============================================================================
# Unity renders shaders in linear color space when the project is set to
# Linear (the default for VIP-Sim). All shader colour math therefore happens
# on linear values, and the framebuffer is sRGB-encoded on write. To match
# Unity output we mirror that round-trip here.

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


# =============================================================================
# Brightness / Contrast / Gamma (from myBrightnessContrastGamma.shader)
# =============================================================================

def apply_bcg(img: Image.Image, brightness: float = 1.0, contrast: float = 1.0, gamma: float = 1.0) -> Image.Image:
    """Apply brightness, contrast, and gamma adjustment.

    Matches the VIP-Sim shader formula (executed in linear colour space, as
    Unity does when the project Color Space is Linear):
        color *= brightness
        color = (color - 0.5) * contrast + 0.5
        color = pow(color, gamma)

    Args:
        img: Input PIL Image (RGB).
        brightness: Multiplier (1.0 = no change, <1 darker, >1 brighter).
        contrast: Multiplier (1.0 = no change, <1 less contrast, >1 more contrast).
        gamma: Exponent (1.0 = no change, >1 darker midtones, <1 lighter midtones).
    """
    srgb = np.asarray(img, dtype=np.float32) / 255.0
    arr = _srgb_to_linear(srgb)

    # Brightness
    arr = arr * brightness
    # Contrast
    arr = (arr - 0.5) * contrast + 0.5
    # Clamp before gamma to avoid NaN from negative values
    arr = np.clip(arr, 0.0, 1.0)
    # Gamma
    if gamma != 1.0:
        arr = np.power(arr, gamma)

    out = _linear_to_srgb(np.clip(arr, 0.0, 1.0))
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# =============================================================================
# Cataracts — faithful port of myCataract.cs + cfxFrost.shader
# =============================================================================
# Pipeline:
#   1. Per-channel BCG with brightness=(1-severity), contrast=(1-severity),
#      coeffs=(0.7, 0.7, 0.4), gamma=1/Gamma.
#      Per-channel coeff acts as the contrast pivot — blue pivots toward 0.4
#      (yellow lens tint) while R/G pivot toward 0.7.
#   2. Frost: per-pixel UV displacement driven by a tiled Perlin noise texture,
#      scaled by 10*severity (matches `_Scale` in the shader).

_PERLIN_NOISE_CACHE: dict = {}


def _perlin_noise_2d(size: int = 256, freq: float = 20.0, seed: int = 0) -> np.ndarray:
    """Perlin-style smooth noise tile in [0, 1], shape (size, size).

    Mirrors `myCataract.GenerateNoiseTexture`: sample at coord/size * 20.
    Implementation uses value-noise with smoothstep interpolation — visually
    equivalent to Unity's Mathf.PerlinNoise for tile use.
    """
    key = (size, freq, seed)
    if key in _PERLIN_NOISE_CACHE:
        return _PERLIN_NOISE_CACHE[key]

    rng = np.random.default_rng(seed)
    grid_n = int(np.ceil(freq)) + 2
    lattice = rng.random((grid_n, grid_n)).astype(np.float32)

    # Sample lattice with smoothstep interpolation
    coords = np.linspace(0, freq, size, endpoint=False, dtype=np.float32)
    xs, ys = np.meshgrid(coords, coords, indexing="xy")
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = xs - x0
    fy = ys - y0
    # Smoothstep
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)

    v00 = lattice[y0, x0]
    v10 = lattice[y0, x0 + 1]
    v01 = lattice[y0 + 1, x0]
    v11 = lattice[y0 + 1, x0 + 1]
    a = v00 * (1 - sx) + v10 * sx
    b = v01 * (1 - sx) + v11 * sx
    out = a * (1 - sy) + b * sy
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    _PERLIN_NOISE_CACHE[key] = out
    return out


def _apply_bcg_per_channel(arr: np.ndarray, brightness: float, contrast: float,
                            coeffs=(0.5, 0.5, 0.5), gamma: float = 1.0) -> np.ndarray:
    """BCG with per-channel contrast pivot — matches myBrightnessContrastGamma.shader.

    Input/output is sRGB in [0,1]; the shader math runs in linear space to
    mirror Unity's Linear colour-space rendering.

    Formula (per-channel, linear):
        color *= brightness
        color = (color - coeff) * contrast + coeff
        color = clamp(0, 1)
        color = pow(color, gamma)
    """
    coeffs_arr = np.array(coeffs, dtype=np.float32).reshape(1, 1, 3)
    out = _srgb_to_linear(arr) * brightness
    out = (out - coeffs_arr) * contrast + coeffs_arr
    out = np.clip(out, 0.0, 1.0)
    if gamma != 1.0:
        out = np.power(out, gamma)
    return _linear_to_srgb(np.clip(out, 0.0, 1.0))


def apply_cataracts(img: Image.Image, severity: float = 0.5,
                     use_brightness: bool = True, use_contrast: bool = True,
                     use_frosting: bool = True,
                     contrast_coeff=(0.7, 0.7, 0.4),
                     gamma: float = 1.0) -> Image.Image:
    """Faithful port of VIP-Sim cataract effect.

    Source: `Assets/VisualEffects/Scripts/myCataract.cs` +
    `Assets/VisualEffects/Shaders/cfxFrost.shader` (CC BY 4.0).

    Args:
        img: Input PIL Image (RGB).
        severity: 0.0 (no effect) to 1.0 (heavy cataracts).
        use_brightness: Apply brightness reduction (Unity default: True).
        use_contrast: Apply contrast reduction (Unity default: True).
        use_frosting: Apply Perlin-noise UV-displacement frost (default: True).
        contrast_coeff: Per-channel contrast pivot (default (0.7, 0.7, 0.4) —
            blue pivots low, producing the yellow lens tint).
        gamma: Optional gamma. Shader uses 1/gamma; we mirror that.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    # 1. Per-channel BCG (brightness/contrast both shrink linearly with severity)
    brightness = (1.0 - severity) if use_brightness else 1.0
    contrast = (1.0 - severity) if use_contrast else 1.0
    arr = _apply_bcg_per_channel(arr, brightness, contrast,
                                  coeffs=contrast_coeff, gamma=1.0 / gamma)

    # 2. Frost via Perlin-noise UV displacement
    if use_frosting:
        scale = 10.0 * severity  # mirrors `_Scale = 10f * severityIndex`
        # Build a noise field at output resolution (sample noise at uv * 20)
        noise_tile = _perlin_noise_2d(size=256)
        # Per-pixel uv → noise sample (scaled coords mod tile size)
        ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
        xs = np.arange(w, dtype=np.float32) / max(w - 1, 1)
        uy, ux = np.meshgrid(ys, xs, indexing="ij")
        nx = ((ux * 20.0) * 256.0).astype(np.int32) % 256
        ny = ((uy * 20.0) * 256.0).astype(np.int32) % 256
        n = noise_tile[ny, nx]
        # Same fractional offsets as the shader
        dx = -0.005 + (n - 0.008 * np.floor(n / 0.008))
        dy = -0.006 + (n - 0.010 * np.floor(n / 0.010))
        # Source UVs (clamped)
        src_u = np.clip(ux + dx * scale, 0.0, 1.0)
        src_v = np.clip(uy + dy * scale, 0.0, 1.0)
        sx = (src_u * (w - 1)).astype(np.int32)
        sy = (src_v * (h - 1)).astype(np.int32)
        arr = arr[sy, sx]

    out = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(out)


# =============================================================================
# Bloom — faithful port of myBloom.cs + myBloom.shader
# =============================================================================
# Pipeline:
#   1. Downsample by 4 with 4-tap box average.
#   2. Threshold-extract: max(downsampled - threshold, 0) * intensity.
#   3. Separable Gaussian blur (7-tap kernel from shader) horizontal+vertical,
#      repeated `iterations` times with growing radius.
#   4. Upsample to source resolution and add to original.

# Gaussian-ish weights used in myBloom.shader.fragBlur8
_BLOOM_KERNEL = np.array([0.0205, 0.0855, 0.232, 0.324, 0.232, 0.0855, 0.0205],
                          dtype=np.float32)


def _separable_gaussian(arr: np.ndarray, radius_px: float) -> np.ndarray:
    """Separable Gaussian-like blur using the VIP-Sim 7-tap weight curve.

    The shader spaces 7 taps `radius_px` apart in each direction, so
    larger radius = wider blur. We approximate by scaling the kernel via
    PIL's GaussianBlur, which is mathematically equivalent for our purposes
    (separable, normalized, monotonic in radius).
    """
    if radius_px <= 0:
        return arr
    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=float(radius_px)))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def apply_bloom(img: Image.Image, severity: float = 0.5,
                 intensity: float = None, threshold: float = None,
                 blur_size: float = None, iterations: int = 1) -> Image.Image:
    """Faithful port of VIP-Sim bloom effect (light scatter / glare / photophobia).

    Source: `myBloom.cs` + `myBloom.shader` (CC BY 4.0).

    If `severity` is given (0..1), it sets sensible defaults:
        intensity = 0.25 + severity * 2.25   (Unity range 0.0–2.5)
        threshold = max(0.05, 0.5 - severity * 0.45)  (lower → more sources)
        blur_size = 1.0 + severity * 4.5     (Unity range 0.25–5.5)

    Explicit kwargs override the severity-derived defaults.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0 and intensity is None and threshold is None and blur_size is None:
        return img.copy()

    # NOTE: Unity defaults (intensity up to 2.5, low threshold) were tuned for
    # game scenes with sparse bright sources. Web pages are mostly bright
    # background so we use much gentler defaults — only near-white pixels
    # bloom, and the composite is moderate. Override via kwargs for a truly
    # faithful Unity-style bloom (e.g. on dark UI screenshots).
    if intensity is None:
        intensity = 0.1 + severity * 0.6   # 0.1 → 0.7 across severity range
    if threshold is None:
        threshold = max(0.6, 0.95 - severity * 0.3)  # 0.95 → 0.65
    if blur_size is None:
        blur_size = 1.0 + severity * 4.5

    # Unity renders bloom in linear colour space (sRGB framebuffer write).
    arr_srgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = _srgb_to_linear(arr_srgb)
    h, w = arr.shape[:2]

    # 1. Downsample by 4 (Resolution.Low → divider=4, widthMod=0.5)
    divider = 4
    width_mod = 0.5
    rt_w, rt_h = max(1, w // divider), max(1, h // divider)
    down = np.asarray(
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((rt_w, rt_h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    # 2. Threshold-extract: max(color - threshold, 0) * intensity
    bright = np.clip(down - threshold, 0.0, None) * intensity

    # 3. Separable Gaussian, repeated with growing radius (matches C# loop)
    for i in range(iterations):
        radius = blur_size * width_mod + i * 1.0
        # Single Gaussian call covers vertical+horizontal separably
        bright = _separable_gaussian(bright, radius_px=radius * 1.5)

    # 4. Upsample bloom and additive composite (fragBloom: color + bloom)
    bloom_full = np.asarray(
        Image.fromarray((np.clip(bright, 0, 1) * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    out_lin = np.clip(arr + bloom_full, 0.0, 1.0)
    out = _linear_to_srgb(out_lin)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# =============================================================================
# Field Loss — port of myFieldLoss.cs + myFieldLoss.shader
# =============================================================================
# Original samples a "degradation" texture (e.g. macular-degeneration.png),
# converts grayscale to a weight w = 1 - gray, and uses w * MaxLODlevel to
# pick a mip level — high w (dark mask region) = heavy blur, low w = sharp.
#
# We don't have the asset texture, so we substitute a generic radial mask
# (documented Phase-1 deviation). `field_type`:
#   - "central"    → blurred center, sharp periphery (macular-degeneration shape)
#   - "peripheral" → sharp center, blurred periphery (tunnel-vision shape)

def _build_mip_pyramid(arr: np.ndarray, levels: int) -> list:
    """Return `levels` progressively-blurred versions of `arr` (level 0 = sharp).

    Approximates GPU mipmap chain — radius doubles per level, matching how
    each mip level halves resolution.
    """
    pyramid = [arr]
    for lvl in range(1, levels):
        radius = float(2 ** (lvl - 1))
        pyramid.append(_separable_gaussian(arr, radius_px=radius))
    return pyramid


def _radial_weight_mask(h: int, w: int, kind: str = "central",
                         center=(0.5, 0.5), radius: float = 0.45,
                         softness: float = 0.25) -> np.ndarray:
    """Generate a radial weight mask in [0, 1] used in place of the missing
    `macular-degeneration.png` overlay.

    Returns w where 1.0 = max blur, 0.0 = sharp.
    """
    cy, cx = center
    ys = (np.arange(h, dtype=np.float32) / max(h - 1, 1) - cy)
    xs = (np.arange(w, dtype=np.float32) / max(w - 1, 1) - cx)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist = np.sqrt(xx * xx + yy * yy)

    # smoothstep edge
    inner = max(0.0, radius - softness)
    outer = radius
    t = np.clip((dist - inner) / max(outer - inner, 1e-6), 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)  # smoothstep

    if kind == "central":
        # Blur in the center (mask high in center, low at edges)
        w = 1.0 - s
    elif kind == "peripheral":
        # Blur at the periphery (mask low in center, high at edges)
        w = s
    else:
        raise ValueError(f"Unknown field_type: {kind}")
    return w.astype(np.float32)


# Path to the VIP-Sim macular-degeneration overlay (copied from
# VIP-Sim/windows/Assets/Resources/macular-degeneration.jpg). Used as the
# faithful overlay for apply_field_loss; falls back to the radial mask if
# the asset is missing.
_MACULAR_OVERLAY_PATH = (
    Path(__file__).parent / "vipsim_assets" / "macular-degeneration.jpg"
)
_MACULAR_OVERLAY_CACHE: dict = {}


def _load_macular_overlay(h: int, w: int, field_type: str = "central",
                            overlay_scale: float = 1.0,
                            center=(0.5, 0.5)) -> "np.ndarray | None":
    """Load the VIP-Sim macular-degeneration overlay, sampled like the shader.

    Mirrors `myFieldLoss.shader`:
        scaledUV = (uv - 0.5) / overlay_scale + (mousePos - 0.5)/overlay_scale + 0.5
        scaledUV = clamp(scaledUV, 0.0, 1.0)         # IMPORTANT: clamp, not zero-pad
        degradation = tex2D(_Overlay, scaledUV)
        w = 1 - luminance(degradation)               # dark JPG region → high blur

    Clamping (vs Python's old zero-pad) is the bugfix: pixels outside the
    scaled overlay get the *edge* sample, which for the macular-degeneration
    JPG is bright (no blur), so the periphery is left sharp.

    Returns a per-pixel weight mask in [0, 1] (1 = max blur, 0 = sharp), or
    None if the asset isn't on disk.
    """
    if not _MACULAR_OVERLAY_PATH.is_file():
        return None
    cache_key = (h, w, field_type, round(overlay_scale, 4),
                 round(center[0], 4), round(center[1], 4))
    if cache_key in _MACULAR_OVERLAY_CACHE:
        return _MACULAR_OVERLAY_CACHE[cache_key]

    overlay = Image.open(_MACULAR_OVERLAY_PATH).convert("L")
    o_w, o_h = overlay.size
    overlay_arr = np.asarray(overlay, dtype=np.float32) / 255.0

    cy, cx = center
    ys = (np.arange(h, dtype=np.float32) + 0.5) / h
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    # Shader transform: sample point = (uv - 0.5)/scale + 0.5, with mouse
    # offset absorbed (mouse=center → no shift).
    scaled_u = (xx - 0.5) / max(overlay_scale, 1e-6) + cx
    scaled_v = (yy - 0.5) / max(overlay_scale, 1e-6) + cy
    scaled_u = np.clip(scaled_u, 0.0, 1.0)
    scaled_v = np.clip(scaled_v, 0.0, 1.0)
    su = np.clip((scaled_u * (o_w - 1)).astype(np.int32), 0, o_w - 1)
    sv = np.clip((scaled_v * (o_h - 1)).astype(np.int32), 0, o_h - 1)
    sampled = overlay_arr[sv, su]

    w_central = 1.0 - sampled  # dark overlay = high blur weight
    if field_type == "central":
        result = w_central
    elif field_type == "peripheral":
        result = 1.0 - w_central
    else:
        raise ValueError(f"Unknown field_type: {field_type}")

    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    _MACULAR_OVERLAY_CACHE[cache_key] = result
    return result


def apply_field_loss(img: Image.Image, severity: float = 0.5,
                      field_type: str = "central",
                      overlay_scale: float = 0.75,
                      use_overlay_asset: bool = True) -> Image.Image:
    """Field-loss simulation via per-pixel mip-level selection.

    Source: `myFieldLoss.cs` + `myFieldLoss.shader` (CC BY 4.0).
    Uses the real VIP-Sim `macular-degeneration.jpg` overlay if available
    in `vipsim_assets/`; otherwise falls back to a generic radial mask.

    Args:
        img: Input PIL Image (RGB).
        severity: 0.0 (no effect) to 1.0 (max blur in masked region).
        field_type: "central" (macular-style, blurred center) or
            "peripheral" (tunnel-vision-style, blurred periphery).
        overlay_scale: Mirrors Unity `_OverlayScale` — larger = overlay covers
            a larger fraction of the image.
        use_overlay_asset: If True, try to load the VIP-Sim overlay JPG
            (`vipsim_assets/macular-degeneration.jpg`); falls back to a
            radial mask if missing. Set False to force the radial mask.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    # numLevelsOfBlur in Unity = 1 + 1 + floor(ln(max(W,H))). For 2048×1024
    # this is 1+1+floor(ln 2048)=1+1+7=9. Match that so max blur is comparable.
    n_levels = int(2 + np.floor(np.log(max(h, w))))
    pyramid = _build_mip_pyramid(arr, n_levels)

    # Try the real Unity overlay first; fall back to a radial mask
    mask = None
    if use_overlay_asset:
        mask = _load_macular_overlay(h, w, field_type=field_type,
                                       overlay_scale=overlay_scale)
    if mask is None:
        # Fallback: generic radial mask (Phase-1 substitute)
        mask_radius = 0.45 / max(overlay_scale, 0.1)
        mask = _radial_weight_mask(h, w, kind=field_type,
                                    radius=min(mask_radius, 0.7))
    # Severity scales the maximum mip level effectively reached
    w_scaled = mask * severity

    # Bias = w * (n_levels - 1): per-pixel float "level"
    bias = w_scaled * (n_levels - 1)
    lo = np.floor(bias).astype(np.int32)
    hi = np.clip(lo + 1, 0, n_levels - 1)
    frac = (bias - lo)[..., None]

    # Stack pyramid → shape (n_levels, h, w, 3) — small (n_levels=7) so OK
    stack = np.stack(pyramid, axis=0)
    # Sample lo and hi mips per pixel
    iy, ix = np.indices((h, w))
    lo_sample = stack[lo, iy, ix]
    hi_sample = stack[hi, iy, ix]
    out = lo_sample * (1 - frac) + hi_sample * frac

    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# =============================================================================
# Distortion — faithful port of myDistortionMap.cs + myDistortionMap.shader
# =============================================================================
# Original precomputes a 2D warp field from N "warp centers", each defined by
# (x, y, radius, magnitude). At runtime each pixel samples vx/vy and shifts.
# We compute the warp field in numpy and apply directly via nearest-neighbor
# resample, mirroring the shader's UV displacement.

def _compute_warp_field(h: int, w: int,
                         warp_x, warp_y, warp_radius, warp_magn) -> tuple:
    """Reproduces myDistortionMap.compute() and the shader's UV-offset math.

    Returns (vx, vy) arrays of shape (h, w) clipped to [-0.1, 0.1].
    """
    SQRT2 = float(np.sqrt(2.0))
    vx = np.zeros((h, w), dtype=np.float32)
    vy = np.zeros((h, w), dtype=np.float32)
    # Use width as the divisor for both axes — matches the original code
    # (which uses width_px for both, leading to non-square scaling that we
    # preserve for fidelity).
    ys = np.arange(h, dtype=np.float32) / float(w)
    xs = np.arange(w, dtype=np.float32) / float(w)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    for i in range(len(warp_x)):
        magn = warp_magn[i]
        if magn == 0:
            continue
        radius = warp_radius[i]
        if radius <= 0:
            continue
        ox = xx - warp_x[i]
        oy = yy - warp_y[i]
        dist = np.sqrt(ox * ox + oy * oy)
        percent = 1.0 - ((radius - dist) / radius) * magn / SQRT2
        if magn < 0:
            percent = np.maximum(1.0, percent)
        else:
            percent = np.minimum(1.0, percent)
        vx += ox - ox * percent
        vy += oy - oy * percent

    vx = np.clip(vx, -0.1, 0.1)
    vy = np.clip(vy, -0.1, 0.1)
    return vx, vy


def apply_distortion(img: Image.Image, severity: float = 0.5,
                      warp_centers=None) -> Image.Image:
    """Radial UV distortion (e.g. metamorphopsia from macular damage).

    Source: `myDistortionMap.cs` + `myDistortionMap.shader` (CC BY 4.0).
    Gaze-contingent in Unity; we use fixed centers (Phase-1 deviation).

    Args:
        img: Input PIL Image (RGB).
        severity: 0..1. Scales the magnitude of all warp centers.
        warp_centers: Optional list of (x, y, radius, magnitude) tuples in
            normalized [0,1] coords. Defaults to the two centers used in
            Unity's `myDistortionMap` initial config.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    if warp_centers is None:
        # Defaults from myDistortionMap.cs warp_x/warp_y/warp_radius/warp_magn.
        # These coords live in 512×512 warp-texture space (uv ∈ [0,1]).
        warp_centers = [
            (0.0,  0.0,  0.15,  1.0),
            (0.33, 0.66, 0.15, -1.0),
        ]

    wx = [c[0] for c in warp_centers]
    wy = [c[1] for c in warp_centers]
    wr = [c[2] for c in warp_centers]
    wm = [c[3] * severity for c in warp_centers]

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    # Bug-faithful port of myDistortionMap.shader. Four things to reproduce:
    #
    # 1) Y-flip in texture encoding (myDistortionMap.cs:180 stores `vy[x,H-1-y]`).
    #    So warpY sampled at UV (s', t') = vy_field at (s', 1-t').
    #
    # 2) Unity D3D-Blit UV convention: i.uv.y = 0 at bottom of screen, 1 at top.
    #    Our numpy uy = 0 at top of array. So Unity's `i.uv.y = 1 - uy_python`.
    #    This means:
    #      - For vx eval (no Y flip in encoding): degcoords.y = 0.75 - 0.5*uy
    #      - For vy eval (Y flip in encoding):    degcoords.y = 0.25 + 0.5*uy
    #    And the sampling Y direction is inverted: src_uy_python = uy − yOffset.
    #
    # 3) Shader decode bug (myDistortionMap.shader:69-74):
    #      `xOffset = (0.2*w) - 0.0425; xOffset = -xOffset;`
    #      `yOffset = (0.2*w) - 0.0425;`
    #    The correct inverse of `vx*5+0.5` would have been `0.2*w - 0.1`. Off by
    #    +0.0575 in the constant term. Combined with sRGB texture sampling
    #    (Texture2D RGB24 defaults to sRGB), the actual decoded w is the
    #    sRGB→linear of the encoded value — which happens to make the
    #    *zero-warp* case lie inside the deadband (so clean regions don't
    #    shift), while warped regions get ~57% boosted asymmetric displacement.
    #
    # 4) Deadband: `if |offset| < 0.001 → 0`.
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    uy, ux = np.meshgrid(ys, xs, indexing="ij")
    deg_x = ux * 0.5 + 0.25
    deg_y_for_x = 0.75 - 0.5 * uy   # vx_field evaluation coord (Unity UV space)
    deg_y_for_y = 0.25 + 0.5 * uy   # vy_field eval coord (post Y-flip in encoding)

    SQRT2 = float(np.sqrt(2.0))

    def _accumulate_v(field_x, field_y, want_y=False):
        """Run Unity's compute-pass accumulation. Returns vx if want_y=False, vy otherwise."""
        f = np.zeros((h, w), dtype=np.float32)
        for cx, cy, r, m in zip(wx, wy, wr, wm):
            if m == 0 or r <= 0:
                continue
            ox = field_x - cx
            oy = field_y - cy
            dist = np.sqrt(ox * ox + oy * oy)
            percent = 1.0 - ((r - dist) / r) * m / SQRT2
            percent = np.maximum(1.0, percent) if m < 0 else np.minimum(1.0, percent)
            if want_y:
                f += oy - oy * percent
            else:
                f += ox - ox * percent
        return np.clip(f, -0.1, 0.1)

    vx_field = _accumulate_v(deg_x, deg_y_for_x, want_y=False)
    vy_field = _accumulate_v(deg_x, deg_y_for_y, want_y=True)

    # sRGB→linear conversion (Unity samples RGB24 textures as sRGB by default).
    def _srgb_to_linear(s):
        a = 0.055
        s = np.asarray(s, dtype=np.float32)
        return np.where(s <= 0.04045, s / 12.92, ((s + a) / (1.0 + a)) ** 2.4)

    w_x = _srgb_to_linear(5.0 * vx_field + 0.5)
    w_y = _srgb_to_linear(5.0 * vy_field + 0.5)

    # Shader-bug decode (with negation on X).
    x_off = -(0.2 * w_x - 0.0425)
    y_off = (0.2 * w_y - 0.0425)

    # Deadband: zero out very small offsets (mirrors the shader's HACK block).
    x_off = np.where(np.abs(x_off) < 0.001, 0.0, x_off)
    y_off = np.where(np.abs(y_off) < 0.001, 0.0, y_off)

    # Apply: src_ux = ux + xOff (X axis matches), src_uy = uy − yOff (Y inverted).
    src_u = np.clip(ux + x_off, 0.0, 1.0)
    src_v = np.clip(uy - y_off, 0.0, 1.0)
    sx = (src_u * (w - 1)).astype(np.int32)
    sy = (src_v * (h - 1)).astype(np.int32)
    out = arr[sy, sx]
    return Image.fromarray(out.astype(np.uint8))


# =============================================================================
# Double Vision — port of DoubleVision.shader (monocular 2D blend)
# =============================================================================
# Unity's myDoubleVision.cs is largely binocular (rotates eye in 3D); for our
# single-screenshot use we port the monocular `DoubleVision.shader`:
#   sample left = src(uv + (-d, 0)); sample right = src(uv + (d, 0));
#   out = 0.5 * (left + right)

def apply_double_vision(img: Image.Image, severity: float = 0.5,
                          axis: str = "horizontal") -> Image.Image:
    """Monocular double vision (diplopia) via two-sample UV-offset blend.

    Source: `DoubleVision.shader` (CC BY 4.0).

    Args:
        img: Input PIL Image (RGB).
        severity: 0..1. Maps to displacement up to ~2% of image width.
        axis: "horizontal" (default), "vertical", or "diagonal".
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    disp = severity * 0.025  # normalized UV (matches Unity DoubleVisionEffect.displacementAmount = 0.025)

    if axis == "horizontal":
        dx, dy = disp, 0.0
    elif axis == "vertical":
        dx, dy = 0.0, disp
    elif axis == "diagonal":
        s = disp / float(np.sqrt(2.0))
        dx, dy = s, s
    else:
        raise ValueError(f"Unknown axis: {axis}")

    px = int(round(dx * (w - 1)))
    py = int(round(dy * (h - 1)))

    # left/up sample (uv - displacement) and right/down sample (uv + displacement)
    def shifted(arr_, sx_, sy_):
        out = np.zeros_like(arr_)
        H, W = arr_.shape[:2]
        # source range
        src_y0 = max(0, sy_); src_y1 = min(H, H + sy_)
        src_x0 = max(0, sx_); src_x1 = min(W, W + sx_)
        dst_y0 = max(0, -sy_); dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, -sx_); dst_x1 = dst_x0 + (src_x1 - src_x0)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr_[src_y0:src_y1, src_x0:src_x1]
        # fill edges with original (avoids black borders that would skew the blend)
        if dst_y0 > 0:
            out[:dst_y0] = arr_[:dst_y0]
        if dst_y1 < H:
            out[dst_y1:] = arr_[dst_y1:]
        if dst_x0 > 0:
            out[:, :dst_x0] = arr_[:, :dst_x0]
        if dst_x1 < W:
            out[:, dst_x1:] = arr_[:, dst_x1:]
        return out

    left = shifted(arr, -px, -py)
    right = shifted(arr, px, py)
    out = 0.5 * (left + right)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =============================================================================
# Detail Loss — bit-depth posterize (per Phase-1 spec)
# =============================================================================
# Approximates loss of fine luminance/chroma detail (e.g. cataract residual,
# very low contrast sensitivity). Per-channel posterization via PIL.

def apply_detail_loss(img: Image.Image, severity: float = 0.5) -> Image.Image:
    """Reduce per-channel bit depth to simulate detail loss.

    Args:
        img: Input PIL Image (RGB).
        severity: 0..1. Maps to bits-per-channel: 0.0 → 8 bits (no change),
            1.0 → 2 bits (heavy posterization).
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()
    # 0.0 → 8 bits (no change), 0.5 → 4 bits (visible banding), 1.0 → 1 bit
    bits = int(round(8 - severity * 7))
    bits = max(1, min(8, bits))
    return ImageOps.posterize(img.convert("RGB"), bits)


# =============================================================================
# Phase 2 — remaining VIP-Sim filters
# =============================================================================
# Static-clean (faithful pixel math): pixelation, led, vortex, wiggle, noise.
# Static-tricky (faithful with frozen state): glitch, teichopsia (scintillating
#   scotoma), foveal darkness.
# Constrained (frozen-frame approximations of animated effects): nystagmus,
#   flickering_stars, floaters.
# Skipped: inpainter / inpainter2 (depend on field-loss output + a kdtree
#   compute shader + missing resources; not portable to a single-frame numpy
#   pipeline without major Phase-3 work).


# -- Pixelation --------------------------------------------------------------
# Source: PixelationEffect.cs/shader. Faithful: floor(uv * pixel_size) /
# pixel_size with bilinear sampling. We use PIL resize down/up — equivalent
# image content; the shader's bilinear blend between cell corners is
# subsumed by BILINEAR downsample.

def apply_pixelation(img: Image.Image, severity: float = 0.5,
                      pixel_size: float = None) -> Image.Image:
    """Soft-bilinear pixelation — faithful port of PixelationEffect.shader.

    Unity: `_PixelSize = 1000 - pixelRadius` cells across (default 990 →
    barely visible). The shader downsamples to that grid then bilinearly
    interpolates between the 4 surrounding cells when reading back.

    Args:
        severity: 0..1 — if `pixel_size` is None, maps to cells-across
            1000..20 (high sev = blockier).
        pixel_size: Override — cells across the image (Unity's `_PixelSize`).
            Unity default 990 (pixelRadius=10).
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0 and pixel_size is None:
        return img.copy()
    if pixel_size is None:
        pixel_size = float(1000.0 - severity * 980.0)  # 1000..20
    pixel_size = max(2.0, float(pixel_size))

    w, h = img.size
    # Downsample to (pixel_size × pixel_size) — non-square cells like the shader
    cells = max(2, int(round(pixel_size)))
    small = img.resize((cells, cells), Image.BILINEAR)
    # Bilinear back up so adjacent cells blend (matches shader's manual lerp)
    return small.resize((w, h), Image.BILINEAR)


# -- LED grid ----------------------------------------------------------------
# Source: myLed.cs/shader. Pixelate + per-cell sin(uv*PI)*sin(uv*PI) shape
# mask (round bright dot in cell center, dark gaps).

def apply_led(img: Image.Image, severity: float = 0.5,
                scale: float = None, brightness: float = 1.0,
                shape: float = 1.5, margin: float = 0.0,
                automatic_ratio: bool = True) -> Image.Image:
    """LED-board look — faithful port of myLed.cs + myLed.shader.

    Unity algorithm:
      1. Pixelate via `coord = ds * ceil(uv/ds)` with `ds = 1/scale` on X and
         `ds*ratio` on Y (so cells are square when ratio = W/H).
      2. Multiply by Brightness booster.
      3. Cell mask: `mv = abs(sin(uv*scale*PI, uv*scale/ratio*PI)) * Shape`,
         `s = mv.x*mv.y`, `c = step(s, 1.0)` → with Shape=1.5 the bright
         "dot" region is where s>=1, surrounded by sin-falloff into dark gaps.

    Args:
        severity: Convenience knob — if `scale` is None, maps 0..1 → 200..20
            cells across (high sev = blockier).
        scale: Override — cells across the width (Unity default 80).
        brightness: LED brightness multiplier (default 1.0).
        shape: Mask shape, softer→harsher (default 1.5 per Unity).
        margin: Blank black border fraction [0, 0.45] (default 0).
        automatic_ratio: If True, use W/H so cells are square (Unity default).
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0 and scale is None:
        return img.copy()
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    if scale is None:
        # Map severity (subtle→strong) to scale (cells-across): 200 → 20.
        scale = float(200.0 - severity * 180.0)
    ratio = (w / h) if automatic_ratio else 1.0

    # 1. Pixelate: coord = ds * ceil(uv/ds), with ds=1/scale on X
    #    and ds*ratio on Y (so cell aspect matches image aspect).
    ys = (np.arange(h, dtype=np.float32) + 0.5) / h  # pixel centers
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w
    ds_x = 1.0 / scale
    ds_y = ds_x * ratio
    coord_x = ds_x * np.ceil(xs / ds_x)
    coord_y = ds_y * np.ceil(ys / ds_y)
    # Sample _MainTex at (coord_x, coord_y) — nearest texel
    sx = np.clip((coord_x * w).astype(np.int32), 0, w - 1)
    sy = np.clip((coord_y * h).astype(np.int32), 0, h - 1)
    pixelated = arr[sy[:, None], sx[None, :]] * brightness  # shape (h,w,3)

    # 2. LED mask: mv = |sin(uv * (scale, scale/ratio) * PI)| * shape
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mvx = np.abs(np.sin(xx * scale * np.pi)) * shape
    mvy = np.abs(np.sin(yy * (scale / ratio) * np.pi)) * shape
    s = mvx * mvy
    c = (s <= 1.0).astype(np.float32)  # step(s, 1.0): 1 in bright dot, 0 in gaps
    # color = ((1 - c) * color) + ((color * s) * c)
    #       = pixelated where s>1, else pixelated*s
    mask = (1.0 - c) + c * s
    # Unity renders this shader in linear color space; apply the mask in
    # linear and convert back to sRGB so cell-centre brightness matches.
    out_lin = np.clip(_srgb_to_linear(pixelated) * mask[..., None], 0.0, 1.0)
    out = _linear_to_srgb(out_lin)

    # 3. Optional margin (black border)
    if margin > 0:
        m = np.ones((h, w), dtype=np.float32)
        m[(yy < margin) | (yy > 1.0 - margin)] = 0.0
        m[:, (xs < margin) | (xs > 1.0 - margin)] = 0.0
        out = out * m[..., None]

    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


# -- Vortex ------------------------------------------------------------------
# Source: VortexEffect.cs/shader. Radial swirl; gaze-contingent in Unity, we
# use a fixed center (Phase-2 deviation). Inner gray fade preserved.

def apply_vortex(img: Image.Image, severity: float = 0.5,
                  center=(0.5, 0.5),
                  vortex_radius: float = None,
                  suction_strength: float = None,
                  inner_circle_radius: float = 0.01,
                  noise_amount: float = 0.05,
                  noise_scale: float = 10.0,
                  fade_width: float = 0.05) -> Image.Image:
    """Localised radial-suction vortex — faithful port of VortexEffect.shader.

    Unity algorithm (gaze-centered):
      - Aspect-corrected distance from gaze point (UV.x *= W/H).
      - Outside `vortex_radius`: pixel unchanged.
      - Inside: sample further from centre — `r_sample = dist * (1 + k(dist))`
        where `k = suction * (1 - dist/R)`. This produces an inward-pulling /
        fish-eye-like suction (NOT angular rotation).
      - Small noise added to distance and angle (default amount 0.05, scale 10).
      - Tiny inner circle (default 0.01) blends to grey via smoothstep.
      - Outer ring blends back to source via `smoothstep(R - fade, R, dist)`.

    Args:
        severity: 0..1 — if `vortex_radius` is None, maps to radius 0.05..0.5.
            If `suction_strength` is None, maps to strength 0.2..2.0.
        center: (cy, cx) gaze point in normalised UV.
        vortex_radius: Override — fraction of image diagonal-ish (Unity 0.27).
        suction_strength: Override — Unity default 1.0.
        inner_circle_radius: Tiny grey core (Unity default 0.01).
        noise_amount: Edge-wobble magnitude (Unity default 0.05).
        noise_scale: Noise grid frequency (Unity default 10).
        fade_width: Outer smoothstep width (Unity shader default 0.05).
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    cy, cx = center
    if vortex_radius is None:
        vortex_radius = 0.05 + severity * 0.45  # 0.05..0.5
    if suction_strength is None:
        suction_strength = 0.2 + severity * 1.8  # 0.2..2.0

    aspect = w / h

    ys_norm = (np.arange(h, dtype=np.float32) + 0.5) / h
    xs_norm = (np.arange(w, dtype=np.float32) + 0.5) / w
    yy, xx = np.meshgrid(ys_norm, xs_norm, indexing="ij")

    adj_x = xx * aspect
    adj_y = yy
    adj_cx = cx * aspect
    adj_cy = cy
    dx = adj_x - adj_cx
    dy = adj_y - adj_cy
    dist = np.sqrt(dx * dx + dy * dy)

    # Value-noise grid (matches the shader's rand/lerp construction)
    def _value_noise(u, v, scale):
        p_u = np.floor(u * scale)
        p_v = np.floor(v * scale)
        f_u = u * scale - p_u
        f_v = v * scale - p_v

        def _rand(a, b):
            return np.modf(np.sin(a * 12.9898 + b * 78.233) * 43758.5453)[0]

        a = _rand(p_u, p_v)
        b = _rand(p_u + 1, p_v)
        c = _rand(p_u, p_v + 1)
        d = _rand(p_u + 1, p_v + 1)
        su = f_u * f_u * (3.0 - 2.0 * f_u)
        sv = f_v * f_v * (3.0 - 2.0 * f_v)
        return (a + (b - a) * su) + (c - a) * sv * (1.0 - su) + (d - b) * su * sv

    noise_val = _value_noise(xx, yy, noise_scale)
    dist_n = dist + noise_val * noise_amount * vortex_radius

    inside = dist_n < vortex_radius
    out = arr.copy()
    if np.any(inside):
        eps = 1e-8
        angle = np.arctan2(dy, dx)
        # Optional angular noise (small)
        angle_noise = _value_noise(xx, yy, noise_scale)
        angle = angle + angle_noise * noise_amount

        # Radial suction: sample further out
        d_over_R = np.clip(dist_n / max(vortex_radius, eps), 0.0, 1.0)
        adj_suction = suction_strength * (1.0 - d_over_R)
        r_sample = dist_n * (1.0 + adj_suction * (1.0 - d_over_R))

        sample_x = adj_cx + r_sample * np.cos(angle)
        sample_y = adj_cy + r_sample * np.sin(angle)
        sample_x = sample_x / aspect  # undo aspect correction
        sample_x = np.clip(sample_x, 0.0, 1.0)
        sample_y = np.clip(sample_y, 0.0, 1.0)
        sx = (sample_x * (w - 1)).astype(np.int32)
        sy = (sample_y * (h - 1)).astype(np.int32)
        vortexed = arr[sy, sx]

        # Outer fade back to source (smoothstep R-fade..R)
        t_out = np.clip((dist_n - (vortex_radius - fade_width))
                        / max(fade_width, eps), 0.0, 1.0)
        s_out = t_out * t_out * (3.0 - 2.0 * t_out)
        blended = vortexed * (1.0 - s_out[..., None]) + arr * s_out[..., None]

        # Inner grey core
        if inner_circle_radius > 0:
            grey = np.full_like(arr, 0.5)
            t_in = np.clip((dist_n - (inner_circle_radius - fade_width))
                            / max(fade_width, eps), 0.0, 1.0)
            s_in = t_in * t_in * (3.0 - 2.0 * t_in)
            inner_blend = grey * (1.0 - s_in[..., None]) + blended * s_in[..., None]
            blended = np.where(
                (dist_n < inner_circle_radius)[..., None], inner_blend, blended
            )

        out = np.where(inside[..., None], blended, arr)

    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


# -- Wiggle ------------------------------------------------------------------
# Source: myWiggle.cs/shader (Simple mode). Original is animated:
#   t.x += sin(timer + t.x * freq) * amp
# Frozen-frame: timer = 0 → static sinusoidal warp on both axes.

def apply_wiggle(img: Image.Image, severity: float = 0.5,
                  freq: float = 12.0, amplitude: float = 0.01,
                  timer: float = 0.0, mode: str = "complex") -> Image.Image:
    """UV-displacement wiggle — faithful port of myWiggle.shader.

    Unity defaults: Frequency=12, Amplitude=0.01, Mode=Complex. The shader
    uses `Timer * 0.1` for the complex pass. Capture is animation-frozen
    after a couple of frames so `timer` is small (≈0.05). We expose `timer`
    so the validator can sweep it; `severity` scales `amplitude`.

    Args:
        img: input PIL image (RGB).
        severity: 0..1; multiplies `amplitude`.
        freq: Frequency (Unity default 12).
        amplitude: Amplitude (Unity default 0.01).
        timer: Frozen-frame timer (Unity multiplies by 0.1 for complex mode).
        mode: "simple" or "complex" (Unity default complex).
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    amp = amplitude * severity

    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    uy, ux = np.meshgrid(ys, xs, indexing="ij")

    if mode == "simple":
        # t.x += sin(timer + uv.x * freq) * amp
        # t.y += cos(timer + uv.y * freq) * amp - amp
        du = np.sin(timer + ux * freq) * amp
        dv = np.cos(timer + uy * freq) * amp - amp
    else:  # complex (Unity default)
        # Unity multiplies Timer by 0.1 for complex
        t = timer * 0.1
        def shift(u, v):
            fx = freq * (u + t)
            fy = freq * (v + t)
            sx_ = np.cos(np.cos(fx - fy) * np.cos(fy))
            sy_ = np.cos(np.sin(fx + fy) * np.sin(fy))
            return sx_, sy_
        px, py = shift(ux, uy)
        qx, qy = shift(ux + 1.0, uy + 1.0)
        du = amp * (px - qx)
        dv = amp * (py - qy)

    src_u = np.clip(ux + du, 0.0, 1.0)
    src_v = np.clip(uy + dv, 0.0, 1.0)
    sx = (src_u * (w - 1)).astype(np.int32)
    sy = (src_v * (h - 1)).astype(np.int32)
    return Image.fromarray(arr[sy, sx].astype(np.uint8))


# -- Noise -------------------------------------------------------------------
# Source: myNoise.cs/shader. Original cycles through 10 textures; we use a
# single frozen Perlin frame. Shader formula treated as additive noise blend
# (the literal `lerp(color, (color+n)/16, intensity)` collapses the image at
# high intensity which is not the perceptual intent — we use the standard
# additive form, scaled by intensity).

def apply_noise(img: Image.Image, severity: float = 0.5,
                  seed: int = 0) -> Image.Image:
    """Per-pixel additive luminance noise."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    rng = np.random.default_rng(seed)
    n = (rng.random((h, w)).astype(np.float32) - 0.5) * 2.0  # [-1, 1]
    intensity = severity * 0.35
    out = arr + n[..., None] * intensity
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# -- Glitch (RGB channel offset) --------------------------------------------
# Source: myGlitch.cs/shader (Interferences mode). Per-row RGB channel
# horizontal displacement driven by random blocks. Original uses _Time.y
# for animation — we freeze with a fixed seed.

def apply_glitch(img: Image.Image, severity: float = 0.5,
                  max_displace: float = 0.05, seed: int = 0) -> Image.Image:
    """Frozen RGB-channel-shift glitch (interferences-style)."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    rng = np.random.default_rng(seed)

    # Random per-block horizontal displacement (a few wide bands)
    n_blocks = max(1, h // 12)
    disp_blocks = (rng.random(n_blocks) - 0.5) * 2.0 * max_displace * severity
    block_size = max(1, h // n_blocks)
    disp = np.repeat(disp_blocks, block_size)[:h]
    # Pad if shorter
    if disp.size < h:
        disp = np.pad(disp, (0, h - disp.size), mode="edge")
    disp_px = (disp * w).astype(np.int32)

    r = arr[..., 0].copy()
    g = arr[..., 1].copy()
    b = arr[..., 2].copy()
    rows = np.arange(h)
    for y in rows:
        d = int(disp_px[y])
        r[y] = np.roll(arr[y, :, 0], d)
        g[y] = np.roll(arr[y, :, 1], d // 2)
        b[y] = np.roll(arr[y, :, 2], -d)
    out = np.stack([r, g, b], axis=-1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# -- Teichopsia (scintillating scotoma) --------------------------------------
# Source: myTeichopsia.cs + myScintillate.shader (CC BY 4.0).
#
# Unity algorithm:
#  1. CPU generates a 512² mask texture: a ring-shaped polygon (outer spiky
#     polygon minus inner less-spiky polygon, centered at (256, 256)).
#     - outer: 30 points around a circle, radius = 150*(1-0.75) + Uniform[0, 150*0.75]
#     - inner: same angles, radius = outer - Normal(70, 15) but ≥ 30
#  2. Shader samples mask at degcoords = (uv - gazeOffset)*0.5 + 0.25 — mask
#     covers the central 50%×50% of screen, centered on gaze.
#  3. Inside the white mask region: boost saturation x2 in HSV, then apply
#     multiplicative per-channel scintillating noise:
#       nr = frac(n)*2, ng = frac(n*1.2154)*2, nb = frac(n*1.3453)*2
#       result = lerp(color, color * mult, strength * (1 - lum*lumContrib))
#  4. Outside mask: unchanged.
#
# This Python port reproduces all of the above with a seeded RNG so a frozen
# frame is reproducible.

def _box_muller(n: int, rng: "np.random.Generator") -> np.ndarray:
    """Normal(0, 1) samples via Box-Muller (matches Unity's generateNormalRandom)."""
    r1 = np.maximum(rng.random(n), 1e-10).astype(np.float32)
    r2 = rng.random(n).astype(np.float32)
    return np.sqrt(-2.0 * np.log(r1)) * np.cos(2.0 * np.pi * r2)


def apply_teichopsia(img: Image.Image, severity: float = 0.75,
                       lum_contribution: float = 0.75,
                       gaze=(0.5, 0.5), seed: int = 0,
                       sat_boost: float = 2.0) -> Image.Image:
    """Scintillating scotoma — port of myTeichopsia + myScintillate (colored+lum pass).

    The ring is generated at Unity's exact parameters (30 points, outer
    radius 150px / inner-width ~70±15px in a 512² mask space) and remapped
    to cover the central half of the screen, centered on `gaze`.

    Args:
        img: input PIL image (RGB).
        severity: shader `_Params.y` (Strength). Default 0.75 = Unity default.
        lum_contribution: shader `_Params.z` — reduces effect in bright pixels.
        gaze: (x, y) in [0, 1] screen UV. Default (0.5, 0.5) for fixed-center.
        seed: RNG seed for deterministic polygon + noise.
        sat_boost: HSV saturation multiplier inside the ring (Unity hard-codes 2).
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    try:
        from skimage.color import rgb2hsv, hsv2rgb
        from skimage.draw import polygon as skpolygon
    except ImportError as e:
        raise ImportError("teichopsia requires scikit-image") from e

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    # 1) Build the ring mask in 512² mask space (Unity's exact parameters).
    mask_size = 512
    cx_mask = cy_mask = mask_size // 2
    n_points = 30
    max_radius = 150.0
    spikiness = 0.75
    width_mu = 70.0
    width_std = 15.0
    min_radius_px = 30.0
    min_pixels = 10.0

    rng = np.random.default_rng(seed)
    # Outer: rho = maxRadius*(1-spikiness) + UniformInt[0, maxRadius*spikiness)
    rho_out = max_radius * (1 - spikiness) + \
        rng.integers(0, int(max_radius * spikiness), n_points).astype(np.float32)
    thetas = (2 * np.pi / (n_points + 1)) * np.arange(n_points, dtype=np.float32)
    outer_x = cx_mask + rho_out * np.cos(thetas)
    outer_y = cy_mask + rho_out * np.sin(thetas)

    # Inner: subtract Normal(mu, std) clamped to >= min_pixels, then clamp final
    # radius to >= min_radius_px.
    ring_w = np.maximum(width_mu + width_std * _box_muller(n_points, rng), min_pixels)
    rho_in = np.maximum(rho_out - ring_w, min_radius_px)
    inner_x = cx_mask + rho_in * np.cos(thetas)
    inner_y = cy_mask + rho_in * np.sin(thetas)

    ring_mask = np.zeros((mask_size, mask_size), dtype=bool)
    rr_o, cc_o = skpolygon(outer_y, outer_x, ring_mask.shape)
    rr_i, cc_i = skpolygon(inner_y, inner_x, ring_mask.shape)
    ring_mask[rr_o, cc_o] = True
    ring_mask[rr_i, cc_i] = False  # outer ∖ inner = ring

    # 2) Map mask to screen via degcoords = (uv - gazeOffset)*0.5 + 0.25
    # i.e. screen-UV [0, 1] → mask-UV [0.25 - gazeOff, 0.75 - gazeOff].
    gx, gy = gaze
    gaze_off_x = 0.5 - gx
    gaze_off_y = 0.5 - gy
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    uy, ux = np.meshgrid(ys, xs, indexing="ij")
    deg_x = (ux - gaze_off_x) * 0.5 + 0.25
    deg_y = (uy - gaze_off_y) * 0.5 + 0.25
    # Sample mask (NEAREST). Outside [0, 1] sample → False (no ring effect).
    mx = (deg_x * (mask_size - 1)).astype(np.int32)
    my = (deg_y * (mask_size - 1)).astype(np.int32)
    in_bounds = (mx >= 0) & (mx < mask_size) & (my >= 0) & (my < mask_size)
    screen_mask = np.zeros((h, w), dtype=bool)
    screen_mask[in_bounds] = ring_mask[my[in_bounds], mx[in_bounds]]

    # 3) Inside the ring: HSV saturation boost, multiplicative scintillating noise,
    # luminance-attenuated mixing — exactly the colored_lum pass of myScintillate.
    hsv = rgb2hsv(arr)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost, 0.0, 1.0)
    boosted = hsv2rgb(hsv).astype(np.float32)

    rng2 = np.random.default_rng(seed + 1)
    n = rng2.random((h, w)).astype(np.float32)
    nr = (n % 1.0) * 2.0
    ng = ((n * 1.2154) % 1.0) * 2.0
    nb = ((n * 1.3453) % 1.0) * 2.0
    mult = np.stack([nr, ng, nb], axis=-1).astype(np.float32)

    lum = 0.299 * boosted[..., 0] + 0.587 * boosted[..., 1] + 0.114 * boosted[..., 2]
    strength = severity * (1.0 - lum * lum_contribution)
    noisy = boosted * mult
    scint = boosted * (1.0 - strength[..., None]) + noisy * strength[..., None]

    out = arr.copy()
    out[screen_mask] = scint[screen_mask]
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8))


# -- Foveal Darkness ---------------------------------------------------------
# Source: FovealDarkness.cs/shader. Smoothstep-faded dark spot at gaze
# location (we fix to image center).

def apply_foveal_darkness(img: Image.Image, severity: float = 0.5,
                            center=(0.5, 0.5), radius: float = None) -> Image.Image:
    """Central dark spot with smoothstep edge (gaze-fixed)."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    if radius is None:
        radius = 0.05 + severity * 0.20

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    cy, cx = center
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1) - cy
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist = np.sqrt(xx * xx + yy * yy)

    t = np.clip(dist / max(radius, 1e-6), 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)  # smoothstep
    darken = (1.0 - s) * severity  # 1 at center, 0 outside radius

    out = arr * (1.0 - darken[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# -- Nystagmus ---------------------------------------------------------------
# Source: myNystagmus.cs/shader. Animated jerk oscillation. Frozen-frame:
# motion-blur along the oscillation axis (perceptually equivalent to seeing
# constantly-moving content).

def apply_nystagmus(img: Image.Image, severity: float = 0.5,
                      axis: str = "horizontal") -> Image.Image:
    """Frozen-frame motion-blur approximation of jerk nystagmus."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    amp = severity * 0.04
    n_samples = 7
    offsets = np.linspace(-amp, amp, n_samples)

    accum = np.zeros_like(arr)
    for o in offsets:
        if axis == "horizontal":
            shift = int(round(o * (w - 1)))
            sh = np.roll(arr, shift, axis=1)
        elif axis == "vertical":
            shift = int(round(o * (h - 1)))
            sh = np.roll(arr, shift, axis=0)
        else:  # both
            sx = int(round(o * (w - 1)))
            sy = int(round(o * (h - 1)))
            sh = np.roll(np.roll(arr, sx, axis=1), sy, axis=0)
        accum += sh
    out = accum / n_samples
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# -- Flickering Stars --------------------------------------------------------
# Source: FlickeringStars.cs. Random star sprites that fade in/out. Frozen-
# frame: render N stars at fixed brightness (sampled from the fade curve).

def apply_flickering_stars(img: Image.Image, severity: float = 0.5,
                              n_stars: int = None, seed: int = 0) -> Image.Image:
    """Frozen flickering-star overlay (random bright dots)."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    if n_stars is None:
        n_stars = int(20 + severity * 80)
    rng = np.random.default_rng(seed)
    star_y = rng.integers(0, h, n_stars)
    star_x = rng.integers(0, w, n_stars)
    brightness = rng.random(n_stars).astype(np.float32) * severity

    out = arr.copy()
    # Sprite footprint: 5x5 with gaussian falloff
    dy_grid, dx_grid = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing="ij")
    fall = np.clip(1.0 - (dy_grid * dy_grid + dx_grid * dx_grid) / 8.0, 0.0, 1.0)
    for sy_, sx_, b in zip(star_y, star_x, brightness):
        y0, y1 = max(0, sy_ - 2), min(h, sy_ + 3)
        x0, x1 = max(0, sx_ - 2), min(w, sx_ + 3)
        py0 = y0 - (sy_ - 2); py1 = py0 + (y1 - y0)
        px0 = x0 - (sx_ - 2); px1 = px0 + (x1 - x0)
        sprite = (255.0 * b * fall[py0:py1, px0:px1])[..., None]
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], sprite)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# -- Floaters ----------------------------------------------------------------
# Source: myFloaters.cs/shader. Original generates a 1024² texture with N
# elliptical floater shapes (dark/light/scintillating modes), gaze-contingent.
# Frozen-frame: paint random oriented elliptical translucent blobs over the
# image (gaze-fixed, single texture frame).

def apply_floaters(img: Image.Image, severity: float = 0.5,
                     n_floaters: int = None, mode: str = "dark",
                     floater_size: int = 1, circle_radius_uv: float = 0.146,
                     seed: int = 0) -> Image.Image:
    """Vitreous floaters: many tiny grain dots, dense central disc + soft
    falloff + 1% noise everywhere — faithful port of Unity's IsInsideCircle.

    Source: `myFloaters.cs` (CC BY 4.0). Unity samples candidate positions
    uniformly in a 1024² overlay and accepts via three zones:
      - d ≤ R  : accept (P=1.0)
      - R < d ≤ 2R : P = 0.01 + 0.99·(1 − (d−R)/R)  (soft falloff)
      - d > 2R : P = 0.01  (1% noise everywhere)
    where R = circleRadius (default 150). The 1024² overlay is then stretched
    onto the screen UVs, so on a 2048×1024 screen the disc becomes a 2:1
    ellipse. We mirror that here by sampling in Unity's overlay space and
    mapping to screen coords.

    Args:
        img: Input PIL Image (RGB).
        severity: 0..1. Scales number of floaters and per-pixel alpha.
        n_floaters: explicit count override. Default = severity * 50000.
        mode: "dark" (black spots) or "light" (white spots).
        floater_size: pixel size of each speck in overlay space (Unity=1).
        circle_radius_uv: disc radius as fraction of overlay size (Unity
            150/1024 ≈ 0.146).
        seed: RNG seed.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    if n_floaters is None:
        n_floaters = int(severity * 50000)
    n_floaters = int(max(0, n_floaters))
    if n_floaters == 0:
        return img.copy()

    rng = np.random.default_rng(seed)

    # Unity samples in a 1024² overlay; we replicate then stretch to screen.
    overlay_size = 1024.0
    cx_o = cy_o = overlay_size * 0.5
    r_o = circle_radius_uv * overlay_size  # ≈ 150 by default

    # Vectorised rejection sampling per IsInsideCircle's 3-zone acceptance.
    pts_x = []
    pts_y = []
    target = n_floaters
    while len(pts_x) < target:
        batch = max(target * 2, 4096)
        cxs = rng.uniform(0.0, overlay_size, size=batch)
        cys = rng.uniform(0.0, overlay_size, size=batch)
        d = np.hypot(cxs - cx_o, cys - cy_o)
        # Outside-disc soft falloff: 1 at d=R, 0 at d>=2R.
        falloff = np.clip(1.0 - (d - r_o) / r_o, 0.0, 1.0)
        # Combined acceptance: inside disc → 1; outside → 0.01 + 0.99·falloff.
        accept_p = np.where(d <= r_o, 1.0, 0.01 + 0.99 * falloff)
        keep = rng.random(batch) < accept_p
        pts_x.extend(cxs[keep].tolist())
        pts_y.extend(cys[keep].tolist())

    pts_x = np.asarray(pts_x[:target], dtype=np.float64)
    pts_y = np.asarray(pts_y[:target], dtype=np.float64)

    # Map overlay coords → screen pixel coords.
    sx_f = pts_x * (w / overlay_size)
    sy_f = pts_y * (h / overlay_size)
    xs = np.clip(sx_f.astype(int), 0, w - 1)
    ys = np.clip(sy_f.astype(int), 0, h - 1)

    out = arr.copy()
    target_val = 0.0 if mode == "dark" else 255.0
    alpha = severity  # per-pixel blend strength

    if floater_size <= 1:
        # Single-pixel specks (Unity default). Use np.add.at to handle
        # duplicate (y,x) indices correctly (no double-blend overshoot —
        # last-write-wins via direct assignment is fine here since alpha is
        # constant; we just want the binary 'pixel touched' look).
        for c in range(3):
            out[ys, xs, c] = arr[ys, xs, c] * (1.0 - alpha) + target_val * alpha
    else:
        # Small blob ≥ 2 px. Random per-pixel mask inside the box mimics
        # Unity's `Mathf.PerlinNoise(...) > 0.5` per-subpixel gating. Box size
        # is measured in overlay px, so scale to screen.
        s_screen_x = max(1, int(round(floater_size * w / overlay_size)))
        s_screen_y = max(1, int(round(floater_size * h / overlay_size)))
        half_x, half_y = s_screen_x // 2, s_screen_y // 2
        for sx, sy in zip(xs, ys):
            x0, x1 = max(0, sx - half_x), min(w, sx + half_x + 1)
            y0, y1 = max(0, sy - half_y), min(h, sy + half_y + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = rng.random((y1 - y0, x1 - x0)) > 0.5
            for c in range(3):
                tile = out[y0:y1, x0:x1, c]
                tile[mask] = tile[mask] * (1.0 - alpha) + target_val * alpha
                out[y0:y1, x0:x1, c] = tile
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =============================================================================
# Presets — common combinations for study use
# =============================================================================

PRESETS = {
    "mild_low_vision": {
        "description": "Mild low vision: slight blur + reduced contrast",
        "filters": [
            ("blur", {"severity": 0.2}),
            ("bcg", {"brightness": 0.9, "contrast": 0.8, "gamma": 1.1}),
        ],
    },
    "moderate_low_vision": {
        "description": "Moderate low vision: noticeable blur + contrast loss",
        "filters": [
            ("blur", {"severity": 0.4}),
            ("bcg", {"brightness": 0.85, "contrast": 0.7, "gamma": 1.2}),
        ],
    },
    "severe_low_vision": {
        "description": "Severe low vision: heavy blur + major contrast loss",
        "filters": [
            ("blur", {"severity": 0.7}),
            ("bcg", {"brightness": 0.75, "contrast": 0.5, "gamma": 1.4}),
        ],
    },
    "deuteranomaly_moderate": {
        "description": "Moderate red-green color blindness (deuteranomaly)",
        "filters": [
            ("cvd", {"cvd_type": "deuteranomaly", "severity": 0.6}),
        ],
    },
    "protanomaly_moderate": {
        "description": "Moderate red color blindness (protanomaly)",
        "filters": [
            ("cvd", {"cvd_type": "protanomaly", "severity": 0.6}),
        ],
    },
    "cataracts_moderate": {
        "description": "Moderate cataracts: blur + yellowing + contrast loss",
        "filters": [
            ("cataracts", {"severity": 0.5}),
        ],
    },
    "elderly_vision": {
        "description": "Typical elderly vision: mild cataracts + slight CVD",
        "filters": [
            ("cataracts", {"severity": 0.3}),
            ("cvd", {"cvd_type": "tritanomaly", "severity": 0.3}),
        ],
    },
    "photophobia": {
        "description": "Light sensitivity / glare from bright regions (bloom)",
        "filters": [
            ("bloom", {"severity": 0.5}),
        ],
    },
    "macular_degeneration": {
        "description": "Central field loss (mip-blur in center via radial mask)",
        "filters": [
            ("field_loss", {"severity": 0.6, "field_type": "central"}),
        ],
    },
    "tunnel_vision": {
        "description": "Peripheral field loss (mip-blur at edges)",
        "filters": [
            ("field_loss", {"severity": 0.6, "field_type": "peripheral"}),
        ],
    },
    "metamorphopsia": {
        "description": "Visual distortion / wavy lines (radial UV warp)",
        "filters": [
            ("distortion", {"severity": 0.5}),
        ],
    },
    "diplopia": {
        "description": "Double vision (horizontal offset blend)",
        "filters": [
            ("double_vision", {"severity": 0.4}),
        ],
    },
    "low_contrast_sensitivity": {
        "description": "Detail loss via posterization (severe contrast sensitivity loss)",
        "filters": [
            ("detail_loss", {"severity": 0.6}),
        ],
    },
    "low_resolution": {
        "description": "Heavily pixelated view (e.g. very low acuity)",
        "filters": [
            ("pixelation", {"severity": 0.5}),
        ],
    },
    "led_display": {
        "description": "LED-board appearance (cellular dot pattern)",
        "filters": [
            ("led", {"severity": 0.5}),
        ],
    },
    "vortex_distortion": {
        "description": "Local swirling distortion at fixed gaze point",
        "filters": [
            ("vortex", {"severity": 0.5}),
        ],
    },
    "static_wiggle": {
        "description": "Static sinusoidal warp (wiggle frozen)",
        "filters": [
            ("wiggle", {"severity": 0.5}),
        ],
    },
    "visual_noise": {
        "description": "Per-pixel additive noise (snow / interference)",
        "filters": [
            ("noise", {"severity": 0.5}),
        ],
    },
    "rgb_glitch": {
        "description": "Frozen RGB-channel-shift glitch",
        "filters": [
            ("glitch", {"severity": 0.5}),
        ],
    },
    "scintillating_scotoma": {
        "description": "Migraine aura: crescent of shimmering colored noise",
        "filters": [
            ("teichopsia", {"severity": 0.6}),
        ],
    },
    "central_scotoma": {
        "description": "Central dark spot (foveal darkness)",
        "filters": [
            ("foveal_darkness", {"severity": 0.6}),
        ],
    },
    "nystagmus_motion": {
        "description": "Frozen-frame motion blur from involuntary eye oscillation",
        "filters": [
            ("nystagmus", {"severity": 0.5}),
        ],
    },
    "flickering_stars": {
        "description": "Random bright dots scattered across the field",
        "filters": [
            ("flickering_stars", {"severity": 0.5}),
        ],
    },
    "floaters": {
        "description": "Dark elliptical translucent floaters",
        "filters": [
            ("floaters", {"severity": 0.5}),
        ],
    },
}


_FILTER_DISPATCH = {
    "blur": apply_blur,
    "cvd": apply_cvd,
    "bcg": apply_bcg,
    "cataracts": apply_cataracts,
    "bloom": apply_bloom,
    "field_loss": apply_field_loss,
    "distortion": apply_distortion,
    "double_vision": apply_double_vision,
    "detail_loss": apply_detail_loss,
    # Phase 2
    "pixelation": apply_pixelation,
    "led": apply_led,
    "vortex": apply_vortex,
    "wiggle": apply_wiggle,
    "noise": apply_noise,
    "glitch": apply_glitch,
    "teichopsia": apply_teichopsia,
    "foveal_darkness": apply_foveal_darkness,
    "nystagmus": apply_nystagmus,
    "flickering_stars": apply_flickering_stars,
    "floaters": apply_floaters,
}


def apply_preset(img: Image.Image, preset_name: str) -> Image.Image:
    """Apply a named preset (chain of filters)."""
    preset = PRESETS[preset_name]
    result = img.copy()
    for filter_name, kwargs in preset["filters"]:
        fn = _FILTER_DISPATCH[filter_name]
        result = fn(result, **kwargs)
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VipSim image filters for VLM accessibility analysis")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")

    filter_choices = list(_FILTER_DISPATCH.keys())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--filter", choices=filter_choices, help="Single filter to apply")
    group.add_argument("--preset", choices=list(PRESETS.keys()), help="Named preset")
    group.add_argument("--list-presets", action="store_true", help="List available presets")

    # Filter-specific args
    parser.add_argument("--severity", type=float, default=0.5,
                        help="Severity 0.0-1.0 (used by all severity-driven filters)")
    parser.add_argument("--type", dest="cvd_type", default="deuteranomaly",
                        choices=["protanomaly", "deuteranomaly", "tritanomaly", "monochrome"],
                        help="CVD type")
    parser.add_argument("--brightness", type=float, default=1.0, help="BCG brightness")
    parser.add_argument("--contrast", type=float, default=1.0, help="BCG contrast")
    parser.add_argument("--gamma", type=float, default=1.0, help="BCG gamma")
    parser.add_argument("--field-type", dest="field_type", default="central",
                        choices=["central", "peripheral"],
                        help="Field-loss mask type")
    parser.add_argument("--axis", default="horizontal",
                        choices=["horizontal", "vertical", "diagonal", "both"],
                        help="Displacement axis (double-vision / nystagmus)")
    parser.add_argument("--position", default="left",
                        choices=["left", "right"],
                        help="Teichopsia crescent position")
    parser.add_argument("--floater-mode", dest="floater_mode", default="dark",
                        choices=["dark", "light"],
                        help="Floater appearance mode")

    args = parser.parse_args()

    if args.list_presets:
        for name, preset in PRESETS.items():
            print(f"  {name}: {preset['description']}")
        return

    img = Image.open(args.input).convert("RGB")

    if args.preset:
        result = apply_preset(img, args.preset)
    elif args.filter == "cvd":
        result = apply_cvd(img, cvd_type=args.cvd_type, severity=args.severity)
    elif args.filter == "bcg":
        result = apply_bcg(img, brightness=args.brightness, contrast=args.contrast, gamma=args.gamma)
    elif args.filter == "field_loss":
        result = apply_field_loss(img, severity=args.severity, field_type=args.field_type)
    elif args.filter == "double_vision":
        result = apply_double_vision(img, severity=args.severity, axis=args.axis)
    elif args.filter == "nystagmus":
        result = apply_nystagmus(img, severity=args.severity, axis=args.axis)
    elif args.filter == "teichopsia":
        result = apply_teichopsia(img, severity=args.severity, position=args.position)
    elif args.filter == "floaters":
        result = apply_floaters(img, severity=args.severity, mode=args.floater_mode)
    else:
        # Generic severity-only filters
        fn = _FILTER_DISPATCH[args.filter]
        result = fn(img, severity=args.severity)

    result.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
