#!/usr/bin/env python3
"""
VipSim image filters — Python reimplementation of VIP-Sim Unity shaders.

Simulates visual impairments on screenshots for VLM accessibility analysis.
Based on shader code from https://github.com/Max-Raed/VIP-Sim (CC BY 4.0).

Usage (installed: `vipsim ...` or `python -m vipsim ...`):
    vipsim input.png output.png --filter blur --severity 0.5
    vipsim input.png output.png --filter cvd --type deuteranomaly --severity 0.8
    vipsim input.png output.png --filter bcg --brightness 0.8 --contrast 0.7 --gamma 1.5
    vipsim input.png output.png --filter cataracts --severity 0.5
    vipsim input.png output.png --preset moderate_low_vision
"""

import argparse
import math
import warnings
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
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    # VIP-Sim is a Linear-colour-space project, so myRecolour.shader applies
    # the matrices to linear RGB (the space Machado et al. derive them in).
    lin = _image_to_linear(img.convert("RGB"))

    if cvd_type == "monochrome":
        gray_weights = np.array([0.299, 0.587, 0.114])
        gray = np.dot(lin, gray_weights)
        gray_rgb = np.stack([gray] * 3, axis=-1)
        result = lin * (1 - severity) + gray_rgb * severity
    else:
        matrices = CVD_MATRICES[cvd_type]
        # Interpolate between matrix steps (same as VIP-Sim)
        n = len(matrices)
        idx = severity * (n - 1)
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, n - 1)
        w = idx - i0
        mat = matrices[i0] * (1 - w) + matrices[i1] * w  # 3x3
        result = np.dot(lin, mat.T)

    return Image.fromarray(_to_uint8(_linear_to_srgb(result)))


# =============================================================================
# Gaussian Blur / Hyperopia (from myBlur.shader)
# =============================================================================

def _check_blur_radius(radius) -> None:
    """Raise ValueError for exactly the radii Pillow 13's GaussianBlur
    rejects: NaN, infinite, or 2**31 - 64 px or more either way.  Pillow 12
    segfaults on those radii, taking the whole interpreter down.  The
    messages are Pillow 13's ("radius must be >= 0", "radius too large"),
    but beyond ~1.4e15 px Pillow 13 picks between them by float32 rounding,
    so there only the exception type is guaranteed to match."""
    r = float(radius)
    if not np.isfinite(r):
        raise ValueError("radius must be >= 0")
    if abs(r) >= 2147483584.0:
        raise ValueError("radius too large")


def apply_blur(img: Image.Image, severity: float = 0.5) -> Image.Image:
    """Apply Gaussian blur to simulate reduced visual acuity (hyperopia).

    Args:
        img: Input PIL Image.
        severity: 0.0 (no blur) to 1.0 (heavy blur). Maps to blur radius 0-15px.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()
    radius = severity * 15.0
    _check_blur_radius(radius)            # NaN severity survives np.clip
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


# An 8-bit image has only 256 levels, so decode each level once and index the
# table with the image: table[levels] holds exactly the values of
# _srgb_to_linear(levels.astype(dtype) / 255.0), element for element, without
# a per-pixel power.
_SRGB_TO_LINEAR_LUT = {
    np.dtype(dt): _srgb_to_linear(np.arange(256, dtype=dt) / 255.0)
    for dt in (np.float32, np.float64)
}


def _image_to_linear(img, dtype=np.float32) -> np.ndarray:
    """_srgb_to_linear(np.asarray(img, dtype) / 255.0), by table for 8-bit data."""
    levels = np.asarray(img)
    if levels.dtype != np.uint8:  # 1-bit, 16-bit, 32-bit and float modes
        return _srgb_to_linear(levels.astype(dtype) / 255.0)
    return _SRGB_TO_LINEAR_LUT[np.dtype(dtype)][levels]


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    # np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055),
    # evaluated in place: the same arithmetic per element, fewer temporaries.
    out = np.power(x, 1.0 / 2.4)
    if type(out) is not np.ndarray:  # scalar, 0-d input or ndarray subclass
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * out - 0.055)
    out *= 1.055
    out -= 0.055
    np.multiply(x, 12.92, out=out, where=x <= 0.0031308)
    return out


def _to_uint8(x: np.ndarray, scale: float = 255.0) -> np.ndarray:
    """Quantise to uint8 by rounding; `scale` maps the input range to 0..255.

    Truncating with astype(np.uint8) instead biases every output ~0.5 LSB
    darker, and the bias accumulates when filters are chained.
    """
    # The old expression's own product: np.multiply(x, scale) converts big
    # Python ints differently (OverflowError for 2**63 with an int scale).
    y = np.asarray(x) * scale
    if not (type(y) is np.ndarray and y.dtype.kind == "f"):
        return np.clip(np.rint(y), 0, 255).astype(np.uint8)
    np.rint(y, out=y)
    np.clip(y, 0, 255, out=y)
    return y.astype(np.uint8)


# =============================================================================
# Brightness / Contrast / Gamma (from myBrightnessContrastGamma.shader)
# =============================================================================

def apply_bcg(img: Image.Image, brightness: float = 1.0, contrast: float = 1.0, gamma: float = 1.0,
              saturation: float = 1.0,
              color_space: str = "linear") -> Image.Image:
    """Apply brightness, contrast, gamma and (extension) saturation.

    Matches the VIP-Sim shader formula (executed in linear colour space, as
    Unity does when the project Color Space is Linear):
        color *= brightness
        color = (color - 0.5) * contrast + 0.5
        color = pow(color, gamma)

    Args:
        img: Input PIL Image; converted to RGB like every other filter input
            (alpha is dropped, not tone-mapped).
        brightness: Multiplier (1.0 = no change, <1 darker, >1 brighter).
        contrast: Multiplier (1.0 = no change, <1 less contrast, >1 more contrast).
        gamma: Exponent (1.0 = no change, >1 darker midtones, <1 lighter midtones).
        saturation: 1.0 = no change, 0.0 = full greyscale, >1 more vivid.
            NOT in the Unity shader (extension for profile matching): blends
            each pixel toward its luminance, so colors drain without touching
            tone/contrast. Applied after the b/c/g chain, in the same space.
        color_space: "linear" (default, faithful to a Linear-space Unity
            project) or "srgb" — run the same formula directly on the encoded
            sRGB values, as Unity would in a Gamma-space project. In sRGB
            space the identical contrast value desaturates far more
            aggressively and treats highlights differently; kept as an option
            because the original VIP-Sim project's colour-space setting is
            unconfirmed.
    """
    rgb = img.convert("RGB")
    if color_space == "srgb":
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    else:
        arr = _image_to_linear(rgb)

    # Brightness
    arr = arr * brightness
    # Contrast
    arr = (arr - 0.5) * contrast + 0.5
    # Clamp before gamma to avoid NaN from negative values
    arr = np.clip(arr, 0.0, 1.0)
    # Gamma
    if gamma != 1.0:
        arr = np.power(arr, gamma)

    arr = np.clip(arr, 0.0, 1.0)
    # Saturation (extension): lerp toward per-pixel luminance
    if saturation != 1.0:
        lum = (arr @ np.array([0.2126, 0.7152, 0.0722],
                              dtype=np.float32))[..., None]
        arr = np.clip(lum + (arr - lum) * saturation, 0.0, 1.0)
    out = arr if color_space == "srgb" else _linear_to_srgb(arr)
    return Image.fromarray(_to_uint8(out))


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


def _apply_bcg_per_channel(lin: np.ndarray, brightness: float, contrast: float,
                            coeffs=(0.5, 0.5, 0.5), gamma: float = 1.0) -> np.ndarray:
    """BCG with per-channel contrast pivot — matches myBrightnessContrastGamma.shader.

    Input is linear RGB in [0,1] (see _image_to_linear), output is sRGB in
    [0,1]; the shader math runs in linear space to mirror Unity's Linear
    colour-space rendering.

    Formula (per-channel, linear):
        color *= brightness
        color = (color - coeff) * contrast + coeff
        color = clamp(0, 1)
        color = pow(color, gamma)
    """
    coeffs_arr = np.array(coeffs, dtype=np.float32).reshape(1, 1, 3)
    out = lin * brightness
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

    lin = _image_to_linear(img.convert("RGB"))
    h, w = lin.shape[:2]

    # 1. Per-channel BCG (brightness/contrast both shrink linearly with severity)
    brightness = (1.0 - severity) if use_brightness else 1.0
    contrast = (1.0 - severity) if use_contrast else 1.0
    arr = _apply_bcg_per_channel(lin, brightness, contrast,
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

    out = _to_uint8(arr)
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
    _check_blur_radius(radius_px)
    pil = Image.fromarray(_to_uint8(arr))
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
    arr = _image_to_linear(img.convert("RGB"))
    h, w = arr.shape[:2]

    # 1. Downsample by 4 (Resolution.Low → divider=4, widthMod=0.5)
    divider = 4
    width_mod = 0.5
    rt_w, rt_h = max(1, w // divider), max(1, h // divider)
    down = np.asarray(
        Image.fromarray(_to_uint8(arr)).resize((rt_w, rt_h), Image.BILINEAR),
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
        Image.fromarray(_to_uint8(bright)).resize(
            (w, h), Image.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    out_lin = np.clip(arr + bloom_full, 0.0, 1.0)
    out = _linear_to_srgb(out_lin)
    return Image.fromarray(_to_uint8(out))


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
    return [_mip_level(arr, lvl) for lvl in range(levels)]


def _mip_level(arr: np.ndarray, lvl: int) -> np.ndarray:
    """Level `lvl` of _build_mip_pyramid(arr, ...), built on its own."""
    if lvl == 0:
        return arr
    return _separable_gaussian(arr, radius_px=float(2 ** (lvl - 1)))


def _blend_mip_levels(arr: np.ndarray, bias: np.ndarray, n_levels: int) -> np.ndarray:
    """Per-pixel lerp between mip levels floor(bias) and floor(bias) + 1.

    The upper level is clamped to n_levels - 1. Evaluates the same expression
    on the same values as stacking _build_mip_pyramid and fancy-indexing it,
    one floor(bias) value at a time, so the n_levels x frame stack is never
    built. Levels no pixel samples are never blurred, and where every frac of
    a group is 0 the upper level's weight is exactly 0, so the lower level
    stands in for it. Level indices follow the stack's indexing too: a
    negative floor(bias) counts from the top level, and one outside
    [-n_levels, n_levels) raises IndexError (a NaN bias casts to -2**31).
    """
    lo = np.floor(bias).astype(np.int32)
    if lo.size and (lo.min() < -n_levels or lo.max() >= n_levels):
        bad = lo.min() if lo.min() < -n_levels else lo.max()
        raise IndexError(f"index {bad} is out of bounds for axis 0 "
                         f"with size {n_levels}")
    frac = bias - lo  # float32 - int32 promotes to float64
    out = np.empty(arr.shape, dtype=np.result_type(arr, frac))
    built = {}

    def level(i):
        if i not in built:
            built[i] = _mip_level(arr, i)
        return built[i]

    for i in np.unique(lo).tolist():
        sel = lo == i
        f = frac[sel][:, None]
        lower = level(i % n_levels)[sel]
        upper = (level(min(max(i + 1, 0), n_levels - 1))[sel] if f.any()
                 else lower)
        out[sel] = lower * (1 - f) + upper * f
        del built[i % n_levels]  # lower levels are not sampled again
    return out


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
# VIP-Sim/windows/Assets/Resources/macular-degeneration.jpg), shipped as
# package data. Used as the faithful overlay for apply_field_loss; falls back
# to the radial mask if the asset is missing.
_MACULAR_OVERLAY_PATH = (
    Path(__file__).parent / "assets" / "macular-degeneration.jpg"
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
                      use_overlay_asset: bool = True,
                      mask_center=None,
                      mask_radius: float = None) -> Image.Image:
    """Field-loss simulation via per-pixel mip-level selection.

    Source: `myFieldLoss.cs` + `myFieldLoss.shader` (CC BY 4.0).
    Uses the real VIP-Sim `macular-degeneration.jpg` overlay if available
    in the package's `assets/`; otherwise falls back to a generic radial mask.

    Args:
        img: Input PIL Image (RGB).
        severity: 0.0 (no effect) to 1.0 (max blur in masked region).
        field_type: "central" (macular-style, blurred center) or
            "peripheral" (tunnel-vision-style, blurred periphery).
        overlay_scale: Mirrors Unity `_OverlayScale` — larger = overlay covers
            a larger fraction of the image.
        use_overlay_asset: If True, try to load the VIP-Sim overlay JPG
            (`vipsim/assets/macular-degeneration.jpg`); falls back to a
            radial mask if missing. Set False to force the radial mask.
        mask_center: Optional (cy, cx) in normalised coords. When given, the
            radial mask is used (overlay asset skipped) and centred there —
            lets the blur region follow a repositioned scotoma. Default None
            keeps the original behaviour exactly.
        mask_radius: Optional direct radius of the radial mask (fraction of
            image, ~0.05..0.7). Overrides the 0.45/overlay_scale mapping;
            None keeps the original behaviour.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    # numLevelsOfBlur in Unity = 1 + 1 + floor(ln(max(W,H))). For 2048×1024
    # this is 1+1+floor(ln 2048)=1+1+7=9. Match that so max blur is comparable.
    n_levels = int(2 + np.floor(np.log(max(h, w))))

    # Try the real Unity overlay first; fall back to a radial mask
    mask = None
    if use_overlay_asset and mask_center is None:
        mask = _load_macular_overlay(h, w, field_type=field_type,
                                       overlay_scale=overlay_scale)
    if mask is None:
        # Fallback: generic radial mask (Phase-1 substitute)
        r = (mask_radius if mask_radius is not None
             else 0.45 / max(overlay_scale, 0.1))
        mask = _radial_weight_mask(h, w, kind=field_type,
                                    center=mask_center or (0.5, 0.5),
                                    radius=min(r, 0.7))
    # Severity scales the maximum mip level effectively reached
    w_scaled = mask * severity

    # Bias = w * (n_levels - 1): per-pixel float "level"
    bias = w_scaled * (n_levels - 1)
    if type(bias) is np.ndarray and bias.shape == (h, w):
        out = _blend_mip_levels(arr, bias, n_levels)
    else:  # array or masked mask kwargs: the old stacked lookup
        lo = np.floor(bias).astype(np.int32)
        hi = np.clip(lo + 1, 0, n_levels - 1)
        frac = (bias - lo)[..., None]
        stack = np.stack(_build_mip_pyramid(arr, n_levels), axis=0)
        iy, ix = np.indices((h, w))
        out = (stack[lo, iy, ix] * (1 - frac) + stack[hi, iy, ix] * frac)

    return Image.fromarray(_to_uint8(out))


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
    return Image.fromarray(_to_uint8(out, 1.0))


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
                      pixel_size: float = None, hard: bool = False,
                      sampling: str = "average",
                      grid_offset_x: float = 0.0,
                      grid_offset_y: float = 0.0) -> Image.Image:
    """Pixelation / mosaic downsampling.

    Unity: `_PixelSize = 1000 - pixelRadius` cells across (default 990 →
    barely visible). The shader downsamples to that grid then bilinearly
    interpolates between the 4 surrounding cells when reading back.

    Args:
        severity: 0..1 — if `pixel_size` is None, maps to cells-across
            1000..20 (high sev = blockier).
        pixel_size: Override — cells across the WIDTH (Unity's `_PixelSize`).
            Unity default 990 (pixelRadius=10).
        hard: If True, upscale with NEAREST-neighbour so the blocks have HARD
            square edges (true mosaic / detail loss); if False (default), the
            faithful soft-bilinear shader look. Blocks are kept ~square by
            deriving the vertical cell count from the aspect ratio.
        sampling: "average" (default) box-averages all pixels in a cell —
            partially-covered cells turn muddy. "point" reads ONE source
            point per cell like the Unity shader's tex2D lookup: cells
            snap to a single local color, so high-contrast features (e.g.
            crosswalk stripes) become bold solid chunks that can fuse into
            L-shapes with their neighbours.
        grid_offset_x / grid_offset_y: 0..1 — shifts the block grid by that
            fraction of a cell, so the mosaic boundaries can be aligned with
            a reference image without moving the content. 0 = the grid starts
            at the image origin (previous behaviour). Implemented by edge-pad
            shifting the image, pixelating, and shifting back, so the outer
            ring of blocks samples replicated border pixels.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0 and pixel_size is None:
        return img.copy()
    if pixel_size is None:
        pixel_size = float(1000.0 - severity * 980.0)  # 1000..20
    pixel_size = max(2.0, float(pixel_size))

    w, h = img.size
    cells_x = max(2, int(round(pixel_size)))
    cells_y = max(2, int(round(cells_x * h / w))) if hard else cells_x
    ox = float(grid_offset_x) % 1.0
    oy = float(grid_offset_y) % 1.0
    dx = int(round(ox * (w / cells_x)))
    dy = int(round(oy * (h / cells_y)))
    if dx or dy:
        arr = np.asarray(img)
        arr = np.pad(arr, ((dy, 0), (dx, 0)) + ((0, 0),) * (arr.ndim - 2),
                     mode="edge")[:h, :w]
        img = Image.fromarray(arr)

    down = Image.NEAREST if sampling == "point" else Image.BILINEAR
    if hard:
        small = img.resize((cells_x, cells_y), down)
        out = small.resize((w, h), Image.NEAREST)  # HARD square blocks
    else:
        # Faithful shader look: square grid + bilinear up (soft blocks).
        small = img.resize((cells_x, cells_x), down)
        out = small.resize((w, h), Image.BILINEAR)

    if dx or dy:  # shift the content back; the grid keeps the offset
        arr = np.asarray(out)
        arr = np.pad(arr, ((0, dy), (0, dx)) + ((0, 0),) * (arr.ndim - 2),
                     mode="edge")[dy:dy + h, dx:dx + w]
        out = Image.fromarray(arr)
    return out


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
    levels = np.asarray(img.convert("RGB"))
    arr = levels.astype(np.float32) / 255.0
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
    if np.ndim(brightness) == 0 and brightness == 1 and pixelated.dtype == np.float32:
        # Unscaled cells are still 8-bit levels: decode them by table.
        lin = _image_to_linear(levels[sy[:, None], sx[None, :]])
    else:
        lin = _srgb_to_linear(pixelated)
    out_lin = np.clip(lin * mask[..., None], 0.0, 1.0)
    out = _linear_to_srgb(out_lin)

    # 3. Optional margin (black border)
    if margin > 0:
        m = np.ones((h, w), dtype=np.float32)
        m[(yy < margin) | (yy > 1.0 - margin)] = 0.0
        m[:, (xs < margin) | (xs > 1.0 - margin)] = 0.0
        out = out * m[..., None]

    return Image.fromarray(_to_uint8(out))


# -- Vortex ------------------------------------------------------------------
# Source: VortexEffect.cs/shader. Radial swirl; gaze-contingent in Unity, we
# use a fixed center (Phase-2 deviation). Inner gray fade preserved.

def apply_vortex(img: Image.Image, severity: float = 0.5,
                  center=(0.5, 0.5),
                  vortex_radius: float = None,
                  radius_y: float = None,
                  radius_y_down: float = None,
                  suction_strength: float = None,
                  falloff_exponent: float = 2.0,
                  swirl_strength: float = 0.0,
                  inner_circle_radius: float = 0.01,
                  inner_color: float = 0.5,
                  inner_center=None,
                  inner_fade: float = None,
                  inner_opacity: float = 1.0,
                  halo_radius: float = 0.0,
                  halo_center=None,
                  halo_thickness: float = 0.012,
                  halo_color: float = 1.0,
                  halo_opacity: float = 0.5,
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
        radius_y: Vertical semi-axis (same units as vortex_radius, 1.0 = image
            height). Makes the field elliptical (e.g. wide-but-flat so a broad
            band is affected without reaching floors above/below). None keeps
            the original circular field. NOT in the Unity shader (extension
            for profile matching); content-agnostic, transfers to any image.
        radius_y_down: Optional separate vertical reach for the LOWER half
            of the field; `radius_y` then only governs the upper half.
        suction_strength: Override — Unity default 1.0.
        falloff_exponent: Steepness of the pull's decay toward the rim.
            2.0 reproduces the Unity shader exactly; lower values keep
            more pull near the rim (same zero boundary at the radius).
        swirl_strength: Angular rotation (radians) added on top of the radial
            pinch, peaking at the center and decaying quadratically to 0 at
            `vortex_radius`. NOT in the Unity shader (extension for profile
            matching); default 0.0 keeps the original pure-suction behaviour.
        inner_circle_radius: Tiny grey core (Unity default 0.01).
        inner_color: Grey level of that core, 0..1. Unity blends to
            mid-grey (0.5); 1.0 gives the bright 'eye' dot seen in some
            reference renders.
        inner_center: Optional (cy, cx) position of the core dot; None
            keeps it at the vortex center (Unity behaviour). The dot is
            clipped to the vortex field, so keep it inside the radius.
        inner_fade: Edge softness of the dot alone; None inherits
            `fade_width` (Unity behaviour).
        inner_opacity: 0..1 blend of the dot over the vortexed image;
            1.0 = solid (Unity behaviour), lower = translucent.
        halo_radius: Radius of a luminous orb. 0 disables it (Unity
            behaviour). Fully bright inside this radius, then fades
            softly outward (no visible structure inside the orb).
        halo_center: Optional (cy, cx) position of the orb; None uses
            the core dot's center (which itself defaults to the vortex
            center). Lets dot and halo sit at different spots.
        halo_thickness: Gaussian width of the outward glow falloff.
        halo_color: Grey level of the orb, 0..1.
        halo_opacity: 0..1 peak translucency of the orb.
        noise_amount: Edge-wobble magnitude (Unity default 0.05).
        noise_scale: Noise grid frequency (Unity default 10).
        fade_width: Outer smoothstep width (Unity shader default 0.05).
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()
    levels = np.asarray(img.convert("RGB"))
    arr = levels.astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    cy, cx = center
    if vortex_radius is None:
        vortex_radius = 0.05 + severity * 0.45  # 0.05..0.5
    if suction_strength is None:
        suction_strength = 0.2 + severity * 1.8  # 0.2..2.0

    aspect = w / h
    # Before ys_up/ys_dn, as before #6: with several bad kwargs the
    # centre's error is still the one raised.
    adj_cx = cx * aspect
    # Elliptical field: stretch y so the ellipse becomes a circle of radius
    # `vortex_radius` in the working space; sampling positions are unstretched
    # on the way back out. `radius_y_down` allows an asymmetric vertical
    # reach (upper vs lower half of the field).
    ys_up = vortex_radius / max(radius_y, 1e-6) if radius_y else 1.0
    ys_dn = (vortex_radius / max(radius_y_down, 1e-6)
             if radius_y_down else ys_up)

    # Only pixels with dist_n < vortex_radius change (the inner dot and halo
    # are clipped to them too), and |value noise| < 1, so every changed pixel
    # has dist < R + |noise_amount * R|. Compute only inside that ellipse's
    # bounding box (+2 px for rounding), on the same global pixel coordinates.
    use_box = _vortex_box_ok(
        (cy, cx, vortex_radius, radius_y, radius_y_down, suction_strength,
         falloff_exponent, swirl_strength, inner_circle_radius, inner_fade,
         inner_opacity, halo_radius, halo_thickness, halo_opacity,
         noise_amount, noise_scale, fade_width),
        (inner_center, halo_center), (inner_color, halo_color))
    if use_box:
        R, A = float(vortex_radius), float(noise_amount)
        # The field's centre is cx * aspect in cx's own precision (NEP 50:
        # a float16 cx rounds it by up to 2**-11 relative, many pixels far
        # off-frame), so place the box around that value, not around cx.
        with np.errstate(all="ignore"):  # the field below warns as before
            fcx = float(cx * aspect) / aspect
        fcy = float(cy)
        reach = R + abs(A * R)
        reach_up = reach / abs(float(ys_up)) if ys_up else np.inf
        reach_dn = reach / abs(float(ys_dn)) if ys_dn else np.inf
        # The field is computed in float32: keep its rounding (a few ulps of
        # these magnitudes, in pixels) far below the 2 px margin, and keep
        # every value inside float32 range. Also the whole frame:
        # - a reach below 2**-50 (or a negative radius): every pixel outside
        #   the box has a stretched |dy| above reach, and below that dy * dy
        #   can underflow to 0, which puts far pixels inside the field;
        # - a noise_scale whose float32 hash (p_u * 12.9898 + p_v * 78.233)
        #   can overflow: sin(inf) is NaN, and the whole-frame arr[y0, x0]
        #   raises on it even at pixels outside the box.
        scale = h * (aspect * (1.0 + abs(fcx)) + 1.0 + abs(fcy)
                     + reach + reach_up + reach_dn)
        use_box = (reach > 2.0 ** -50 and np.isfinite(scale)
                   and scale < 2.0 ** 18 and max(abs(R), abs(A)) < 1e30
                   and (noise_scale is None or abs(float(noise_scale)) < 1e30))
    if use_box:
        # |dx| < reach  <=>  |col + 0.5 - cx * w| < reach * h, likewise rows
        c0 = int(min(w, max(0, np.floor(fcx * w - 0.5 - reach * h) - 2)))
        c1 = int(max(0, min(w, np.ceil(fcx * w - 0.5 + reach * h) + 3)))
        r0 = int(min(h, max(0, np.floor(fcy * h - 0.5 - reach_up * h) - 2)))
        r1 = int(max(0, min(h, np.ceil(fcy * h - 0.5 + reach_dn * h) + 3)))
    else:  # anything unusual: the whole frame, exactly as before
        r0, r1, c0, c1 = 0, h, 0, w
    box = (slice(r0, r1), slice(c0, c1))

    ys_norm = (np.arange(h, dtype=np.float32) + 0.5) / h
    xs_norm = (np.arange(w, dtype=np.float32) + 0.5) / w
    yy, xx = np.meshgrid(ys_norm[r0:r1], xs_norm[c0:c1], indexing="ij")

    adj_x = xx * aspect
    adj_y = yy
    adj_cy = cy
    dx = adj_x - adj_cx
    dy_raw = adj_y - adj_cy
    ys = np.where(dy_raw < 0, ys_up, ys_dn)
    dy = dy_raw * ys
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
    # Pixels outside the field pass through: _to_uint8(level / 255) == level
    # for every 8-bit level, so they keep their input value.
    out = levels.copy()
    if np.any(inside):
        eps = 1e-8
        src = arr[box]
        angle = np.arctan2(dy, dx)
        # Optional angular noise (small); the same field as noise_val
        angle = angle + noise_val * noise_amount

        # Radial suction: sample further out
        d_over_R = np.clip(dist_n / max(vortex_radius, eps), 0.0, 1.0)
        r_sample = dist_n * (1.0 + suction_strength
                             * (1.0 - d_over_R) ** falloff_exponent)

        # Swirl: rotate the sample angle, strongest at center, 0 at the rim
        if swirl_strength:
            angle = angle + swirl_strength * (1.0 - d_over_R) ** 2

        sample_x = adj_cx + r_sample * np.cos(angle)
        sin_a = np.sin(angle)
        ys_back = np.where(sin_a < 0, ys_up, ys_dn)
        sample_y = adj_cy + r_sample * sin_a / ys_back
        sample_x = sample_x / aspect  # undo aspect correction
        sample_x = np.clip(sample_x, 0.0, 1.0)
        sample_y = np.clip(sample_y, 0.0, 1.0)
        # Bilinear sampling (Unity tex2D is bilinear; nearest-neighbour
        # shreds strong stretches into aliased filaments)
        fx = sample_x * (w - 1)
        fy = sample_y * (h - 1)
        x0 = np.floor(fx).astype(np.int32)
        y0 = np.floor(fy).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        wx = (fx - x0)[..., None]
        wy = (fy - y0)[..., None]
        top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
        bot = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
        vortexed = top * (1.0 - wy) + bot * wy

        # Outer fade back to source (smoothstep R-fade..R)
        t_out = np.clip((dist_n - (vortex_radius - fade_width))
                        / max(fade_width, eps), 0.0, 1.0)
        s_out = t_out * t_out * (3.0 - 2.0 * t_out)
        blended = vortexed * (1.0 - s_out[..., None]) + src * s_out[..., None]

        # Inner core dot (grey in Unity; position/brightness/softness/opacity
        # configurable — only rendered inside the vortex field)
        has_halo = halo_radius > 0 and halo_opacity > 0
        if inner_circle_radius > 0 or has_halo:
            if inner_center is not None:
                icy, icx = inner_center
                idx = adj_x - icx * aspect
                idy = adj_y - icy
                idist = np.sqrt(idx * idx + idy * idy)
            else:
                idist = dist_n

        if inner_circle_radius > 0:
            i_fade = fade_width if inner_fade is None else inner_fade
            grey = np.full_like(src, inner_color)
            t_in = np.clip((idist - (inner_circle_radius - i_fade))
                            / max(i_fade, eps), 0.0, 1.0)
            s_in = t_in * t_in * (3.0 - 2.0 * t_in)
            alpha = (1.0 - s_in)[..., None] * inner_opacity
            alpha = np.where((idist < inner_circle_radius)[..., None], alpha, 0.0)
            blended = grey * alpha + blended * (1.0 - alpha)

        # Halo: luminous orb (solid glow inside halo_radius, Gaussian
        # falloff outward); own center if given
        if has_halo:
            if halo_center is not None:
                hcy, hcx = halo_center
                hdx = adj_x - hcx * aspect
                hdy = adj_y - hcy
                hdist = np.sqrt(hdx * hdx + hdy * hdy)
            else:
                hdist = idist
            band = np.where(
                hdist <= halo_radius, 1.0,
                np.exp(-0.5 * ((hdist - halo_radius)
                               / max(halo_thickness, eps)) ** 2))
            h_alpha = band[..., None] * halo_opacity
            halo_col = np.full_like(src, halo_color)
            blended = halo_col * h_alpha + blended * (1.0 - h_alpha)

        res = np.where(inside[..., None], blended, src)
        # Without the box, return what the whole-frame code returned, even
        # where array kwargs broadcast it to another shape.
        if use_box:
            out[box] = _to_uint8(res)
        else:
            out = _to_uint8(res)

    return Image.fromarray(out)


def _vortex_box_ok(scalars, centres, colours) -> bool:
    """True when apply_vortex may compute only inside its bounding box.

    Every per-pixel kwarg must be a real (int or float) scalar or None, the
    centres pairs of them, and the colours broadcast to one value per
    channel. Array kwargs broadcast the field to their own shape, so they
    take the whole frame.
    """
    def real(v):
        return v is None or (np.ndim(v) == 0
                             and np.asarray(v).dtype.kind in "fiu")
    try:
        vals = list(scalars)
        for c in centres:
            if c is not None:
                if np.ndim(c) != 1 or len(c) != 2:
                    return False
                vals.extend(c)
        return (all(real(v) for v in vals)
                and all(np.broadcast_shapes(np.shape(c), (3,)) == (3,)
                        for c in colours))
    except Exception:  # let the whole-frame code raise what it raised
        return False


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
    return Image.fromarray(_to_uint8(out))


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
    return Image.fromarray(_to_uint8(out, 1.0))


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
        raise ImportError("teichopsia requires scikit-image: "
                          "pip install 'vipsim[teichopsia]'") from e

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
    return Image.fromarray(_to_uint8(out))


# -- Foveal Darkness ---------------------------------------------------------
# Source: FovealDarkness.cs/shader. Smoothstep-faded dark spot at gaze
# location (we fix to image center).

def apply_foveal_darkness(img: Image.Image, severity: float = 0.5,
                            center=(0.5, 0.5), radius: float = None,
                            radius_x: float = None,
                            radius_y: float = None,
                            edge_width: float = None,
                            mode: str = "dark",
                            invert: bool = False,
                            band_width: float = None) -> Image.Image:
    """Central dark spot with smoothstep edge (gaze-fixed).

    `radius` is in NORMALISED units (fraction of each axis: radius 0.2 =>
    the darkened ellipse spans ~0.4 of the width AND ~0.4 of the height).
    Because the image is usually wider than tall, a single radius renders as a
    horizontal oval in pixels. Pass `radius_x` / `radius_y` to size the two
    axes independently (fractions of width / height respectively) — e.g.
    radius_x=0.15, radius_y=0.20 gives a compact oval ~30% wide x ~40% tall.

    `edge_width` (0..1, optional): fraction of the radius used as the edge
    feather. Default None keeps the original profile (smoothstep across the
    FULL radius: darkest at center, fading continuously). With e.g. 0.3 the
    interior is a flat plateau at full severity and only the outer 30% of the
    ellipse feathers to 0 — a uniform semi-transparent patch with a defined
    boundary rather than a center-peaked cloud.

    `mode`: "dark" (default) blends toward black — the original behaviour.
    "light" blends toward white instead (whitish veil / halo ellipse).
    "desat" drains color toward per-pixel grey (luminance) — a localized
    desaturation ellipse; combine with `invert` to grey a corner/edge.

    `invert` (default False): apply the effect OUTSIDE the ellipse instead
    of inside — e.g. mode='light' + invert gives a peripheral white glow
    ring/veil that fades in across the edge feather. False keeps the
    original behaviour exactly.

    `band_width` (only with invert, default None): limit the inverted
    effect to a BAND just outside the ellipse. Units are fractions of the
    radius: 0.5 means the effect dies out again by 1.5x the radius, so the
    far corners return to untouched. None = the inverted effect extends
    all the way out (previous behaviour).
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()

    if radius is None:
        radius = 0.05 + severity * 0.20
    rx = radius_x if radius_x is not None else radius
    ry = radius_y if radius_y is not None else radius

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    cy, cx = center
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1) - cy
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    # Elliptical normalised distance: 1.0 exactly on the ellipse boundary.
    t_raw = np.sqrt((xx / max(rx, 1e-6)) ** 2 + (yy / max(ry, 1e-6)) ** 2)
    t = np.clip(t_raw, 0.0, 1.0)
    if edge_width is not None:
        e = float(np.clip(edge_width, 1e-3, 1.0))
        u = np.clip((t - (1.0 - e)) / e, 0.0, 1.0)  # 0 inside plateau, 1 at rim
        s = u * u * (3.0 - 2.0 * u)
    else:
        s = t * t * (3.0 - 2.0 * t)  # smoothstep
    darken = (s if invert else (1.0 - s)) * severity
    if invert and band_width:
        u2 = np.clip((t_raw - 1.0) / max(band_width, 1e-6), 0.0, 1.0)
        darken = darken * (1.0 - u2 * u2 * (3.0 - 2.0 * u2))

    if mode == "desat":
        lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        out = arr + (lum[..., None] - arr) * darken[..., None]
    elif mode == "light":
        out = arr + (255.0 - arr) * darken[..., None]
    else:
        out = arr * (1.0 - darken[..., None])
    return Image.fromarray(_to_uint8(out, 1.0))


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
    return Image.fromarray(_to_uint8(out))


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
    return Image.fromarray(_to_uint8(out, 1.0))


# -- Floaters ----------------------------------------------------------------
# Source: myFloaters.cs/shader. Original generates a 1024² texture with N
# elliptical floater shapes (dark/light/scintillating modes), gaze-contingent.
# Frozen-frame: paint random oriented elliptical translucent blobs over the
# image (gaze-fixed, single texture frame).

# Mask draws _paint_blobs makes per chunk: about 40 MiB of temporaries.
_BLOB_DRAWS_PER_CHUNK = 1 << 20


def _paint_blobs(out: np.ndarray, xs, ys, sizes, rng: "np.random.Generator",
                 alpha: float, target_val: float,
                 overlay_size: float = 1024.0) -> None:
    """Blend `target_val` into a random ~60% of each blob's box, in place.

    A blob of overlay size `sz` centred on pixel (y, x) covers the box of
    pixels within s // 2 of it on each axis, s = max(1, round(sz * frame
    side / overlay_size)), clipped to the frame, and blends the box pixels
    where its rng.random(box shape) > 0.4. Vectorised over the blobs, with
    the same result as painting them one at a time:
      - PCG64 spends one draw per double, so rng.random calls over all
        boxes in blob order, in chunks of any size, yield the same values
        (empty boxes draw nothing);
      - every hit applies the same float32 blend, so a pixel hit n times gets
        it n times in a row.
    """
    if len(sizes) == 0:  # no blobs, no draws
        return
    h, w = out.shape[:2]
    f_x = np.asarray(sizes * w / overlay_size, dtype=np.float64)
    f_y = np.asarray(sizes * h / overlay_size, dtype=np.float64)
    bad = ~(np.isfinite(f_x) & np.isfinite(f_y)
            & (np.abs(f_x) < 2.0 ** 52) & (np.abs(f_y) < 2.0 ** 52))
    if bad.any():
        # astype(int) would wrap these; paint one blob at a time with Python
        # ints as before, which raises (or not) exactly as it did.
        for sx, sy, sz in zip(xs, ys, sizes):
            s_x = max(1, int(round(sz * w / overlay_size)))
            s_y = max(1, int(round(sz * h / overlay_size)))
            hx, hy = s_x // 2, s_y // 2
            x0, x1 = max(0, sx - hx), min(w, sx + hx + 1)
            y0, y1 = max(0, sy - hy), min(h, sy + hy + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = rng.random((y1 - y0, x1 - x0)) > 0.4
            for c in range(3):
                tile = out[y0:y1, x0:x1, c]
                tile[mask] = tile[mask] * (1.0 - alpha) + target_val * alpha
                out[y0:y1, x0:x1, c] = tile
        return
    s_x = np.maximum(1, np.rint(f_x).astype(np.int64))
    s_y = np.maximum(1, np.rint(f_y).astype(np.int64))
    hx, hy = s_x // 2, s_y // 2
    x0, x1 = np.maximum(0, xs - hx), np.minimum(w, xs + hx + 1)
    y0, y1 = np.maximum(0, ys - hy), np.minimum(h, ys + hy + 1)
    keep = (x1 > x0) & (y1 > y0)
    bw, bh = (x1 - x0)[keep], (y1 - y0)[keep]
    x0, y0 = x0[keep], y0[keep]
    area = bw * bh
    ends = np.cumsum(area)
    starts = ends - area
    total = int(ends[-1]) if ends.size else 0
    # The draws for all boxes are made in chunks of _BLOB_DRAWS_PER_CHUNK, so
    # memory stays bounded however large or many the blobs are. Consecutive
    # rng.random calls continue the same PCG64 stream (one draw per double),
    # and hit counts add up across chunks.
    hits = np.zeros(h * w, dtype=np.int64)
    for s in range(0, total, _BLOB_DRAWS_PER_CHUNK):
        e = min(total, s + _BLOB_DRAWS_PER_CHUNK)
        d = s + np.flatnonzero(rng.random(e - s) > 0.4)  # draws that hit
        b = np.searchsorted(ends, d, side="right")       # blob of each draw
        k = d - starts[b]   # row-major position within the blob's box
        bw_k = bw[b]
        hits += np.bincount((y0[b] + k // bw_k) * w + x0[b] + k % bw_k,
                            minlength=h * w)
    hits = hits.reshape(h, w)

    r, c = np.nonzero(hits)
    left = hits[r, c]
    while r.size:
        out[r, c] = out[r, c] * (1.0 - alpha) + target_val * alpha
        left -= 1
        more = left > 0
        r, c, left = r[more], c[more], left[more]


# apply_floaters' Gaussian and disc rejection samplers loop until enough
# points land in the frame, so parameters under which (almost) no draw can
# land would loop forever.  They are checked before sampling, and nothing is
# drawn, so every run that does not raise samples exactly as before:
# - a non-finite centre, spread or angle, or a disc radius of NaN or -inf,
#   raises ValueError;
# - so does a Gaussian whose draws land in the frame with probability below
#   _FLOATER_MIN_ACCEPTANCE, unless the expected number of draws for all its
#   points, n_floaters / p, stays within _FLOATER_MAX_EXPECTED_DRAWS (a few
#   floaters far off-frame still finish, in seconds to minutes);
# - a Gaussian that passes but would need more than _FLOATER_DRAW_BUDGET
#   draws (about 40 s) warns with its estimated run time, then samples.
_FLOATER_MIN_ACCEPTANCE = 1e-7
_FLOATER_MAX_EXPECTED_DRAWS = 1e8
_FLOATER_DRAW_BUDGET = 1e9
_FLOATER_DRAWS_PER_SEC = 2.5e7     # vectorised rng.normal pair + mask
_FLOATER_PASS_COST = 300.0         # fixed cost of one pass, in draws


def _gaussian_sampling_cost(p: float, n: int) -> float:
    """Expected cost, in draws, of sampling n points at acceptance p:
    n / p draws, plus about H_n / (2 p) passes (each draws 2 * remaining)."""
    h_n = math.log(n) + 0.5772156649 + 0.5 / n if n > 1 else 1.0
    return n / p + _FLOATER_PASS_COST * h_n / (2.0 * p)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, elementwise (NumPy has no erf; math.erfc keeps
    the core dependencies at numpy and Pillow)."""
    flat = [0.5 * math.erfc(-float(t) / math.sqrt(2.0)) for t in z.ravel()]
    return np.asarray(flat, dtype=np.float64).reshape(z.shape)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
# Where a conditional bound is this many sds from the conditional mean the
# normal CDF is saturated (Phi(-8.3) ~ 5e-17); pieces end at these.
_GL_T = np.array([-8.3, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.3])


def _gaussian_frame_probability(mu_x, mu_y, sigma_x, sigma_y, cos_a, sin_a,
                                size) -> float:
    """Probability that one draw of apply_floaters' rotated Gaussian lands
    in [0, size)^2.

    A draw (x, y) = mu + sigma_x*u*(cos_a, sin_a) + sigma_y*v*(-sin_a, cos_a)
    with u, v ~ N(0, 1) is bivariate normal.  Given x, y is normal with mean
    m(x) = mu_y + k*(x - mu_x) and sd s, so the probability is the integral
    over x in [0, size) of pdf_x(x) * P(0 <= y < size | x).  The integrand
    is smooth there; its only sharp features are where m(x) crosses a y edge
    (width s/|k|).  Gauss-Legendre (16 nodes) on pieces that end at those
    crossings, +-1..8.3 conditional sds around them, and every sd_x / 2
    integrates it to ~1e-10 relative."""
    sd_x = math.hypot(sigma_x * cos_a, sigma_y * sin_a)
    sd_y = math.hypot(sigma_x * sin_a, sigma_y * cos_a)
    # A marginal density never exceeds 1/(sqrt(2 pi) sd): the frame then
    # holds less than 1e-12 (and squaring the spreads below could overflow).
    if size / (math.sqrt(2.0 * math.pi) * max(sd_x, sd_y)) < 1e-12:
        return 0.0
    lo, hi = max(0.0, mu_x - 38.0 * sd_x), min(size, mu_x + 38.0 * sd_x)
    if not lo < hi:                     # frame beyond 38 sigma along x
        return 0.0
    k = (sigma_x * sigma_x - sigma_y * sigma_y) * sin_a * cos_a / (sd_x * sd_x)
    s = sigma_x * sigma_y / sd_x        # sd of y given x
    cuts = [mu_x + 0.5 * sd_x * np.arange(-76.0, 77.0)]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if k != 0.0:
            for edge in (0.0, size):
                cuts.append(mu_x + (edge - mu_y - s * _GL_T) / k)
        cuts = np.concatenate(cuts)
        cuts = np.unique(np.clip(cuts[np.isfinite(cuts)], lo, hi))
    cuts = np.unique(np.concatenate(([lo, hi], cuts)))
    half = 0.5 * np.diff(cuts)
    x = ((cuts[:-1] + half)[:, None] + half[:, None] * _GL_X).ravel()
    w = (half[:, None] * _GL_W).ravel()
    m = mu_y + k * (x - mu_x)
    a, b = -m / s, (size - m) / s       # P(a <= z < b), a < b
    flip = a > 0.0                      # both in the upper tail: mirror
    mass = np.clip(_normal_cdf(np.where(flip, -a, b))
                   - _normal_cdf(np.where(flip, -b, a)), 0.0, None)
    pdf = np.exp(-0.5 * ((x - mu_x) / sd_x) ** 2) / (math.sqrt(2.0 * math.pi) * sd_x)
    # Rounding can take a near-certain draw a few ulps past 1.
    return min(1.0, float(np.sum(w * pdf * mass)))


def _check_gaussian_reachable(mu_x, mu_y, sigma_x, sigma_y, cos_a, sin_a,
                              size, target: int = None) -> None:
    """Check the Gaussian sampler's parameters before it draws target points
    (see the rules above _FLOATER_MIN_ACCEPTANCE): raise ValueError where it
    would (practically) never finish, warn where it would take long.  Without
    target, only the _FLOATER_MIN_ACCEPTANCE floor applies."""
    n = 1 if target is None else target
    limit = _FLOATER_MIN_ACCEPTANCE
    if target is not None:
        limit = min(limit, target / _FLOATER_MAX_EXPECTED_DRAWS)
    vals = (mu_x, mu_y, sigma_x, sigma_y, cos_a, sin_a)
    if any(np.ndim(v) and np.size(v) != 1 for v in vals):
        _check_per_draw_gaussian(vals, size, limit)
        return
    # A 1-element array (np.array([30.]) as angle, say) broadcasts like a
    # scalar in the sampler, which then draws the scalar's points: check its
    # value.
    vals = [np.asarray(v).reshape(-1)[0] if np.ndim(v) and np.size(v) == 1
            else v for v in vals]
    mu_x, mu_y, sigma_x, sigma_y, cos_a, sin_a = vals
    params = np.array(vals, dtype=np.float64)
    if not np.isfinite(params).all():
        p = 0.0
    else:
        # The disc of radius d (centre to nearest edge) lies in the frame and
        # holds at least 1 - exp(-d^2 / 2 sigma_max^2) of the mass: when that
        # bound already passes both checks, skip the integral.
        d = min(mu_x, size - mu_x, mu_y, size - mu_y)
        if d > 0:
            p_lb = -math.expm1(-0.5 * (d / max(sigma_x, sigma_y)) ** 2)
            if (p_lb >= _FLOATER_MIN_ACCEPTANCE
                    and _gaussian_sampling_cost(p_lb, n) <= _FLOATER_DRAW_BUDGET):
                return
        p = _gaussian_frame_probability(*params, size)
    if not p >= limit:
        raise ValueError(
            "apply_floaters: no point would land in the frame (a draw lands "
            f"there with probability {p:.3g}); check center, spread and angle")
    cost = _gaussian_sampling_cost(p, n)
    if cost > _FLOATER_DRAW_BUDGET:
        warnings.warn(
            f"apply_floaters: a draw lands in the frame with probability "
            f"{p:.3g}, so {n} floaters need about {n / p:.3g} "
            f"draws (roughly {cost / _FLOATER_DRAWS_PER_SEC:,.0f} s); lower "
            "n_floaters, move center into the frame or widen spread",
            RuntimeWarning, stacklevel=3)


def _check_per_draw_gaussian(vals, size, limit) -> None:
    """_check_gaussian_reachable for a centre or angle given per draw (an
    array of other than one element).

    The sampler broadcasts such an array against its (2 * remaining,)
    draws, so draw i takes element i.  It is left to the sampler as before
    (which raises where the shapes do not broadcast) unless even the best
    element's draws practically never land: p is at most the chance that x,
    and that y, lands in [0, size), and the best such bound must reach
    limit.  No run time is estimated here.
    """
    try:
        mu_x, mu_y, sigma_x, sigma_y, cos_a, sin_a = np.broadcast_arrays(
            *(np.asarray(v, dtype=np.float64) for v in vals))
    except (TypeError, ValueError):
        return
    with np.errstate(all="ignore"):
        sd_x = np.hypot(sigma_x * cos_a, sigma_y * sin_a)
        sd_y = np.hypot(sigma_x * sin_a, sigma_y * cos_a)
        p_x = _normal_cdf((size - mu_x) / sd_x) - _normal_cdf(-mu_x / sd_x)
        p_y = _normal_cdf((size - mu_y) / sd_y) - _normal_cdf(-mu_y / sd_y)
        p = np.nanmax(np.minimum(p_x, p_y), initial=0.0)
    if not p >= limit:
        raise ValueError(
            "apply_floaters: no point would land in the frame (a draw lands "
            f"there with probability at most {p:.3g}); check center, spread "
            "and angle")


def _check_disc_reachable(r_o) -> None:
    """Raise when the disc sampler's radius (circle_radius_uv * overlay
    size) is NaN or -inf: its acceptance is then NaN everywhere.  Any other
    radius accepts at least 1% of draws."""
    if np.isnan(r_o) or r_o == -np.inf:
        raise ValueError(
            "apply_floaters: no point would land in the frame "
            f"(circle_radius_uv * 1024 = {r_o!r}); check circle_radius_uv")


def apply_floaters(img: Image.Image, severity: float = 0.5,
                     n_floaters: int = None, mode: str = "dark",
                     floater_size: int = 1, circle_radius_uv: float = 0.146,
                     seed: int = 0, distribution: str = "disc",
                     center=(0.5, 0.5), spread: float = 0.16,
                     spread_x: float = None, spread_y: float = None,
                     size_variance: bool = False,
                     dot_brightness: float = None,
                     angle: float = 0.0,
                     region=None,
                     region_feather: float = 0.02,
                     region_center_bias: float = 0.0) -> Image.Image:
    """Vitreous floaters: many tiny grain dots clustered over the image.

    Source: `myFloaters.cs` (CC BY 4.0).

    Two spatial distributions:
      - "disc" (default, faithful Unity port): uniform sampling accepted via
        IsInsideCircle's 3 zones — dense disc + soft falloff + 1% noise
        EVERYWHERE (so some specks land full-frame).
      - "gaussian": a 2-D Gaussian-weighted spray — highest density at `center`,
        decaying with `spread` (std as a fraction of the overlay), and NO
        uniform background noise, so the periphery thins out. Reads as a
        concentrated central BURST rather than full-frame grain.

    Args:
        severity: 0..1. Scales number of floaters and per-pixel alpha.
        n_floaters: explicit count override. Default = severity * 50000.
        mode: "dark" (black spots) or "light" (white spots).
        floater_size: base speck size in overlay (1024²) px (Unity=1).
        circle_radius_uv: disc radius as fraction of overlay ("disc" mode).
        distribution: "disc" or "gaussian".
        center: (cy, cx) cluster centre in normalised [0,1] ("gaussian" mode).
        spread: Gaussian std as a fraction of the overlay ("gaussian" mode).
        spread_x, spread_y: override the horizontal / vertical std separately
            (fraction of overlay). E.g. spread_x < spread_y gives a tall
            vertical PLUME rather than a round/horizontal cluster.
        size_variance: if True, mix larger blobs among the fine specks.
        dot_brightness: grey level of the dots, 0..1. Default None keeps
            the Unity behaviour (pure white in 'light' mode, pure black in
            'dark' mode); e.g. 0.85 gives softer off-white specks.
        angle: rotation of the gaussian ellipse in degrees (0 = spread_x
            horizontal / spread_y vertical; positive tilts clockwise).
            Only affects 'gaussian' distribution.
        region: optional polygon [[y, x], ...] in normalised coords. When
            given it OVERRIDES `distribution`: dots are scattered uniformly
            inside the polygon (a hand-drawn cluster shape). This is a
            visual-field shape (like center/spread), so it transfers to
            arbitrary images.
        region_feather: soft edge of the region mask (fraction of overlay;
            0 = hard boundary).
        region_center_bias: 0 = dots spread evenly across the region
            (uniform); toward 1 = dots concentrate around the region's
            middle, thinning out toward its outline (dense-core look).
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
    target = n_floaters

    region_mask = None
    if region is not None and len(region) >= 3:
        from PIL import ImageDraw, ImageFilter
        m = Image.new("L", (int(overlay_size), int(overlay_size)), 0)
        poly = [(float(x) * overlay_size, float(y) * overlay_size)
                for y, x in region]
        ImageDraw.Draw(m).polygon(poly, fill=255)
        if region_feather:
            _check_blur_radius(region_feather * overlay_size)
            m = m.filter(ImageFilter.GaussianBlur(region_feather * overlay_size))
        region_mask = np.asarray(m, np.float32) / 255.0
        if region_mask.max() <= 0:
            region_mask = None  # degenerate polygon: fall back
        elif region_center_bias > 0:
            ys_m, xs_m = np.nonzero(region_mask > 0.05)
            cy_m, cx_m = ys_m.mean(), xs_m.mean()
            gy, gx = np.mgrid[0:region_mask.shape[0], 0:region_mask.shape[1]]
            dist_c = np.hypot(gy - cy_m, gx - cx_m)
            dmax = max(dist_c[region_mask > 0.05].max(), 1.0)
            falloff = np.clip(1.0 - dist_c / dmax, 0.0, 1.0) \
                ** (1.0 + 5.0 * float(np.clip(region_center_bias, 0, 1)))
            region_mask = region_mask * falloff
            region_mask /= max(region_mask.max(), 1e-6)

    if region_mask is not None:
        pts_x, pts_y = [], []
        ov = int(overlay_size)
        for _ in range(200):
            if len(pts_x) >= target:
                break
            n = max((target - len(pts_x)) * 3, 1024)
            bx = rng.uniform(0.0, overlay_size, n)
            by = rng.uniform(0.0, overlay_size, n)
            pa = region_mask[np.clip(by.astype(int), 0, ov - 1),
                             np.clip(bx.astype(int), 0, ov - 1)]
            keep = rng.random(n) < pa
            pts_x.extend(bx[keep].tolist())
            pts_y.extend(by[keep].tolist())
        pts_x = np.asarray(pts_x[:target], dtype=np.float64)
        pts_y = np.asarray(pts_y[:target], dtype=np.float64)
        target = len(pts_x)
    elif distribution == "uniform":
        # Even scatter across the WHOLE frame (denser toward the periphery
        # relative to a centre-gaussian) — stray floaters everywhere.
        pts_x = rng.uniform(0.0, overlay_size, size=target)
        pts_y = rng.uniform(0.0, overlay_size, size=target)
    elif distribution == "gaussian":
        # 2-D Gaussian spray centred at `center`, std = `spread` of overlay.
        # Reject out-of-bounds samples (keeps the periphery thin, no edge pile).
        cyN, cxN = center
        mu_x = cxN * overlay_size
        mu_y = cyN * overlay_size
        sigma_x = max(1.0, (spread_x if spread_x is not None else spread) * overlay_size)
        sigma_y = max(1.0, (spread_y if spread_y is not None else spread) * overlay_size)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        _check_gaussian_reachable(mu_x, mu_y, sigma_x, sigma_y, cos_a, sin_a,
                                  overlay_size, target)
        pts_x, pts_y = [], []
        while len(pts_x) < target:
            n = (target - len(pts_x)) * 2
            dx = rng.normal(0.0, sigma_x, n)
            dy = rng.normal(0.0, sigma_y, n)
            bx = mu_x + dx * cos_a - dy * sin_a
            by = mu_y + dx * sin_a + dy * cos_a
            ok = (bx >= 0) & (bx < overlay_size) & (by >= 0) & (by < overlay_size)
            pts_x.extend(bx[ok].tolist())
            pts_y.extend(by[ok].tolist())
        pts_x = np.asarray(pts_x[:target], dtype=np.float64)
        pts_y = np.asarray(pts_y[:target], dtype=np.float64)
    else:
        # Faithful disc rejection sampling (3-zone IsInsideCircle).
        cx_o = cy_o = overlay_size * 0.5
        r_o = circle_radius_uv * overlay_size
        _check_disc_reachable(r_o)
        pts_x, pts_y = [], []
        while len(pts_x) < target:
            batch = max(target * 2, 4096)
            cxs = rng.uniform(0.0, overlay_size, size=batch)
            cys = rng.uniform(0.0, overlay_size, size=batch)
            d = np.hypot(cxs - cx_o, cys - cy_o)
            falloff = np.clip(1.0 - (d - r_o) / r_o, 0.0, 1.0)
            accept_p = np.where(d <= r_o, 1.0, 0.01 + 0.99 * falloff)
            keep = rng.random(batch) < accept_p
            pts_x.extend(cxs[keep].tolist())
            pts_y.extend(cys[keep].tolist())
        pts_x = np.asarray(pts_x[:target], dtype=np.float64)
        pts_y = np.asarray(pts_y[:target], dtype=np.float64)

    # Map overlay coords → screen pixel coords.
    xs = np.clip((pts_x * (w / overlay_size)).astype(int), 0, w - 1)
    ys = np.clip((pts_y * (h / overlay_size)).astype(int), 0, h - 1)

    # Per-floater size: constant, or a weighted mix (mostly fine, some blobs).
    if size_variance:
        pool = np.array([1, 1, 1, 1, 2, 2, 3, floater_size + 2, floater_size + 4])
        sizes = rng.choice(pool, size=target)
    else:
        sizes = np.full(target, floater_size, dtype=int)

    out = arr.copy()
    if dot_brightness is not None:
        target_val = float(np.clip(dot_brightness, 0.0, 1.0)) * 255.0
    else:
        target_val = 0.0 if mode == "dark" else 255.0
    alpha = severity

    # Fast path: single-pixel specks (size <= 1) painted vectorised.
    small = sizes <= 1
    if np.any(small):
        smx, smy = xs[small], ys[small]
        for c in range(3):
            out[smy, smx, c] = arr[smy, smx, c] * (1.0 - alpha) + target_val * alpha

    # Larger blobs painted with a random sub-pixel mask each.
    _paint_blobs(out, xs[~small], ys[~small], sizes[~small], rng,
                 alpha, target_val, overlay_size)
    return Image.fromarray(_to_uint8(out, 1.0))


# =============================================================================
# Vignette — peripheral darkening (tunnel-vision / peripheral field loss)
# =============================================================================
# Not a distinct VIP-Sim shader, but the "Vision loss, peripheral" percept
# (glaucoma / retinitis-pigmentosa tunnel vision) darkens the periphery in
# addition to the mip-blur that apply_field_loss(peripheral) already provides.
# This multiplies a radial darkening mask: ~1 in the centre, falling to
# (1 - strength) toward the corners via a smoothstep beyond `radius`.

def apply_vignette(img: Image.Image, severity: float = 0.5,
                    strength: float = None, radius: float = 0.5,
                    softness: float = 0.5, mode: str = "dark") -> Image.Image:
    """Radial peripheral shading (vignette).

    `mode="dark"` darkens the periphery (tunnel-vision, e.g. glaucoma).
    `mode="light"` brightens/blooms the periphery toward white (a bright
    veiling fog that lightens the edges — e.g. some RP / hazy peripheral loss).

    Args:
        img: Input PIL Image (RGB).
        severity: 0..1 — if `strength` is None, maps directly to how strong the
            corner shading is (0 = none, 1 = corners fully black / white).
        strength: Override for the corner shading amount (0..1).
        radius: Normalised distance (0=centre, 1=corner) where shading starts.
        softness: Width of the smoothstep falloff from `radius` outward.
        mode: "dark" (multiply toward black) or "light" (blend toward white).
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    if strength is None:
        strength = severity
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return img.copy()

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1) - 0.5
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1) - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    # Normalise so the corner distance is 1.0 (0.707 = sqrt(0.5^2+0.5^2)).
    dist = np.sqrt(xx * xx + yy * yy) / 0.7071
    t = np.clip((dist - radius) / max(softness, 1e-6), 0.0, 1.0)
    s = (t * t * (3.0 - 2.0 * t)) * strength  # smoothstep * strength

    if mode == "light":
        # Blend toward white at the periphery (bright veiling fog).
        out = arr * (1.0 - s[..., None]) + 1.0 * s[..., None]
    elif mode == "dark":
        out = arr * (1.0 - s[..., None])
    else:
        raise ValueError(f"Unknown vignette mode: {mode}")
    return Image.fromarray(_to_uint8(out))


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
    "vignette": apply_vignette,
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
    # Optional at parse time so --list-presets works without paths; enforced below.
    parser.add_argument("input", nargs="?", help="Input image path")
    parser.add_argument("output", nargs="?", help="Output image path")

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
                        help="Displacement axis (double-vision: horizontal/vertical/diagonal; "
                             "nystagmus: horizontal/vertical/both)")
    parser.add_argument("--floater-mode", dest="floater_mode", default="dark",
                        choices=["dark", "light"],
                        help="Floater appearance mode")

    args = parser.parse_args()

    if args.list_presets:
        for name, preset in PRESETS.items():
            print(f"  {name}: {preset['description']}")
        return
    if args.input is None or args.output is None:
        parser.error("the following arguments are required: input, output")
    if args.filter == "double_vision" and args.axis == "both":
        parser.error("double_vision supports --axis horizontal, vertical or diagonal")

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
