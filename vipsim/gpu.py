#!/usr/bin/env python3
"""
vipsim.gpu — GPU (PyTorch) port of the VIP-Sim visual-impairment filters.

Every filter of vipsim/filters.py (all 21 in _FILTER_DISPATCH) is a VipSimGPU
method with the CPU name, keyword arguments, defaults and positional order:
`sim.cvd(x, cvd_type="protanomaly", severity=0.6)` is
`apply_cvd(img, cvd_type="protanomaly", severity=0.6)` on a batch.  Presets
and vipsim_profiles_v2 profiles run with VipSimGPU.apply_preset /
apply_profile.

Design goals
------------
* Operate on BATCHED, channels-first tensors  x: [N, 3, H, W], float32 in [0,1],
  living on the GPU.  N = number of parallel envs (or frame stack).
* The NumPy/PIL filters in vipsim/filters.py are the REFERENCE.  A GPU filter
  reproduces the CPU filter as it is today, quirks included (nearest-index
  truncation, 8-bit intermediates, PIL's integer arithmetic); CPU bugs are
  fixed in filters.py first, never only here.
* Every twin is bit-exact with its CPU filter where tested (tests/test_gpu_*.py
  assert equality; TWIN_TOLERANCE lists the documented bound).  The CPU only
  ever sees a uint8 image, so each twin reads x as its 8-bit levels and
  returns level/255 in x's dtype: a chain of GPU filters sees the CPU chain's
  images.
* No GLSL/CUDA.  Every original filter reduces to a few primitives:
      1. per-level tables / 3-channel mixes -> the CPU's numpy code on the
         256 levels, looked up on the device; _channel_dot  (cvd, bcg, ...)
      2. separable blur / resize    -> pil_gaussian_blur, pil_resize
      3. per-pixel UV displacement  -> gather_nearest, gather_bilinear
* Anything that does NOT depend on the image (warp fields, index maps, masks,
  noise fields, lookup tables) is computed ONCE with the CPU's own numpy code
  and kept on the device in a bounded LRU cache (VipSimGPU.cached /
  VipSimGPU.numpy_field).

Shared primitives (the API the filter ports build on)
-----------------------------------------------------
  clamp_severity(s)                      float(np.clip(s, 0, 1)), as every apply_* does
  quantize_u8(x, scale=255.0)            filters._to_uint8, result back in [0,1]
  exact_div(x, d)                        numpy's x / d (CUDA's x / float is x * (1/d))
  pil_box_radius(radius, passes=3)       Pillow's _gaussian_blur_radius (float32)
  pil_gaussian_blur(x, radius, quantize) img.filter(ImageFilter.GaussianBlur(radius))
  gather_nearest(x, iy, ix)              arr[sy, sx] with integer index maps
  gather_bilinear(x, fy, fx)             the 4-tap blend of apply_vortex
  skimage_rgb2hsv(x), skimage_hsv2rgb(x) skimage.color's HSV round trip (float32)
  VipSimGPU.pil_resize(x, width, height, resample, quantize)
                                         img.resize((width, height), BILINEAR/NEAREST),
                                         in the installed Pillow's pass order
  VipSimGPU.cached(name, params, hw, build)
  VipSimGPU.numpy_field(name, params, hw, fn, dtype)
  VipSimGPU.to_device(array, dtype)

Blurs and resizes deliberately avoid conv2d: cuDNN runs float32 convolutions
in TF32 by default on Ampere+ GPUs (torch.backends.cudnn.allow_tf32), which
keeps ~3 decimal digits, and float32 cannot hold PIL's 24-bit fixed-point
products anyway.  They replay PIL's integer arithmetic with index gathers and
cumulative sums in int32/int64 instead, which is exact.  Colour maths
(sRGB <-> linear, pow) never runs on the device: torch's pow is ~1 ulp off
numpy's, so the CPU's own functions are evaluated on the 256 levels and the
device looks the result up.  The porting notes at the bottom list which
primitive each filter uses.
"""

from __future__ import annotations
import functools
import inspect
import math
from collections import OrderedDict

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from .filters import (CVD_MATRICES, PRESETS, _FILTER_DISPATCH,
                      _box_muller, _linear_to_srgb,
                      _load_macular_overlay, _perlin_noise_2d,
                      _radial_weight_mask, _srgb_to_linear, _to_uint8,
                      apply_cataracts)


# -----------------------------------------------------------------------------
# Severity and 8-bit quantisation
# -----------------------------------------------------------------------------
# np.asarray(img, np.float32) / 255.0 for every 8-bit level.
_LEVELS = np.arange(256, dtype=np.float32) / 255.0


def clamp_severity(severity) -> float:
    """float(np.clip(severity, 0, 1)) — the first line of every CPU apply_*."""
    if type(severity) is float and 0.0 < severity < 1.0:
        return severity              # what np.clip returns, without its ~10 us
    return float(np.clip(severity, 0.0, 1.0))


def exact_div(x: torch.Tensor, divisor: float) -> torch.Tensor:
    """x / divisor, correctly rounded like numpy's division.

    On CUDA, `tensor / python_float` is computed as tensor * (1/divisor),
    which is 1 ulp off for about half of k/255; a device-tensor divisor
    takes the true-division kernel.
    """
    return x / _device_scalar(divisor, math.copysign(1.0, divisor),
                              x.dtype, x.device)


@functools.lru_cache(maxsize=64)
def _device_scalar(value, sign, dtype, device) -> torch.Tensor:
    """torch.tensor(value) on the device, built once: each build copies from
    host memory, which waits for all queued GPU work.  `sign` keeps 0.0 and
    -0.0, which compare equal, apart."""
    return torch.tensor(value, dtype=dtype, device=device)


def _u8(x: torch.Tensor, scale: float = 255.0) -> torch.Tensor:
    """Integer 8-bit levels (int64) of x, rounded like filters._to_uint8."""
    return (x * scale).round().clamp(0, 255).long()


def _from_u8(levels: torch.Tensor, dtype) -> torch.Tensor:
    """8-bit levels -> levels/255, the same floats as np.float32(levels)/255."""
    return exact_div(levels.to(dtype), 255.0)


def quantize_u8(x: torch.Tensor, scale: float = 255.0) -> torch.Tensor:
    """Twin of filters._to_uint8, returned as levels/255 in x's dtype.

    np.rint and torch.round both round half to even.  `scale` maps the input
    range to 0..255 exactly as in _to_uint8 (scale=1.0 for 0..255 data).
    Use it wherever the CPU filter round-trips through a uint8 image (every
    filter output, bloom's intermediates, field-loss mip levels).
    """
    return _from_u8(_u8(x, scale), x.dtype)


# -----------------------------------------------------------------------------
# Pillow GaussianBlur twin (src/libImaging/BoxBlur.c)
# -----------------------------------------------------------------------------
# ImageFilter.GaussianBlur(radius) is not a sampled Gaussian: Pillow runs
# `passes`=3 "extended box" blurs per axis (all horizontal passes, then all
# vertical ones) with replicated edges.  Each pass computes
#     out[x] = (ww * sum(in[x-r .. x+r]) + fw * (in[x-r-1] + in[x+r+1])
#               + 2**23) >> 24
# in 24-bit fixed point and stores uint8, so every pass rounds to 8 bits.
# r, ww and fw come from the float32 box radius below.  Pillow only blurs
# 8-bit modes (an 'F' image raises "image has wrong mode"), so the CPU
# _separable_gaussian also blurs a uint8 image.

def check_pil_blur_radius(radius: float) -> None:
    """Raise ValueError where Pillow 13's GaussianBlur does: a NaN or
    infinite radius ("radius must be >= 0"), or one of 2**31 - 64 px or more
    either way ("radius too large").  Pillow 12 segfaults on those radii."""
    r = float(radius)
    if not math.isfinite(r):
        raise ValueError("radius must be >= 0")
    if abs(r) >= 2147483584.0:
        raise ValueError("radius too large")


def pil_box_radius(radius: float, passes: int = 3) -> float:
    """Pillow's _gaussian_blur_radius, evaluated in float32 like the C code.

    The box radius whose `passes`-fold repetition has the variance of a
    Gaussian with sigma=`radius` (Gwosdek et al., SSVM 2011).
    """
    f32 = np.float32
    sigma2 = f32(radius) * f32(radius) / f32(passes)
    L = f32(math.sqrt(12.0 * float(sigma2) + 1.0))
    l = f32(math.floor((float(L) - 1.0) / 2.0))
    a = (f32(2) * l + f32(1)) * (l * (l + f32(1)) - f32(3) * sigma2)
    a = a / (f32(6) * (sigma2 - (l + f32(1)) * (l + f32(1))))
    return float(l + a)


def _box_blur_weights(box_radius: float) -> tuple:
    """(r, ww, fw) of ImagingHorizontalBoxBlur for a float box radius."""
    f32 = np.float32
    r = int(box_radius)
    ww = int(f32(1 << 24) / (f32(box_radius) * f32(2) + f32(1)))
    fw = ((1 << 24) - (2 * r + 1) * ww) // 2
    return r, ww, fw


def _box_blur_index(n: int, r: int, device) -> torch.Tensor:
    """clamp(k - r - 1, 0, n - 1) for k = 0 .. n + 2r + 1: the replicated-edge
    source of every padded position, built once per (n, r, device) (at most
    64 x 512 KiB kept; longer ones, from huge radii, are built per call)."""
    if n + 2 * r + 2 > 1 << 16:
        return torch.arange(-r - 1, n + r + 1, device=device).clamp_(0, n - 1)
    return _box_blur_index_cached(n, r, device)


@functools.lru_cache(maxsize=64)
def _box_blur_index_cached(n: int, r: int, device) -> torch.Tensor:
    return torch.arange(-r - 1, n + r + 1, device=device).clamp_(0, n - 1)


# Up to this many elements a pass keeps its levels in int64 (fewest kernels:
# small batches are launch-bound); larger ones sum in int32 and convert back,
# which moves half the bytes (measured on an RTX 4050: int64 throughout was
# 10-25% slower from ~2M elements, where int32 intermediates still fit in L2).
_BOX_BLUR_INT64_NUMEL = 1 << 20


def _box_blur_pass(t: torch.Tensor, r: int, ww: int, fw: int,
                   dim: int) -> torch.Tensor:
    """One extended-box pass along `dim` (replicated edges).

    Integer levels: Pillow's fixed-point pass, bit-exact.  Window sums in
    int64, or in int32 for large tensors (wrapping like Pillow's UINT32
    accumulator); either is read unsigned only where a window of 255s can
    overflow it.  The 24-bit products are int64.  8 kernels per pass (6 for
    r = 0), 10 (8) for large tensors.  float64: the same weights without
    the 8-bit rounding.  Window sums come from a cumulative sum, so the cost
    does not grow with the radius.
    """
    n = t.shape[dim]
    p = t.index_select(dim, _box_blur_index(n, r, t.device))  # p[k] = in[clamp(k - r - 1)]
    edges = p.narrow(dim, 0, n) + p.narrow(dim, 2 * r + 2, n)  # in[x-r-1], in[x+r+1]
    if t.is_floating_point():
        cs = p.cumsum(dim)
        acc = cs.narrow(dim, 2 * r + 1, n) - cs.narrow(dim, 0, n)   # in[x-r..x+r]
        return (acc * ww + edges * fw) / (1 << 24)
    wide = t.numel() <= _BOX_BLUR_INT64_NUMEL
    if r == 0:
        # Sub-pixel box (radius < ~1 px): the window is the pixel itself.
        bulk = p.narrow(dim, 1, n).long()
    else:
        cs = p.cumsum(dim) if wide else p.cumsum(dim, dtype=p.dtype)
        bulk = cs.narrow(dim, 2 * r + 1, n) - cs.narrow(dim, 0, n)    # in[x-r..x+r]
        limit = 0x7FFFFFFF if cs.dtype == torch.int32 else 0xFFFFFFFF
        bulk = bulk.long()
        if 255 * (2 * r + 1) > limit:    # the window sum can wrap
            bulk.bitwise_and_(0xFFFFFFFF)
    bulk = bulk.mul_(ww).add_(edges, alpha=fw).add_(1 << 23).bitwise_right_shift_(24)
    return bulk if wide else bulk.int()


def pil_gaussian_blur(x: torch.Tensor, radius: float, quantize: bool = True,
                      passes: int = 3) -> torch.Tensor:
    """Twin of PIL `img.filter(ImageFilter.GaussianBlur(radius))`.

    x: [..., H, W] float in [0,1].  quantize=True treats x as an 8-bit image
    (rounded to k/255 first, as the CPU always hands PIL a uint8 image) and
    replays Pillow's integer arithmetic: bit-exact with PIL 12 and 13 (max
    difference 0, tests/test_gpu_primitives.py).  quantize=False runs the
    same box cascade in float64 without the per-pass rounding, for float data
    that never goes through a uint8 image; against PIL on 8-bit input it is
    within 1 LSB from radius 2.25 px, 2 LSB at 1.2-2 px and up to 3 LSB on
    noise at sub-pixel radii, where PIL's six roundings outweigh the blur.
    """
    if quantize:
        return _from_u8(_blur_levels(_u8(x), radius, passes), x.dtype)
    check_pil_blur_radius(radius)
    if radius == 0:                    # PIL returns a copy for radius 0
        return x
    r, ww, fw = _box_blur_weights(pil_box_radius(radius, passes))
    t = x.double()
    for dim in (-1, -2):                             # all horizontal passes first
        for _ in range(passes):
            t = _box_blur_pass(t, r, ww, fw, dim)
    return t.to(x.dtype)


def _blur_levels(levels: torch.Tensor, radius: float, passes: int = 3) -> torch.Tensor:
    """pil_gaussian_blur on integer 8-bit levels, returning integer levels.

    Filters that blur an image they already hold as levels (bloom, the
    field_loss pyramid) call this directly: _u8(_from_u8(t)) is t for levels
    0..255, so skipping that round trip changes no value.
    """
    check_pil_blur_radius(radius)
    if radius == 0:                    # PIL returns a copy for radius 0
        return levels
    r, ww, fw = _box_blur_weights(pil_box_radius(radius, passes))
    t = levels.int()
    for dim in (-1, -2):                             # all horizontal passes first
        for _ in range(passes):
            t = _box_blur_pass(t, r, ww, fw, dim)
    return t


# -----------------------------------------------------------------------------
# PIL resize twin (src/libImaging/Resample.c, Geometry.c)
# -----------------------------------------------------------------------------
# The tap tables only depend on (in_size, out_size), so VipSimGPU.pil_resize
# builds them once with these numpy functions and caches them on the device.
_PIL_VERSION = tuple(int(v) for v in PIL.__version__.split(".")[:2])


def _pil_vertical_first(in_w: int, in_h: int, out_w: int, out_h: int) -> bool:
    """Whether Pillow resamples (in_w, in_h) -> (out_w, out_h) vertically first.

    Each pass rounds to 8 bits, so the order shows.  Pillow 13 goes vertical
    first when the height shrinks by more than twice as many pixels as the
    width (ImagingResampleInner's horizontal_first).  Pillow 12.2 does it in
    Image.resize instead, for images more than 100 times taller than wide
    whose height shrinks: a vertical-only resize, then a horizontal-only one.
    Earlier versions always go horizontal first.
    """
    if _PIL_VERSION >= (13, 0):
        return in_h - out_h > 0 and in_h - out_h > (in_w - out_w) * 2
    if _PIL_VERSION >= (12, 2):
        return in_h > in_w * 100 and out_h < in_h
    return False


def _pil_bilinear_taps(in_size: int, out_size: int) -> tuple:
    """Pillow's precompute_coeffs + normalize_coeffs_8bpc for BILINEAR.

    Returns (idx, k), both int64 [out_size, ksize]: source indices and 22-bit
    fixed-point weights.  When downscaling, the triangle filter is stretched
    by the scale factor (Pillow's antialiasing); padding taps get weight 0.
    """
    scale = float(in_size) / out_size
    filterscale = max(scale, 1.0)
    support = 1.0 * filterscale                      # bilinear support is 1
    ksize = int(math.ceil(support)) * 2 + 1
    idx = np.zeros((out_size, ksize), np.int64)
    kk = np.zeros((out_size, ksize), np.float64)
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        ss = 1.0 / filterscale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        w = [max(0.0, 1.0 - abs((x + xmin - center + 0.5) * ss))
             for x in range(xmax)]
        ww = 0.0
        for wi in w:                 # sequential, like C (builtin sum() is not)
            ww += wi
        idx[xx, :] = xmin
        idx[xx, :xmax] = np.arange(xmin, xmin + xmax)
        kk[xx, :xmax] = [wi / ww for wi in w] if ww != 0.0 else w
    k = np.trunc(kk * (1 << 22) + np.where(kk < 0, -0.5, 0.5)).astype(np.int64)
    return np.minimum(idx, in_size - 1), k


def _pil_nearest_index(in_size: int, out_size: int) -> np.ndarray:
    """Source index of each output pixel in Pillow's NEAREST resize.

    ImagingScaleAffine accumulates the position (xo += scale) instead of
    multiplying, so the table is built the same way.
    """
    scale = float(in_size) / out_size
    idx = np.empty(out_size, np.int64)
    xo = scale * 0.5
    for x in range(out_size):
        idx[x] = int(xo)
        xo += scale
    return np.minimum(idx, in_size - 1)


# Up to this many gathered taps ([..., out, ksize]) a resample pass sums one
# int64 gather (fewest kernels); beyond it, it adds one tap at a time in
# int32, which never holds more than two output-sized arrays.  Pillow's
# accumulator is int32 too, and a BILINEAR sum (non-negative 22-bit weights
# summing to ~2**22, times 255) stays below 2**31, so both are exact.
_RESAMPLE_GATHER_NUMEL = 1 << 22


def _resample_pass(t: torch.Tensor, idx: torch.Tensor,
                   k: torch.Tensor) -> torch.Tensor:
    """One fixed-point resample pass along the last dim (integer levels)."""
    if t.numel() // t.shape[-1] * idx.numel() <= _RESAMPLE_GATHER_NUMEL:
        v = (t[..., idx] * k).sum(-1) + (1 << 21)
        return (v >> 22).clamp_(0, 255)
    # Contiguous along `out`: the vertical pass arrives transposed, and ksize
    # strided gathers would cost more than this one copy.
    t32 = t.to(torch.int32, memory_format=torch.contiguous_format)
    k32 = k.int()
    v = torch.full((*t.shape[:-1], idx.shape[0]), 1 << 21, dtype=torch.int32,
                   device=t.device)
    for j in range(idx.shape[1]):
        v.add_(t32[..., idx[:, j]] * k32[:, j])
    return (v >> 22).clamp_(0, 255).to(t.dtype)


# -----------------------------------------------------------------------------
# Pixel-coordinate gathers (twins of numpy fancy indexing)
# -----------------------------------------------------------------------------
def gather_nearest(x: torch.Tensor, iy: torch.Tensor,
                   ix: torch.Tensor) -> torch.Tensor:
    """out[n, c, i, j] = x[n, c, iy[i, j], ix[i, j]]  — numpy's arr[sy, sx].

    iy, ix: integer tensors that broadcast to [Ho, Wo] (one map shared by the
    batch; e.g. sy[:, None] and sx[None, :] as in apply_led) or to
    [N, Ho, Wo] (one map per sample).  Indices are clamped to the edge; numpy
    would wrap negative indices instead, but the CPU filters only pass
    in-range indices (_check_numpy_index raises where numpy would).  Pure
    copies, so the result is exact.
    """
    H, W = x.shape[-2:]
    return _gather_flat(x, _flat_index(iy, ix, H, W))


def _flat_index(iy: torch.Tensor, ix: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """The edge-clamped flat source index gather_nearest reads, [Ho, Wo] or
    [N, Ho, Wo].  It depends only on the maps, so filters whose maps are
    cached keep this instead of rebuilding it on every call."""
    return iy.long().clamp(0, H - 1) * W + ix.long().clamp(0, W - 1)


def _gather_flat(x: torch.Tensor, lin: torch.Tensor) -> torch.Tensor:
    """gather_nearest with its flat index (_flat_index) already built."""
    N, C, _, _ = x.shape
    flat = x.flatten(2)                              # [N, C, H*W]
    if lin.dim() == 2:
        return flat.index_select(2, lin.reshape(-1)).view(N, C, *lin.shape)
    lin = lin.expand(N, -1, -1)
    out = flat.gather(2, lin.reshape(N, 1, -1).expand(-1, C, -1))
    return out.view(N, C, *lin.shape[1:])


def _bilinear_plan(fy: torch.Tensor, fx: torch.Tensor, H: int, W: int) -> tuple:
    """Everything of gather_bilinear that depends only on the coordinates:
    the four corners' flat indices and the weights wx, 1 - wx, wy, 1 - wy,
    computed by the same ops, so cached coordinates need no per-call work."""
    fx = fx.clamp(0, W - 1)
    fy = fy.clamp(0, H - 1)
    x0 = fx.floor()
    y0 = fy.floor()
    wx = fx - x0
    wy = fy - y0
    x0 = x0.long()
    y0 = y0.long()
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    if wx.dim() == 3:                                # per-sample maps
        wx, wy = wx.unsqueeze(1), wy.unsqueeze(1)
    return (_flat_index(y0, x0, H, W), _flat_index(y0, x1, H, W),
            _flat_index(y1, x0, H, W), _flat_index(y1, x1, H, W),
            wx, 1.0 - wx, wy, 1.0 - wy)


def _gather_bilinear_plan(x: torch.Tensor, plan: tuple) -> torch.Tensor:
    """gather_bilinear with its _bilinear_plan already built."""
    l00, l01, l10, l11, wx, omwx, wy, omwy = plan
    top = _gather_flat(x, l00) * omwx + _gather_flat(x, l01) * wx
    bot = _gather_flat(x, l10) * omwx + _gather_flat(x, l11) * wx
    return top * omwy + bot * wy


def gather_bilinear(x: torch.Tensor, fy: torch.Tensor,
                    fx: torch.Tensor) -> torch.Tensor:
    """Bilinear sample at float pixel coordinates, as apply_vortex does it:

        x0 = floor(fx); x1 = min(x0 + 1, W - 1); wx = fx - x0   (same for y)
        top = a[y0,x0]*(1-wx) + a[y0,x1]*wx;  bot = a[y1,x0]*(1-wx) + a[y1,x1]*wx
        out = top*(1-wy) + bot*wy

    fy, fx: float tensors broadcasting to [Ho, Wo] or [N, Ho, Wo], in pixel
    units (0 .. H-1 / 0 .. W-1); out-of-range coordinates clamp to the edge.
    The blend runs in torch's promoted dtype of x and the coordinates, which
    mirrors numpy's promotion: in apply_vortex `fx - x0` is float32 minus
    int32, i.e. float64, so pass float64 coordinates to match it bit for bit.
    Same op order as the CPU, so equal-dtype inputs give identical results.
    """
    H, W = x.shape[-2:]
    return _gather_bilinear_plan(x, _bilinear_plan(fy, fx, H, W))


def _check_numpy_index(shape: tuple, *index: np.ndarray) -> None:
    """Raise the IndexError numpy raises for arr[index] on an array of this
    shape, if it raises one.

    The CPU's index maps come from astype(np.int32), which turns NaN (a NaN
    severity, timer or centre; freq=inf through 0 * inf) into -2**31: numpy
    rejects that index, while gather_nearest would clamp it to the edge and
    return an image.  Indexing a broadcast view raises numpy's own message
    (same index, axis and size as for the image).
    """
    if any(i.size and (i.min() < 0 or i.max() >= n) for i, n in zip(index, shape)):
        np.broadcast_to(np.empty((), np.bool_), shape)[index]


# -----------------------------------------------------------------------------
# Bounded cache for image-independent tensors
# -----------------------------------------------------------------------------
def _freeze(params):
    """An exact, hashable cache key for filter parameters, or None.

    Parameters are keyed exactly, never rounded (two severities that differ
    in the 4th decimal must not share a warp field), and by type as well as
    value, because equal values can give different CPU fields: under NEP 50
    a float32 array times np.float64(0.5) is float64 while times 0.5 it stays
    float32, numpy rejects Decimal("0.5") although it hashes like 0.5, and a
    seed of 1.0 raises where 1 does not.  So Python floats are keyed by
    float.hex() (exact, and -0.0 apart from 0.0), numpy scalars and arrays by
    np.dtype (which tells structured layouts apart) and their bytes, and
    object arrays by their values (their bytes are PyObject pointers, which a
    freed number's successor can reuse).  Any other type (Decimal, Fraction,
    array.array, deque, subclasses, torch tensors) gives None: VipSimGPU.cached
    then builds on every call, so the CPU code returns or raises on it
    whatever was cached before.
    """
    kind = type(params)
    if kind is float:
        return (kind, params.hex())
    if kind in (bool, int, str, type(None)):
        return (kind, params)
    if kind is dict:
        keys = tuple((k, _freeze(v)) for k, v in sorted(params.items()))
        return None if any(v is None for _, v in keys) else (kind, keys)
    if kind in (list, tuple):
        keys = tuple(_freeze(v) for v in params)
        return None if any(k is None for k in keys) else (kind, keys)
    if kind is np.ndarray or isinstance(params, np.generic):
        if not params.dtype.hasobject:
            return (kind, params.dtype, np.shape(params), params.tobytes())
        if kind is np.ndarray:
            keys = _freeze(params.tolist())
            return None if keys is None else (kind, params.dtype, params.shape, keys)
    return None


class TensorCache:
    """LRU cache with at most `maxsize` entries; the oldest is evicted first.

    Severity sweeps and varying resolutions would otherwise grow a plain dict
    without bound (one H x W field per distinct severity).
    """

    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._entries: OrderedDict = OrderedDict()

    def get(self, key, build):
        """Return the entry for `key`, calling build() on a miss."""
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        value = build()
        self._entries[key] = value
        if len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
        return value

    def clear(self):
        self._entries.clear()

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key):
        return key in self._entries


# -----------------------------------------------------------------------------
# Filter twins: shared helpers
# -----------------------------------------------------------------------------
# The CPU input is an 8-bit image, so every step that maps one channel value
# to another is a 256-entry table, evaluated once with the CPU's own numpy
# code.  Only channel mixes, pixel moves and position-dependent weights run as
# torch arithmetic, in numpy's dtypes and summation order.
_NUMPY_FLOAT = {torch.float32: np.dtype(np.float32),
                torch.float64: np.dtype(np.float64)}


def _torch_dtype(dtype) -> torch.dtype:
    """The torch dtype of a numpy dtype (float32 -> torch.float32, ...)."""
    return getattr(torch, np.dtype(dtype).name)


def _channel_dot(x: torch.Tensor, m: torch.Tensor, sdot: bool = False) -> torch.Tensor:
    """out[:, i] = x[:, 0]*m[i, 0] + x[:, 1]*m[i, 1] + x[:, 2]*m[i, 2]

    — numpy's np.dot(arr, m.T) or arr @ m[0] on an [H, W, 3] image.
    x: [N, 3, H, W], m: [K, 3]  ->  [N, K, H, W], in their promoted dtype.
    The products are summed left to right without fused multiply-adds, as
    numpy's float64 np.dot (ddot) and float32 matmul (sgemv) do (measured on
    numpy 2.5 with OpenBLAS 0.3.34); einsum and matmul reorder the sum.
    sdot=True: float32 products summed in float64 and rounded once, which is
    what float32 `arr @ w` does on a 1-pixel-wide image (numpy calls
    cblas_sdot per pixel there instead of sgemv).
    """
    w = m.t()[:, None, :, None, None]               # w[j] = m[:, j] as [1, K, 1, 1]
    p0, p1, p2 = (x[:, j:j + 1] * w[j] for j in range(3))
    if sdot:
        return ((p0.double() + p1) + p2).to(p0.dtype)
    return (p0 + p1) + p2


def _srgb_u8_steps(dtype) -> np.ndarray:
    """For k = 1..255, the smallest linear value v of `dtype` that the CPU's
    _to_uint8(_linear_to_srgb(v)) maps to level >= k  ([255], ascending).

    The number of steps at or below v is then the CPU's level of v, without
    evaluating pow on the device (torch's float32 pow differs from numpy's in
    the last bit for a few percent of inputs).  Each step is bisected on the
    bit pattern, which sorts like the value for non-negative floats, with the
    CPU functions themselves.
    """
    itype = np.int64 if dtype == np.float64 else np.int32
    k = np.arange(1, 256)
    lo = np.zeros(255, itype)                             # 0.0 maps below k
    hi = np.full(255, np.array(1.0, dtype).view(itype))   # 1.0 maps to 255
    while np.any(hi - lo > 1):
        mid = lo + (hi - lo) // 2
        up = _to_uint8(_linear_to_srgb(mid.view(dtype))) >= k
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    return hi.view(dtype)


def _fixed_seed(seed) -> bool:
    """True when np.random.default_rng(seed) gives the same draws on every
    call (an int, a sequence of ints, a SeedSequence), so its field can be
    cached.  seed=None (fresh OS entropy) and a Generator or BitGenerator
    (whose state advances) give new draws on every CPU call.
    """
    return seed is not None and not isinstance(
        seed, (np.random.Generator, np.random.BitGenerator))


# -- cvd -----------------------------------------------------------------------
# BT.601 weights (float64): apply_cvd's monochrome gray, and in float32
# apply_foveal_darkness' desat luminance.  apply_bcg's saturation luma is
# BT.709 (float32).  Both as [1, 3] rows for _channel_dot.
_BT601_WEIGHTS = np.array([[0.299, 0.587, 0.114]])
_BCG_LUMA = np.array([[0.2126, 0.7152, 0.0722]], dtype=np.float32)


def _cvd_matrix(cvd_type: str, severity: float) -> np.ndarray:
    """apply_cvd's interpolated 3x3 Machado matrix (its numpy code, float64)."""
    matrices = CVD_MATRICES[cvd_type]
    # Interpolate between matrix steps (same as VIP-Sim)
    n = len(matrices)
    idx = severity * (n - 1)
    i0 = int(np.floor(idx))
    i1 = min(i0 + 1, n - 1)
    w = idx - i0
    return matrices[i0] * (1 - w) + matrices[i1] * w


# -- bcg -----------------------------------------------------------------------
def _bcg_tone(brightness, contrast, gamma, color_space: str) -> np.ndarray:
    """apply_bcg up to its saturation step (its numpy code) on the 256 levels.

    These steps act on each channel value alone, so on an 8-bit image they
    are this table.  numpy evaluates it, so its pow and its dtype promotion
    of NumPy-scalar kwargs (float64 under NEP 50) carry over as they are.
    """
    srgb = _LEVELS
    arr = srgb if color_space == "srgb" else _srgb_to_linear(srgb)
    # Brightness
    arr = arr * brightness
    # Contrast
    arr = (arr - 0.5) * contrast + 0.5
    # Clamp before gamma to avoid NaN from negative values
    arr = np.clip(arr, 0.0, 1.0)
    # Gamma
    if gamma != 1.0:
        arr = np.power(arr, gamma)
    return np.clip(arr, 0.0, 1.0)


# -- cataracts -----------------------------------------------------------------
def _channel_lookup(lut: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """lut[c, level] for every pixel of x: [N, C, H, W] read as 8-bit levels,
    with one [256] table per channel (lut: [C, 256])."""
    c = torch.arange(lut.shape[0], device=x.device).view(-1, 1, 1)
    return lut[c, _u8(x)]


def _cataract_levels(severity: float, use_brightness, use_contrast,
                     contrast_coeff, gamma) -> np.ndarray:
    """uint8 [3, 256]: apply_cataracts' BCG step on every level (the frost step
    after it only copies pixels), read from apply_cataracts itself on a
    1x256 grey ramp with frosting off."""
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    out = apply_cataracts(Image.fromarray(ramp), severity,
                          use_brightness=use_brightness,
                          use_contrast=use_contrast, use_frosting=False,
                          contrast_coeff=contrast_coeff, gamma=gamma)
    return np.asarray(out)[0].T.copy()


def _cataract_frost_maps(h: int, w: int, severity: float) -> tuple:
    """(sy, sx) int32 source pixels of apply_cataracts' frost step, its numpy
    code verbatim: the noise tile is read at int(uv * 20 * 256) % 256 and the
    displaced uv truncated to a pixel."""
    scale = 10.0 * severity
    noise_tile = _perlin_noise_2d(size=256)
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    uy, ux = np.meshgrid(ys, xs, indexing="ij")
    nx = ((ux * 20.0) * 256.0).astype(np.int32) % 256
    ny = ((uy * 20.0) * 256.0).astype(np.int32) % 256
    n = noise_tile[ny, nx]
    dx = -0.005 + (n - 0.008 * np.floor(n / 0.008))
    dy = -0.006 + (n - 0.010 * np.floor(n / 0.010))
    src_u = np.clip(ux + dx * scale, 0.0, 1.0)
    src_v = np.clip(uy + dy * scale, 0.0, 1.0)
    sx = (src_u * (w - 1)).astype(np.int32)
    sy = (src_v * (h - 1)).astype(np.int32)
    _check_numpy_index((h, w), sy, sx)                       # arr[sy, sx]
    return sy, sx


# -- bloom ---------------------------------------------------------------------
def _bloom_tables() -> tuple:
    """apply_bloom's colour maths for every level, by the CPU functions.

    lin [256]: _to_uint8(_srgb_to_linear(level)), the 8-bit linear image that
    gets downsampled.  composite [256, 256]: the output level
    _to_uint8(_linear_to_srgb(clip(lin + bloom))) for each input level and
    upsampled bloom level; the add uses the unrounded float32 linear input.
    """
    lin = _srgb_to_linear(_LEVELS)
    out_lin = np.clip(lin[:, None] + _LEVELS[None, :], 0.0, 1.0)
    return _to_uint8(lin), _to_uint8(_linear_to_srgb(out_lin))


def _numpy_operand(t: torch.Tensor, value) -> tuple:
    """(t, value) cast the way numpy combines the CPU's HWC array with value.

    NEP 50: a Python number takes the array's dtype, but a numpy scalar or
    array promotes it (float32 - np.float64(0.9) is float64, float32 -
    np.int64(1) too); torch would read a numpy scalar as a Python number and
    stay in float32.  A scalar comes back as a Python float in numpy's dtype
    (a bool as 0.0 or 1.0: numpy computes float32 - True, torch refuses
    tensor - bool).  A Python int becomes float(value), as numpy converts it
    through a C double, beyond int64 too (float32 - 2**70 is float32);
    torch cannot hold such an int, and would round an int64 straight to
    float32 where numpy rounds it twice.  Arrays broadcast against the CPU's
    (H, W, C) layout, so they come back as [1, C, H, W] tensors (a (3,)
    array is per channel), and
    fail where the CPU fails: ValueError if numpy cannot broadcast them,
    TypeError (Image.fromarray's) if they add a dimension.
    """
    if isinstance(value, (bool, np.generic)) or not isinstance(value, (int, float)):
        value = np.asarray(value)
    np_dtype = np.result_type(_NUMPY_FLOAT[t.dtype], value)
    if np_dtype not in _NUMPY_FLOAT.values():
        raise NotImplementedError(f"bloom operand of dtype {np_dtype}")
    t = t.to(_torch_dtype(np_dtype))
    if not isinstance(value, np.ndarray):
        value = float(value)
    elif value.ndim == 0:
        value = value.astype(np_dtype).item()
    else:
        np.broadcast_shapes(value.shape, (t.shape[-2], t.shape[-1], t.shape[-3]))
        if value.ndim > 3:                           # a 4-D image for PIL
            raise TypeError(f"bloom operand of shape {value.shape}")
        a = value.reshape((1,) * (3 - value.ndim) + value.shape)  # (H, W, C)
        value = torch.as_tensor(a, dtype=t.dtype, device=t.device)
        value = value.permute(2, 0, 1)[None]
    return t, value


# -- field_loss ----------------------------------------------------------------
def _field_loss_maps(h: int, w: int, severity: float, field_type: str,
                     overlay_scale, use_overlay_asset, mask_center,
                     mask_radius) -> tuple:
    """(lo, hi, frac) of apply_field_loss, its numpy code verbatim: the two
    pyramid levels each pixel blends and the weight of the upper one (float64,
    as numpy promotes float32 - int32)."""
    n_levels = int(2 + np.floor(np.log(max(h, w))))
    mask = None
    if use_overlay_asset and mask_center is None:
        mask = _load_macular_overlay(h, w, field_type=field_type,
                                     overlay_scale=overlay_scale)
    if mask is None:
        r = (mask_radius if mask_radius is not None
             else 0.45 / max(overlay_scale, 0.1))
        mask = _radial_weight_mask(h, w, kind=field_type,
                                   center=mask_center or (0.5, 0.5),
                                   radius=min(r, 0.7))
    w_scaled = mask * severity
    bias = w_scaled * (n_levels - 1)
    lo = np.floor(bias).astype(np.int32)
    hi = np.clip(lo + 1, 0, n_levels - 1)
    frac = bias - lo
    _check_numpy_index((n_levels,), lo)                      # stack[lo, iy, ix]
    return lo, hi, frac


# -- distortion ----------------------------------------------------------------
def _distortion_centres(severity: float, warp_centers) -> tuple:
    """(wx, wy, wr, wm) of apply_distortion, its code verbatim.

    These lists are all the map code reads from warp_centers, so the maps
    are keyed by them: deques, array.array rows and np.void records come
    down to their numbers.  They are made once per call, by the CPU's four
    passes, so an iterator gives the CPU's result (x from the first pass,
    no centres after it).
    """
    if warp_centers is None:
        # Defaults from myDistortionMap.cs warp_x/warp_y/warp_radius/warp_magn.
        warp_centers = [
            (0.0,  0.0,  0.15,  1.0),
            (0.33, 0.66, 0.15, -1.0),
        ]

    wx = [c[0] for c in warp_centers]
    wy = [c[1] for c in warp_centers]
    wr = [c[2] for c in warp_centers]
    wm = [c[3] * severity for c in warp_centers]
    return wx, wy, wr, wm


def _distortion_maps(h: int, w: int, wx: list, wy: list, wr: list,
                     wm: list) -> tuple:
    """(sy, sx) int32 maps of apply_distortion, its numpy code verbatim.

    See apply_distortion for the four shader quirks reproduced here: the
    Y-flip of the warp texture, the D3D UV convention, the sRGB decode bug
    (0.2*w - 0.0425) and the 0.001 deadband.
    """
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    uy, ux = np.meshgrid(ys, xs, indexing="ij")
    deg_x = ux * 0.5 + 0.25
    deg_y_for_x = 0.75 - 0.5 * uy   # vx_field evaluation coord (Unity UV space)
    deg_y_for_y = 0.25 + 0.5 * uy   # vy_field eval coord (post Y-flip in encoding)

    SQRT2 = float(np.sqrt(2.0))

    def _accumulate_v(field_x, field_y, want_y=False):
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

    # The CPU's local sRGB->linear (casts to float32 first, unlike filters'
    # module-level _srgb_to_linear).
    def _srgb_to_linear(s):
        a = 0.055
        s = np.asarray(s, dtype=np.float32)
        return np.where(s <= 0.04045, s / 12.92, ((s + a) / (1.0 + a)) ** 2.4)

    w_x = _srgb_to_linear(5.0 * vx_field + 0.5)
    w_y = _srgb_to_linear(5.0 * vy_field + 0.5)

    x_off = -(0.2 * w_x - 0.0425)
    y_off = (0.2 * w_y - 0.0425)

    x_off = np.where(np.abs(x_off) < 0.001, 0.0, x_off)
    y_off = np.where(np.abs(y_off) < 0.001, 0.0, y_off)

    src_u = np.clip(ux + x_off, 0.0, 1.0)
    src_v = np.clip(uy - y_off, 0.0, 1.0)
    sx = (src_u * (w - 1)).astype(np.int32)
    sy = (src_v * (h - 1)).astype(np.int32)
    _check_numpy_index((h, w), sy, sx)                       # arr[sy, sx]
    return sy, sx


# -- double_vision -------------------------------------------------------------
def _shifted(t: torch.Tensor, sx_: int, sy_: int) -> torch.Tensor:
    """`shifted` of apply_double_vision on [..., H, W] tensors.

    out[y, x] = t[y + sy_, x + sx_] where that pixel exists, else t[y, x]:
    the CPU pastes the shifted rectangle into zeros and fills every row and
    column outside it with the original, which is the same as pasting it
    into a copy of t.
    """
    H, W = t.shape[-2:]
    src_y0 = max(0, sy_); src_y1 = min(H, H + sy_)
    src_x0 = max(0, sx_); src_x1 = min(W, W + sx_)
    dst_y0 = max(0, -sy_); dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, -sx_); dst_x1 = dst_x0 + (src_x1 - src_x0)
    out = t.clone()
    out[..., dst_y0:dst_y1, dst_x0:dst_x1] = t[..., src_y0:src_y1, src_x0:src_x1]
    return out


# -- pixelation ----------------------------------------------------------------
def _pixelation_grid(w: int, h: int, severity, pixel_size, hard,
                     grid_offset_x, grid_offset_y):
    """apply_pixelation's grid, (cells_x, cells_y, dx, dy), from its own lines;
    None where the CPU returns the image unchanged."""
    severity = clamp_severity(severity)
    if severity == 0.0 and pixel_size is None:
        return None
    if pixel_size is None:
        pixel_size = float(1000.0 - severity * 980.0)  # 1000..20
    pixel_size = max(2.0, float(pixel_size))
    cells_x = max(2, int(round(pixel_size)))
    cells_y = max(2, int(round(cells_x * h / w))) if hard else cells_x
    ox = float(grid_offset_x) % 1.0
    oy = float(grid_offset_y) % 1.0
    dx = int(round(ox * (w / cells_x)))
    dy = int(round(oy * (h / cells_y)))
    return cells_x, cells_y, dx, dy


# -- led -----------------------------------------------------------------------
def _led_fields(h: int, w: int, scale, shape, margin, automatic_ratio) -> tuple:
    """apply_led's image-independent arrays, from its own lines.

    Returns (sy, sx, mask, keep): the texel row/column of each cell (the
    ceil-snapped coordinate truncated by astype(np.int32)), the LED mask in
    numpy's dtype, and the margin's 0/1 map as bool (None without a margin).
    Under NumPy 2 a np.float64 `shape` or `scale` makes the mask float64.
    """
    ratio = (w / h) if automatic_ratio else 1.0
    ys = (np.arange(h, dtype=np.float32) + 0.5) / h  # pixel centers
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w
    ds_x = 1.0 / scale
    ds_y = ds_x * ratio
    coord_x = ds_x * np.ceil(xs / ds_x)
    coord_y = ds_y * np.ceil(ys / ds_y)
    sx = np.clip((coord_x * w).astype(np.int32), 0, w - 1)
    sy = np.clip((coord_y * h).astype(np.int32), 0, h - 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mvx = np.abs(np.sin(xx * scale * np.pi)) * shape
    mvy = np.abs(np.sin(yy * (scale / ratio) * np.pi)) * shape
    s = mvx * mvy
    c = (s <= 1.0).astype(np.float32)
    mask = (1.0 - c) + c * s
    keep = None
    if margin > 0:
        m = np.ones((h, w), dtype=np.float32)
        m[(yy < margin) | (yy > 1.0 - margin)] = 0.0
        m[:, (xs < margin) | (xs > 1.0 - margin)] = 0.0
        keep = m != 0.0
    return sy, sx, mask, keep


def _led_lut(brightness) -> np.ndarray:
    """_srgb_to_linear(level/255 * brightness) for the 256 input levels, in
    the dtype apply_led computes it in."""
    return _srgb_to_linear(_LEVELS * brightness)


# -- vortex --------------------------------------------------------------------
def _vortex_fields(h, w, severity, center, vortex_radius, radius_y,
                   radius_y_down, suction_strength, falloff_exponent,
                   swirl_strength, inner_circle_radius, inner_color,
                   inner_center, inner_fade, inner_opacity, halo_radius,
                   halo_center, halo_thickness, halo_color, halo_opacity,
                   noise_amount, noise_scale, fade_width):
    """Image-independent part of apply_vortex, at the pixels inside the field.

    Its numpy code, line for line.  Returns None when no pixel is inside (the
    CPU then returns the image), else a dict of flat arrays over the M inside
    pixels: idx; fy, fx (sample coordinates, [1, M]); keep_out = 1 - s_out and
    s_out (outer fade); inner_add / halo_add = colour * alpha as [3, M] with
    inner_keep / halo_keep = 1 - alpha, or None when that layer is off.
    """
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
    ys_up = vortex_radius / max(radius_y, 1e-6) if radius_y else 1.0
    ys_dn = (vortex_radius / max(radius_y_down, 1e-6)
             if radius_y_down else ys_up)
    dx = adj_x - adj_cx
    dy_raw = adj_y - adj_cy
    ys = np.where(dy_raw < 0, ys_up, ys_dn)
    dy = dy_raw * ys
    dist = np.sqrt(dx * dx + dy * dy)

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
    if not np.any(inside):
        return None
    eps = 1e-8
    angle = np.arctan2(dy, dx)
    angle_noise = _value_noise(xx, yy, noise_scale)
    angle = angle + angle_noise * noise_amount

    d_over_R = np.clip(dist_n / max(vortex_radius, eps), 0.0, 1.0)
    r_sample = dist_n * (1.0 + suction_strength
                         * (1.0 - d_over_R) ** falloff_exponent)

    if swirl_strength:
        angle = angle + swirl_strength * (1.0 - d_over_R) ** 2

    sample_x = adj_cx + r_sample * np.cos(angle)
    sin_a = np.sin(angle)
    ys_back = np.where(sin_a < 0, ys_up, ys_dn)
    sample_y = adj_cy + r_sample * sin_a / ys_back
    sample_x = sample_x / aspect  # undo aspect correction
    sample_x = np.clip(sample_x, 0.0, 1.0)
    sample_y = np.clip(sample_y, 0.0, 1.0)
    fx = sample_x * (w - 1)
    fy = sample_y * (h - 1)

    t_out = np.clip((dist_n - (vortex_radius - fade_width))
                    / max(fade_width, eps), 0.0, 1.0)
    s_out = t_out * t_out * (3.0 - 2.0 * t_out)

    has_halo = halo_radius > 0 and halo_opacity > 0
    if inner_circle_radius > 0 or has_halo:
        if inner_center is not None:
            icy, icx = inner_center
            idx = adj_x - icx * aspect
            idy = adj_y - icy
            idist = np.sqrt(idx * idx + idy * idy)
        else:
            idist = dist_n

    sel = np.flatnonzero(inside)

    def at(a):                       # [h, w] or [h, w, 3] -> [M] or [3, M]
        return a.reshape(h * w)[sel] if a.ndim == 2 else a.reshape(h * w, -1)[sel].T

    f = dict(idx=sel, fy=at(fy)[None], fx=at(fx)[None],
             keep_out=at(1.0 - s_out), s_out=at(s_out),
             inner_add=None, inner_keep=None, halo_add=None, halo_keep=None)

    if inner_circle_radius > 0:
        i_fade = fade_width if inner_fade is None else inner_fade
        grey = np.full((h, w, 3), inner_color, dtype=np.float32)  # full_like(arr)
        t_in = np.clip((idist - (inner_circle_radius - i_fade))
                       / max(i_fade, eps), 0.0, 1.0)
        s_in = t_in * t_in * (3.0 - 2.0 * t_in)
        alpha = (1.0 - s_in)[..., None] * inner_opacity
        alpha = np.where((idist < inner_circle_radius)[..., None], alpha, 0.0)
        f.update(inner_add=at(grey * alpha), inner_keep=at((1.0 - alpha)[..., 0]))

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
        halo_col = np.full((h, w, 3), halo_color, dtype=np.float32)
        f.update(halo_add=at(halo_col * h_alpha), halo_keep=at((1.0 - h_alpha)[..., 0]))
    return f


# -- wiggle --------------------------------------------------------------------
def _wiggle_maps(h: int, w: int, severity: float, freq, amplitude, timer,
                 mode) -> tuple:
    """(sy, sx) int32 maps of apply_wiggle, its numpy code verbatim."""
    amp = amplitude * severity

    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    uy, ux = np.meshgrid(ys, xs, indexing="ij")

    if mode == "simple":
        du = np.sin(timer + ux * freq) * amp
        dv = np.cos(timer + uy * freq) * amp - amp
    else:  # complex (Unity default)
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
    _check_numpy_index((h, w), sy, sx)                       # arr[sy, sx]
    return sy, sx


# -- noise ---------------------------------------------------------------------
# noise, glitch, flickering_stars and floaters draw all their randomness from
# np.random.default_rng(seed) and never look at the image.  Each helper runs
# the RNG part of its CPU filter (same draws in the same order, same dtypes)
# and returns a [C, H, W] field, C = 1 (one value for R, G and B) or 3.
def _noise_field(h: int, w: int, seed) -> np.ndarray:
    """apply_noise's float32 noise in [-1, 1): [1, h, w]."""
    rng = np.random.default_rng(seed)
    n = (rng.random((h, w)).astype(np.float32) - 0.5) * 2.0  # [-1, 1]
    return n[None]


# -- glitch --------------------------------------------------------------------
def _glitch_columns(h: int, w: int, severity: float, max_displace: float,
                    seed) -> np.ndarray:
    """Source column of every output pixel of apply_glitch: int64 [3, h, w].

    The per-row displacement is apply_glitch's code verbatim.  Its
    np.roll(row, s) reads row[(j - s) % w], with s = d, d // 2 and -d for
    R, G and B (floor division, like the CPU's int(d) // 2).
    """
    rng = np.random.default_rng(seed)
    n_blocks = max(1, h // 12)
    disp_blocks = (rng.random(n_blocks) - 0.5) * 2.0 * max_displace * severity
    block_size = max(1, h // n_blocks)
    disp = np.repeat(disp_blocks, block_size)[:h]
    if disp.size < h:
        disp = np.pad(disp, (0, h - disp.size), mode="edge")
    d = (disp * w).astype(np.int32).astype(np.int64)
    shifts = np.stack([d, d // 2, -d])[..., None]          # [3, h, 1]
    return (np.arange(w) - shifts) % w


# -- teichopsia ----------------------------------------------------------------
def _teichopsia_fields(h, w, gaze, seed):
    """Ring pixels and scintillation multipliers of apply_teichopsia.

    Its numpy code, line for line.  Returns (idx, mult): flat indices of the
    screen pixels inside the ring and the float32 per-channel noise
    multipliers there, [3, M].
    """
    try:
        from skimage.draw import polygon as skpolygon
    except ImportError as e:
        raise ImportError("teichopsia requires scikit-image: "
                          "pip install 'vipsim[teichopsia]'") from e

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
    rho_out = max_radius * (1 - spikiness) + \
        rng.integers(0, int(max_radius * spikiness), n_points).astype(np.float32)
    thetas = (2 * np.pi / (n_points + 1)) * np.arange(n_points, dtype=np.float32)
    outer_x = cx_mask + rho_out * np.cos(thetas)
    outer_y = cy_mask + rho_out * np.sin(thetas)

    ring_w = np.maximum(width_mu + width_std * _box_muller(n_points, rng), min_pixels)
    rho_in = np.maximum(rho_out - ring_w, min_radius_px)
    inner_x = cx_mask + rho_in * np.cos(thetas)
    inner_y = cy_mask + rho_in * np.sin(thetas)

    ring_mask = np.zeros((mask_size, mask_size), dtype=bool)
    rr_o, cc_o = skpolygon(outer_y, outer_x, ring_mask.shape)
    rr_i, cc_i = skpolygon(inner_y, inner_x, ring_mask.shape)
    ring_mask[rr_o, cc_o] = True
    ring_mask[rr_i, cc_i] = False  # outer ∖ inner = ring

    # 2) Map mask to screen (NEAREST, int32 truncation toward zero).
    gx, gy = gaze
    gaze_off_x = 0.5 - gx
    gaze_off_y = 0.5 - gy
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    uy, ux = np.meshgrid(ys, xs, indexing="ij")
    deg_x = (ux - gaze_off_x) * 0.5 + 0.25
    deg_y = (uy - gaze_off_y) * 0.5 + 0.25
    mx = (deg_x * (mask_size - 1)).astype(np.int32)
    my = (deg_y * (mask_size - 1)).astype(np.int32)
    in_bounds = (mx >= 0) & (mx < mask_size) & (my >= 0) & (my < mask_size)
    screen_mask = np.zeros((h, w), dtype=bool)
    screen_mask[in_bounds] = ring_mask[my[in_bounds], mx[in_bounds]]

    # 3) Scintillation noise, drawn for the whole frame as on the CPU.
    rng2 = np.random.default_rng(seed + 1)
    n = rng2.random((h, w)).astype(np.float32)
    nr = (n % 1.0) * 2.0
    ng = ((n * 1.2154) % 1.0) * 2.0
    nb = ((n * 1.3453) % 1.0) * 2.0
    mult = np.stack([nr, ng, nb], axis=0).astype(np.float32)

    sel = np.flatnonzero(screen_mask)
    return sel, mult.reshape(3, -1)[:, sel]


def _scalar_product_dtype(scalar) -> torch.dtype:
    """dtype of numpy's `float32_array * scalar`.

    A Python number leaves the product float32, but a numpy scalar keeps its
    dtype (NEP 50): np.float64(2.0) or np.int64(2) make it float64.  torch
    reads any numpy scalar as a Python number, so the twins use this to
    promote exactly where numpy does.
    """
    return _torch_dtype((np.zeros(1, np.float32) * scalar).dtype)


def skimage_rgb2hsv(x: torch.Tensor) -> torch.Tensor:
    """Twin of skimage.color.rgb2hsv for channels at dim 1 ([N, 3, ...]).

    Same float ops in the same dtype; where skimage overwrites the hue of
    pixels whose max is in several channels (red, then green, then blue
    wins), the nested where gives blue priority too.  Bit-exact in float32.
    """
    r, g, b = x.unbind(1)
    v = torch.maximum(torch.maximum(r, g), b)
    delta = v - torch.minimum(torch.minimum(r, g), b)            # np.ptp
    h = torch.where(b == v, 4.0 + (r - g) / delta,
                    torch.where(g == v, 2.0 + (b - r) / delta, (g - b) / delta))
    h = torch.where(delta == 0.0, 0.0, exact_div(h, 6.0) % 1.0)
    s = torch.where(delta == 0.0, 0.0, delta / v)
    hsv = torch.stack((h, s, v), 1)
    return torch.where(torch.isnan(hsv), 0.0, hsv)               # out[isnan] = 0


# hsv2rgb's np.choose table: (r, g, b) as indices into (v, p, q, t), by sector.
_HSV_SECTORS = ((0, 3, 1), (2, 0, 1), (1, 0, 3), (1, 2, 0), (3, 1, 0), (0, 1, 2))


def skimage_hsv2rgb(x: torch.Tensor) -> torch.Tensor:
    """Twin of skimage.color.hsv2rgb for channels at dim 1 ([N, 3, ...])."""
    h, s, v = x.unbind(1)
    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    sectors = _device_scalar(_HSV_SECTORS, 1.0, torch.long, x.device)
    pick = sectors[hi.long() % 6].movedim(-1, 1)                  # [N, 3, ...]
    return torch.stack((v, p, q, t), 1).gather(1, pick)


# -- foveal_darkness -----------------------------------------------------------
def _foveal_darken(h, w, severity, center, radius, radius_x, radius_y,
                   edge_width, invert, band_width):
    """The per-pixel blend weight `darken` of apply_foveal_darkness, [h, w]:
    float32, or float64 when a keyword is a numpy float64."""
    if radius is None:
        radius = 0.05 + severity * 0.20
    rx = radius_x if radius_x is not None else radius
    ry = radius_y if radius_y is not None else radius

    cy, cx = center
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1) - cy
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
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
    return darken


# -- flickering_stars ----------------------------------------------------------
def _star_levels(h: int, w: int, severity: float, n_stars, seed) -> np.ndarray:
    """apply_flickering_stars' sprites drawn on black, rounded: uint8 [1, h, w].

    The CPU keeps max(pixel, sprite) per star in float32 and rounds once at
    the end.  The pixels are whole levels, and float32 rounding and rint are
    monotonic, so its output is max(level, this field) for every image.
    RNG and sprite loop are the CPU code verbatim.
    """
    if n_stars is None:
        n_stars = int(20 + severity * 80)
    rng = np.random.default_rng(seed)
    star_y = rng.integers(0, h, n_stars)
    star_x = rng.integers(0, w, n_stars)
    brightness = rng.random(n_stars).astype(np.float32) * severity

    out = np.zeros((h, w), np.float32)
    dy_grid, dx_grid = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing="ij")
    fall = np.clip(1.0 - (dy_grid * dy_grid + dx_grid * dx_grid) / 8.0, 0.0, 1.0)
    for sy_, sx_, b in zip(star_y, star_x, brightness):
        y0, y1 = max(0, sy_ - 2), min(h, sy_ + 3)
        x0, x1 = max(0, sx_ - 2), min(w, sx_ + 3)
        py0 = y0 - (sy_ - 2); py1 = py0 + (y1 - y0)
        px0 = x0 - (sx_ - 2); px1 = px0 + (x1 - x0)
        sprite = 255.0 * b * fall[py0:py1, px0:px1]
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], sprite)
    return _to_uint8(out, 1.0)[None]


# -- floaters ------------------------------------------------------------------
# apply_floaters' rejection samplers loop until enough points land in the
# frame; parameters that leave no part of it to sample (a NaN or -inf
# circle_radius_uv, a Gaussian centred far off-frame, NaN angle or spread)
# would loop forever.  This many draws in a row with none accepted is
# beyond any acceptance rate above ~1e-6, so it raises instead.
_FLOATER_MAX_REJECTED = 10_000_000


def _check_floater_rejections(rejected: int) -> None:
    if rejected >= _FLOATER_MAX_REJECTED:
        raise ValueError(
            f"apply_floaters: no point landed in the frame in {rejected:,} "
            "draws; check center, spread, angle and circle_radius_uv")


def _floater_counts(h: int, w: int, n_floaters: int, floater_size, circle_radius_uv,
                    seed, distribution, center, spread, spread_x, spread_y,
                    size_variance, angle, region, region_feather,
                    region_center_bias) -> np.ndarray:
    """How many times apply_floaters blends each pixel: int64 [1, h, w].

    n_floaters is the resolved count (> 0).  Region mask, point sampling,
    sizes and blob masks are apply_floaters' code verbatim.  Single-pixel
    specks blend the original pixel once however many land on it; every blob
    mask then blends the running result again, so overlapping blobs compound
    and the output is the same blend applied `count` times.
    """
    rng = np.random.default_rng(seed)
    overlay_size = 1024.0
    target = n_floaters

    region_mask = None
    if region is not None and len(region) >= 3:
        m = Image.new("L", (int(overlay_size), int(overlay_size)), 0)
        poly = [(float(x) * overlay_size, float(y) * overlay_size)
                for y, x in region]
        ImageDraw.Draw(m).polygon(poly, fill=255)
        if region_feather:
            check_pil_blur_radius(region_feather * overlay_size)
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
        pts_x = rng.uniform(0.0, overlay_size, size=target)
        pts_y = rng.uniform(0.0, overlay_size, size=target)
    elif distribution == "gaussian":
        cyN, cxN = center
        mu_x = cxN * overlay_size
        mu_y = cyN * overlay_size
        sigma_x = max(1.0, (spread_x if spread_x is not None else spread) * overlay_size)
        sigma_y = max(1.0, (spread_y if spread_y is not None else spread) * overlay_size)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        pts_x, pts_y = [], []
        rejected = 0
        while len(pts_x) < target:
            n = (target - len(pts_x)) * 2
            dx = rng.normal(0.0, sigma_x, n)
            dy = rng.normal(0.0, sigma_y, n)
            bx = mu_x + dx * cos_a - dy * sin_a
            by = mu_y + dx * sin_a + dy * cos_a
            ok = (bx >= 0) & (bx < overlay_size) & (by >= 0) & (by < overlay_size)
            pts_x.extend(bx[ok].tolist())
            pts_y.extend(by[ok].tolist())
            rejected = 0 if ok.any() else rejected + n
            _check_floater_rejections(rejected)
        pts_x = np.asarray(pts_x[:target], dtype=np.float64)
        pts_y = np.asarray(pts_y[:target], dtype=np.float64)
    else:
        cx_o = cy_o = overlay_size * 0.5
        r_o = circle_radius_uv * overlay_size
        pts_x, pts_y = [], []
        rejected = 0
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
            rejected = 0 if keep.any() else rejected + batch
            _check_floater_rejections(rejected)
        pts_x = np.asarray(pts_x[:target], dtype=np.float64)
        pts_y = np.asarray(pts_y[:target], dtype=np.float64)

    xs = np.clip((pts_x * (w / overlay_size)).astype(int), 0, w - 1)
    ys = np.clip((pts_y * (h / overlay_size)).astype(int), 0, h - 1)
    if size_variance:
        pool = np.array([1, 1, 1, 1, 2, 2, 3, floater_size + 2, floater_size + 4])
        sizes = rng.choice(pool, size=target)
    else:
        sizes = np.full(target, floater_size, dtype=int)

    counts = np.zeros((h, w), np.int64)
    small = sizes <= 1
    counts[ys[small], xs[small]] = 1
    for sx, sy, sz in zip(xs[~small], ys[~small], sizes[~small]):
        s_x = max(1, int(round(sz * w / overlay_size)))
        s_y = max(1, int(round(sz * h / overlay_size)))
        hx, hy = s_x // 2, s_y // 2
        x0, x1 = max(0, sx - hx), min(w, sx + hx + 1)
        y0, y1 = max(0, sy - hy), min(h, sy + hy + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = rng.random((y1 - y0, x1 - x0)) > 0.4
        counts[y0:y1, x0:x1] += mask
    return counts[None]


def _floater_levels(alpha: float, target_val: float, max_count: int) -> np.ndarray:
    """Level k after c floater blends: uint8 [max_count + 1, 256], row c.

    Iterates the CPU's float32 blend `v * (1.0 - alpha) + target_val * alpha`
    on every level and rounds each row like its final _to_uint8(out, 1.0).
    """
    v = np.arange(256, dtype=np.float32)
    rows = [_to_uint8(v, 1.0)]
    for _ in range(max_count):
        v = v * (1.0 - alpha) + target_val * alpha
        rows.append(_to_uint8(v, 1.0))
    return np.stack(rows)


# -- vignette ------------------------------------------------------------------
def _vignette_ramp(h: int, w: int, radius, softness) -> np.ndarray:
    """apply_vignette's smoothstep falloff (its numpy code, before
    `* strength`): float32 [h, w], float64 if radius or softness is a NumPy
    float64 scalar."""
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1) - 0.5
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1) - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    # Normalise so the corner distance is 1.0 (0.707 = sqrt(0.5^2+0.5^2)).
    dist = np.sqrt(xx * xx + yy * yy) / 0.7071
    t = np.clip((dist - radius) / max(softness, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class VipSimGPU:
    """Holds device + cached static tensors so per-frame work is minimal."""

    def __init__(self, device="cuda", dtype=torch.float32, cache_size: int = 64):
        self.device = torch.device(device)
        self.dtype = dtype
        # Image-independent tensors of every filter: warp fields, index maps,
        # masks, noise fields, resize taps, lookup tables.
        self.cache = TensorCache(cache_size)
        # The Machado tables as float32 device tensors, for callers; cvd()
        # interpolates the CPU's float64 CVD_MATRICES itself, as apply_cvd does.
        self.cvd_tables = {name: torch.from_numpy(m.astype(np.float32)).to(self.device, dtype)
                           for name, m in CVD_MATRICES.items()}
        self.cvd_deuter = self.cvd_tables["deuteranomaly"]

    # ---- Cache helpers ------------------------------------------------------
    def cached(self, name: str, params, hw, build, dtype=None):
        """Return build() for this key, computed once and kept in self.cache.

        Key: (name, params, (H, W), device, dtype).  params is any nesting of
        numbers, strings, None, tuples, lists, dicts and numpy scalars or
        arrays, keyed exactly by _freeze (by type and value, no rounding).
        build() returns a tensor or a tuple of tensors already on the device.
        Params that _freeze cannot key exactly are built on every call and
        not cached.
        """
        frozen = _freeze(params)
        if frozen is None:
            return build()
        key = (name, frozen, tuple(hw), str(self.device), dtype or self.dtype)
        return self.cache.get(key, build)

    def to_device(self, a, dtype=None) -> torch.Tensor:
        """numpy array -> device tensor (copied, so read-only arrays are fine).

        Default dtype: bool stays bool, integers become int64 (index maps),
        floats become self.dtype.
        """
        a = np.asarray(a)
        if dtype is None:
            dtype = (torch.bool if a.dtype == np.bool_ else
                     torch.long if a.dtype.kind in "iu" else self.dtype)
        return torch.tensor(a, dtype=dtype, device=self.device)

    def numpy_field(self, name: str, params, hw, fn, dtype=None):
        """Run fn() once — the CPU filter's own numpy code, RNG included — and
        keep its result on the device.

        fn() returns an array or a tuple of arrays (e.g. (sy, sx) index maps
        or a noise field from np.random.default_rng(seed)); each is converted
        with to_device(…, dtype).  Reusing the CPU expression verbatim is what
        makes nearest-index truncation and seeded noise match exactly.
        """
        def build():
            out = fn()
            if isinstance(out, tuple):
                return tuple(self.to_device(a, dtype) for a in out)
            return self.to_device(out, dtype)
        return self.cached(name, params, hw, build, dtype)

    def _flat_map(self, maps) -> torch.Tensor:
        """(sy, sx) int32 numpy maps of an [H, W] image -> the device flat
        index gather_nearest(x, sy, sx) reads (_flat_index), for caching."""
        sy, sx = maps
        H, W = sy.shape
        return _flat_index(self.to_device(sy), self.to_device(sx), H, W)

    def _random_field(self, name: str, params, hw, seed, n: int, fn, convert=None):
        """convert(fn()), where fn() draws a [C, H, W] field from `seed`.

        A fixed seed runs fn() once per key and caches the result, and every
        image in the batch gets it, as N CPU calls with that seed would.
        seed=None or a Generator draws a new field on every CPU call, so fn()
        then runs once per image ([n, C, H, W], in batch order), uncached.
        In a batched chain, a Generator shared by two such filters is drawn
        filter by filter (A(f0), A(f1), B(f0), B(f1)), as the CPU running
        each filter over the whole batch in turn would draw it, not as the
        CPU chain run frame by frame (A(f0), B(f0), A(f1), B(f1)).
        """
        convert = convert or self.to_device
        if _fixed_seed(seed):
            return self.cached(name, params, hw, lambda: convert(fn()))
        return convert(np.stack([fn() for _ in range(n)]))

    # ---- PIL resize twin ----------------------------------------------------
    def pil_resize(self, x: torch.Tensor, width: int, height: int,
                   resample: str = "bilinear", quantize: bool = True) -> torch.Tensor:
        """Twin of PIL `img.resize((width, height), Image.BILINEAR/NEAREST)`.

        Note PIL's (width, height) order.  x: [N, C, H, W] float in [0,1].
        quantize=True treats x as an 8-bit image (rounded to k/255 first) and
        replays Pillow's fixed-point resampler, horizontal pass then vertical
        pass (the reverse where the installed Pillow does, see
        _pil_vertical_first), each rounded to 8 bits: bit-exact with PIL,
        including the antialiased (support-stretched) filter when
        downscaling.  NEAREST is exact either way.  quantize=False with
        BILINEAR uses F.interpolate(antialias=True), which computes the same
        filter in float without the intermediate rounding: measured max 1 LSB,
        mean <= 0.20 LSB after rounding (tests/test_gpu_primitives.py).
        """
        H, W = x.shape[-2:]
        if quantize:
            return _from_u8(self._resize_levels(_u8(x), width, height, resample),
                            x.dtype)
        if (width, height) == (W, H):                # PIL returns a copy
            return x
        if resample == "nearest":
            return self._resize_levels(x, width, height, resample)
        if resample != "bilinear":
            raise ValueError(f"Unsupported resample: {resample}")
        return F.interpolate(x, size=(height, width), mode="bilinear",
                             align_corners=False, antialias=True)

    def _resize_levels(self, t: torch.Tensor, width: int, height: int,
                       resample: str = "bilinear") -> torch.Tensor:
        """pil_resize on integer 8-bit levels (int64), returning levels.

        Filters that resize an image they already hold as levels (bloom,
        pixelation) call this directly: _u8(_from_u8(t)) is t for levels
        0..255, so skipping that round trip changes no value.  NEAREST only
        copies, so it also takes float images.
        """
        H, W = t.shape[-2:]
        if (width, height) == (W, H):                # PIL returns a copy
            return t
        if resample == "nearest":
            lin = self.cached(
                "pil_resize_nearest", (width, height), (H, W),
                lambda: _flat_index(self.to_device(_pil_nearest_index(H, height))[:, None],
                                    self.to_device(_pil_nearest_index(W, width))[None, :],
                                    H, W))
            return _gather_flat(t, lin)
        if resample != "bilinear":
            raise ValueError(f"Unsupported resample: {resample}")
        ix, kx, iy, ky = self.cached(
            "pil_resize_bilinear", (width, height), (H, W),
            lambda: tuple(self.to_device(a) for a in
                          _pil_bilinear_taps(W, width) + _pil_bilinear_taps(H, height)))
        vertical_first = _pil_vertical_first(W, H, width, height)
        if vertical_first:
            t = _resample_pass(t.transpose(-1, -2), iy, ky).transpose(-1, -2)
        if width != W:                               # Pillow skips unchanged axes
            t = _resample_pass(t, ix, kx)
        if height != H and not vertical_first:
            t = _resample_pass(t.transpose(-1, -2), iy, ky).transpose(-1, -2)
        return t

    # ---- sRGB <-> linear on 8-bit images ------------------------------------
    def _srgb_to_linear_u8(self, x: torch.Tensor) -> torch.Tensor:
        """filters._srgb_to_linear(np.float32(img) / 255), float32."""
        table = self.numpy_field("srgb_to_linear", (), (256,),
                                 lambda: _srgb_to_linear(_LEVELS), torch.float32)
        return table[_u8(x)]

    def _linear_to_srgb_u8(self, v: torch.Tensor) -> torch.Tensor:
        """filters._to_uint8(filters._linear_to_srgb(v)) as int64 levels, for
        float32 or float64 v (the CPU's dtype decides where the steps fall).

        The level is the number of _srgb_u8_steps at or below v, so no pow
        runs on the device.  NaN gets the level numpy's uint8 cast gives it
        on this platform (a SIMD-sized array, as for an image).
        """
        def build():
            dtype = np.float64 if v.dtype == torch.float64 else np.float32
            with np.errstate(invalid="ignore"):
                nan = _to_uint8(_linear_to_srgb(np.full(256, np.nan, dtype)))[0]
            return self.to_device(_srgb_u8_steps(dtype), v.dtype), int(nan)
        steps, nan_level = self.cached("srgb_u8_steps", (), (), build, v.dtype)
        # bucketize would copy a strided v itself, with a warning.
        levels = torch.bucketize(v.contiguous(), steps, right=True)
        return levels.masked_fill_(v.isnan(), nan_level)

    # ---- cvd: a Machado matrix in linear light ------------------------------
    def cvd(self, x: torch.Tensor, cvd_type: str = "deuteranomaly",
            severity: float = 1.0) -> torch.Tensor:
        """Twin of apply_cvd: a Machado matrix, or the monochrome blend, in
        linear light.  cvd_type: 'protanomaly', 'deuteranomaly', 'tritanomaly'
        or 'monochrome'; others raise KeyError, as on the CPU."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        # Linear light, as in Unity's Linear-colour-space myRecolour pass.
        lin = self._srgb_to_linear_u8(x)                      # float32
        if cvd_type == "monochrome":
            weights = self.numpy_field("bt601_weights", (), (),
                                       lambda: _BT601_WEIGHTS, torch.float64)
            gray = _channel_dot(lin.double(), weights)        # [N,1,H,W] float64
            # lin * (1 - severity) stays float32 on the CPU; the sum is float64.
            result = (lin * (1 - severity)).double() + gray * severity
        else:
            mat = self.numpy_field("cvd_matrix", (cvd_type, severity), (),
                                   lambda: _cvd_matrix(cvd_type, severity),
                                   torch.float64)
            result = _channel_dot(lin.double(), mat)          # np.dot(lin, mat.T)
        return _from_u8(self._linear_to_srgb_u8(result), x.dtype)

    # ---- blur: PIL GaussianBlur (twin of apply_blur) ------------------------
    def blur(self, x: torch.Tensor, severity: float = 0.5) -> torch.Tensor:
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        # Same 0..15 px radius mapping as apply_blur, through PIL's box cascade.
        return pil_gaussian_blur(x, severity * 15.0)

    # ---- bcg: brightness / contrast / gamma / saturation --------------------
    def bcg(self, x: torch.Tensor, brightness: float = 1.0, contrast: float = 1.0,
            gamma: float = 1.0, saturation: float = 1.0,
            color_space: str = "linear") -> torch.Tensor:
        """Twin of apply_bcg.  Brightness, contrast and gamma are a table of
        the 256 levels; any color_space other than 'srgb' runs in linear
        light, as on the CPU."""
        # The tables only depend on these four keywords, so they are built
        # (by numpy, ~0.1 ms) once per exact key, never per frame.  _freeze
        # keys NumPy scalars by dtype, so float64 tables from NumPy-scalar
        # kwargs never share an entry with float32 ones.
        tone_key = (brightness, contrast, gamma, color_space)
        levels = _u8(x)
        if saturation == 1.0:
            # No channel mixing: the whole filter is a table of the levels.
            def lut_table():
                tone = _bcg_tone(brightness, contrast, gamma, color_space)  # numpy, [256]
                return _to_uint8(tone if color_space == "srgb" else _linear_to_srgb(tone))
            lut = self.numpy_field("bcg_lut", tone_key, (256,), lut_table)
            return _from_u8(lut[levels], x.dtype)
        arr = self.cached("bcg_tone", tone_key, (256,),
                          lambda: torch.from_numpy(_bcg_tone(
                              brightness, contrast, gamma, color_space)).to(self.device))[levels]
        # Saturation (extension): lerp toward per-pixel luminance
        luma = self.numpy_field("bcg_luma", (), (), lambda: _BCG_LUMA, torch.float32)
        # numpy's float32 `arr @ luma` calls sdot per pixel on 1-px-wide images.
        sdot = arr.dtype == torch.float32 and x.shape[-1] == 1
        lum = _channel_dot(arr, luma, sdot)
        diff = arr - lum
        if np.result_type(np.float32, saturation) == np.float64:   # NumPy scalar
            diff = diff.double()
        arr = (lum + diff * float(saturation)).clamp(0.0, 1.0)
        if color_space == "srgb":
            return _from_u8(_u8(arr), x.dtype)
        return _from_u8(self._linear_to_srgb_u8(arr), x.dtype)

    # ---- cataracts: per-channel BCG, then the frost gather ------------------
    def cataracts(self, x: torch.Tensor, severity: float = 0.5,
                  use_brightness: bool = True, use_contrast: bool = True,
                  use_frosting: bool = True, contrast_coeff=(0.7, 0.7, 0.4),
                  gamma: float = 1.0) -> torch.Tensor:
        """Twin of apply_cataracts: per-channel BCG, then the frost gather."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        H, W = x.shape[-2:]
        # 1. Per-channel BCG in linear light, as a [3, 256] table.
        lut = self.numpy_field(
            "cataracts_bcg",
            (severity, use_brightness, use_contrast, contrast_coeff, gamma),
            (), lambda: _cataract_levels(severity, use_brightness, use_contrast,
                                         contrast_coeff, gamma))
        out = _from_u8(_channel_lookup(lut, x), x.dtype)
        # 2. Frost: noise-driven uv displacement, truncated to source pixels.
        if use_frosting:
            lin = self.cached("cataracts_frost", (severity,), (H, W),
                              lambda: self._flat_map(_cataract_frost_maps(H, W, severity)))
            out = _gather_flat(out, lin)
        return out

    # ---- bloom: 8-bit linear threshold, PIL blur, linear-light add ----------
    def bloom(self, x: torch.Tensor, severity: float = 0.5,
              intensity: float = None, threshold: float = None,
              blur_size: float = None, iterations: int = 1) -> torch.Tensor:
        """Twin of apply_bloom, 8-bit linear intermediates included."""
        severity = clamp_severity(severity)
        if severity == 0.0 and intensity is None and threshold is None and blur_size is None:
            return x
        if intensity is None:
            intensity = 0.1 + severity * 0.6
        if threshold is None:
            threshold = max(0.6, 0.95 - severity * 0.3)
        if blur_size is None:
            blur_size = 1.0 + severity * 4.5
        lin, composite = self.numpy_field("bloom_tables", (), (), _bloom_tables)
        H, W = x.shape[-2:]
        levels = _u8(x)
        # 1. Downsample the 8-bit linear image by 4 (PIL BILINEAR), as the
        # CPU's float32 k/255 whatever x's dtype: fp16 k/255 would move the
        # threshold's rounding boundaries.
        # The 8-bit intermediates stay integer levels from here on (blur and
        # resize on levels): _u8(_from_u8(t)) is t, so no value changes.
        down = _from_u8(self._resize_levels(lin[levels], max(1, W // 4),
                                            max(1, H // 4)), torch.float32)
        # 2. Threshold-extract in numpy's dtypes (float32 unless an operand is
        # a numpy scalar or array that promotes it).
        bright, threshold = _numpy_operand(down, threshold)
        bright = (bright - threshold).clamp(min=0.0)
        bright, intensity = _numpy_operand(bright, intensity)
        bright = _u8(bright * intensity)
        # 3. _separable_gaussian: PIL GaussianBlur of the 8-bit image, with a
        # radius growing by 1.5 px per iteration; radius <= 0 is skipped.
        for i in range(iterations):
            radius = blur_size * 0.5 + i * 1.0
            if radius * 1.5 <= 0:      # NaN is not skipped: PIL raises on it
                continue
            bright = _blur_levels(bright, radius * 1.5)
        # 4. Upsample the 8-bit bloom and add it in linear light.
        bloom = self._resize_levels(bright, W, H)
        return _from_u8(composite[levels, bloom], x.dtype)

    # ---- field_loss: per-pixel blend of two blur-pyramid levels -------------
    def field_loss(self, x: torch.Tensor, severity: float = 0.5,
                   field_type: str = "central", overlay_scale: float = 0.75,
                   use_overlay_asset: bool = True, mask_center=None,
                   mask_radius: float = None) -> torch.Tensor:
        """Twin of apply_field_loss: per-pixel blend of two pyramid levels."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        H, W = x.shape[-2:]
        params = (severity, field_type, overlay_scale, use_overlay_asset,
                  mask_center, mask_radius)

        def build():
            lo, hi, frac = _field_loss_maps(H, W, *params)
            used = tuple(int(lvl) for lvl in np.union1d(lo, hi))
            lo, hi = self.to_device(lo), self.to_device(hi)
            frac = self.to_device(frac, torch.float64)
            # Level 0 is where every sample starts, so only the blurred
            # levels need a mask.
            masks = tuple((lvl, lo == lvl, hi == lvl) for lvl in used if lvl != 0)
            return masks, frac, 1 - frac
        masks, frac, keep = self.cached("field_loss_maps", params, (H, W), build)
        # _build_mip_pyramid: level 0 is the float32 input, level l its PIL
        # blur with radius 2**(l-1).  Only the levels some pixel reads are built.
        levels = _u8(x)
        arr = _from_u8(levels, torch.float32)
        lo_sample = hi_sample = arr
        for lvl, lo_mask, hi_mask in masks:
            # pil_gaussian_blur(arr, ...), on the levels arr was made from.
            mip = _from_u8(_blur_levels(levels, float(2 ** (lvl - 1))), torch.float32)
            lo_sample = torch.where(lo_mask, mip, lo_sample)
            hi_sample = torch.where(hi_mask, mip, hi_sample)
        out = lo_sample * keep + hi_sample * frac           # float64, as on the CPU
        return _from_u8(_u8(out), x.dtype)

    # ---- distortion: one gather through the CPU's index maps ----------------
    # distortion and wiggle read a uint8 image on the CPU, so they round with
    # quantize_u8 as well: a no-op on frames_to_gpu output.
    def distortion(self, x: torch.Tensor, severity: float = 0.5,
                   warp_centers=None) -> torch.Tensor:
        """Twin of apply_distortion: one gather through the CPU's index maps."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        H, W = x.shape[-2:]
        centres = _distortion_centres(severity, warp_centers)
        lin = self.cached("distortion", centres, (H, W),
                          lambda: self._flat_map(_distortion_maps(H, W, *centres)))
        return quantize_u8(_gather_flat(x, lin))

    # ---- double_vision: mean of two whole-pixel shifts ----------------------
    def double_vision(self, x: torch.Tensor, severity: float = 0.5,
                      axis: str = "horizontal") -> torch.Tensor:
        """Twin of apply_double_vision: mean of two whole-pixel shifts."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        H, W = x.shape[-2:]
        disp = severity * 0.025

        if axis == "horizontal":
            dx, dy = disp, 0.0
        elif axis == "vertical":
            dx, dy = 0.0, disp
        elif axis == "diagonal":
            s = disp / float(np.sqrt(2.0))
            dx, dy = s, s
        else:
            raise ValueError(f"Unknown axis: {axis}")

        px = int(round(dx * (W - 1)))
        py = int(round(dy * (H - 1)))
        # On 8-bit levels, as the CPU's float32 copy of the uint8 image: the
        # sum of two levels and its half are exact, so rounding half to even
        # gives _to_uint8(out, 1.0), .5 ties included (in [0,1] floats the
        # ties land 1 ulp either side of .5).
        levels = _u8(x).to(x.dtype)
        out = 0.5 * (_shifted(levels, -px, -py) + _shifted(levels, px, py))
        return quantize_u8(out, scale=1.0)

    # ---- detail_loss: posterize ---------------------------------------------
    def detail_loss(self, x: torch.Tensor, severity: float = 0.5) -> torch.Tensor:
        """Twin of apply_detail_loss: ImageOps.posterize keeps the top `bits`
        bits of each 8-bit level (it truncates, it does not round)."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        # 0.0 → 8 bits (no change), 0.5 → 4 bits (visible banding), 1.0 → 1 bit
        bits = int(round(8 - severity * 7))
        bits = max(1, min(8, bits))
        mask = ~(2 ** (8 - bits) - 1)                       # posterize's LUT is i & mask
        return _from_u8(_u8(x) & mask, x.dtype)

    # ---- pixelation: PIL resize down, then up -------------------------------
    def pixelation(self, x: torch.Tensor, severity: float = 0.5,
                   pixel_size: float = None, hard: bool = False,
                   sampling: str = "average", grid_offset_x: float = 0.0,
                   grid_offset_y: float = 0.0) -> torch.Tensor:
        """Twin of apply_pixelation (bit-exact; tests/test_gpu_resample.py).

        x is treated as an 8-bit image, as pil_resize does.  The grid offsets
        shift the frame with np.pad(..., mode="edge") semantics, which is
        gather_nearest's edge clamp.
        """
        H, W = x.shape[-2:]
        grid = _pixelation_grid(W, H, severity, pixel_size, hard,
                                grid_offset_x, grid_offset_y)
        if grid is None:
            return x
        cells_x, cells_y, dx, dy = grid
        if dx or dy:
            def shifts():
                ry, rx = (torch.arange(n, device=self.device) for n in (H, W))
                return (_flat_index((ry - dy)[:, None], (rx - dx)[None, :], H, W),
                        _flat_index((ry + dy)[:, None], (rx + dx)[None, :], H, W))
            lin_in, lin_back = self.cached("pixelation_shift", (dx, dy), (H, W), shifts)
        # On 8-bit levels throughout (gathers only copy, and the small image
        # is never needed as floats: _u8(_from_u8(t)) is t).
        t = _u8(x)
        if dx or dy:
            t = _gather_flat(t, lin_in)
        down = "nearest" if sampling == "point" else "bilinear"
        if hard:
            small = self._resize_levels(t, cells_x, cells_y, down)
            if small is t or down == "nearest":
                # Not rounded by a BILINEAR pass: clamp as _u8(_from_u8(small))
                # would: a no-op on levels 0..255, and NaN input keeps its old result.
                small = small.clamp(0, 255)
            out = self._resize_levels(small, W, H, "nearest")  # HARD square blocks
        else:
            # Same cell count on both axes, as on the CPU, then bilinear up.
            small = self._resize_levels(t, cells_x, cells_x, down)
            if small is t or down == "nearest":
                # Not rounded by a BILINEAR pass: clamp as _u8(_from_u8(small))
                # would: a no-op on levels 0..255, and NaN input keeps its old result.
                small = small.clamp(0, 255)
            out = self._resize_levels(small, W, H)
        if dx or dy:  # shift the content back; the grid keeps the offset
            out = _gather_flat(out, lin_back)
        return _from_u8(out, x.dtype)

    # ---- led: cell texels times the LED mask, in linear light ---------------
    def led(self, x: torch.Tensor, severity: float = 0.5, scale: float = None,
            brightness: float = 1.0, shape: float = 1.5, margin: float = 0.0,
            automatic_ratio: bool = True) -> torch.Tensor:
        """Twin of apply_led (bit-exact; tests/test_gpu_resample.py).

        Per frame: gather each cell's texel level, look up
        _srgb_to_linear(level/255 * brightness) in a numpy-built table,
        multiply by the numpy-built mask and clamp, then convert back with
        _linear_to_srgb_u8.  No pow runs on the device, and NaN/inf inputs
        follow numpy's arithmetic.  x is treated as an 8-bit image.
        """
        severity = clamp_severity(severity)
        if severity == 0.0 and scale is None:
            return x
        H, W = x.shape[-2:]
        if scale is None:
            # Map severity (subtle->strong) to scale (cells-across): 200 -> 20.
            scale = float(200.0 - severity * 180.0)

        def fields():
            sy, sx, mask, keep = _led_fields(H, W, scale, shape, margin,
                                             automatic_ratio)
            return (_flat_index(self.to_device(sy)[:, None],
                                self.to_device(sx)[None, :], H, W),
                    torch.from_numpy(mask).to(self.device),       # numpy's dtype
                    None if keep is None else self.to_device(keep))
        lin, mask, keep = self.cached(
            "led_fields", (scale, shape, margin, automatic_ratio), (H, W), fields)
        lut = self.cached("led_lut", (brightness,), (),
                          lambda: torch.from_numpy(_led_lut(brightness)).to(self.device))
        level = _gather_flat(_u8(x), lin)
        # Linear light, clipped, in numpy's promoted dtype of lut and mask.
        v = (lut[level] * mask).clamp(0.0, 1.0)
        out = self._linear_to_srgb_u8(v)
        if keep is not None:  # optional margin (black border); NaN * 0 stays NaN
            out = out * (keep | v.isnan())
        return _from_u8(out, x.dtype)

    # ---- vortex: bilinear samples of the pixels inside the field ------------
    def vortex(self, x: torch.Tensor, severity: float = 0.5,
               center=(0.5, 0.5), vortex_radius: float = None,
               radius_y: float = None, radius_y_down: float = None,
               suction_strength: float = None, falloff_exponent: float = 2.0,
               swirl_strength: float = 0.0, inner_circle_radius: float = 0.01,
               inner_color: float = 0.5, inner_center=None,
               inner_fade: float = None, inner_opacity: float = 1.0,
               halo_radius: float = 0.0, halo_center=None,
               halo_thickness: float = 0.012, halo_color: float = 1.0,
               halo_opacity: float = 0.5, noise_amount: float = 0.05,
               noise_scale: float = 10.0,
               fade_width: float = 0.05) -> torch.Tensor:
        """Twin of apply_vortex (same keywords), bit-exact.

        The CPU blends in float64 (its sample weights are float64), so the
        pixels inside the field are blended in float64 here too.  The fade
        and dot weights keep the field's numpy dtype, so `arr * s_out` stays
        float32 where the CPU's does (a numpy float32 radius_y makes the
        whole field float32).
        """
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        N, C, H, W = x.shape
        params = dict(
            severity=severity, center=center, vortex_radius=vortex_radius,
            radius_y=radius_y, radius_y_down=radius_y_down,
            suction_strength=suction_strength, falloff_exponent=falloff_exponent,
            swirl_strength=swirl_strength, inner_circle_radius=inner_circle_radius,
            inner_color=inner_color, inner_center=inner_center,
            inner_fade=inner_fade, inner_opacity=inner_opacity,
            halo_radius=halo_radius, halo_center=halo_center,
            halo_thickness=halo_thickness, halo_color=halo_color,
            halo_opacity=halo_opacity, noise_amount=noise_amount,
            noise_scale=noise_scale, fade_width=fade_width)

        def build():
            f = _vortex_fields(H, W, **params)
            # fy, fx go up as float64: the CPU's wx = fx - x0 is float32 minus
            # int32, i.e. float64.  The weights keep their numpy dtype, and
            # torch promotes them with the pixels exactly as numpy does.
            if f is None:
                return None
            f = {k: None if a is None else
                 self.to_device(a, torch.long if a.dtype.kind == "i" else
                                torch.float64 if k in ("fy", "fx") else
                                _torch_dtype(a.dtype))
                 for k, a in f.items()}
            f["plan"] = _bilinear_plan(f["fy"], f["fx"], H, W)
            return f
        f = self.cached("vortex", params, (H, W), build)
        if f is None:                                # no pixel inside the field
            return x
        levels = _u8(x)                              # the CPU's uint8 image
        arr = _from_u8(levels, torch.float32)        # np.float32(img) / 255
        src = arr.flatten(2).index_select(2, f["idx"])
        vortexed = _gather_bilinear_plan(arr, f["plan"]).flatten(2)
        blended = vortexed * f["keep_out"] + src * f["s_out"]
        if f["inner_add"] is not None:
            blended = f["inner_add"] + blended * f["inner_keep"]
        if f["halo_add"] is not None:
            blended = f["halo_add"] + blended * f["halo_keep"]
        # Pixels outside the field keep their level, as _to_uint8(arr) does.
        levels = levels.flatten(2).index_copy(2, f["idx"], _u8(blended))
        return _from_u8(levels.view(N, C, H, W), x.dtype)

    # ---- wiggle: one gather through the CPU's index maps --------------------
    def wiggle(self, x: torch.Tensor, severity: float = 0.5, freq: float = 12.0,
               amplitude: float = 0.01, timer: float = 0.0,
               mode: str = "complex") -> torch.Tensor:
        """Twin of apply_wiggle: one gather through the CPU's index maps."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        H, W = x.shape[-2:]
        lin = self.cached(
            "wiggle", (severity, freq, amplitude, timer, mode), (H, W),
            lambda: self._flat_map(_wiggle_maps(H, W, severity, freq, amplitude,
                                                timer, mode)))
        return quantize_u8(_gather_flat(x, lin))

    # ---- noise: the CPU's float32 noise field, added ------------------------
    def noise(self, x: torch.Tensor, severity: float = 0.5,
              seed: int = 0) -> torch.Tensor:
        """Twin of apply_noise (bit-exact)."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        H, W = x.shape[-2:]
        n = self._random_field("noise", (seed,), (H, W), seed, x.shape[0],
                               lambda: _noise_field(H, W, seed),
                               lambda a: self.to_device(a, torch.float32))
        # arr + n * intensity in float32, like numpy with a Python float.
        out = _from_u8(_u8(x), torch.float32) + n * (severity * 0.35)
        return _from_u8(_u8(out), x.dtype)

    # ---- glitch: per-row R/G/B rolls as one gather --------------------------
    def glitch(self, x: torch.Tensor, severity: float = 0.5,
               max_displace: float = 0.05, seed: int = 0) -> torch.Tensor:
        """Twin of apply_glitch: the per-row R/G/B rolls as one gather (bit-exact)."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        H, W = x.shape[-2:]
        cols = self._random_field(
            "glitch", (severity, max_displace, seed), (H, W), seed, x.shape[0],
            lambda: _glitch_columns(H, W, severity, max_displace, seed))
        return quantize_u8(x.gather(-1, cols.expand_as(x)))

    # ---- teichopsia: HSV boost and scintillation inside the ring ------------
    def teichopsia(self, x: torch.Tensor, severity: float = 0.75,
                   lum_contribution: float = 0.75, gaze=(0.5, 0.5),
                   seed: int = 0, sat_boost: float = 2.0) -> torch.Tensor:
        """Twin of apply_teichopsia (same keywords), bit-exact.

        Ring mask and scintillation noise come from the CPU's seeded numpy
        code; the saturation boost is skimage's HSV round trip in float32.
        Building the mask needs scikit-image, like the CPU filter.
        """
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        N, C, H, W = x.shape

        def build():
            idx, mult = _teichopsia_fields(H, W, gaze, seed)
            return self.to_device(idx), self.to_device(mult, torch.float32)
        idx, mult = self.cached("teichopsia", (gaze, seed), (H, W), build)
        if idx.numel() == 0:                         # ring misses the screen
            return x
        levels = _u8(x).flatten(2)                   # the CPU's uint8 image
        hsv = skimage_rgb2hsv(_from_u8(levels.index_select(2, idx), torch.float32))
        # A numpy float64 sat_boost or lum_contribution makes the CPU's
        # product float64 (_scalar_product_dtype); the result is still stored
        # as float32, in hsv here and in `out` below.
        sat = hsv[:, 1].to(_scalar_product_dtype(sat_boost))
        hsv[:, 1] = (sat * float(sat_boost)).clamp(0.0, 1.0)
        boosted = skimage_hsv2rgb(hsv)

        lum = 0.299 * boosted[:, 0] + 0.587 * boosted[:, 1] + 0.114 * boosted[:, 2]
        lum = lum.to(_scalar_product_dtype(lum_contribution))
        strength = (severity * (1.0 - lum * float(lum_contribution))).unsqueeze(1)
        noisy = boosted * mult
        scint = boosted * (1.0 - strength) + noisy * strength
        levels = levels.index_copy(2, idx, _u8(scint.float()))
        return _from_u8(levels.view(N, C, H, W), x.dtype)

    # ---- foveal_darkness: blend toward dark, gray or white ------------------
    def foveal_darkness(self, x: torch.Tensor, severity: float = 0.5,
                        center=(0.5, 0.5), radius: float = None,
                        radius_x: float = None, radius_y: float = None,
                        edge_width: float = None, mode: str = "dark",
                        invert: bool = False,
                        band_width: float = None) -> torch.Tensor:
        """Twin of apply_foveal_darkness (same keywords), bit-exact.

        Like the CPU it works on the 0..255 levels in float32 and rounds with
        _to_uint8(out, 1.0).
        """
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        H, W = x.shape[-2:]
        params = dict(severity=severity, center=center, radius=radius,
                      radius_x=radius_x, radius_y=radius_y,
                      edge_width=edge_width, invert=invert,
                      band_width=band_width)

        def build():
            # darken keeps its numpy dtype: float32, or float64 when a keyword
            # is a numpy float64 (e.g. a centre taken from a gaze array).  The
            # CPU then blends in float64, and so does torch below.
            darken = _foveal_darken(H, W, **params)
            return self.to_device(darken, _torch_dtype(darken.dtype))
        darken = self.cached("foveal_darkness", params, (H, W), build)
        arr = _u8(x).float()                         # the CPU's float32 0..255 array
        if mode == "desat":
            # arr @ [0.299, 0.587, 0.114] in float32, summed as numpy's BLAS
            # sums it (sdot on a 1-pixel-wide image; see _channel_dot).
            luma = self.numpy_field("bt601_weights", (), (),
                                    lambda: _BT601_WEIGHTS, torch.float32)
            lum = _channel_dot(arr, luma, sdot=W == 1)
            out = arr + (lum - arr) * darken
        elif mode == "light":
            out = arr + (255.0 - arr) * darken
        else:
            out = arr * (1.0 - darken)
        return _from_u8(_u8(out, 1.0), x.dtype)

    # ---- nystagmus: mean of 7 wrapped shifts --------------------------------
    def nystagmus(self, x: torch.Tensor, severity: float = 0.5,
                  axis: str = "horizontal") -> torch.Tensor:
        """Twin of apply_nystagmus: mean of 7 wrapped shifts (np.roll)."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        arr = _from_u8(_u8(x), torch.float32)        # the CPU's uint8 / 255.0
        H, W = x.shape[-2:]
        amp = severity * 0.04
        n_samples = 7
        offsets = np.linspace(-amp, amp, n_samples)

        # float32 sums in the CPU's order, so every rounding is the same.
        accum = torch.zeros_like(arr)
        for o in offsets:
            if axis == "horizontal":
                shift = int(round(o * (W - 1)))
                sh = torch.roll(arr, shift, dims=-1)
            elif axis == "vertical":
                shift = int(round(o * (H - 1)))
                sh = torch.roll(arr, shift, dims=-2)
            else:  # both
                sx = int(round(o * (W - 1)))
                sy = int(round(o * (H - 1)))
                sh = torch.roll(arr, (sy, sx), dims=(-2, -1))
            accum += sh
        return _from_u8(_u8(exact_div(accum, n_samples)), x.dtype)

    # ---- flickering_stars: max(level, star level) ---------------------------
    def flickering_stars(self, x: torch.Tensor, severity: float = 0.5,
                         n_stars: int = None, seed: int = 0) -> torch.Tensor:
        """Twin of apply_flickering_stars: max(level, star level) (bit-exact)."""
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        H, W = x.shape[-2:]
        stars = self._random_field(
            "flickering_stars", (severity, n_stars, seed), (H, W), seed, x.shape[0],
            lambda: _star_levels(H, W, severity, n_stars, seed))
        return _from_u8(torch.maximum(_u8(x), stars), x.dtype)

    # ---- floaters: one [blends, level] table lookup -------------------------
    def floaters(self, x: torch.Tensor, severity: float = 0.5,
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
                 region_center_bias: float = 0.0) -> torch.Tensor:
        """Twin of apply_floaters, every distribution and region (bit-exact).

        A pixel's output depends only on its input level and on how many
        times the CPU blends it, so the call is one lookup in a
        [blends, level] table built with the CPU's float32 blend.
        """
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        if n_floaters is None:
            n_floaters = int(severity * 50000)
        n_floaters = int(max(0, n_floaters))
        if n_floaters == 0:
            return x
        H, W = x.shape[-2:]
        # Everything that places the dots; severity only enters via n_floaters.
        geometry = dict(n_floaters=n_floaters, floater_size=floater_size,
                        circle_radius_uv=circle_radius_uv, seed=seed,
                        distribution=distribution, center=center, spread=spread,
                        spread_x=spread_x, spread_y=spread_y,
                        size_variance=size_variance, angle=angle, region=region,
                        region_feather=region_feather,
                        region_center_bias=region_center_bias)
        # The largest count is read on the host, so lookups never sync.
        counts, max_count = self._random_field(
            "floaters", geometry, (H, W), seed, x.shape[0],
            lambda: _floater_counts(H, W, **geometry),
            lambda c: (self.to_device(c), int(c.max())))
        if dot_brightness is not None:
            target_val = float(np.clip(dot_brightness, 0.0, 1.0)) * 255.0
        else:
            target_val = 0.0 if mode == "dark" else 255.0
        levels = self.cached(
            "floaters_levels", (severity, target_val, max_count), (),
            lambda: self.to_device(_floater_levels(severity, target_val, max_count)))
        return _from_u8(levels.view(-1)[counts * levels.shape[1] + _u8(x)], x.dtype)

    # ---- vignette: radial darkening or veiling glare ------------------------
    def vignette(self, x: torch.Tensor, severity: float = 0.5,
                 strength: float | None = None, radius: float = 0.5,
                 softness: float = 0.5, mode: str = "dark") -> torch.Tensor:
        """Twin of apply_vignette: radial darkening ('dark') or a blend toward
        white ('light'); the falloff comes from the CPU's numpy code."""
        severity = clamp_severity(severity)
        if strength is None:
            strength = severity
        strength = clamp_severity(strength)                 # same clip as severity
        if strength == 0.0:
            return x
        if mode not in ("light", "dark"):
            raise ValueError(f"Unknown vignette mode: {mode}")
        H, W = x.shape[-2:]
        # A NumPy float64 radius or softness makes the CPU's ramp float64 from
        # that step on (NEP 50); _freeze keys it apart from the equal Python
        # float, and the ramp keeps numpy's dtype.
        ramp = self.cached(
            "vignette_ramp", (radius, softness), (H, W),
            lambda: torch.from_numpy(_vignette_ramp(H, W, radius, softness)).to(self.device))
        s = ramp * strength                                 # smoothstep * strength
        out = _from_u8(_u8(x), torch.float32) * (1.0 - s)
        if mode == "light":
            # Blend toward white at the periphery (bright veiling fog).
            out = out + s
        return _from_u8(_u8(out), x.dtype)

    # ---- Chains: presets and profiles ---------------------------------------
    def apply_chain(self, x: torch.Tensor, chain) -> torch.Tensor:
        """Run [(method_name, kwargs), ...] in order, like apply_preset's loop.

        Each filter runs on the whole batch before the next one: see
        _random_field for what that means for a shared numpy Generator.
        """
        for name, kwargs in chain:
            x = getattr(self, name)(x, **kwargs)
        return x

    def apply_preset(self, x: torch.Tensor, preset_name: str) -> torch.Tensor:
        """Twin of apply_preset: the named chain of filters.PRESETS."""
        return self.apply_chain(x, PRESETS[preset_name]["filters"])

    def apply_profile(self, x: torch.Tensor, profile) -> torch.Tensor:
        """Run a vipsim_profiles_v2 profile's filter stack (see profile_chain),
        as validation_scripts/render_profiles_v2.apply_stack does on the CPU."""
        return self.apply_chain(x, profile_chain(profile))


# -----------------------------------------------------------------------------
# Profiles (vipsim_profiles_v2) and the parity contract
# -----------------------------------------------------------------------------
def profile_chain(profile) -> list:
    """A vipsim_profiles_v2 profile as a chain of (method_name, kwargs).

    profile: the parsed pN.json, or its "filters" list.  It is read the way
    validation_scripts/render_profiles_v2.apply_stack reads it: steps with
    "_disabled" set are skipped, "filter" names the filter and keys starting
    with "_" are notes; a "severity" the filter does not take (bcg) is
    dropped, where apply_stack retries the call without it.  The chain is
    what VipSimGPU.apply_chain and VisualImpairmentWrapper take.

    Invalid steps fail like apply_stack's: an unknown filter name raises
    KeyError.  An invalid severity (None, a string, a list) raises here,
    where apply_stack's retry on any TypeError renders the filter at its
    default severity instead.
    """
    steps = profile["filters"] if isinstance(profile, dict) else profile
    chain = []
    for step in steps:
        if step.get("_disabled"):
            continue
        kwargs = {k: v for k, v in step.items()
                  if k != "filter" and not k.startswith("_")}
        if step["filter"] not in _FILTER_DISPATCH:  # as apply_stack's lookup
            raise KeyError(step["filter"])
        method = getattr(VipSimGPU, step["filter"])
        if "severity" not in inspect.signature(method).parameters:
            kwargs.pop("severity", None)
        chain.append((step["filter"], kwargs))
    return chain


# Tolerance of each twin against its CPU filter: (max, mean) of |gpu - cpu| in
# 8-bit levels over an image, for frames_to_gpu input.  Every twin is
# bit-exact where tested (numpy 2.5 + OpenBLAS 0.3.34 on x86-64, Pillow 12.2
# and 13.0, CUDA and CPU torch).  cvd, bcg (saturation != 1) and
# foveal_darkness (mode="desat") mix the channels in the order numpy's BLAS
# sums a 3-element dot (_channel_dot); a BLAS build that fuses multiply-adds
# rounds a few of those sums otherwise, so they keep GpuParityCase's bound.
TWIN_TOLERANCE = {name: (1, 0.02) if name in ("cvd", "bcg", "foveal_darkness")
                  else (0, 0.0) for name in _FILTER_DISPATCH}


# -----------------------------------------------------------------------------
# Conversion helpers: MuJoCo renderer (HxWx3 uint8 numpy) <-> GPU tensor
# -----------------------------------------------------------------------------
def frames_to_gpu(frames: np.ndarray, device) -> torch.Tensor:
    """frames: [N,H,W,3] or [H,W,3] uint8  ->  [N,3,H,W] float32 in [0,1]."""
    if frames.dtype != np.uint8:
        raise TypeError(f"frames must be uint8, got {frames.dtype}")
    if frames.ndim not in (3, 4) or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [N,H,W,3] or [H,W,3], got {frames.shape}")
    if frames.ndim == 3:
        frames = frames[None]
    if not (frames.flags.writeable and frames.flags.c_contiguous):
        # torch.from_numpy warns on read-only memory (np.asarray(PIL image),
        # some renderer buffers): copy those once, share everything else.
        frames = np.array(frames, order="C")
    t = torch.from_numpy(frames).to(device)
    # Same floats as the CPU's np.asarray(img, np.float32) / 255.0.
    return _from_u8(t.permute(0, 3, 1, 2), torch.float32)


def gpu_to_frames(x: torch.Tensor) -> np.ndarray:
    """[N,3,H,W] float [0,1]  ->  [N,H,W,3] uint8 (back on CPU)."""
    x = (x.clamp(0, 1) * 255).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous().cpu().numpy()


# -----------------------------------------------------------------------------
# MyoSuite / Gym integration (Path A: standard renderer -> GPU -> filters)
# -----------------------------------------------------------------------------
try:
    import gymnasium as gym
    _Wrapper = gym.ObservationWrapper
except Exception:                                   # fall back to classic gym
    try:
        import gym
        _Wrapper = gym.ObservationWrapper
    except Exception:
        _Wrapper = object


class VisualImpairmentWrapper(_Wrapper):
    """Applies a chain of GPU filters to the image part of the observation.

    Works with any env whose observation is an HxWx3 uint8 image (or a dict
    containing one).  MyoSuite vision envs render on CPU, so there is exactly
    one host->device copy per step here; the filtering itself is on GPU.

    chain: list of (method_name, kwargs), e.g.
        [("cvd", {"severity": 0.8}), ("blur", {"severity": 0.4})]
    Every filter of vipsim.filters is a VipSimGPU method with the CPU's name
    and keyword arguments, and the output matches the CPU chain (see
    TWIN_TOLERANCE).  A preset's chain is filters.PRESETS[name]["filters"];
    profile_chain(json.load(f)) turns a vipsim_profiles_v2 profile into one.
    """

    def __init__(self, env, chain, device="cuda", image_key=None):
        if _Wrapper is object:
            raise ImportError("VisualImpairmentWrapper requires gymnasium: "
                              "pip install 'vipsim[gym]'")
        super().__init__(env)
        self.sim = VipSimGPU(device=device)
        self.chain = chain
        self.image_key = image_key          # set if obs is a dict

    def _filter(self, frame: np.ndarray) -> np.ndarray:
        x = self.sim.apply_chain(frames_to_gpu(frame, self.sim.device), self.chain)
        out = gpu_to_frames(x)
        return out[0] if frame.ndim == 3 else out

    def observation(self, obs):
        if self.image_key is not None:
            obs = dict(obs)
            obs[self.image_key] = self._filter(obs[self.image_key])
            return obs
        return self._filter(obs)


# -----------------------------------------------------------------------------
# Porting notes — which primitive each filter uses
# -----------------------------------------------------------------------------
# Recipe: severity = clamp_severity(severity); build every image-independent
# array with self.numpy_field(...) by calling/copying the CPU numpy code
# verbatim (same dtypes, same RNG); read x as its 8-bit levels (_u8), as the
# CPU only ever gets a uint8 image; do the per-frame work with torch ops in
# the CPU's order; end where the CPU calls _to_uint8 (quantize_u8, or
# _from_u8 of integer levels).  Check against the CPU with
# np.testing.assert_array_equal on frames_to_gpu/gpu_to_frames round trips
# (tests/test_gpu_*.py), or GpuParityCase.assert_close (tests/test_rounding.py:
# max <= 1 LSB, mean < 0.02) where TWIN_TOLERANCE allows it.
#
# Where bit-equality is lost, and how to keep it:
#   * dtypes follow numpy: float32 array op Python float stays float32, but
#     float32 - int32 (or any float64 array, or a numpy float64 scalar under
#     NEP 50) promotes to float64; torch reads a numpy scalar as a Python
#     number, so mirror the CPU dtype of each array, and key cached fields by
#     type (_freeze does).
#   * on CUDA, `t / 2.0` is t * (1/2.0); use exact_div where numpy divides.
#   * pow/exp/sin/arctan2 differ from numpy by ~1 ulp on both devices: run
#     them on the host (numpy_field), or tabulate them over the 256 levels.
#   * 3-element dots: numpy's BLAS sums left to right without FMA (sdot in
#     float64 on 1-pixel-wide images); einsum/matmul reorder (_channel_dot).
#   * every Pillow pass rounds to 8 bits, in the installed version's order.
#   * don't call grid_sample or conv2d for CPU twins (coordinate rounding
#     differs from numpy's; TF32); use the primitives above.
#
# per-level tables / channel mixes (bucket 1):
#   cvd: sRGB->linear table, the float64 Machado matrix by _channel_dot,
#     linear->sRGB levels by bucketize over _srgb_u8_steps
#   bcg: tone table of the 256 levels (a uint8 LUT when saturation == 1),
#     BT.709 luma by _channel_dot
#   detail_loss: posterize truncates, level & ~(2**(8 - bits) - 1)
#   vignette: the CPU's float32 ramp * strength, blended on k/255
#   noise: seeded float32 noise field, float32 multiply then add
#   flickering_stars: max(level, star-level field)
#   floaters: [blend count, level] table iterated with the CPU's float32 blend
#   foveal_darkness: the CPU's darken weight on 0..255 float32 levels
#   teichopsia: ring pixels and noise from the CPU's seeded code, then
#     skimage_rgb2hsv / skimage_hsv2rgb
# blur/resize (bucket 2):
#   blur: pil_gaussian_blur(severity * 15)
#   bloom: pil_resize(BILINEAR) down by 4 -> threshold -> pil_gaussian_blur
#     -> pil_resize up -> add in linear light, 8-bit between steps (a
#     [256, 256] composite table)
#   field_loss: pil_gaussian_blur per used mip level (radius 2**(lvl-1)),
#     blended by the numpy (lo, hi, frac) maps
#   cataracts: per-channel BCG table, then the frost gather (bucket 3)
#   pixelation: pil_resize down (BILINEAR or NEAREST) then up, with
#     edge-padded grid-offset shifts by gather_nearest
#   led: cell texels by gather_nearest, sRGB->linear table * numpy mask,
#     linear->sRGB levels as for cvd
# gathers (bucket 3):
#   distortion, wiggle, cataract frost: numpy_field returns the (sy, sx) maps
#     from the CPU expression `(src_u * (w - 1)).astype(np.int32)`
#     -> gather_nearest
#   vortex: float64 pixel coords of the pixels inside the field
#     -> gather_bilinear, blended in float64
#   double_vision: two whole-pixel shifts (slices); nystagmus: 7 torch.rolls;
#     glitch: one per-row, per-channel column gather


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sim = VipSimGPU(device=dev)
    x = torch.rand(8, 3, 256, 256, device=dev)     # batch of 8 fake frames
    for fn, kw in [("cvd", {"severity": 0.8}), ("bcg", {"brightness": 0.8, "contrast": 0.7}),
                   ("blur", {"severity": 0.4}), ("vortex", {"severity": 0.5})]:
        y = getattr(sim, fn)(x, **kw)
        print(f"{fn:8s} -> {tuple(y.shape)}  range[{y.min():.3f},{y.max():.3f}]")
