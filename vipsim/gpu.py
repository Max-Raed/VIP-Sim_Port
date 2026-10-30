#!/usr/bin/env python3
"""
vipsim.gpu — GPU (PyTorch) port of the VIP-Sim visual-impairment filters.

Design goals
------------
* Operate on BATCHED, channels-first tensors  x: [N, 3, H, W], float32 in [0,1],
  living on the GPU.  N = number of parallel envs (or frame stack).
* The NumPy/PIL filters in vipsim/filters.py are the REFERENCE.  A GPU filter
  reproduces the CPU filter as it is today, quirks included (nearest-index
  truncation, 8-bit intermediates, PIL's integer arithmetic); CPU bugs are
  fixed in filters.py first, never only here.
* No GLSL/CUDA.  Every original filter reduces to one of three primitives:
      1. elementwise / 3x3 matmul   -> einsum + arithmetic   (cvd, bcg, ...)
      2. separable blur / resize    -> pil_gaussian_blur, pil_resize
      3. per-pixel UV displacement  -> gather_nearest, gather_bilinear
* Anything that does NOT depend on the image (warp fields, index maps, masks,
  noise tiles) is computed ONCE with the CPU's own numpy code and kept on the
  device in a bounded LRU cache (VipSimGPU.cached / VipSimGPU.numpy_field).

Shared primitives (the API the filter ports build on)
-----------------------------------------------------
  clamp_severity(s)                      float(np.clip(s, 0, 1)), as every apply_* does
  quantize_u8(x, scale=255.0)            filters._to_uint8, result back in [0,1]
  exact_div(x, d)                        numpy's x / d (CUDA's x / float is x * (1/d))
  pil_box_radius(radius, passes=3)       Pillow's _gaussian_blur_radius (float32)
  pil_gaussian_blur(x, radius, quantize) img.filter(ImageFilter.GaussianBlur(radius))
  gather_nearest(x, iy, ix)              arr[sy, sx] with integer index maps
  gather_bilinear(x, fy, fx)             the 4-tap blend of apply_vortex
  VipSimGPU.pil_resize(x, width, height, resample, quantize)
                                         img.resize((width, height), BILINEAR/NEAREST)
  VipSimGPU.cached(name, params, hw, build)
  VipSimGPU.numpy_field(name, params, hw, fn, dtype)
  VipSimGPU.to_device(array, dtype)

Blurs and resizes deliberately avoid conv2d: cuDNN runs float32 convolutions
in TF32 by default on Ampere+ GPUs (torch.backends.cudnn.allow_tf32), which
keeps ~3 decimal digits, and float32 cannot hold PIL's 24-bit fixed-point
products anyway.  They replay PIL's integer arithmetic with index gathers and
cumulative sums in int32/int64 instead, which is exact.

This file ports a representative filter from each of the three buckets
(cvd, bcg, blur, and a grid_sample warp) plus the MyoSuite hook.  The
remaining ~15 filters in vipsim/filters.py follow the exact same patterns —
see the porting notes at the bottom.
"""

from __future__ import annotations
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from .filters import CVD_MATRICES


# -----------------------------------------------------------------------------
# sRGB <-> linear  (same round-trip as the NumPy version, just torch ops)
# -----------------------------------------------------------------------------
def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


# -----------------------------------------------------------------------------
# Severity and 8-bit quantisation
# -----------------------------------------------------------------------------
def clamp_severity(severity) -> float:
    """float(np.clip(severity, 0, 1)) — the first line of every CPU apply_*."""
    return float(np.clip(severity, 0.0, 1.0))


def exact_div(x: torch.Tensor, divisor: float) -> torch.Tensor:
    """x / divisor, correctly rounded like numpy's division.

    On CUDA, `tensor / python_float` is computed as tensor * (1/divisor),
    which is 1 ulp off for about half of k/255; a device-tensor divisor
    takes the true-division kernel.
    """
    return x / torch.tensor(divisor, dtype=x.dtype, device=x.device)


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


def _box_blur_pass(t: torch.Tensor, r: int, ww: int, fw: int,
                   dim: int) -> torch.Tensor:
    """One extended-box pass along `dim` (replicated edges).

    int32 levels: Pillow's fixed-point pass, bit-exact (window sums in
    int32, the 24-bit products in int64).  float64: the same weights
    without the 8-bit rounding.  Window sums come from a cumulative sum, so
    the cost does not grow with the radius.
    """
    n = t.shape[dim]
    idx = torch.arange(-r - 1, n + r + 1, device=t.device).clamp_(0, n - 1)
    p = t.index_select(dim, idx)                     # p[k] = in[clamp(k - r - 1)]
    cs = p.cumsum(dim, dtype=p.dtype)
    acc = cs.narrow(dim, 2 * r + 1, n) - cs.narrow(dim, 0, n)   # in[x-r..x+r]
    edges = p.narrow(dim, 0, n) + p.narrow(dim, 2 * r + 2, n)  # in[x-r-1], in[x+r+1]
    if t.is_floating_point():
        return (acc * ww + edges * fw) / (1 << 24)
    bulk = acc.long().mul_(ww).add_(edges, alpha=fw)
    return bulk.add_(1 << 23).bitwise_right_shift_(24).int()


def pil_gaussian_blur(x: torch.Tensor, radius: float, quantize: bool = True,
                      passes: int = 3) -> torch.Tensor:
    """Twin of PIL `img.filter(ImageFilter.GaussianBlur(radius))`.

    x: [..., H, W] float in [0,1].  quantize=True treats x as an 8-bit image
    (rounded to k/255 first, as the CPU always hands PIL a uint8 image) and
    replays Pillow's integer arithmetic: bit-exact with PIL 12 and 13 (max
    difference 0, tests/test_gpu_primitives.py).  quantize=False runs the
    same box cascade in float64 without the per-pass rounding, for float data
    that never goes through a uint8 image; against PIL on 8-bit input it is
    within 1 LSB for radius >= 1.5 px, but up to 3 LSB on noise at sub-pixel
    radii, where PIL's six roundings outweigh the blur itself.
    """
    if radius == 0:                    # PIL returns a copy for radius 0
        return quantize_u8(x) if quantize else x
    r, ww, fw = _box_blur_weights(pil_box_radius(radius, passes))
    t = _u8(x).int() if quantize else x.double()
    for dim in (-1, -2):                             # all horizontal passes first
        for _ in range(passes):
            t = _box_blur_pass(t, r, ww, fw, dim)
    return _from_u8(t, x.dtype) if quantize else t.to(x.dtype)


# -----------------------------------------------------------------------------
# PIL resize twin (src/libImaging/Resample.c, Geometry.c)
# -----------------------------------------------------------------------------
# The tap tables only depend on (in_size, out_size), so VipSimGPU.pil_resize
# builds them once with these numpy functions and caches them on the device.

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


def _resample_pass(t: torch.Tensor, idx: torch.Tensor,
                   k: torch.Tensor) -> torch.Tensor:
    """One fixed-point resample pass along the last dim (int64 levels)."""
    v = (t[..., idx] * k).sum(-1) + (1 << 21)
    return (v >> 22).clamp_(0, 255)


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
    in-range indices.  Pure copies, so the result is exact.
    """
    N, C, H, W = x.shape
    lin = iy.long().clamp(0, H - 1) * W + ix.long().clamp(0, W - 1)
    flat = x.flatten(2)                              # [N, C, H*W]
    if lin.dim() == 2:
        return flat.index_select(2, lin.reshape(-1)).view(N, C, *lin.shape)
    lin = lin.expand(N, -1, -1)
    out = flat.gather(2, lin.reshape(N, 1, -1).expand(-1, C, -1))
    return out.view(N, C, *lin.shape[1:])


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
    top = gather_nearest(x, y0, x0) * (1.0 - wx) + gather_nearest(x, y0, x1) * wx
    bot = gather_nearest(x, y1, x0) * (1.0 - wx) + gather_nearest(x, y1, x1) * wx
    return top * (1.0 - wy) + bot * wy


# -----------------------------------------------------------------------------
# Bounded cache for image-independent tensors
# -----------------------------------------------------------------------------
def _freeze(params):
    """Make filter parameters hashable without rounding them: two severities
    that differ in the 4th decimal must not share a warp field."""
    if isinstance(params, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in params.items()))
    if isinstance(params, (list, tuple)):
        return tuple(_freeze(v) for v in params)
    if isinstance(params, np.ndarray):
        return (params.shape, params.dtype.str, params.tobytes())
    if isinstance(params, np.generic):
        return params.item()
    return params


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


class VipSimGPU:
    """Holds device + cached static tensors so per-frame work is minimal."""

    def __init__(self, device="cuda", dtype=torch.float32, cache_size: int = 64):
        self.device = torch.device(device)
        self.dtype = dtype
        # Image-independent tensors of every filter: warp fields, index maps,
        # masks, noise fields, resize taps.
        self.cache = TensorCache(cache_size)
        # Full Machado tables from the CPU module (float32, as before).
        self.cvd_tables = {name: torch.from_numpy(m.astype(np.float32)).to(self.device, dtype)
                           for name, m in CVD_MATRICES.items()}
        self.cvd_deuter = self.cvd_tables["deuteranomaly"]

    # ---- Cache helpers ------------------------------------------------------
    def cached(self, name: str, params, hw, build, dtype=None):
        """Return build() for this key, computed once and kept in self.cache.

        Key: (name, params, (H, W), device, dtype).  params is any nesting of
        numbers/strings/tuples/lists/dicts and is used exactly (no rounding).
        build() returns a tensor or a tuple of tensors already on the device.
        """
        key = (name, _freeze(params), tuple(hw), str(self.device),
               dtype or self.dtype)
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

    # ---- PIL resize twin ----------------------------------------------------
    def pil_resize(self, x: torch.Tensor, width: int, height: int,
                   resample: str = "bilinear", quantize: bool = True) -> torch.Tensor:
        """Twin of PIL `img.resize((width, height), Image.BILINEAR/NEAREST)`.

        Note PIL's (width, height) order.  x: [N, C, H, W] float in [0,1].
        quantize=True treats x as an 8-bit image (rounded to k/255 first) and
        replays Pillow's fixed-point resampler, horizontal pass then vertical
        pass, each rounded to 8 bits: bit-exact with PIL, including the
        antialiased (support-stretched) filter when downscaling.  NEAREST is
        exact either way.  quantize=False with BILINEAR uses
        F.interpolate(antialias=True), which computes the same filter in float
        without the intermediate rounding: measured max 1 LSB, mean <= 0.20 LSB
        after rounding (tests/test_gpu_primitives.py).
        """
        H, W = x.shape[-2:]
        if (width, height) == (W, H):                # PIL returns a copy
            return quantize_u8(x) if quantize else x
        if resample == "nearest":
            iy, ix = self.cached(
                "pil_resize_nearest", (width, height), (H, W),
                lambda: (self.to_device(_pil_nearest_index(H, height)),
                         self.to_device(_pil_nearest_index(W, width))))
            return gather_nearest(quantize_u8(x) if quantize else x,
                                  iy[:, None], ix[None, :])
        if resample != "bilinear":
            raise ValueError(f"Unsupported resample: {resample}")
        if not quantize:
            return F.interpolate(x, size=(height, width), mode="bilinear",
                                 align_corners=False, antialias=True)
        ix, kx, iy, ky = self.cached(
            "pil_resize_bilinear", (width, height), (H, W),
            lambda: tuple(self.to_device(a) for a in
                          _pil_bilinear_taps(W, width) + _pil_bilinear_taps(H, height)))
        t = _u8(x)
        if width != W:                               # Pillow skips unchanged axes
            t = _resample_pass(t, ix, kx)
        if height != H:
            t = _resample_pass(t.transpose(-1, -2), iy, ky).transpose(-1, -2)
        return _from_u8(t, x.dtype)

    # ---- CVD: interpolate the 3x3 matrix by severity, then einsum -----------
    def cvd(self, x: torch.Tensor, severity: float = 1.0,
            table: torch.Tensor | None = None) -> torch.Tensor:
        if severity <= 0:
            return x
        table = self.cvd_deuter if table is None else table
        n = table.shape[0]
        idx = float(np.clip(severity, 0, 1)) * (n - 1)
        i0 = int(np.floor(idx)); i1 = min(i0 + 1, n - 1); w = idx - i0
        mat = table[i0] * (1 - w) + table[i1] * w            # [3,3]
        # Linear light, as in Unity's Linear-colour-space myRecolour pass.
        # x:[N,3,H,W], mat:[3,3]  ->  out[N,i,H,W] = sum_j mat[i,j] x[N,j,H,W]
        lin = torch.einsum("ij,njhw->nihw", mat, srgb_to_linear(x))
        return linear_to_srgb(lin)

    # ---- BCG: linear-space brightness/contrast/gamma ------------------------
    def bcg(self, x: torch.Tensor, brightness=1.0, contrast=1.0, gamma=1.0,
            pivot=0.5) -> torch.Tensor:
        lin = srgb_to_linear(x) * brightness
        lin = (lin - pivot) * contrast + pivot
        lin = lin.clamp(0, 1)
        if gamma != 1.0:
            lin = lin ** gamma
        return linear_to_srgb(lin.clamp(0, 1))

    # ---- Bucket 2: Gaussian blur (twin of apply_blur) -----------------------
    def blur(self, x: torch.Tensor, severity: float = 0.5) -> torch.Tensor:
        severity = clamp_severity(severity)
        if severity == 0.0:
            return x
        # Same 0..15 px radius mapping as apply_blur, through PIL's box cascade.
        return pil_gaussian_blur(x, severity * 15.0)

    # ---- Bucket 3: per-pixel UV displacement via grid_sample ----------------
    # grid_sample expects a sampling grid in normalized [-1,1] coords,
    # shape [N,H,W,2] with (x,y) order.  Fine for new effects, but ports of CPU
    # filters use gather_nearest / gather_bilinear on numpy-built coordinate
    # maps instead: grid_sample's coordinate rounding is not numpy's.
    def _identity_grid(self, H: int, W: int) -> torch.Tensor:
        ys = torch.linspace(-1, 1, H, device=self.device, dtype=self.dtype)
        xs = torch.linspace(-1, 1, W, device=self.device, dtype=self.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((gx, gy), dim=-1).unsqueeze(0)    # [1,H,W,2]

    def _vortex_grid(self, severity: float, H: int, W: int) -> torch.Tensor:
        """Example image-independent warp: a swirl around the center.
        Cached because it only depends on (severity,H,W)."""
        def build():
            base = self._identity_grid(H, W)[0]              # [H,W,2]
            gx, gy = base[..., 0], base[..., 1]
            r = torch.sqrt(gx ** 2 + gy ** 2)
            ang = severity * 3.0 * torch.exp(-r * 2.0)       # swirl falls off radially
            cos, sin = torch.cos(ang), torch.sin(ang)
            wx = cos * gx - sin * gy
            wy = sin * gx + cos * gy
            return torch.stack((wx, wy), dim=-1).unsqueeze(0)  # [1,H,W,2]
        return self.cached("vortex_demo_grid", (severity,), (H, W), build)

    def warp(self, x: torch.Tensor, grid: torch.Tensor,
             mode="bilinear") -> torch.Tensor:
        # grid is [1,H,W,2]; expand to batch.  padding_mode='border' == clamp.
        g = grid.expand(x.shape[0], -1, -1, -1)
        return F.grid_sample(x, g, mode=mode, padding_mode="border",
                             align_corners=True)

    def vortex(self, x: torch.Tensor, severity: float = 0.5) -> torch.Tensor:
        if severity <= 0:
            return x
        return self.warp(x, self._vortex_grid(severity, x.shape[2], x.shape[3]))


# -----------------------------------------------------------------------------
# Conversion helpers: MuJoCo renderer (HxWx3 uint8 numpy) <-> GPU tensor
# -----------------------------------------------------------------------------
def frames_to_gpu(frames: np.ndarray, device) -> torch.Tensor:
    """frames: [N,H,W,3] or [H,W,3] uint8  ->  [N,3,H,W] float32 in [0,1]."""
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
    return x.permute(0, 2, 3, 1).cpu().numpy()


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
        x = frames_to_gpu(frame, self.sim.device)
        for name, kw in self.chain:
            x = getattr(self.sim, name)(x, **kw)
        out = gpu_to_frames(x)
        return out[0] if frame.ndim == 3 else out

    def observation(self, obs):
        if self.image_key is not None:
            obs = dict(obs)
            obs[self.image_key] = self._filter(obs[self.image_key])
            return obs
        return self._filter(obs)


# -----------------------------------------------------------------------------
# Porting the remaining filters — which primitive each one needs
# -----------------------------------------------------------------------------
# Recipe: severity = clamp_severity(severity); build every image-independent
# array with self.numpy_field(...) by calling/copying the CPU numpy code
# verbatim (same dtypes, same RNG); do the per-frame work with torch ops in the
# CPU's order; end with quantize_u8 where the CPU calls _to_uint8 (gpu_to_frames
# does the final rounding).  Check against the CPU with GpuParityCase
# (tests/test_rounding.py): max <= 1 LSB, mean < 0.02.
#
# Where bit-equality is lost, and how to keep it:
#   * dtypes follow numpy: float32 array op Python float stays float32, but
#     float32 - int32 (or any float64 array) promotes to float64; torch
#     promotes tensors the same way, so mirror the CPU dtype of each array.
#   * on CUDA, `t / 2.0` is t * (1/2.0); use exact_div where numpy divides.
#   * pow/exp/sin/arctan2 differ from numpy by ~1 ulp on both devices, so
#     values sitting exactly on an 8-bit rounding boundary can flip by 1 LSB.
#   * don't call grid_sample or conv2d for CPU twins (coordinate rounding
#     differs from numpy's; TF32); use the primitives above.
#
# matmul/elementwise (bucket 1):
#   monochrome, noise, glitch, foveal_darkness, teichopsia/floater blends,
#   detail_loss (posterize = round(x*L)/L), led/pixelation tint
# blur/resize (bucket 2):
#   bloom: pil_resize(BILINEAR) down by 4 -> threshold -> pil_gaussian_blur
#     -> pil_resize up -> add; quantize_u8 between steps (8-bit linear
#     intermediates, exactly like the CPU's Image.fromarray(_to_uint8(...)))
#   field_loss: pil_gaussian_blur per mip level (radius 2**(lvl-1)), blend by
#     the overlay mask from numpy_field
#   pixelation: pil_resize down (BILINEAR or NEAREST) then up
# gathers (bucket 3):
#   distortion, wiggle, cataract frost, led, teichopsia mask: numpy_field
#     returns the (sy, sx) maps from the CPU expression
#     `(src_u * (w - 1)).astype(np.int32)` -> gather_nearest
#   vortex: numpy_field returns float pixel coords -> gather_bilinear
#   double_vision, nystagmus, glitch: integer shifts/rolls -> gather_nearest
#     with shifted index maps (or torch.roll)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sim = VipSimGPU(device=dev)
    x = torch.rand(8, 3, 256, 256, device=dev)     # batch of 8 fake frames
    for fn, kw in [("cvd", {"severity": 0.8}), ("bcg", {"brightness": 0.8, "contrast": 0.7}),
                   ("blur", {"severity": 0.4}), ("vortex", {"severity": 0.5})]:
        y = getattr(sim, fn)(x, **kw)
        print(f"{fn:8s} -> {tuple(y.shape)}  range[{y.min():.3f},{y.max():.3f}]")
