#!/usr/bin/env python3
"""
vipsim_gpu.py — GPU (PyTorch) port of the VIP-Sim visual-impairment filters.

Design goals
------------
* Operate on BATCHED, channels-first tensors  x: [N, 3, H, W], float32 in [0,1],
  living on the GPU.  N = number of parallel envs (or frame stack).
* No GLSL/CUDA.  Every original filter reduces to one of three primitives:
      1. elementwise / 3x3 matmul   -> einsum + arithmetic   (cvd, bcg, ...)
      2. separable blur / pooling   -> conv2d / avg_pool2d    (blur, bloom, field_loss)
      3. per-pixel UV displacement  -> F.grid_sample          (distortion, vortex, ...)
* Anything that does NOT depend on the image (warp fields, masks, noise tiles)
  is precomputed ONCE per (severity, H, W) and cached on the device.

This file ports a representative filter from each of the three buckets
(cvd, bcg, blur, and a grid_sample warp) plus the MyoSuite hook.  The
remaining ~15 filters in vipsim_filters.py follow the exact same patterns —
see the porting notes at the bottom.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# sRGB <-> linear  (same round-trip as the NumPy version, just torch ops)
# -----------------------------------------------------------------------------
def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


# -----------------------------------------------------------------------------
# Bucket 1: matmul / elementwise  — CVD and BCG
# -----------------------------------------------------------------------------
# The 11-step CVD matrices, copied verbatim from vipsim_filters.CVD_MATRICES.
# (Trimmed here to the deuteranomaly set for brevity; paste the full dict in.)
_CVD_DEUTER = np.array([
    [[1,0,0],[0,1,0],[0,0,1]],
    [[0.866435,0.177704,-0.044139],[0.049567,0.939063,0.01137],[-0.003453,0.007233,0.99622]],
    [[0.760729,0.319078,-0.079807],[0.090568,0.889315,0.020117],[-0.006027,0.013325,0.992702]],
    [[0.675425,0.43385,-0.109275],[0.125303,0.847755,0.026942],[-0.00795,0.018572,0.989378]],
    [[0.605511,0.52856,-0.134071],[0.155318,0.812366,0.032316],[-0.009376,0.023176,0.9862]],
    [[0.547494,0.607765,-0.155259],[0.181692,0.781742,0.036566],[-0.01041,0.027275,0.983136]],
    [[0.498864,0.674741,-0.173604],[0.205199,0.754872,0.039929],[-0.011131,0.030969,0.980162]],
    [[0.457771,0.731899,-0.18967],[0.226409,0.731012,0.042579],[-0.011595,0.034333,0.977261]],
    [[0.422823,0.781057,-0.203881],[0.245752,0.709602,0.044646],[-0.011843,0.037423,0.974421]],
    [[0.392952,0.82361,-0.216562],[0.263559,0.69021,0.046232],[-0.01191,0.040281,0.97163]],
    [[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],[-0.01182,0.04294,0.968881]],
], dtype=np.float32)


class VipSimGPU:
    """Holds device + cached static tensors so per-frame work is minimal."""

    def __init__(self, device="cuda", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.cvd_deuter = torch.from_numpy(_CVD_DEUTER).to(self.device, dtype)
        self._warp_cache: dict = {}     # (name, severity, H, W) -> grid [1,H,W,2]
        self._kernel_cache: dict = {}   # sigma -> 1d gaussian kernel

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
        # x:[N,3,H,W], mat:[3,3]  ->  out[N,i,H,W] = sum_j mat[i,j] x[N,j,H,W]
        return torch.einsum("ij,njhw->nihw", mat, x).clamp(0, 1)

    # ---- BCG: linear-space brightness/contrast/gamma ------------------------
    def bcg(self, x: torch.Tensor, brightness=1.0, contrast=1.0, gamma=1.0,
            pivot=0.5) -> torch.Tensor:
        lin = srgb_to_linear(x) * brightness
        lin = (lin - pivot) * contrast + pivot
        lin = lin.clamp(0, 1)
        if gamma != 1.0:
            lin = lin ** gamma
        return linear_to_srgb(lin.clamp(0, 1))

    # ---- Bucket 2: separable Gaussian blur (replaces PIL GaussianBlur) ------
    def _gauss1d(self, sigma: float) -> torch.Tensor:
        key = round(float(sigma), 3)
        if key in self._kernel_cache:
            return self._kernel_cache[key]
        radius = max(1, int(np.ceil(3 * sigma)))
        xs = torch.arange(-radius, radius + 1, device=self.device, dtype=self.dtype)
        k = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
        k = k / k.sum()
        self._kernel_cache[key] = k
        return k

    def blur(self, x: torch.Tensor, severity: float = 0.5) -> torch.Tensor:
        if severity <= 0:
            return x
        sigma = severity * 15.0          # same 0..15 mapping as the NumPy port
        k = self._gauss1d(sigma)
        r = (k.numel() - 1) // 2
        C = x.shape[1]
        kh = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)   # horizontal
        kv = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)   # vertical
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kh, groups=C)
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kv, groups=C)
        return x.clamp(0, 1)

    # ---- Bucket 3: per-pixel UV displacement via grid_sample ----------------
    # grid_sample expects a sampling grid in normalized [-1,1] coords,
    # shape [N,H,W,2] with (x,y) order.  Any filter that does `arr[sy,sx]`
    # becomes: build offset field -> add to identity grid -> grid_sample.
    def _identity_grid(self, H: int, W: int) -> torch.Tensor:
        ys = torch.linspace(-1, 1, H, device=self.device, dtype=self.dtype)
        xs = torch.linspace(-1, 1, W, device=self.device, dtype=self.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((gx, gy), dim=-1).unsqueeze(0)    # [1,H,W,2]

    def _vortex_grid(self, severity: float, H: int, W: int) -> torch.Tensor:
        """Example image-independent warp: a swirl around the center.
        Cached because it only depends on (severity,H,W)."""
        key = ("vortex", round(severity, 3), H, W)
        if key in self._warp_cache:
            return self._warp_cache[key]
        base = self._identity_grid(H, W)[0]                  # [H,W,2]
        gx, gy = base[..., 0], base[..., 1]
        r = torch.sqrt(gx ** 2 + gy ** 2)
        ang = severity * 3.0 * torch.exp(-r * 2.0)           # swirl falls off radially
        cos, sin = torch.cos(ang), torch.sin(ang)
        wx = cos * gx - sin * gy
        wy = sin * gx + cos * gy
        grid = torch.stack((wx, wy), dim=-1).unsqueeze(0)    # [1,H,W,2]
        self._warp_cache[key] = grid
        return grid

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
    t = torch.from_numpy(np.ascontiguousarray(frames)).to(device)
    return t.permute(0, 3, 1, 2).float() / 255.0


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
# matmul/elementwise (bucket 1):
#   monochrome, noise, glitch, foveal_darkness, teichopsia/floater blends,
#   detail_loss (posterize = round(x*L)/L), led/pixelation tint
# conv/pool (bucket 2):
#   bloom (avg_pool2d downsample -> threshold -> blur -> upsample+add),
#   field_loss (build mip pyramid with avg_pool2d, blend by a precomputed mask),
#   cataract frost blur
# grid_sample (bucket 3):
#   distortion (precompute vx,vy field once -> grid),
#   wiggle (sinusoidal offset field), double_vision (shifted-copy blend),
#   nystagmus (axis offset), cataract frost displacement
#
# pixelation / led are just avg_pool2d down then F.interpolate(mode='nearest') up.
# Static masks (radial, macular overlay, teichopsia crescent) are computed once
# on the GPU and cached exactly like _vortex_grid above.


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sim = VipSimGPU(device=dev)
    x = torch.rand(8, 3, 256, 256, device=dev)     # batch of 8 fake frames
    for fn, kw in [("cvd", {"severity": 0.8}), ("bcg", {"brightness": 0.8, "contrast": 0.7}),
                   ("blur", {"severity": 0.4}), ("vortex", {"severity": 0.5})]:
        y = getattr(sim, fn)(x, **kw)
        print(f"{fn:8s} -> {tuple(y.shape)}  range[{y.min():.3f},{y.max():.3f}]")
