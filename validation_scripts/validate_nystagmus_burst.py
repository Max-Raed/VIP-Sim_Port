#!/usr/bin/env python3
"""Burst-validation for nystagmus (time-varying filter).

Compares a Unity burst (`vipsim_assets/unity_refs/nystagmus_burst/`) against a
matching Python burst. The Unity capture is forced into `artificialRotation`
mode (UV-shift via shader, no camera-transform rotation) and
`baselineErr_deg=0` (no UnityRNG jitter that numpy cannot replay), so the
saccade cycle is fully deterministic from `timer_secs` and the static params.

This script implements the time-aware saccade cycle here, standalone. It does
NOT use the frozen-frame `apply_nystagmus()` in vipsim_filters.py, because
that one is a static motion-blur approximation, not a time-aware port.

Usage:
    python scripts/validate_nystagmus_burst.py \
        --unity-dir vipsim_assets/unity_refs/nystagmus_burst \
        --baseline vipsim_assets/unity_refs/baseline.png \
        --out abs_diff_out/nystagmus_burst

    # optional: also produce side-by-side MP4 (uses bundled ffmpeg)
    python scripts/validate_nystagmus_burst.py --make-video
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as _ssim


# -----------------------------------------------------------------------------
# Time-aware nystagmus (faithful port of myNystagmus.OnUpdate saccade math)
# -----------------------------------------------------------------------------
# Unity logic in plain English (with baselineErr_deg = 0 and useNullingField
# disabled, both forced by NystagmusBurstCapture for reproducibility):
#
#   total_d = foveat_d + rise_d
#   t_cycle = timer_secs % total_d
#   if t_cycle <= foveat_d:
#       shift_deg = 0
#   else:
#       p = ((t_cycle - foveat_d) / rise_d) ** rise_exp
#       shift_deg = p * amp_deg
#
# With direction_deg = 0 (horizontal only), in artificialRotation mode the
# shader receives:
#   xOffset_px = -shift_deg * (screenWidth_px / viewingAngle_deg)
#   yOffset_px = 0
# and shifts UVs by (xOffset_px / source.width, 0). The image moves to the
# right by `xOffset_px` pixels (negative sign in xOffset_deg flips it).

def shift_deg_at(timer_secs: float, foveat_d: float, rise_d: float,
                  rise_exp: float, amp_deg: float) -> float:
    total_d = foveat_d + rise_d
    if total_d <= 0:
        return 0.0
    t_cycle = math.fmod(timer_secs, total_d)
    if t_cycle <= foveat_d:
        return 0.0
    p = (t_cycle - foveat_d) / rise_d
    p = p ** rise_exp
    return p * amp_deg


def apply_uv_shift(img: np.ndarray, dx_px: float, dy_px: float) -> np.ndarray:
    """Shift `img` by (dx_px, dy_px) using bilinear sampling with edge clamp.

    Matches Unity's shader: tex2D with uv shifted by (_Displace.x, _Displace.y)
    in normalised coords; out-of-bounds samples are clamped (the default Unity
    texture wrap is Clamp on Capture's render target).
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Unity shader convention: sample at uv + _Displace, so a negative
    # _Displace.x (Unity default for nystagmus with direction_deg=0) moves
    # the visible image to the right. Match that by ADDING dx_px / dy_px.
    src_x = np.clip(xx + dx_px, 0.0, w - 1.0)
    src_y = np.clip(yy + dy_px, 0.0, h - 1.0)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (src_x - x0)[..., None]
    fy = (src_y - y0)[..., None]
    a = img[y0, x0].astype(np.float32)
    b = img[y0, x1].astype(np.float32)
    c = img[y1, x0].astype(np.float32)
    d = img[y1, x1].astype(np.float32)
    top = a * (1 - fx) + b * fx
    bot = c * (1 - fx) + d * fx
    out = top * (1 - fy) + bot * fy
    return np.clip(out, 0, 255).astype(np.uint8)


def render_nystagmus(baseline: np.ndarray, timer_secs: float, *,
                       foveat_d: float, rise_d: float, rise_exp: float,
                       amp_deg: float, direction_deg: float,
                       screen_width_px: int,
                       viewing_angle_deg: float) -> np.ndarray:
    """Render one Unity-equivalent frame at `timer_secs`."""
    deg = shift_deg_at(timer_secs, foveat_d, rise_d, rise_exp, amp_deg)
    if deg == 0.0:
        return baseline.copy()
    pixel_per_dg = screen_width_px / viewing_angle_deg
    rad = math.radians(direction_deg)
    x_off_deg = -deg * math.cos(rad)
    y_off_deg = deg * math.sin(rad)
    dx_px = x_off_deg * pixel_per_dg
    dy_px = y_off_deg * pixel_per_dg
    return apply_uv_shift(baseline, dx_px, dy_px)


# -----------------------------------------------------------------------------
# Comparison + reporting
# -----------------------------------------------------------------------------

def ssim_blur(a: np.ndarray, b: np.ndarray, sigma: float = 4.0) -> float:
    a01 = a.astype(np.float32) / 255.0
    b01 = b.astype(np.float32) / 255.0
    af = gaussian_filter(a01, sigma=(sigma, sigma, 0))
    bf = gaussian_filter(b01, sigma=(sigma, sigma, 0))
    return float(_ssim(af, bf, channel_axis=2, data_range=1.0))


def make_side_by_side(unity: np.ndarray, python: np.ndarray) -> np.ndarray:
    return np.vstack([unity, python])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unity-dir", type=Path,
                    default=Path("vipsim_assets/unity_refs/nystagmus_burst"))
    ap.add_argument("--baseline", type=Path,
                    default=Path("vipsim_assets/unity_refs/baseline.png"))
    ap.add_argument("--out", type=Path,
                    default=Path("abs_diff_out/nystagmus_burst"))
    ap.add_argument("--make-video", action="store_true")
    args = ap.parse_args()

    if not args.unity_dir.exists():
        sys.exit(f"Unity burst dir not found: {args.unity_dir}")
    if not args.baseline.exists():
        sys.exit(f"Baseline not found: {args.baseline}")

    meta_path = args.unity_dir / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"metadata.json missing in {args.unity_dir}; re-run Unity burst capture.")
    meta = json.loads(meta_path.read_text())

    foveat_d = float(meta["foveat_d"])
    rise_d = float(meta["rise_d"])
    rise_exp = float(meta["rise_exp"])
    amp_deg = float(meta["amp_deg"])
    direction_deg = float(meta.get("direction_deg", 0.0))
    screen_width_px = int(meta.get("screenWidth_px", 2048))
    viewing_angle_deg = float(meta.get("viewingAngle_deg", 100.0))
    frame_times = meta["frame_times_s"]
    timer_secs_list = meta["timer_secs"]

    args.out.mkdir(parents=True, exist_ok=True)
    compare_dir = args.out / "compare"; compare_dir.mkdir(exist_ok=True)
    python_dir = args.out / "python";  python_dir.mkdir(exist_ok=True)

    baseline = np.asarray(Image.open(args.baseline).convert("RGB"))
    print(f"Loaded baseline: {baseline.shape}, foveat_d={foveat_d} rise_d={rise_d} "
          f"rise_exp={rise_exp} amp_deg={amp_deg}")

    scores = []
    for i, (t, timer) in enumerate(zip(frame_times, timer_secs_list)):
        unity_path = args.unity_dir / f"frame_{i:03d}.png"
        if not unity_path.exists():
            print(f"  skip {i:03d}: {unity_path.name} missing")
            continue
        unity = np.asarray(Image.open(unity_path).convert("RGB"))

        python = render_nystagmus(
            baseline, timer_secs=float(timer),
            foveat_d=foveat_d, rise_d=rise_d, rise_exp=rise_exp,
            amp_deg=amp_deg, direction_deg=direction_deg,
            screen_width_px=screen_width_px,
            viewing_angle_deg=viewing_angle_deg,
        )

        Image.fromarray(python).save(python_dir / f"frame_{i:03d}.png")
        Image.fromarray(make_side_by_side(unity, python)).save(
            compare_dir / f"frame_{i:03d}.png")

        s = ssim_blur(unity, python)
        deg = shift_deg_at(float(timer), foveat_d, rise_d, rise_exp, amp_deg)
        scores.append({"frame": i, "t": t, "timer_secs": timer,
                       "shift_deg": deg, "ssim_blur": s})
        print(f"  frame {i:03d} t={t:.2f}s timer={timer:.3f} shift_deg={deg:.2f}  SSIM_blur={s:.3f}")

    summary = {
        "meta": meta,
        "scores": scores,
        "ssim_blur_mean": float(np.mean([s["ssim_blur"] for s in scores])) if scores else None,
        "ssim_blur_min": float(np.min([s["ssim_blur"] for s in scores])) if scores else None,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nMean SSIM_blur: {summary['ssim_blur_mean']:.3f}  (min {summary['ssim_blur_min']:.3f})")
    print(f"Outputs: {args.out}")

    if args.make_video:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"Using bundled ffmpeg: {ffmpeg}")
            except Exception:
                print("ffmpeg not found and imageio-ffmpeg not installed; skipping --make-video")
                return
        mp4 = args.out / "compare.mp4"
        cmd = [ffmpeg, "-y", "-framerate", "2",
               "-i", str(compare_dir / "frame_%03d.png"),
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
        print(f"Video: {mp4}")


if __name__ == "__main__":
    main()
