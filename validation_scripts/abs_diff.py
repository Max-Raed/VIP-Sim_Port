#!/usr/bin/env python3
"""Image-similarity validation: compare our Python filter output to a reference.

Reports three complementary metrics per comparison:
  - MAE (mean ABS diff, 0–255): raw pixel difference. Cheap, intuitive.
  - SSIM (0–1): structural similarity on luminance. Robust to small global
    shifts; >0.98 means "visually identical".
  - ΔE2000 (Lab colour distance): perceptual colour difference. <1 imperceptible,
    1–3 perceptible on close inspection, >5 clearly different. Best metric for
    colour-shifting filters (CVD, cataracts).

Picking metrics per filter family (rule of thumb):
  blur, bloom         → MAE + SSIM (SSIM primary)
  BCG, gamma          → MAE  (Tier 1: should be ≈0; SSIM/ΔE corroborate)
  CVD, cataract       → ΔE2000 + per-channel MAE
  field/double/foveal → MAE on masked region + SSIM full image
  floaters, flicker   → SSIM (occluder shape) + MAE

Used to validate that vipsim_filters.py produces (near-)identical output
to the Unity VIP-Sim reference. Two reference modes:

1. **participants_panel** mode (interim, no Unity required):
   crop the relevant panel from `vipsim_assets/participants.png` and
   diff against our Python output on the matching cropped Trafficscene.

2. **unity_render** mode (gold standard, requires Unity render):
   diff a directly-paired (input.png, unity_output.png) against our
   Python pipeline output for the same input + matched filter parameters.

Usage:
    # Interim (participant panel reference)
    python scripts/abs_diff.py --profile p1 --mode participants_panel

    # Gold standard (paired Unity render)
    python scripts/abs_diff.py \\
        --input ref/scene.png \\
        --unity ref/scene_p1_unity.png \\
        --profile p1 --mode unity_render

Outputs (under abs_diff_out/):
    diff_<profile>_<mode>.png       — side-by-side input | python | unity | diff
    diff_<profile>_<mode>.json      — mean/max/p95/p99 ABS diff per channel
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.color import deltaE_ciede2000, rgb2lab
    from skimage.metrics import structural_similarity as ssim
    from scipy.ndimage import gaussian_filter
    from scipy.stats import wasserstein_distance
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False

# LPIPS: pretrained CNN perceptual distance. Best correlate with human
# judgement for noisy / texture-heavy filters (cataract, noise, floaters).
# Optional — install with `pip install lpips torch`. Falls back gracefully.
try:
    import lpips as _lpips_mod
    import torch as _torch
    _HAVE_LPIPS = True
    _LPIPS_NET = None  # lazy-init so we only pay model-load cost once
except Exception:
    _HAVE_LPIPS = False

# Make imports work regardless of cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import vipsim_filters as vf  # noqa: E402

ASSETS = ROOT / "vipsim_assets"
PROFILES_DIR = ROOT / "vipsim_profiles"
OUT = ROOT / "abs_diff_out"


# -----------------------------------------------------------------------------
# Profile loader — JSON files describe a participant's filter chain
# -----------------------------------------------------------------------------

def load_profile(profile_id: str) -> dict:
    """Load a profile JSON from vipsim_profiles/<profile_id>.json."""
    path = PROFILES_DIR / f"{profile_id}.json"
    if not path.is_file():
        sys.exit(
            f"Missing profile file: {path}\n"
            f"Create it (see vipsim_profiles/_template.json) before running."
        )
    return json.loads(path.read_text())


def apply_profile(img: Image.Image, profile: dict) -> Image.Image:
    """Apply a profile's filter chain to img."""
    result = img
    for step in profile["filters"]:
        name = step["filter"]
        kwargs = {k: v for k, v in step.items() if k != "filter"}
        fn = vf._FILTER_DISPATCH[name]
        result = fn(result, **kwargs)
    return result


# -----------------------------------------------------------------------------
# Reference loaders
# -----------------------------------------------------------------------------

def load_participants_reference(profile_id: str) -> tuple[Image.Image, Image.Image]:
    """Load (input, reference_output) for the participants_panel mode.

    Crops the matching panel from vipsim_assets/participants.png. The crop
    rectangles must be defined in the profile JSON (`panel_crop` field) since
    they're hand-measured per profile.
    """
    profile = load_profile(profile_id)
    crop = profile.get("panel_crop")
    if not crop:
        sys.exit(
            f"Profile {profile_id} has no `panel_crop` field. "
            f"Add (left, top, right, bottom) pixel coords for the participants.png panel."
        )

    # The Trafficscene is the input scene used in participants.png
    input_path = ASSETS / "Trafficscene.png"
    panel_path = ASSETS / "participants.png"
    if not input_path.is_file() or not panel_path.is_file():
        sys.exit(f"Missing assets in {ASSETS}/. Need Trafficscene.png + participants.png.")

    panel = Image.open(panel_path).convert("RGB").crop(tuple(crop))
    # Resize Trafficscene down to the panel's resolution so ABS diff is
    # apples-to-apples (the panel is heavily downscaled JPEG).
    input_img = Image.open(input_path).convert("RGB").resize(panel.size, Image.BILINEAR)
    return input_img, panel


def load_unity_reference(input_path: Path, unity_path: Path) -> tuple[Image.Image, Image.Image]:
    """Load (input, reference_output) for the unity_render mode."""
    if not input_path.is_file():
        sys.exit(f"Missing input: {input_path}")
    if not unity_path.is_file():
        sys.exit(f"Missing Unity reference: {unity_path}")
    inp = Image.open(input_path).convert("RGB")
    ref = Image.open(unity_path).convert("RGB")
    if inp.size != ref.size:
        # Match Unity ref size — assume the Unity output is the canonical resolution
        inp = inp.resize(ref.size, Image.BILINEAR)
    return inp, ref


# -----------------------------------------------------------------------------
# ABS-diff scoring
# -----------------------------------------------------------------------------

def _wasserstein_per_channel(a: np.ndarray, b: np.ndarray) -> dict:
    """1-D Wasserstein-1 distance per channel via `scipy.stats.wasserstein_distance`.

    Alignment-free: only the distribution of pixel values matters, not where
    the pixels sit. Useful for "did the overall colour / brightness
    distribution shift the same way" comparisons (cataract, BCG, noise).
    Units: pixel-value steps on the 0-255 scale.

    Library: scipy.stats.wasserstein_distance (standard 1-D Earth Mover's
    Distance). No hand-rolled metric.
    """
    out = {}
    for i, ch in enumerate("RGB"):
        out[ch] = float(wasserstein_distance(a[..., i].ravel(), b[..., i].ravel()))
    out["mean"] = float(np.mean([out["R"], out["G"], out["B"]]))
    return out


def _lpips_distance(a01: np.ndarray, b01: np.ndarray) -> float | None:
    """Pretrained AlexNet LPIPS perceptual distance. ~0 identical, 1+ very different."""
    if not _HAVE_LPIPS:
        return None
    global _LPIPS_NET
    if _LPIPS_NET is None:
        _LPIPS_NET = _lpips_mod.LPIPS(net="alex", verbose=False)
        _LPIPS_NET.eval()
    # LPIPS expects (1, 3, H, W) in [-1, 1]
    ta = _torch.from_numpy(a01.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    tb = _torch.from_numpy(b01.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    with _torch.no_grad():
        d = _LPIPS_NET(ta, tb)
    return float(d.item())


def score_abs_diff(ours: Image.Image, ref: Image.Image) -> dict:
    """Compute pixel + perceptual similarity metrics.

    Metrics (all kept; pick the one(s) right for your filter family):
      - MAE (overall + per-channel): pixel-level difference, units 0–255.
        Sensitive to *any* pixel misalignment, including stochastic noise.
      - SSIM (luminance): structural similarity on grayscale, range [0, 1].
        >0.98 ≈ visually identical. Still alignment-sensitive at high freqs.
      - ΔE2000 (CIEDE2000): perceptual colour distance in Lab. <1 imperceptible,
        1–3 perceptible on close inspection, >5 clearly different. Best for
        colour-shifting filters (CVD, cataract, BCG).
      - SSIM_blur: SSIM after Gaussian σ=4 blur on both images. Removes the
        high-frequency noise-alignment penalty so structural similarity of
        stochastic textures is comparable. Best companion to SSIM for noise /
        cataract / floaters / teichopsia.
      - Wasserstein (per-channel 1-D Earth Mover's Distance):
        alignment-free distribution distance via scipy.stats.wasserstein_distance.
        Great for "did the colour / brightness distribution shift the same
        way" — cataract, BCG, noise. Units: pixel-value steps (0-255).
      - LPIPS (optional): pretrained AlexNet perceptual distance. Gold standard
        for noisy / textured similarity. Requires `pip install lpips torch`.

    SSIM-family + ΔE2000 + Wasserstein require scikit-image / scipy; LPIPS
    requires lpips + torch. Keys are None if the underlying lib is missing.
    """
    if ours.size != ref.size:
        ours = ours.resize(ref.size, Image.BILINEAR)
    a = np.asarray(ours, dtype=np.float32)
    b = np.asarray(ref, dtype=np.float32)
    diff = np.abs(a - b)  # in [0, 255]
    flat = diff.reshape(-1, 3)

    def _stats(x):
        return {
            "mean": float(x.mean()),
            "max": float(x.max()),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
        }

    result = {
        "mae": {
            "overall": _stats(diff),
            "per_channel": {
                "R": _stats(flat[:, 0]),
                "G": _stats(flat[:, 1]),
                "B": _stats(flat[:, 2]),
            },
        },
        "ssim": None,
        "ssim_blur": None,
        "deltaE2000": None,
        "wasserstein": None,
        "lpips": None,
        "image_size": [ours.size[0], ours.size[1]],
    }

    # Backwards-compat shim for callers reading stats["overall"] / ["per_channel"]
    result["overall"] = result["mae"]["overall"]
    result["per_channel"] = result["mae"]["per_channel"]

    if _HAVE_SKIMAGE:
        a01 = a / 255.0
        b01 = b / 255.0
        # SSIM on luminance (channel_axis=-1 averages across RGB; use that).
        result["ssim"] = float(ssim(a01, b01, channel_axis=-1, data_range=1.0))

        # Blurred SSIM: σ=4 Gaussian on each, then SSIM. Removes high-freq
        # noise mismatch so the structural-similarity signal isn't drowned out.
        a_blur = gaussian_filter(a01, sigma=(4, 4, 0))
        b_blur = gaussian_filter(b01, sigma=(4, 4, 0))
        result["ssim_blur"] = float(ssim(a_blur, b_blur, channel_axis=-1, data_range=1.0))

        # ΔE2000 in Lab. rgb2lab expects [0,1] sRGB.
        lab_a = rgb2lab(a01)
        lab_b = rgb2lab(b01)
        de = deltaE_ciede2000(lab_a, lab_b)
        result["deltaE2000"] = {
            "mean": float(de.mean()),
            "max": float(de.max()),
            "p50": float(np.percentile(de, 50)),
            "p95": float(np.percentile(de, 95)),
            "p99": float(np.percentile(de, 99)),
        }

    # Per-channel 1-D Wasserstein-1 via scipy.stats.wasserstein_distance.
    if _HAVE_SKIMAGE:
        result["wasserstein"] = _wasserstein_per_channel(a, b)

    # LPIPS (optional). Run on the original 0..1 RGB float arrays.
    if _HAVE_LPIPS:
        result["lpips"] = _lpips_distance(a / 255.0, b / 255.0)

    return result


def make_comparison_image(input_img: Image.Image, ours: Image.Image,
                            ref: Image.Image) -> Image.Image:
    """Build an Input | Python | Unity | Diff×4 strip."""
    # Match sizes
    target_h = ref.size[1]
    target_w = ref.size[0]
    inp = input_img.resize((target_w, target_h), Image.BILINEAR)
    ours = ours.resize((target_w, target_h), Image.BILINEAR)

    a = np.asarray(ours, dtype=np.int16)
    b = np.asarray(ref, dtype=np.int16)
    diff = np.clip(np.abs(a - b) * 4, 0, 255).astype(np.uint8)  # ×4 to make visible
    diff_img = Image.fromarray(diff)

    strip = Image.new("RGB", (target_w * 4, target_h))
    strip.paste(inp, (0, 0))
    strip.paste(ours, (target_w, 0))
    strip.paste(ref, (target_w * 2, 0))
    strip.paste(diff_img, (target_w * 3, 0))
    return strip


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate Python filters vs Unity reference.")
    parser.add_argument("--profile", required=True,
                        help="Profile id, looked up in vipsim_profiles/<id>.json")
    parser.add_argument("--mode", choices=["participants_panel", "unity_render"],
                        default="participants_panel")
    parser.add_argument("--input", type=Path,
                        help="(unity_render mode) Path to input image")
    parser.add_argument("--unity", type=Path,
                        help="(unity_render mode) Path to Unity-rendered reference")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    profile = load_profile(args.profile)

    if args.mode == "participants_panel":
        input_img, ref = load_participants_reference(args.profile)
    else:  # unity_render
        if not args.input or not args.unity:
            sys.exit("--input and --unity required in unity_render mode")
        input_img, ref = load_unity_reference(args.input, args.unity)

    print(f"[{args.profile}] input={input_img.size}  ref={ref.size}  filters={len(profile['filters'])}")
    ours = apply_profile(input_img, profile)

    stats = score_abs_diff(ours, ref)
    stats["profile"] = args.profile
    stats["mode"] = args.mode
    stats["filters"] = profile["filters"]

    base = f"diff_{args.profile}_{args.mode}"
    (OUT / f"{base}.json").write_text(json.dumps(stats, indent=2))
    make_comparison_image(input_img, ours, ref).save(OUT / f"{base}.png")

    o = stats["mae"]["overall"]
    print(f"  MAE:       mean={o['mean']:6.2f}  p95={o['p95']:6.2f}  max={o['max']:6.2f}")
    if stats["ssim"] is not None:
        de = stats["deltaE2000"]
        print(f"  SSIM:      {stats['ssim']:.4f}   (1.0 = identical, >0.98 ≈ visually identical)")
        print(f"  SSIM_blur: {stats['ssim_blur']:.4f}   (σ=4 Gaussian, robust to noise misalignment)")
        print(f"  MS-SSIM:   {stats['ms_ssim']:.4f}   (3-scale approximation)")
        print(f"  ΔE2000:    mean={de['mean']:5.2f}  p95={de['p95']:5.2f}  max={de['max']:5.2f}")
    else:
        print("  SSIM/ΔE2000 skipped (install scikit-image: pip install scikit-image)")
    if stats["hist_emd"] is not None:
        he = stats["hist_emd"]
        print(f"  HistEMD:   mean={he['mean']:5.2f}  R={he['R']:5.2f} G={he['G']:5.2f} B={he['B']:5.2f}  (alignment-free dist. shift)")
    if stats["lpips"] is not None:
        print(f"  LPIPS:     {stats['lpips']:.4f}   (~0 identical, >0.5 very different — perceptual)")
    elif not _HAVE_LPIPS:
        print("  LPIPS:     skipped (pip install lpips torch  — best perceptual metric for noisy filters)")
    print(f"  → {OUT}/{base}.png  +  {base}.json")


if __name__ == "__main__":
    main()
