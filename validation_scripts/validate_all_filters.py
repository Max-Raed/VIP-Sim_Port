#!/usr/bin/env python3
"""Sweep severity for each Python filter and report best match vs Unity ref.

For every Unity reference PNG under vipsim_assets/unity_refs/, applies the
matching Python filter to the shared input (Trafficscene_2048x1024.png) over a
severity sweep and records the severity that minimises mean ABS diff.

Outputs:
  abs_diff_out/validation_summary.json   — full per-filter sweep results
  abs_diff_out/validation_summary.csv    — one row per filter (best severity)
  abs_diff_out/diff_<filter>_best.png    — side-by-side strip at best severity

Usage:
  python scripts/validate_all_filters.py
  python scripts/validate_all_filters.py --filters blur,noise   # subset
  python scripts/validate_all_filters.py --sweep 0.1,1.0,0.05   # custom range
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from validation_scripts.abs_diff import (  # noqa: E402
    load_unity_reference,
    score_abs_diff,
)
import vipsim_filters as vf  # noqa: E402

ASSETS = ROOT / "vipsim_assets"
UNITY_REFS = ASSETS / "unity_refs"
INPUT_PNG = ASSETS / "Trafficscene_2048x1024.png"
OUT = ROOT / "abs_diff_out"
COMPARE_DIR = OUT / "unityVSpython"
STRIP_DIR = COMPARE_DIR / "compare"
STACK_DIR = COMPARE_DIR / "stacked"  # Unity on top, Python on bottom


# Mapping: Python filter name -> (Unity ref filename, extra kwargs, sweep-key, criterion)
# `sweep_key` is the kwarg name that gets swept. For filters whose Unity render
# was captured with fixed params (bcg, cvd_deutan), we still sweep severity for
# the colour case and pin the type. For bcg there's no severity at all so we do
# a single point evaluation only.
#
# `criterion` selects which metric to minimise/maximise when picking the "best"
# severity. Default is "mae_mean" (minimise). For stochastic filters where pixel
# alignment is meaningless (noise patterns differ between RNGs even at correct
# amplitude), use "ssim_blur" (maximise) — that scores the structural look after
# σ=4 Gaussian smoothing, removing the per-pixel noise mismatch penalty.
FILTERS = {
    "blur":            {"ref": "blur_cpd28.png",            "kwargs": {}, "sweep_key": "severity"},
    # Unity bloom captured at intensity=2.0, threshold=0.3 (defaults: 0.75, 0.25).
    # Pin those and sweep severity (scales nothing; left unused via explicit
    # intensity/threshold/blur_size overrides).
    "bloom":           {"ref": "bloom_int20_thr03.png",     "kwargs": {"intensity": 2.0, "threshold": 0.3, "blur_size": 1.0, "iterations": 1}, "sweep_key": None},
    # Cataracts: stochastic dust/blur pattern; per-pixel alignment is meaningless.
    "cataracts":       {"ref": "cataract_sev04.png",        "kwargs": {}, "sweep_key": "severity", "criterion": "ssim_blur"},
    "cvd":             {"ref": "cvd_deutan_sev06.png",      "kwargs": {"cvd_type": "deuteranomaly"}, "sweep_key": "severity"},
    # Distortion: Unity at default magnitude=1.0. Pin Python sev=1.0 (no sweep).
    "distortion":      {"ref": "distortion_default.png",    "kwargs": {"severity": 1.0}, "sweep_key": None},
    # Note: use DoubleVisionEffect's render (clean monocular UV-shift via
    # Hidden/DoubleVision shader). myDoubleVision's monocular path is broken
    # upstream (references BasicShader, a surface shader incompatible with Blit).
    "double_vision":   {"ref": "doublevisioneffect.png",     "kwargs": {}, "sweep_key": "severity"},
    "field_loss":      {"ref": "fieldloss_default.png",     "kwargs": {}, "sweep_key": "severity"},
    # Pin to Unity capture: 15 stars at full brightness.
    "flickering_stars": {"ref": "flickeringstars.png",      "kwargs": {"n_stars": 15, "severity": 1.0}, "sweep_key": None},
    # Pin to Unity defaults: ~50000 1-px dark grains inside central disc (radius ≈ 0.146 of min dim).
    "floaters":        {"ref": "floaters_dense.png",        "kwargs": {"n_floaters": 50000, "mode": "dark", "floater_size": 1, "circle_radius_uv": 0.146, "severity": 1.0}, "sweep_key": None, "criterion": "ssim_blur"},
    "foveal_darkness": {"ref": "fovealdarkness_max.png",    "kwargs": {}, "sweep_key": "severity"},
    "glitch":          {"ref": "glitch_default.png",        "kwargs": {}, "sweep_key": "severity"},
    # Unity LED ref captured at default Scale=80 — pin Python to scale=80 too
    # (severity is a convenience knob that doesn't map cleanly to the same value).
    "led":             {"ref": "led_default.png",           "kwargs": {"scale": 80.0, "brightness": 1.0, "shape": 1.5, "margin": 0.0, "automatic_ratio": True}, "sweep_key": None},
    # Unity noise captured at intensity=0.25 (1.0 collapses scene to 1/16 brightness).
    # Stochastic — score by SSIM_blur not MAE.
    "noise":           {"ref": "noise_int025.png",          "kwargs": {}, "sweep_key": "severity", "criterion": "ssim_blur"},
    # Nystagmus is a fundamental model-difference, not a tuning problem:
    # Unity = instantaneous frame of a rotating camera (single moment in the
    # oscillation, motion blur on one side only); Python = integrated motion blur
    # over the oscillation window (closer to subjective percept). They cannot
    # match pixel-wise. Score is for record-keeping only.
    "nystagmus":       {"ref": "nystagmus_amp20.png",       "kwargs": {"severity": 1.0}, "sweep_key": None},
    "pixelation":      {"ref": "pixelation_default.png",    "kwargs": {}, "sweep_key": "severity"},
    # Pin to Unity capture: Strength=0.75, LumContribution=0.75.
    "teichopsia":      {"ref": "teichopsia_default.png",    "kwargs": {"severity": 0.75, "lum_contribution": 0.75}, "sweep_key": None},
    "vortex":          {"ref": "vortex_default.png",        "kwargs": {}, "sweep_key": "severity"},
    # Unity wiggle captured at amplitude=0.03 (default 0.01 is invisible at this size).
    # Animation-frozen at capture; pin amplitude and let severity scale further.
    # Score by SSIM_blur: at higher severity the wave is visible but pixel-aligned
    # MAE penalises any displacement, so MAE would always pick the minimum sweep.
    # Wiggle is a time-driven wave; Unity captures live so its phase depends on
    # _Time.y at capture (unknown). Severity is fixed at 1.0 so amplitude=0.03
    # matches Unity; we sweep `timer` instead to align wave phase.
    "wiggle":          {"ref": "wiggle_amp03.png",          "kwargs": {"freq": 12.0, "amplitude": 0.03, "severity": 1.0, "mode": "complex"}, "sweep_key": "timer", "criterion": "ssim_blur", "sweep_override": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]},
    # bcg has no `severity` kwarg — single point eval matching Unity capture
    # (Unity ref was brightness=-25, contrast=+25 in VIP-Sim units; map to our
    # multiplicative scale: brightness 0.75, contrast 1.25, gamma 1.0)
    "bcg":             {"ref": "bcg_b-25_c25.png",          "kwargs": {"brightness": 0.75, "contrast": 1.25, "gamma": 1.0}, "sweep_key": None},
}


def sweep_filter(name: str, cfg: dict, input_img: Image.Image,
                 sweep_values: list[float], save_strip: bool) -> dict:
    ref_path = UNITY_REFS / cfg["ref"]
    if not ref_path.is_file():
        return {"error": f"missing reference {ref_path}"}

    _, ref = load_unity_reference(INPUT_PNG, ref_path)
    fn = vf._FILTER_DISPATCH[name]

    results = []
    sweep_key = cfg["sweep_key"]
    # Per-filter override of the sweep range (e.g. wiggle sweeps timer, not severity).
    points = cfg.get("sweep_override", sweep_values) if sweep_key else [None]

    for v in points:
        kwargs = dict(cfg["kwargs"])
        if sweep_key:
            kwargs[sweep_key] = v
        ours = fn(input_img, **kwargs)
        stats = score_abs_diff(ours, ref)
        results.append({
            "params": kwargs,
            "mae_mean": stats["mae"]["overall"]["mean"],
            "mae_p95": stats["mae"]["overall"]["p95"],
            "ssim": stats["ssim"],
            "ssim_blur": stats["ssim_blur"],
            "deltaE2000_mean": stats["deltaE2000"]["mean"] if stats["deltaE2000"] else None,
            "wasserstein_mean": stats["wasserstein"]["mean"] if stats["wasserstein"] else None,
            "lpips": stats["lpips"],
        })

    # Pick best by configured criterion. Default: minimise mae_mean.
    # SSIM / SSIM_blur are maximise; everything else (MAE, ΔE, Wasserstein,
    # LPIPS) is minimise. Falls back to mae_mean if the chosen metric is
    # None (e.g. ssim_blur with skimage missing).
    criterion = cfg.get("criterion", "mae_mean")
    maximise = criterion in ("ssim", "ssim_blur")
    if all(r.get(criterion) is None for r in results):
        criterion = "mae_mean"
        maximise = False
    best = (max if maximise else min)(
        results, key=lambda r: r[criterion] if r.get(criterion) is not None else (float("-inf") if maximise else float("inf"))
    )
    out = {
        "filter": name,
        "unity_ref": cfg["ref"],
        "sweep_key": sweep_key,
        "criterion": criterion,
        "best": best,
        "all": results,
    }

    if save_strip:
        kwargs = dict(cfg["kwargs"])
        if sweep_key:
            kwargs[sweep_key] = best["params"][sweep_key]
        ours = fn(input_img, **kwargs)
        # Standalone PNGs
        ours.save(COMPARE_DIR / f"{name}_python.png")
        ref.save(COMPARE_DIR / f"{name}_unity.png")
        # Side-by-side: Unity (left) | Python (right)
        w, h = ref.size
        strip = Image.new("RGB", (w * 2, h))
        strip.paste(ref, (0, 0))
        strip.paste(ours, (w, 0))
        strip.save(STRIP_DIR / f"{name}_compare.png")
        # Stacked: Unity (top) | Python (bottom) — full-width detail comparison
        stack = Image.new("RGB", (w, h * 2))
        stack.paste(ref, (0, 0))
        stack.paste(ours, (0, h))
        stack.save(STACK_DIR / f"{name}_stacked.png")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filters", default="",
                        help="Comma-separated subset (default: all)")
    parser.add_argument("--sweep", default="0.1,1.0,0.1",
                        help="lo,hi,step for severity sweep (default 0.1,1.0,0.1)")
    parser.add_argument("--no-strips", action="store_true",
                        help="Skip writing per-filter comparison PNGs")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    COMPARE_DIR.mkdir(exist_ok=True)
    STRIP_DIR.mkdir(exist_ok=True)
    STACK_DIR.mkdir(exist_ok=True)
    lo, hi, step = (float(x) for x in args.sweep.split(","))
    sweep_values = [round(lo + i * step, 4)
                    for i in range(int((hi - lo) / step) + 1)]

    targets = list(FILTERS) if not args.filters else args.filters.split(",")
    input_img = Image.open(INPUT_PNG).convert("RGB")

    print(f"Input: {INPUT_PNG.name}  size={input_img.size}")
    print(f"Sweep ({len(sweep_values)} pts): {sweep_values}")
    print(f"Filters: {len(targets)}\n")

    summary = {}
    rows = []
    for name in targets:
        if name not in FILTERS:
            print(f"  [skip] unknown filter: {name}")
            continue
        cfg = FILTERS[name]
        print(f"  [{name}] ref={cfg['ref']} ...", end=" ", flush=True)
        result = sweep_filter(name, cfg, input_img, sweep_values, not args.no_strips)
        summary[name] = result
        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue
        b = result["best"]
        sb = f" SSIM_blur={b['ssim_blur']:.4f}" if b["ssim_blur"] is not None else ""
        lp = f" LPIPS={b['lpips']:.3f}" if b["lpips"] is not None else ""
        crit = result["criterion"]
        print(f"best({crit}) {cfg['sweep_key'] or 'fixed'}={b['params'].get(cfg['sweep_key']) if cfg['sweep_key'] else '-'}  "
              f"MAE={b['mae_mean']:.2f}  SSIM={b['ssim']:.4f}{sb}{lp}")
        rows.append({
            "filter": name,
            "unity_ref": cfg["ref"],
            "best_sweep_value": b["params"].get(cfg["sweep_key"]) if cfg["sweep_key"] else None,
            "mae_mean": round(b["mae_mean"], 3),
            "mae_p95": round(b["mae_p95"], 3),
            "ssim": round(b["ssim"], 4) if b["ssim"] is not None else None,
            "ssim_blur": round(b["ssim_blur"], 4) if b["ssim_blur"] is not None else None,
            "deltaE2000_mean": round(b["deltaE2000_mean"], 3) if b["deltaE2000_mean"] is not None else None,
            "wasserstein_mean": round(b["wasserstein_mean"], 3) if b["wasserstein_mean"] is not None else None,
            "lpips": round(b["lpips"], 4) if b["lpips"] is not None else None,
        })

    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    csv_path = OUT / "validation_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote:\n  {OUT}/validation_summary.json\n  {csv_path}")
    if not args.no_strips:
        print(f"  {COMPARE_DIR}/<filter>_unity.png      (Unity reference)")
        print(f"  {COMPARE_DIR}/<filter>_python.png     (Python output at best severity)")
        print(f"  {STRIP_DIR}/<filter>_compare.png    (side-by-side: Unity | Python)")


if __name__ == "__main__":
    main()
