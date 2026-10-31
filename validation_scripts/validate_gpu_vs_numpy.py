#!/usr/bin/env python3
"""Equivalence check: GPU (PyTorch) port vs the validated NumPy port.

The NumPy port (`vipsim/filters.py`) is already validated against the Unity
reference renders (see vipsim_equivalence_report.md). This script closes the
chain for the GPU port (`vipsim/gpu.py`) by comparing each GPU filter's output
against the matching NumPy filter on the same input. If GPU == NumPy and
NumPy ~= Unity, then transitively GPU ~= Unity.

Every VipSimGPU method whose name is in vipsim.filters._FILTER_DISPATCH is
checked automatically: with the kwargs in KWARGS (the operating points of the
Unity validation in validate_all_filters.py), or severity 0.5 for a filter not
listed there. Both sides get the same kwargs (the GPU methods take the CPU's).
Each filter must stay within its documented tolerance,
vipsim.gpu.TWIN_TOLERANCE: the max and mean of |gpu - numpy| in 8-bit levels.
Dispatch filters without a GPU method are reported as not ported. The exit
status is 1 when a filter exceeds its tolerance or is not ported, so this
doubles as a porting checklist and a CI gate.

Runs on CUDA if available, else on the CPU (or --device). Requires torch
(optional dep; `pip install "vipsim[gpu]"`).

Usage:
    python validation_scripts/validate_gpu_vs_numpy.py
    python validation_scripts/validate_gpu_vs_numpy.py --device cpu
    python validation_scripts/validate_gpu_vs_numpy.py --input vipsim_assets/Trafficscene_2048x1024.png
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from validation_scripts.abs_diff import score_abs_diff  # noqa: E402
from vipsim import filters as vf  # noqa: E402

OUT = ROOT / "abs_diff_out"
DEFAULT_INPUT = ROOT / "vipsim_assets" / "Trafficscene_2048x1024.png"

# Representative kwargs per filter, the SAME on both sides so the comparison
# is apples-to-apples. They mirror the points used in the Unity validation
# (validate_all_filters.py), so the GPU output is checked at a meaningful
# operating point, not an arbitrary one. Filters not listed run at
# severity 0.5.
KWARGS = {
    "cvd":              {"cvd_type": "deuteranomaly", "severity": 0.6},
    "bcg":              {"brightness": 0.75, "contrast": 1.25, "gamma": 1.0},
    "blur":             {"severity": 0.4},
    "bloom":            {"intensity": 2.0, "threshold": 0.3, "blur_size": 1.0,
                         "iterations": 1},
    "cataracts":        {"severity": 0.4},
    "distortion":       {"severity": 1.0},
    "flickering_stars": {"n_stars": 15, "severity": 1.0},
    "floaters":         {"n_floaters": 50000, "mode": "dark", "floater_size": 1,
                         "circle_radius_uv": 0.146, "severity": 1.0},
    "led":              {"scale": 80.0, "brightness": 1.0, "shape": 1.5,
                         "margin": 0.0, "automatic_ratio": True},
    "nystagmus":        {"severity": 1.0},
    "teichopsia":       {"severity": 0.75, "lum_contribution": 0.75},
    "vortex":           {"severity": 0.5},
    "wiggle":           {"freq": 12.0, "amplitude": 0.03, "severity": 1.0,
                         "mode": "complex", "timer": 1.0},
}


def _missing_optional_dep(e: BaseException) -> bool:
    """True when e (or its cause) is scikit-image not being installed."""
    while e is not None:
        if isinstance(e, ModuleNotFoundError) and (e.name or "").split(".")[0] == "skimage":
            return True
        e = e.__cause__
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--device", default=None,
                    help="torch device (default: cuda if available, else cpu)")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="directory for gpu_vs_numpy_summary.{json,csv}")
    ap.add_argument("--allow-skip", action="store_true",
                    help="exit 0 even if a filter was skipped for a missing optional dependency")
    args = ap.parse_args()

    try:
        import torch
        from vipsim import gpu as vg
    except Exception as e:
        sys.exit(f"GPU check needs torch + vipsim.gpu: {e}\n"
                 f"Install with: pip install 'vipsim[gpu]'")

    if not args.input.is_file():
        sys.exit(f"Missing input: {args.input}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sim = vg.VipSimGPU(device=device)
    print(f"Device: {sim.device}"
          + (f" ({torch.cuda.get_device_name(sim.device)})" if sim.device.type == "cuda" else ""))
    print(f"Input:  {args.input.name}\n")

    img = Image.open(args.input).convert("RGB")
    frame = np.asarray(img)  # [H, W, 3] uint8
    x = vg.frames_to_gpu(frame, sim.device)

    names = list(vf._FILTER_DISPATCH)
    checked = [n for n in names if callable(getattr(sim, n, None))]
    not_ported = [n for n in names if n not in checked]

    rows, summary, skipped, failed = [], {}, {}, []
    for name in checked:
        kwargs = KWARGS.get(name, {"severity": 0.5})
        try:
            # GPU path: uint8 frame -> tensor -> filter -> uint8 frame -> PIL
            gpu_img = Image.fromarray(vg.gpu_to_frames(getattr(sim, name)(x, **kwargs))[0])
            # NumPy path: PIL -> filter -> PIL
            np_img = vf._FILTER_DISPATCH[name](img, **kwargs)
        except ImportError as e:           # optional extra (teichopsia: scikit-image)
            if not _missing_optional_dep(e):
                raise
            skipped[name] = str(e)
            print(f"  [{name:16s}] skipped: {e}")
            continue

        diff = np.abs(np.asarray(gpu_img, int) - np.asarray(np_img, int))
        tol_max, tol_mean = vg.TWIN_TOLERANCE[name]
        ok = bool(diff.max() <= tol_max and diff.mean() <= tol_mean)
        if not ok:
            failed.append(name)
        stats = score_abs_diff(gpu_img, np_img)
        summary[name] = {
            "kwargs": kwargs,
            "max_abs_diff": int(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "differing_values": int(np.count_nonzero(diff)),
            "tolerance": {"max": tol_max, "mean": tol_mean},
            "within_tolerance": ok,
            **stats,
        }
        rows.append({
            "filter": name,
            "kwargs": json.dumps(kwargs),
            "max_abs_diff": int(diff.max()),
            "mean_abs_diff": round(float(diff.mean()), 6),
            "tol_max": tol_max,
            "tol_mean": tol_mean,
            "within_tolerance": ok,
            "ssim": round(stats["ssim"], 4) if stats["ssim"] is not None else None,
            "ssim_blur": round(stats["ssim_blur"], 4) if stats["ssim_blur"] is not None else None,
        })
        print(f"  [{name:16s}] vs NumPy:  max={diff.max():3d}  mean={diff.mean():.6f}  "
              f"(tolerance {tol_max}, {tol_mean})  {'ok' if ok else 'EXCEEDS TOLERANCE'}")

    print(f"\nGPU port covers {len(checked)}/{len(names)} filters; "
          f"{len(summary)} checked, {len(skipped)} skipped.")
    if not_ported:
        print(f"Not ported to GPU yet ({len(not_ported)}): {', '.join(not_ported)}")
    if failed:
        print(f"Exceeding tolerance ({len(failed)}): {', '.join(failed)}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gpu_vs_numpy_summary.json").write_text(json.dumps({
        "device": sim.device.type,
        "input": args.input.name,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pillow": PIL.__version__, "torch": torch.__version__},
        "checked": summary,
        "skipped": skipped,
        "not_ported": not_ported,
        "exceeding_tolerance": failed,
    }, indent=2))
    csv_path = args.out / "gpu_vs_numpy_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote:\n  {args.out}/gpu_vs_numpy_summary.json\n  {csv_path}")
    print("\nNote: max/mean = |GPU - NumPy| in 8-bit levels; every twin is "
          "bit-exact where tested, so anything above 0 flags a GPU-port "
          "discrepancy (or another BLAS's dot order, which cvd, bcg and "
          "foveal_darkness are allowed 1 LSB for).")
    if skipped and not args.allow_skip:
        print(f"Skipped ({len(skipped)}): {', '.join(skipped)}; pass --allow-skip to accept")
    sys.exit(1 if failed or not_ported or (skipped and not args.allow_skip) else 0)


if __name__ == "__main__":
    main()
