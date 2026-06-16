#!/usr/bin/env python3
"""Equivalence check: GPU (PyTorch) port vs the validated NumPy port.

The NumPy port (`vipsim_filters.py`) is already validated against the Unity
reference renders (see vipsim_equivalence_report.md). This script closes the
chain for the GPU port (`vipsim_gpu.py`) by comparing each GPU filter's output
against the matching NumPy filter on the same input. If GPU ~= NumPy and
NumPy ~= Unity, then transitively GPU ~= Unity.

Only the filters currently implemented in vipsim_gpu.VipSimGPU are checked
(today: cvd, bcg, blur, vortex). The rest are reported as "not ported yet",
so this doubles as a porting checklist — it auto-covers new filters as they
are added to vipsim_gpu.py.

Runs on GPU if available, else CPU (torch picks automatically). Requires
torch (optional dep; `pip install torch`).

Usage:
    python validation_scripts/validate_gpu_vs_numpy.py
    python validation_scripts/validate_gpu_vs_numpy.py --input vipsim_assets/Trafficscene_2048x1024.png
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

from validation_scripts.abs_diff import score_abs_diff  # noqa: E402
import vipsim_filters as vf  # noqa: E402

OUT = ROOT / "abs_diff_out"
DEFAULT_INPUT = ROOT / "vipsim_assets" / "Trafficscene_2048x1024.png"

# Filters implemented in vipsim_gpu.VipSimGPU. Each entry pairs the GPU method
# (name + kwargs) with the matching NumPy filter (name + kwargs), using the
# SAME parameters on both sides so the comparison is apples-to-apples.
#
# The parameter values mirror the points used in the Unity validation
# (validate_all_filters.py) so the GPU output is checked at a meaningful
# operating point, not an arbitrary one.
GPU_FILTERS = {
    "cvd":    {"gpu_kwargs": {"severity": 0.6},
               "np_name": "cvd",
               "np_kwargs": {"cvd_type": "deuteranomaly", "severity": 0.6}},
    "bcg":    {"gpu_kwargs": {"brightness": 0.75, "contrast": 1.25, "gamma": 1.0},
               "np_name": "bcg",
               "np_kwargs": {"brightness": 0.75, "contrast": 1.25, "gamma": 1.0}},
    "blur":   {"gpu_kwargs": {"severity": 0.4},
               "np_name": "blur",
               "np_kwargs": {"severity": 0.4}},
    "vortex": {"gpu_kwargs": {"severity": 0.5},
               "np_name": "vortex",
               "np_kwargs": {"severity": 0.5}},
}

# Full filter list (for the "not ported yet" report), taken from the NumPy
# dispatch so the checklist stays in sync with what exists.
ALL_NUMPY_FILTERS = list(vf._FILTER_DISPATCH.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        import vipsim_gpu as vg
    except Exception as e:
        sys.exit(f"GPU check needs torch + vipsim_gpu: {e}\n"
                 f"Install with: pip install torch")

    if not args.input.is_file():
        sys.exit(f"Missing input: {args.input}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Input:  {args.input.name}\n")

    sim = vg.VipSimGPU(device=device)
    img = Image.open(args.input).convert("RGB")
    frame = np.asarray(img)  # [H, W, 3] uint8

    rows = []
    summary = {}
    for name, cfg in GPU_FILTERS.items():
        # GPU path: uint8 frame -> tensor -> filter -> uint8 frame -> PIL
        x = vg.frames_to_gpu(frame, sim.device)
        y = getattr(sim, name)(x, **cfg["gpu_kwargs"])
        gpu_img = Image.fromarray(vg.gpu_to_frames(y)[0])

        # NumPy path: PIL -> filter -> PIL
        np_fn = vf._FILTER_DISPATCH[cfg["np_name"]]
        np_img = np_fn(img, **cfg["np_kwargs"])

        stats = score_abs_diff(gpu_img, np_img)
        rows.append({
            "filter": name,
            "gpu_kwargs": json.dumps(cfg["gpu_kwargs"]),
            "mae_mean": round(stats["mae"]["overall"]["mean"], 3),
            "ssim": round(stats["ssim"], 4) if stats["ssim"] is not None else None,
            "ssim_blur": round(stats["ssim_blur"], 4) if stats["ssim_blur"] is not None else None,
        })
        summary[name] = stats
        sb = f" SSIM_blur={stats['ssim_blur']:.4f}" if stats["ssim_blur"] is not None else ""
        print(f"  [{name:7s}] vs NumPy:  MAE={stats['mae']['overall']['mean']:6.2f}  "
              f"SSIM={stats['ssim']:.4f}{sb}")

    # Porting checklist: NumPy filters not yet present in the GPU port.
    not_ported = [f for f in ALL_NUMPY_FILTERS if f not in GPU_FILTERS]
    print(f"\nGPU port covers {len(GPU_FILTERS)}/{len(ALL_NUMPY_FILTERS)} filters.")
    print(f"Not ported to GPU yet ({len(not_ported)}): {', '.join(not_ported)}")

    OUT.mkdir(exist_ok=True)
    (OUT / "gpu_vs_numpy_summary.json").write_text(json.dumps({
        "device": device,
        "input": args.input.name,
        "checked": summary,
        "not_ported": not_ported,
    }, indent=2))
    csv_path = OUT / "gpu_vs_numpy_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote:\n  {OUT}/gpu_vs_numpy_summary.json\n  {csv_path}")
    print("\nNote: high SSIM (>0.98) = GPU matches the validated NumPy port "
          "(=> matches Unity transitively). A low score flags a GPU-port "
          "discrepancy to investigate.")


if __name__ == "__main__":
    main()
