#!/usr/bin/env python3
"""Auto-initialize `panel_crop` fields in vipsim_profiles/p*.json.

The participants.png reference is laid out as a regular grid:
  - 4 rows tall, 4 columns wide
  - Each participant occupies 2 columns: [Amsler-grid-annotation | scene-render]
  - Scene render (right column of each pair) is what we diff against
  - Layout: row 0=P1,P2  row 1=P3,P4  row 2=P5,P6  row 3=P7,(empty)

This computes approximate crops from the image dimensions and writes them
into each profile JSON if its current `panel_crop` is the [0,0,0,0] sentinel.

After running this, eyeball the produced crops by running:
    python scripts/abs_diff.py --profile p1 --mode participants_panel
and refine each profile's `panel_crop` manually if needed.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "vipsim_assets"
PROFILES = ROOT / "vipsim_profiles"

# (row_index, col_pair_index)  — col_pair 0 = left half, 1 = right half
LAYOUT = {
    "p1": (0, 0),
    "p2": (0, 1),
    "p3": (1, 0),
    "p4": (1, 1),
    "p5": (2, 0),
    "p6": (2, 1),
    "p7": (3, 0),
}

N_ROWS = 4
N_COL_PAIRS = 2  # 2 participants per row, each = (amsler + scene)


def compute_scene_crop(W: int, H: int, row: int, col_pair: int) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) for the scene panel of (row, col_pair)."""
    row_h = H // N_ROWS
    pair_w = W // N_COL_PAIRS
    panel_w = pair_w // 2  # half a pair = one panel (amsler or scene)

    pair_left = col_pair * pair_w
    scene_left = pair_left + panel_w  # right half of the pair = scene
    scene_right = pair_left + pair_w
    top = row * row_h
    bottom = top + row_h
    return (scene_left, top, scene_right, bottom)


def main():
    panel_path = ASSETS / "participants.png"
    if not panel_path.is_file():
        raise SystemExit(f"Missing {panel_path}")
    W, H = Image.open(panel_path).size
    print(f"participants.png: {W}x{H}")

    for pid, (row, col_pair) in LAYOUT.items():
        pjson = PROFILES / f"{pid}.json"
        if not pjson.is_file():
            print(f"  {pid}: profile JSON missing, skipped")
            continue
        prof = json.loads(pjson.read_text())
        crop = compute_scene_crop(W, H, row, col_pair)
        current = prof.get("panel_crop", [0, 0, 0, 0])
        if current and any(c != 0 for c in current):
            print(f"  {pid}: existing crop {current} kept")
            continue
        prof["panel_crop"] = list(crop)
        pjson.write_text(json.dumps(prof, indent=2) + "\n")
        print(f"  {pid}: set crop {crop}")


if __name__ == "__main__":
    main()
