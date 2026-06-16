#!/usr/bin/env python3
"""Center-crop + resize Trafficscene.png to a power-of-two 2:1 size for
deterministic Python-vs-Unity ABS-diff validation.

Trafficscene.png is 4896x3264 (3:2). Unity's field-loss / double-vision
shaders silently round non-power-of-two textures up to the next POT, which
introduces resampling artifacts. To avoid that, we feed both pipelines the
same pre-cropped POT 2:1 image.

Output: vipsim_assets/Trafficscene_2048x1024.png
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "vipsim_assets" / "Trafficscene.png"
DST = ROOT / "vipsim_assets" / "Trafficscene_2048x1024.png"

TARGET_W = 2048
TARGET_H = 1024


def main():
    if not SRC.is_file():
        raise SystemExit(f"Missing {SRC}")
    im = Image.open(SRC).convert("RGB")
    W, H = im.size  # 4896, 3264

    # Center-crop to 2:1 aspect (W:H = 2:1). Source is 3:2, so we crop
    # height: keep H_new = W / 2 = 2448, drop (3264 - 2448)/2 = 408 from
    # top and bottom.
    target_aspect = TARGET_W / TARGET_H  # 2.0
    src_aspect = W / H                     # 1.5

    if src_aspect > target_aspect:
        # source is wider than target → crop width
        new_w = int(round(H * target_aspect))
        x0 = (W - new_w) // 2
        box = (x0, 0, x0 + new_w, H)
    else:
        # source is taller than target → crop height
        new_h = int(round(W / target_aspect))
        y0 = (H - new_h) // 2
        box = (0, y0, W, y0 + new_h)

    cropped = im.crop(box)
    print(f"Source {W}x{H} → cropped to {cropped.size} (box={box})")

    out = cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    out.save(DST)
    print(f"Wrote {DST} ({out.size})")


if __name__ == "__main__":
    main()
