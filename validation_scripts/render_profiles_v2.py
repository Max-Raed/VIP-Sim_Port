#!/usr/bin/env python3
"""Regenerate the paper-vs-ours comparison renders for vipsim_profiles_v2/.

For each profile pN.json:
  - load the paper's reference panel from renders/ref-Images/pN.png (clean,
    whitespace-free extractions); falls back to cropping
    vipsim_assets/participants.png via the profile's `panel_crop` if missing,
  - apply the profile's filter stack to vipsim_assets/Trafficscene.png,
    resized to the panel resolution (the paper's street scene),
  - stitch paper (left) vs ours (right) with a small label strip,
  - write vipsim_profiles_v2/renders/pN.png.

Usage:  python scripts/render_profiles_v2.py [p1 p2 ...]   (default: all)
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import vipsim_filters as vf  # noqa: E402

PROFILE_DIR = ROOT / "vipsim_profiles_v2"
RENDER_DIR = PROFILE_DIR / "renders"
REF_DIR = RENDER_DIR / "ref-Images"
PARTICIPANTS = ROOT / "vipsim_assets" / "participants.png"
SCENE = ROOT / "vipsim_assets" / "Trafficscene.png"

# The paper's Figure-6 panels each show a slightly different ~92-96% crop of
# the Trafficscene (measured per panel by gradient cross-correlation against
# the clean refs, 2026-07-09). Crop the source to the matching ref framing
# before filtering, else the comparison is misaligned by a few percent.
SCENE_CROP_FRAC = {
    "p1": (0.0225, 0.0225, 0.9725, 0.9725),
    "p2": (0.0312, 0.0316, 0.9762, 0.9766),
    "p3": (0.0369, 0.0325, 0.9719, 0.9675),
    "p4": (0.0340, 0.0382, 0.9740, 0.9782),
    "p5": (0.0595, 0.0315, 0.9895, 0.9615),
    "p6": (0.0180, 0.0230, 0.9780, 0.9830),
    "p7": (0.0225, 0.0230, 0.9775, 0.9780),
}
DEFAULT_CROP = (0.0231, 0.0230, 0.9730, 0.9727)


def scene_ref_framing(scene: Image.Image, pid: str = "") -> Image.Image:
    # variants (p1b, p3b, ...) inherit their base profile's framing
    frac = SCENE_CROP_FRAC.get(pid) or SCENE_CROP_FRAC.get(pid[:2], DEFAULT_CROP)
    w, h = scene.size
    x0, y0, x1, y1 = frac
    return scene.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))

HEADER_H = 0
GAP = 0


def apply_stack(img, steps):
    out = img
    for step in steps:
        if step.get("_disabled"):
            continue
        kwargs = {k: v for k, v in step.items()
                  if k != "filter" and not k.startswith("_")}
        fn = vf._FILTER_DISPATCH[step["filter"]]
        try:
            out = fn(out.copy(), **kwargs)
        except TypeError:
            kwargs.pop("severity", None)
            out = fn(out.copy(), **kwargs)
        out = out.convert("RGB")
    return out


def render(pid, participants, scene):
    prof = json.loads((PROFILE_DIR / f"{pid}.json").read_text())
    ref_path = REF_DIR / f"{pid}.png"
    if ref_path.is_file():
        paper = Image.open(ref_path).convert("RGB")
    else:
        paper = participants.crop(tuple(prof["panel_crop"]))
    framed = scene_ref_framing(scene, pid)
    ours = apply_stack(framed.resize(paper.size, Image.LANCZOS), prof["filters"])

    w, h = paper.size
    canvas = Image.new("RGB", (w * 2 + GAP, h + HEADER_H), "white")
    canvas.paste(paper, (0, HEADER_H))
    canvas.paste(ours, (w + GAP, HEADER_H))

    out_path = RENDER_DIR / f"{pid}.png"
    canvas.save(out_path)
    print(f"{pid}: {out_path}")


def main():
    ids = sys.argv[1:] or [f"p{i}" for i in range(1, 8)]
    participants = Image.open(PARTICIPANTS).convert("RGB")
    scene = Image.open(SCENE).convert("RGB")
    RENDER_DIR.mkdir(exist_ok=True)
    for pid in ids:
        render(pid, participants, scene)


if __name__ == "__main__":
    main()
