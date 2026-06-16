#!/usr/bin/env python3
"""Build per-filter grid PNG + animated GIF from burst compare frames.

For each of flickering_stars, wiggle, nystagmus, reads the 30 stacked
Unity/Python compare PNGs from abs_diff_out/<filter>_burst/compare/ and
emits two visualisation artefacts under abs_diff_out/<filter>_burst/:

  - grid.png : 2x3 panel of 6 evenly-spaced frames, with a per-frame caption
  - loop.gif : animated GIF cycling through all 30 frames

Downscaled to keep file sizes reasonable for GitHub markdown rendering.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
FILTERS = ["flickeringstars", "wiggle", "nystagmus"]

# Render targets (per-frame)
GRID_CELL_W = 600
GIF_FRAME_W = 360
GIF_DURATION_MS = 200  # 5 fps loop
GIF_COLORS = 64


def load_frame(path: Path, target_w: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    new_h = int(h * target_w / w)
    return im.resize((target_w, new_h), Image.LANCZOS)


def build_grid(compare_dir: Path, out_path: Path, frame_count: int = 30,
                cols: int = 3, rows: int = 2) -> None:
    picks = [int(round(i * (frame_count - 1) / (cols * rows - 1)))
             for i in range(cols * rows)]
    frames = [load_frame(compare_dir / f"frame_{i:03d}.png", GRID_CELL_W)
              for i in picks]
    cell_w, cell_h = frames[0].size
    caption_h = 28
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + caption_h)),
                       (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for idx, (frame_idx, im) in enumerate(zip(picks, frames)):
        r, c = divmod(idx, cols)
        x = c * cell_w
        y = r * (cell_h + caption_h)
        draw.rectangle([x, y, x + cell_w, y + caption_h], fill=(30, 30, 30))
        draw.text((x + 8, y + 4), f"frame {frame_idx:03d}",
                  fill=(255, 255, 255), font=font)
        canvas.paste(im, (x, y + caption_h))
    canvas.save(out_path, optimize=True)
    print(f"  grid -> {out_path.relative_to(REPO)} ({canvas.size})")


def build_gif(compare_dir: Path, out_path: Path, frame_count: int = 30) -> None:
    frames = [load_frame(compare_dir / f"frame_{i:03d}.png", GIF_FRAME_W)
              for i in range(frame_count)]
    # Convert to a shared palette so the GIF stays small.
    palette_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=GIF_COLORS)
                      for f in frames]
    palette_frames[0].save(
        out_path, save_all=True, append_images=palette_frames[1:],
        duration=GIF_DURATION_MS, loop=0, optimize=True, disposal=2,
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"  gif  -> {out_path.relative_to(REPO)} ({frames[0].size}, {size_kb:.0f} KB)")


def main() -> None:
    for name in FILTERS:
        compare_dir = REPO / "abs_diff_out" / f"{name}_burst" / "compare"
        if not compare_dir.exists():
            print(f"skip {name}: {compare_dir} not found")
            continue
        out_dir = compare_dir.parent
        print(f"[{name}]")
        build_grid(compare_dir, out_dir / "grid.png")
        build_gif(compare_dir, out_dir / "loop.gif")


if __name__ == "__main__":
    main()
