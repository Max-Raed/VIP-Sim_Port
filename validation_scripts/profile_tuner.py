#!/usr/bin/env python3
"""Interactive tuner for the vipsim_profiles_v2 drafts.

Run:    python validation_scripts/profile_tuner.py   (then open http://localhost:8765)

Left: paper ref (renders/ref-Images/pN.png). Right: our render, re-applied live
as you drag sliders. Every numeric parameter of every filter step in the
profile JSON gets a slider; Save writes the values back into the JSON
(all documentation fields are preserved). Then regenerate the comparison
PNGs with validation_scripts/render_profiles_v2.py.
"""
import base64
import io
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vipsim import filters as vf  # noqa: E402

PROFILE_DIR = ROOT / "vipsim_profiles_v2"
REF_DIR = PROFILE_DIR / "renders" / "ref-Images"
SCENE_FULL = Image.open(ROOT / "vipsim_assets" / "Trafficscene.png").convert("RGB")
PORT = 8765
# Profile ids name files under PROFILE_DIR / REF_DIR, so they must never
# carry path separators, "..", or a drive.
_PID_RE = re.compile(r"p\d[a-z0-9_-]*")


def valid_pid(pid) -> bool:
    return isinstance(pid, str) and _PID_RE.fullmatch(pid) is not None

_render_lock = threading.Lock()
_scene_cache = {}
# Session-only overrides from the upload buttons: pid -> PIL Image.
# Not persisted; restart the tuner to reset everything.
_custom = {"ref": {}, "base": {}}


def scene_for(pid, size):
    """Scene cropped to this profile's paper framing, resized to ref size."""
    if pid in _custom["base"]:
        # custom base image: no paper crop, plain resize
        return _custom["base"][pid].resize(size, Image.LANCZOS)
    from render_profiles_v2 import scene_ref_framing
    key = (pid[:2], size)
    if key not in _scene_cache:
        _scene_cache[key] = scene_ref_framing(SCENE_FULL, pid).resize(
            size, Image.LANCZOS)
    return _scene_cache[key]


def ref_image(pid):
    """Reference image for pid: custom upload > own ref > base profile ref."""
    if pid in _custom["ref"]:
        return _custom["ref"][pid]
    if not valid_pid(pid):
        return None
    f = REF_DIR / f"{pid}.png"
    if not f.is_file() and len(pid) >= 2:
        f = REF_DIR / f"{pid[:2]}.png"
    return Image.open(f).convert("RGB") if f.is_file() else None


# Starting values for steps added via the UI / wizard. Anything not listed
# gets {"severity": 0.3}.
NEW_STEP_DEFAULTS = {
    "foveal_darkness": {"severity": 0.4, "center": [0.5, 0.5],
                        "radius_x": 0.15, "radius_y": 0.12, "edge_width": 0.3},
    "field_loss": {"severity": 0.5, "field_type": "central"},
    "vortex": {"severity": 0.4, "center": [0.5, 0.5], "vortex_radius": 0.2,
               "suction_strength": 1.0},
    "cvd": {"severity": 0.5, "cvd_type": "deuteranomaly"},
    "bcg": {"brightness": 1.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0},
    "pixelation": {"pixel_size": 80, "hard": True, "sampling": "point"},
    "floaters": {"severity": 0.3, "n_floaters": 1500, "mode": "light",
                 "center": [0.5, 0.5], "spread_x": 0.2, "spread_y": 0.2, "seed": 7},
    "double_vision": {"severity": 0.1, "axis": "horizontal"},
    "nystagmus": {"severity": 0.05, "axis": "horizontal"},
    "blur": {"severity": 0.05},
    "cataracts": {"severity": 0.2},
    "bloom": {"severity": 0.3},
    "vignette": {"severity": 0.3, "radius": 0.4, "softness": 0.5, "mode": "dark"},
    "distortion": {"severity": 0.1},
    "detail_loss": {"severity": 0.3},
    "teichopsia": {"severity": 0.3},
    "flickering_stars": {"severity": 0.3},
}


# ---------------------------------------------------------------------------
# Plain-language documentation. Single source of truth: injected into the page
# JS AND rendered on the /docs page. Style rule: no em dashes, short sentences.
# ---------------------------------------------------------------------------
FILTER_DESC = {
    "distortion": "Makes the whole picture wavy, like looking through uneven glass.",
    "vortex": "Pulls and smears the picture around one point, like a whirlpool.",
    "nystagmus": "Smears the picture in one direction, as if the eyes were shaking.",
    "blur": "Puts the whole picture out of focus.",
    "bcg": "Changes overall lightness and how strong the colors are.",
    "bloom": "Makes bright spots glow outward and swallow nearby detail.",
    "cataracts": "Lays a grainy fog over everything, like a dirty window.",
    "foveal_darkness": "Puts a soft oval spot on the picture (dark, white, or grey).",
    "field_loss": "Makes part of the picture blurry: one patch, or all the edges.",
    "floaters": "Adds small drifting specks, like dust floating in the eye.",
    "teichopsia": "Adds a shimmering zigzag arc, as during a migraine.",
    "flickering_stars": "Scatters little sparkles over the picture.",
    "double_vision": "Adds a faint second copy of the picture, shifted sideways.",
    "cvd": "Simulates color blindness: some colors become hard to tell apart.",
    "pixelation": "Turns the picture into square blocks.",
    "led": "Makes the picture look like a screen of round LED dots.",
    "wiggle": "Makes the picture wave back and forth like heat shimmer.",
    "noise": "Adds random grain, like an old TV without signal.",
    "glitch": "Adds digital artifacts: shifted lines and broken colors.",
    "detail_loss": "Removes fine details; big shapes stay.",
    "vignette": "Darkens (or lightens) the corners and edges.",
}
FILTER_MORE = {
    "distortion": "Like looking through uneven glass. severity 0.05 gives a slight shimmer, 0.15 is clearly wavy, 0.4 and up makes reading impossible.",
    "vortex": "Everything near the center gets pulled in and smeared, like water going down a drain. severity 0.3 is a soft smear, 0.6 a strong pull. The optional dot and glowing orb imitate what some patients see at the center.",
    "nystagmus": "The picture looks smeared in one direction, as if the eyes moved during a photo. severity 0.03 is slight, 0.1 a strong smear.",
    "blur": "The whole picture goes out of focus. severity 0.01 is barely visible, 0.05 feels like missing your glasses, 0.2 leaves only rough shapes.",
    "bcg": "brightness: 1 = unchanged, 1.5 = much brighter, 2.5+ = heavy white-out. contrast: 1 = unchanged, 0.5 = pale, 0 = flat grey. saturation: 1 = normal colors, 0 = black-and-white. gamma: below 1 lifts dark areas, above 1 darkens them.",
    "bloom": "Bright spots like sky or lamps glow outward and swallow nearby detail. severity 0.2 is a soft glow, 0.5 gives strong halos.",
    "cataracts": "A grainy, foggy veil over everything, like a dirty window. This is a VERY strong filter: severity 0.05 is already visible, 0.2 is heavy fog.",
    "foveal_darkness": "This is the classic scotoma (blind spot). Position it with the buttons below or by dragging on the image when the step is selected. edge_width makes the border softer. mode picks dark, milky white, or grey. invert applies the effect AROUND the oval instead of inside it.",
    "field_loss": "central = one blurry patch or an uneven fog, peripheral = blurry edges (tunnel vision). severity sets how blurry it gets.",
    "floaters": "Small drifting specks like dust in the eye. n_floaters sets how many, severity how visible they are. mode light/dark makes white or black dots.",
    "teichopsia": "A shimmering zigzag arc, as seen during a migraine aura.",
    "flickering_stars": "Little sparkles scattered over the picture.",
    "double_vision": "A faint second copy of the picture, shifted to the side. severity 0.05 is a slight ghost, 0.2 clearly double.",
    "cvd": "Color blindness. Some colors become hard to tell apart, for example red and green. severity 1 = full deficiency, 0.5 = partial.",
    "pixelation": "The picture becomes square blocks. pixel_size is the number of blocks across: 20 = huge blocks, 200 = fine grain.",
    "led": "Looks like a screen made of round LED dots.",
    "wiggle": "The picture waves back and forth like heat shimmer.",
    "noise": "Random grain like an old TV without signal.",
    "glitch": "Digital artifacts: shifted lines and broken colors.",
    "detail_loss": "Fine details vanish while big shapes stay. severity 0.2 softens textures, 0.5 leaves only outlines.",
    "vignette": "The corners and edges get darker (or lighter), like looking through a tube.",
}
PARAM_DESC = {
    "severity": "Strength: 0 = off, 0.2 = subtle, 0.5 = clearly visible, 0.8+ = strong. (Cataracts is much stronger: 0.05 is already visible.)",
    "center": "Position of the effect [y, x]. Use the buttons below, or select the step and drag on the image.",
    "mask_center": "Position of the region. Use the buttons below to place it.",
    "mask_radius": "Size of the region circle (fraction of the image).",
    "radius": "Size (fraction of the image).",
    "radius_x": "Width of the oval.",
    "radius_y": "Height of the oval (vortex: upper reach).",
    "radius_y_down": "Vertical reach of the lower half.",
    "edge_width": "How soft the border fades out. 0 = hard edge, higher = softer.",
    "band_width": "How far the band reaches outward before fading back to normal.",
    "vortex_radius": "How far the pull reaches.",
    "suction_strength": "How hard it pulls inward.",
    "swirl_strength": "Rotation added to the pull. 0 = straight inward.",
    "falloff_exponent": "How fast the pull weakens toward the rim. 2 = Unity original; lower = more pull at the rim.",
    "fade_width": "Softness of the outer rim. 0 = hard cut.",
    "inner_circle_radius": "Size of the center dot. 0 = no dot.",
    "inner_color": "Dot brightness: 0 = black, 1 = white.",
    "inner_center": "Position of the dot.",
    "inner_fade": "Softness of the dot edge.",
    "inner_opacity": "Dot opacity: lower = more see-through.",
    "halo_radius": "Size of the glowing orb core. 0 = no orb.",
    "halo_thickness": "How far the orb glow bleeds outward.",
    "halo_color": "Orb brightness: 0 = black, 1 = white.",
    "halo_opacity": "Orb strength.",
    "halo_center": "Position of the orb.",
    "brightness": "Overall lightness: 1 = unchanged, above 1 = brighter.",
    "contrast": "Contrast: 1 = unchanged. LOW values also wash colors toward grey.",
    "saturation": "Color amount: 1 = normal, 0 = black-and-white.",
    "gamma": "Midtones: below 1 lifts dark areas, above 1 darkens them.",
    "threshold": "Only pixels brighter than this glow. Lower = more things glow.",
    "intensity": "Glow strength.",
    "blur_size": "How far the glow spreads.",
    "iterations": "How many times the glow is applied. More = stronger.",
    "overlay_scale": "Coverage of the overlay texture: bigger = covers more.",
    "field_type": "central = blurry patch or fog, peripheral = blurry edges.",
    "mode": "dark, light, or desat (desat drains the color to grey).",
    "invert": "On = the effect hits everything AROUND the oval; the oval stays normal.",
    "axis": "Direction of the effect.",
    "cvd_type": "Which color deficiency to simulate.",
    "pixel_size": "Number of blocks across the image. Smaller number = bigger blocks.",
    "hard": "On = blocks with hard square edges (true mosaic). Off = soft, faithful shader look.",
    "sampling": "point = each block takes one local color (bold chunky blocks). average = blocks are mixed, muddier.",
    "n_floaters": "How many dots.",
    "floater_size": "Base dot size.",
    "size_variance": "On = dots vary in size. Off = all the same size.",
    "dot_brightness": "Dot color: 1 = pure white, 0.8 = off-white, 0 = black.",
    "spread": "Cloud size (spread of the spray).",
    "spread_x": "Cloud width.",
    "spread_y": "Cloud height.",
    "seed": "Random arrangement. Change it for a different dot pattern.",
    "angle": "Tilt of the cluster in degrees. 0 = upright.",
    "region": "Hand-drawn cluster shape. Dots are placed inside this outline.",
    "region_feather": "Soft edge of the drawn region. 0 = hard boundary.",
    "region_center_bias": "Dot arrangement in the drawn area: 0 = even, 1 = packed toward the middle.",
    "distribution": "How the dots spread: gaussian = cloud around center, uniform = everywhere, disc = filled circle.",
    "use_overlay_asset": "On = use the paper's irregular overlay texture instead of a plain shape.",
    "warp_centers": "Local wave spots: [x, y, radius, strength] each.",
    "noise_amount": "Wobble on the rim.",
    "noise_scale": "Rim wobble frequency.",
    "strength": "How dark or bright the corners get.",
    "softness": "How gradual the corner shading fades.",
    "scale": "Size of the LED dots.",
    "shape": "Roundness of each LED dot.",
    "margin": "Dark gap between the LED dots.",
    "automatic_ratio": "On = keep the dots square regardless of image shape.",
    "amplitude": "How far the waves push the picture around.",
    "freq": "How many waves fit across the picture.",
    "circle_radius_uv": "Size of the zigzag arc (fraction of the image).",
    "color_space": "linear = faithful to the Unity shader, srgb = simpler math directly on screen colors.",
    "contrast_coeff": "How strongly the contrast is reduced.",
    "gaze": "Where the eye is looking [y, x]. The effect is anchored there.",
    "grid_offset_x": "Shifts the block grid sideways by a fraction of one block (to line the blocks up with a reference).",
    "grid_offset_y": "Shifts the block grid down by a fraction of one block.",
    "lum_contribution": "How much the brightness feeds into the effect.",
    "max_displace": "How far the glitch lines shift sideways.",
    "n_stars": "How many sparkles.",
    "sat_boost": "Extra color boost.",
    "timer": "Frozen animation time: changes which moment of the movement you see.",
    "use_brightness": "On = also change the brightness.",
    "use_contrast": "On = also change the contrast.",
    "use_frosting": "On = add the frosty grain layer.",
}
# Choice parameters rendered as dropdowns: "filter.param" -> valid values.
ENUM_OPTIONS = {
    "cvd.cvd_type": ["protanomaly", "deuteranomaly", "tritanomaly", "monochrome"],
    "field_loss.field_type": ["central", "peripheral"],
    "double_vision.axis": ["horizontal", "vertical", "diagonal"],
    "nystagmus.axis": ["horizontal", "vertical"],
    "floaters.distribution": ["gaussian", "uniform", "disc"],
    "pixelation.sampling": ["point", "average"],
    "foveal_darkness.mode": ["dark", "light", "desat"],
    "vignette.mode": ["dark", "light"],
    "floaters.mode": ["light", "dark"],
    "bcg.color_space": ["linear", "srgb"],
    "wiggle.mode": ["complex", "simple"],
}


def filter_signatures():
    """Full parameter list per filter (introspected), so every instance shows
    the SAME controls even when the profile JSON omits a parameter."""
    import inspect
    out = {}
    for name, fn in vf._FILTER_DISPATCH.items():
        params = {}
        for pname, p in inspect.signature(fn).parameters.items():
            if pname == "img":
                continue
            d = p.default
            if d is inspect.Parameter.empty:
                d = None
            if isinstance(d, tuple):
                d = list(d)
            params[pname] = d
        out[name] = params
    return out


def docs_html():
    """Human documentation page for every filter, generated from the dicts."""
    sigs = filter_signatures()
    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>VipSim filter documentation</title><style>
body{font-family:system-ui,sans-serif;max-width:860px;margin:0 auto;padding:28px 18px;
  color:#1c1e26;line-height:1.55}
h1{font-size:22px} h2{font-size:17px;margin:26px 0 4px;color:#4f46e5}
p{margin:6px 0} .lead{font-size:14px}
.more{font-size:13px;color:#4a4d57}
table{border-collapse:collapse;font-size:12.5px;margin:8px 0 4px;width:100%}
td,th{border:1px solid #e2e4ea;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f4f5f8} td:first-child{white-space:nowrap;font-family:ui-monospace,monospace}
.top{font-size:12px;color:#8a8d98;margin-bottom:18px}
</style></head><body>
<h1>VipSim filter documentation</h1>
<p class="top">Generated from the tuner's built-in explanations. The same texts
appear as tooltips in the tuner.</p>
<div style="background:#eef0ff;border:1px solid #c7cbf5;border-radius:10px;
  padding:10px 14px;font-size:13.5px;margin-bottom:14px">
<b>The order of the steps matters.</b> Filters run top to bottom, and each step
works on the result of the one before it. Example: a scotoma placed BEFORE a fog
step gets softened by the fog; placed AFTER it, the scotoma stays crisp on top.
Blur before floaters keeps the dots sharp; blur after floaters blurs the dots too.
Reorder steps in the tuner by dragging their handle.</div>"""]
    for name in sorted(vf._FILTER_DISPATCH):
        parts.append(f"<h2>{name}</h2>")
        parts.append(f'<p class="lead">{FILTER_DESC.get(name, "")}</p>')
        more = FILTER_MORE.get(name, "")
        if more:
            parts.append(f'<p class="more">{more}</p>')
        rows = []
        for pname, dflt in sigs[name].items():
            desc = PARAM_DESC.get(pname, "")
            enum = ENUM_OPTIONS.get(f"{name}.{pname}")
            if enum:
                desc += f" Choices: {', '.join(enum)}."
            shown = "" if dflt is None else repr(dflt)
            rows.append(f"<tr><td>{pname}</td><td>{shown}</td><td>{desc}</td></tr>")
        parts.append("<table><tr><th>parameter</th><th>default</th><th>meaning</th></tr>"
                     + "".join(rows) + "</table>")
    parts.append("</body></html>")
    return "".join(parts)


def _tune_metric(pid, filters, ref_arr, ref_lab, size):
    import numpy as np
    from skimage.metrics import structural_similarity
    from skimage.color import rgb2lab, deltaE_ciede2000
    with _render_lock:
        ours = apply_stack(scene_for(pid, size), filters)
    b = np.asarray(ours, np.float32)
    gray = np.array([0.299, 0.587, 0.114], np.float32)
    ssim_val = float(structural_similarity(ref_arr @ gray, b @ gray,
                                           data_range=255.0))
    mae = float(np.abs(ref_arr - b).mean())
    d_e = float(deltaE_ciede2000(ref_lab, rgb2lab(b / 255.0)).mean())
    return (max(0.0, (1.0 - mae / 255.0) * 100.0)
            + max(0.0, ssim_val * 100.0)
            + max(0.0, 100.0 - 2.0 * d_e)) / 3.0


def autotune(pid, filters, seconds=25.0):
    """Time-boxed coordinate descent over every numeric param of the ENABLED
    steps, maximizing the tuner similarity metric. Returns (filters, before,
    after, evals). Same method as the 2026-07-10 offline sweep."""
    import copy
    import time
    import numpy as np
    from skimage.color import rgb2lab

    ref = ref_image(pid)
    if ref is None:
        raise ValueError("no ref image for " + pid)
    ref_arr = np.asarray(ref, np.float32)
    ref_lab = rgb2lab(ref_arr / 255.0)
    size = ref.size
    filters = copy.deepcopy(filters)
    deadline = time.monotonic() + float(seconds)

    INT_KEYS = {"pixel_size", "n_floaters", "blur_size"}
    UNIT_KEYS = {"severity", "opacity", "inner_opacity", "halo_opacity",
                 "dot_brightness"}

    def paths():
        for i, step in enumerate(filters):
            if step.get("_disabled"):
                continue
            for k, v in step.items():
                if k.startswith("_") or k == "filter" or isinstance(v, (bool, str)):
                    continue
                if isinstance(v, (int, float)):
                    yield (i, k, None)
                elif (isinstance(v, list) and len(v) == 2
                      and all(isinstance(x, (int, float)) for x in v)):
                    yield (i, k, 0)
                    yield (i, k, 1)

    def candidates(key, v):
        if key == "seed":
            return [1, 7, 13, 21, 33, 47, 61, 84]
        out = {0.01, 0.03} if v == 0 else (
            {v * m for m in (0.8, 0.9, 1.1, 1.2)} | {v - 0.02, v + 0.02})
        res = []
        for c in out:
            if key in UNIT_KEYS:
                c = min(1.0, max(0.0, c))
            if v >= 0:
                c = max(0.0, c)
            if key in INT_KEYS or (isinstance(v, int) and not isinstance(v, bool)):
                c = max(1, int(round(c)))
            if c != v:
                res.append(c)
        return res

    best = _tune_metric(pid, filters, ref_arr, ref_lab, size)
    start = best
    evals = 1
    for _ in range(4):
        improved = False
        for i, k, s in list(paths()):
            if time.monotonic() > deadline:
                break
            cur = filters[i][k][s] if s is not None else filters[i][k]
            for c in candidates(k, cur):
                if time.monotonic() > deadline:
                    break
                if s is not None:
                    filters[i][k][s] = c
                else:
                    filters[i][k] = c
                evals += 1
                sc = _tune_metric(pid, filters, ref_arr, ref_lab, size)
                if sc > best + 1e-3:
                    best, cur, improved = sc, c, True
                else:
                    if s is not None:
                        filters[i][k][s] = cur
                    else:
                        filters[i][k] = cur
        if not improved or time.monotonic() > deadline:
            break
    return filters, round(start, 2), round(best, 2), evals


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


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>VipSim profile tuner</title>
<style>
:root{--accent:#4f46e5;--bg:#f4f5f8;--line:#e2e4ea;--txt:#1c1e26}
body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh;color:var(--txt)}
#panel{width:500px;overflow-y:auto;padding:14px;border-right:1px solid var(--line);
  box-sizing:border-box;background:var(--bg)}
#view{flex:1;display:flex;flex-direction:column;align-items:center;padding:14px;overflow:auto}
#topbar{position:sticky;top:-14px;z-index:20;background:var(--bg);margin:-14px -14px 12px;
  padding:10px 14px 10px;border-bottom:1px solid var(--line);
  display:flex;flex-direction:column;gap:7px}
#topbar .row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
#topbar button{height:32px;padding:0 12px;font-size:13.5px;white-space:nowrap}
#topbar select{height:32px;font-size:14px}
.btngroup{display:inline-flex;gap:0;border:1px solid var(--line);border-radius:9px;
  overflow:hidden;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.btngroup button{border:none;border-right:1px solid var(--line);border-radius:0;margin:0}
.btngroup button:last-child{border-right:none}
#viewbar{width:100%;max-width:1500px;display:flex;justify-content:flex-end;gap:8px;margin-bottom:6px}
#viewinner{display:flex;flex-direction:column;align-items:center;width:100%}
.zlabel{align-self:center;padding:0 9px;font-size:11.5px;color:#8a8d98}
#viewbar button{font-size:11.5px;padding:4px 9px}
#imgs{display:grid;grid-template-columns:1fr 1fr;gap:10px;width:100%;max-width:1500px}
#imgs.stacked{grid-template-columns:1fr;max-width:980px}
#imgs figure{margin:0;position:relative;min-width:0}
#imgs img{width:100%;height:auto;display:block;border-radius:6px}
canvas.overlay{position:absolute;left:0;top:0;pointer-events:none;z-index:5}
#imgs figcaption{font-size:12px;color:#666;text-align:center;margin-top:3px}
img{image-rendering:auto;border-radius:6px}
#cmpWrap{margin-top:14px;text-align:center;width:100%;max-width:1500px}
#cmp{position:relative;display:inline-block;user-select:none;touch-action:none;
  width:min(100%,860px);border-radius:6px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,.12)}
#cmp img{display:block;width:100%;border-radius:0}
#cmpOurs{position:absolute;left:0;top:0;width:100%;height:100%;clip-path:inset(0 0 0 50%)}
#cmpBar{position:absolute;top:0;bottom:0;left:50%;width:2px;background:var(--accent);
  cursor:ew-resize;z-index:6}
#cmpBar::after{content:'\\2194';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  background:var(--accent);color:#fff;border-radius:50%;width:24px;height:24px;
  line-height:24px;font-size:13px;box-shadow:0 1px 4px rgba(0,0,0,.3)}
#cmpWrap figcaption{font-size:12px;color:#666;margin-top:4px}
#simWrap{margin-top:8px;display:inline-block;background:#fff;border:1px solid var(--line);
  border-radius:999px;padding:5px 14px;font-size:12px;color:#333;cursor:help}
#simline b{color:var(--accent)}
.step{border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin-bottom:10px;
  background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.step h3{margin:0;font-size:14.5px;letter-spacing:.01em;cursor:pointer;display:flex;
  justify-content:space-between;align-items:center;gap:6px}
.step h3 .loc{opacity:.3;font-size:12px;transition:opacity .12s}
.step:hover h3 .loc{opacity:.75}
.step.hl{border-color:var(--accent);box-shadow:0 0 0 2px rgba(79,70,229,.18)}
.step.hl h3 .loc{opacity:1}
#oursH{z-index:4}
.stepdesc{font-size:12.5px;color:#8a8d98;margin:2px 0 8px;line-height:1.4}
.infoBtn{cursor:pointer;color:var(--accent);font-size:12px}
.morebox{display:none;font-size:12.5px;color:#4a4d57;background:var(--bg);
  border-radius:8px;padding:7px 10px;margin:0 0 8px;line-height:1.5}
.param select{width:100%;font-weight:400;padding:3px 6px;font-size:12px}
.param input[type=checkbox]{justify-self:start;accent-color:var(--accent)}
.param{display:grid;grid-template-columns:128px 1fr 68px;gap:7px;align-items:center;
  margin:4px 0;font-size:13px}
.param label{color:#4a4d57;cursor:help;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.param input[type=range]{width:100%;accent-color:var(--accent)}
.param input[type=number]{width:66px;font-size:12.5px;border:1px solid var(--line);
  border-radius:6px;padding:3px 4px}
button{padding:6px 11px;cursor:pointer;font-size:13px;background:#fff;color:var(--txt);
  border:1px solid var(--line);border-radius:9px;transition:background .12s,border-color .12s}
button:hover{border-color:var(--accent);color:var(--accent)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.primary:hover{background:#4338ca;color:#fff}
button.active{background:#eef0ff;border-color:var(--accent);color:var(--accent)}
#status{font-size:11px;color:#166534;background:#e7f6ec;border-radius:999px;
  padding:3px 10px;margin-left:auto}
select{font-size:13px;padding:6px 8px;border:1px solid var(--line);border-radius:9px;
  background:#fff;font-weight:600}
#panel{flex:none}
#divider{width:6px;flex:none;cursor:col-resize;background:var(--line);transition:background .12s}
#divider:hover{background:var(--accent)}
.step.off{opacity:.45}
.step.dropAbove{box-shadow:0 -3px 0 var(--accent)}
.step.dropBelow{box-shadow:0 3px 0 var(--accent)}
.grab{cursor:grab;user-select:none;opacity:.45;font-size:14px;padding:0 3px}
.grab:hover{opacity:1;color:var(--accent)}
.step h3 .ctrl{display:inline-flex;gap:2px;align-items:center}
.step h3 .ctrl button{padding:1px 7px;font-size:11px;border-radius:6px;line-height:1.5}
.step h3 input[type=checkbox]{accent-color:var(--accent);cursor:pointer}
#addrow{display:flex;gap:6px;margin:4px 0 18px}
#addrow select{flex:1;font-weight:400}
#legend{font-size:11px;color:#8a8d98;margin:8px auto 0;max-width:560px;line-height:1.6;
  text-align:center}
#spark{vertical-align:middle;margin-left:10px;background:#fff;border:1px solid var(--line);
  border-radius:6px}
#wizard{position:fixed;inset:0;background:rgba(20,22,30,.4);display:none;z-index:50;
  align-items:center;justify-content:center}
#wizard .box{background:#fff;border-radius:14px;padding:18px 22px;width:460px;
  max-height:82vh;overflow:auto;box-shadow:0 10px 44px rgba(0,0,0,.3)}
#wizard label{display:flex;gap:8px;font-size:13px;margin:5px 0;align-items:baseline;cursor:pointer}
#wizard input[type=checkbox]{accent-color:var(--accent)}
</style></head><body>
<div id="panel">
  <div id="topbar">
    <div class="row">
      <select id="pid" title="profile"></select>
      <div class="btngroup">
        <button id="save" class="primary" title="write values into the profile JSON (asks before overwriting)">Save</button>
        <button id="saveas" title="save the current values as a NEW profile">Save as…</button>
        <button id="reload" title="reload values from disk">Reload</button>
      </div>
      <span id="status"></span>
    </div>
    <div class="row">
      <div class="btngroup">
        <button id="undoBtn" title="undo the last filter change (Ctrl+Z)">Undo</button>
        <button id="autotune" title="optimize every numeric value around the current state to maximize similarity (~25 s; Ctrl+Z reverts)">Auto-tune</button>
      </div>
      <div class="btngroup">
        <button id="newprof" title="create a new profile from symptom tick boxes">New</button>
        <button id="docsBtn" title="open the filter documentation page in a new tab">Docs</button>
      </div>
      <div class="btngroup" title="custom images for THIS profile. Session only: gone on tuner restart.">
        <button id="upRef" title="load a custom REFERENCE image (similarity target, left panel)">Load ref</button>
        <button id="upBase" title="load a custom BASE image the filters run on (e.g. a web screenshot)">Load base</button>
        <button id="upClear" title="back to the paper scene + ref">Reset</button>
      </div>
      <div class="btngroup" title="size of the filter list below">
        <span class="zlabel">Sidebar</span>
        <button id="panelZoomOut" title="smaller">−</button>
        <span class="zlabel" id="panelZoomPct" style="padding:0 4px">100%</span>
        <button id="panelZoomIn" title="bigger">+</button>
      </div>
      <div class="btngroup" title="sketch on the images (never affects the render)">
        <button id="draw" title="draw strokes">Draw</button>
        <button id="erase" title="erase strokes you click/drag over">Erase</button>
        <button id="undoDraw" title="undo last stroke">Undo</button>
        <button id="clearDraw" title="remove all strokes">Clear</button>
      </div>
    </div>
    <input type="file" id="fileRef" accept="image/*" style="display:none">
    <input type="file" id="fileBase" accept="image/*" style="display:none">
  </div>
  <div id="steps"></div>
</div>
<div id="divider" title="drag to resize the panel"></div>
<div id="view"><div id="viewbar"><div class="btngroup" title="zoom of the image area">
<span class="zlabel">Images</span>
<button id="viewZoomOut" title="smaller">−</button>
<span class="zlabel" id="viewZoomPct" style="padding:0 4px">100%</span>
<button id="viewZoomIn" title="bigger">+</button></div>
<button id="layoutBtn" title="switch between images side by side and images above each other">Stack images</button></div><div id="viewinner"><div id="imgs">
  <figure><img id="ref"><canvas class="overlay" id="refC"></canvas><figcaption>REF (similarity target)</figcaption></figure>
  <figure><img id="ours"><canvas class="overlay" id="oursH"></canvas><canvas class="overlay" id="oursC"></canvas><figcaption>OURS (live) · hold <b>B</b> = base image, <b>R</b> = ref · selected step: drag = move, scroll = resize</figcaption></figure>
</div>
<div id="cmpWrap"><div id="cmp">
  <img id="cmpRef"><img id="cmpOurs"><div id="cmpBar"></div>
</div><figcaption>wipe compare: drag the bar (left: ref, right: ours)</figcaption>
<div><span id="simWrap"><span id="simline">similarity: computing…</span></span></div>
<div id="legend">The score compares OUR image with the REF image.
<b>overall</b> is the average of the three parts.
<b>structure</b> (SSIM): are the shapes in the same places?
<b>pixel</b>: how close is the brightness?
<b>color</b> (ΔE00): how close are the colors?
Green and red numbers show what your last change did. The score is a guide; your eye decides.</div></div>
</div></div>
<div id="wizard"><div class="box">
  <h3 style="margin-top:0">New profile from symptoms</h3>
  <p style="font-size:12px;color:#8a8d98;margin:4px 0 10px">Tick what the person
  experiences. A starting stack is prefilled (one filter per symptom, Table-4
  shader vocabulary); tune values afterwards.</p>
  <div id="wizlist"></div>
  <div style="display:flex;gap:8px;margin-top:14px;justify-content:flex-end">
    <button id="wizCancel">cancel</button>
    <button id="wizCreate" class="primary">create&hellip;</button>
  </div>
</div></div>
<script>
let profile = null, renderSeq = 0, renderTimer = null;

const RANGES = {  // param -> [min, max, step]
  brightness: [0, 3, 0.01], contrast: [0, 3, 0.01], gamma: [0.2, 3, 0.01],
  saturation: [0, 2, 0.01],
  pixel_size: [2, 400, 1], n_floaters: [0, 60000, 50], floater_size: [1, 20, 1],
  dot_brightness: [0, 1, 0.01], spread: [0.01, 0.6, 0.005],
  spread_x: [0.01, 0.6, 0.005], spread_y: [0.01, 0.6, 0.005], seed: [0, 99, 1],
  angle: [-90, 90, 1], region_feather: [0, 0.15, 0.002],
  region_center_bias: [0, 1, 0.01],
  suction_strength: [0, 5, 0.05], swirl_strength: [-6.3, 6.3, 0.05],
  inner_circle_radius: [0, 0.15, 0.002], inner_fade: [0, 0.2, 0.002],
  inner_color: [0, 1, 0.01], inner_opacity: [0, 1, 0.01],
  halo_radius: [0, 0.2, 0.002], halo_thickness: [0, 0.1, 0.002],
  halo_color: [0, 1, 0.01], halo_opacity: [0, 1, 0.01],
  band_width: [0.05, 2, 0.01],
  noise_scale: [1, 40, 0.5],
  overlay_scale: [0.1, 5, 0.05], mask_radius: [0.03, 0.7, 0.005],
};
// Heavy filters get finer steps at slider level (filter.param overrides).
const FILTER_RANGES = {
  'cataracts.severity': [0, 1, 0.002],
  'bcg.contrast': [0, 3, 0.005],
  'blur.severity': [0, 1, 0.002],
};
function rangeFor(name, value, filt){
  if (filt && FILTER_RANGES[filt + '.' + name]) return FILTER_RANGES[filt + '.' + name];
  if (RANGES[name]) return RANGES[name];
  const m = Math.max(1, Math.abs(value) * 2);
  return [0, m, m > 10 ? 1 : 0.01];
}

// Role-aware description: looks at the step's parameters to say what THIS
// instance does (several steps can share a filter type but do different jobs).
// Always ONE plain sentence; extra detail lives in FILTER_MORE (tooltip).
function describeStep(step){
  const f = step.filter;
  if (f === 'field_loss'){
    if (step.field_type === 'peripheral')
      return 'Keeps the middle sharp and makes the edges blurry (tunnel vision).';
    if (step.mask_center !== undefined)
      return 'Makes one round spot of the picture blurry.';
    return 'Lays an uneven fog over the picture, some patches thicker than others.';
  }
  if (f === 'foveal_darkness'){
    const what = step.mode === 'light' ? 'milky white'
               : step.mode === 'desat' ? 'grey (color drained out)' : 'dark';
    if (step.invert)
      return `Makes everything AROUND one oval spot ${what}; the oval itself stays normal.`;
    return `Puts a soft oval spot on the picture and makes it ${what}.`;
  }
  if (f === 'bcg'){
    const grey = (step.contrast ?? 1) < 0.95 || (step.saturation ?? 1) < 0.95;
    const tone = (step.brightness ?? 1) !== 1 || (step.gamma ?? 1) !== 1;
    if (grey && !tone) return 'Washes the colors of the whole picture out toward pale grey.';
    if (tone && !grey) return 'Makes the whole picture lighter or darker.';
    return 'Changes overall lightness and how strong the colors are.';
  }
  return FILTER_DESC[f] || '';
}

const FILTER_MORE = __FILTER_MORE__;

function roleTag(step){
  if (step.filter === 'field_loss')
    return step.field_type === 'peripheral' ? 'edge blur'
         : step.mask_center !== undefined ? 'central blob' : 'patchy haze';
  if (step.filter === 'foveal_darkness')
    return step.invert
         ? (step.mode === 'light' ? 'white frame' : step.mode === 'desat' ? 'grey frame' : 'dark frame')
         : (step.mode === 'light' ? 'milky ellipse' : step.mode === 'desat' ? 'grey ellipse' : 'dark spot');
  if (step.filter === 'bcg')
    return (step.contrast ?? 1) < 0.95 ? 'grey knob' : 'tone';
  return step.mode || '';
}

const FILTER_DESC = __FILTER_DESC__;
const PARAM_DESC = __PARAM_DESC__;
const ENUM_OPTIONS = __ENUM_OPTIONS__;

async function loadList(selectId){
  const ids = await (await fetch('/list')).json();
  const sel = document.getElementById('pid');
  sel.innerHTML = ids.map(i => `<option>${i}</option>`).join('');
  sel.onchange = () => loadProfile(sel.value);
  const pick = (selectId && ids.includes(selectId)) ? selectId
             : (ids.includes('p4') ? 'p4' : ids[0]);
  sel.value = pick;
  loadProfile(pick);
}

async function loadProfile(pid){
  hlIndex = null;
  undoStack = []; lastVals = null; sessionBest = null;
  profile = await (await fetch('/profile?id=' + pid)).json();
  const refUrl = '/ref?id=' + pid + '&t=' + Date.now();
  document.getElementById('ref').src = refUrl;
  document.getElementById('cmpRef').src = refUrl;
  buildUI();
  requestRender();
}

// -- wipe compare slider ------------------------------------------------------
(() => {
  const cmp = document.getElementById('cmp');
  const bar = document.getElementById('cmpBar');
  const top = document.getElementById('cmpOurs');
  let dragging = false;
  const setPct = pct => {
    pct = Math.max(0, Math.min(100, pct));
    top.style.clipPath = 'inset(0 0 0 ' + pct + '%)';
    bar.style.left = pct + '%';
  };
  const fromEvent = e => {
    const r = cmp.getBoundingClientRect();
    setPct((e.clientX - r.left) / r.width * 100);
  };
  cmp.addEventListener('pointerdown', e => { dragging = true; cmp.setPointerCapture(e.pointerId); fromEvent(e); });
  cmp.addEventListener('pointermove', e => { if (dragging) fromEvent(e); });
  cmp.addEventListener('pointerup', () => { dragging = false; });
})();

function buildUI(){
  const holder = document.getElementById('steps');
  holder.innerHTML = '';
  profile.filters.forEach((step, si) => {
    const div = document.createElement('div');
    div.className = 'step' + (step._disabled ? ' off' : '');
    const tag = roleTag(step);
    const desc = describeStep(step);
    const more = FILTER_MORE[step.filter] || '';
    div.innerHTML = `<h3><span><input type="checkbox" class="onoff" ${step._disabled ? '' : 'checked'} title="on/off: unticked = this step is skipped"> ${si + 1}. ${step.filter}${tag ? ` <span style="color:#4f46e5;font-weight:600">· ${tag}</span>` : ''}</span><span class="ctrl"><span class="grab" title="drag to reorder. The order of the steps changes the result.">⠿</span><button class="del" title="remove this step">×</button><span class="loc" title="show which image areas this step affects">◎</span></span></h3>`
      + (desc ? `<p class="stepdesc">${desc}${more ? ' <span class="infoBtn" title="click for details and value examples">ⓘ</span>' : ''}</p>` : '')
      + (more ? `<div class="morebox">${more}</div>` : '');
    const info = div.querySelector('.infoBtn');
    if (info) info.onclick = e => {
      e.stopPropagation();
      const b = div.querySelector('.morebox');
      b.style.display = b.style.display === 'block' ? 'none' : 'block';
    };
    div.querySelector('h3').onclick = () => {
      hlIndex = hlIndex === si ? null : si;
      updateHighlight();
    };
    div.querySelector('.onoff').onclick = e => {
      e.stopPropagation();
      pushUndo();
      if (e.target.checked) delete step._disabled; else step._disabled = true;
      div.classList.toggle('off', !e.target.checked);
      requestRender();
    };
    // drag-and-drop reorder via the ⠿ handle
    const grab = div.querySelector('.grab');
    grab.onclick = e => e.stopPropagation();
    grab.addEventListener('pointerdown', () => div.setAttribute('draggable', 'true'));
    div.addEventListener('dragstart', e => {
      dragSrc = si; e.dataTransfer.effectAllowed = 'move'; div.style.opacity = 0.5;
    });
    div.addEventListener('dragend', () => {
      div.removeAttribute('draggable'); div.style.opacity = '';
      dragSrc = null; clearDropMarkers();
    });
    div.addEventListener('dragover', e => {
      if (dragSrc === null) return;
      e.preventDefault();
      const r = div.getBoundingClientRect();
      const below = e.clientY >= r.top + r.height / 2;
      div.classList.toggle('dropAbove', !below);
      div.classList.toggle('dropBelow', below);
    });
    div.addEventListener('dragleave', () => div.classList.remove('dropAbove', 'dropBelow'));
    div.addEventListener('drop', e => {
      e.preventDefault();
      const r = div.getBoundingClientRect();
      let j = si + (e.clientY >= r.top + r.height / 2 ? 1 : 0);
      if (dragSrc === null || j === dragSrc || j === dragSrc + 1){
        dragSrc = null; clearDropMarkers(); return;
      }
      pushUndo();
      const [moved] = profile.filters.splice(dragSrc, 1);
      if (j > dragSrc) j--;
      profile.filters.splice(j, 0, moved);
      dragSrc = null; hlIndex = null; buildUI(); requestRender();
    });
    div.querySelector('.del').onclick = e => {
      e.stopPropagation();
      if (!confirm('Remove step ' + (si + 1) + ' (' + step.filter + ')?')) return;
      pushUndo();
      profile.filters.splice(si, 1);
      hlIndex = null; buildUI(); requestRender();
    };
    // Uniform controls: union of the filter's FULL parameter list (from the
    // Python signature) and whatever extra keys the step carries. Every
    // instance of a filter shows the same fields; values the JSON does not
    // set display the filter's default and are only written on change.
    const sig = (FILTER_CATALOG.signature || {})[step.filter] || {};
    const keys = [...Object.keys(sig),
                  ...Object.keys(step).filter(k => !(k in sig))];
    for (const key of keys){
      if (key === 'filter' || key.startsWith('_')) continue;
      const inStep = key in step;
      const val = inStep ? step[key] : sig[key];
      const enums = ENUM_OPTIONS[step.filter + '.' + key];
      if (enums){
        div.appendChild(enumRow(si, key, String(val ?? enums[0]), enums));
      } else if (typeof val === 'boolean'){
        div.appendChild(boolRow(si, key, val));
      } else if (Array.isArray(val)){
        if (val.some(v => Array.isArray(v))){
          const p = document.createElement('div');
          p.className = 'param';
          p.innerHTML = `<label title="${PARAM_DESC[key] || ''}">${key}</label><span>${val.length} points</span><span></span>`;
          div.appendChild(p);
          continue;
        }
        const ensure = () => {
          if (!(key in profile.filters[si]))
            profile.filters[si][key] = JSON.parse(JSON.stringify(val));
        };
        val.forEach((v, vi) => div.appendChild(paramRow(si, key, vi, v, key + '[' + (vi ? 'x' : 'y') + ']', step.filter, ensure)));
        if (val.length === 2) div.appendChild(pickRow(si, key));
      } else if (typeof val === 'number'){
        div.appendChild(paramRow(si, key, null, val, key, step.filter));
      } else if (val === null || val === undefined){
        continue;  // derived internally from severity unless explicitly set
      } else {
        const p = document.createElement('div');
        p.className = 'param';
        p.innerHTML = `<label title="${PARAM_DESC[key] || ''}">${key}</label><span>${val}</span><span></span>`;
        div.appendChild(p);
      }
    }
    if (step.filter === 'floaters') div.appendChild(regionRow(si, step));
    holder.appendChild(div);
  });
  // -- add-filter row ---------------------------------------------------------
  const row = document.createElement('div');
  row.id = 'addrow';
  row.innerHTML = `<select id="addsel" title="pick a filter to append to the stack"></select>
    <button id="addbtn" title="append the filter with sensible starting values">＋ add</button>`;
  holder.appendChild(row);
  fillAddSel();
  document.getElementById('addbtn').onclick = () => {
    const name = document.getElementById('addsel').value;
    if (!name) return;
    pushUndo();
    const dflt = FILTER_CATALOG.defaults[name] || {severity: 0.3};
    profile.filters.push(Object.assign({filter: name}, JSON.parse(JSON.stringify(dflt))));
    buildUI(); requestRender();
  };
}

let dragSrc = null;
const clearDropMarkers = () =>
  document.querySelectorAll('.step').forEach(el => el.classList.remove('dropAbove', 'dropBelow'));

let FILTER_CATALOG = {names: [], defaults: {}};
fetch('/filters').then(r => r.json()).then(d => { FILTER_CATALOG = d; fillAddSel(); });
function fillAddSel(){
  const sel = document.getElementById('addsel');
  if (!sel || !FILTER_CATALOG.names.length) return;
  if (sel.options.length === FILTER_CATALOG.names.length) return;
  sel.innerHTML = FILTER_CATALOG.names.map(n => `<option>${n}</option>`).join('');
}

function regionRow(si, step){
  const row = document.createElement('div');
  row.className = 'param';
  const btn = document.createElement('button');
  btn.textContent = 'Dots into drawn area';
  btn.style.fontSize = '11px';
  btn.title = 'Draw an outline with the Draw tool first, then click: the dots of this layer arrange inside the drawn shape (overrides center/spread).';
  btn.onclick = () => {
    const path = sketchPaths[sketchPaths.length - 1];
    if (!path || path.length < 3){
      document.getElementById('status').textContent = 'draw an outline with the Draw tool first';
      return;
    }
    const stride = Math.max(1, Math.floor(path.length / 80));
    pushUndo();
    profile.filters[si].region = path.filter((_, i) => i % stride === 0)
      .map(([x, y]) => [Math.round(y * 1000) / 1000, Math.round(x * 1000) / 1000]);
    if (profile.filters[si].region_feather === undefined)
      profile.filters[si].region_feather = 0.02;
    if (profile.filters[si].region_center_bias === undefined)
      profile.filters[si].region_center_bias = 0.4;
    buildUI(); requestRender();
  };
  row.appendChild(document.createElement('label'));
  row.appendChild(btn);
  if (step.region){
    const clr = document.createElement('button');
    clr.textContent = 'Remove region';
    clr.style.fontSize = '11px';
    clr.title = 'Remove the drawn region: back to center/spread placement.';
    clr.onclick = () => {
      pushUndo();
      delete profile.filters[si].region;
      buildUI(); requestRender();
    };
    row.appendChild(clr);
  }
  return row;
}

function paramRow(si, key, idx, val, label, filt, ensure){
  const [mn, mx, st] = rangeFor(key, val, filt);
  const row = document.createElement('div');
  row.className = 'param';
  row.innerHTML = `<label title="${PARAM_DESC[key] || ''}">${label}</label>
    <input type="range" min="${mn}" max="${mx}" step="${st}" value="${val}">
    <input type="number" step="${st}" value="${val}">`;
  const [slider, num] = row.querySelectorAll('input');
  const set = v => {
    v = parseFloat(v); if (isNaN(v)) return;
    if (ensure) ensure();
    if (idx === null) profile.filters[si][key] = v; else profile.filters[si][key][idx] = v;
    slider.value = v; num.value = v;
    requestRender();
  };
  // one undo snapshot per interaction, taken BEFORE the first change
  slider.addEventListener('pointerdown', pushUndo);
  slider.addEventListener('keydown', pushUndoOnce);
  slider.oninput = () => set(slider.value);
  num.onchange = () => { pushUndo(); set(num.value); };
  return row;
}

function enumRow(si, key, val, options){
  const row = document.createElement('div');
  row.className = 'param';
  row.innerHTML = `<label title="${PARAM_DESC[key] || ''}">${key}</label>
    <select>${options.map(o => `<option${o === val ? ' selected' : ''}>${o}</option>`).join('')}</select><span></span>`;
  row.querySelector('select').onchange = e => {
    pushUndo();
    profile.filters[si][key] = e.target.value;
    buildUI(); requestRender();   // the step label and description may change
  };
  return row;
}

function boolRow(si, key, val){
  const row = document.createElement('div');
  row.className = 'param';
  row.innerHTML = `<label title="${PARAM_DESC[key] || ''}">${key}</label>
    <input type="checkbox"${val ? ' checked' : ''}><span></span>`;
  row.querySelector('input').onchange = e => {
    pushUndo();
    profile.filters[si][key] = e.target.checked;
    buildUI(); requestRender();
  };
  return row;
}

let pickTarget = null;
function pickRow(si, key){
  const row = document.createElement('div');
  row.className = 'param';
  const btn = document.createElement('button');
  btn.textContent = 'Set by click';
  btn.style.fontSize = '11px';
  btn.title = 'then click on either image to place it there';
  btn.onclick = () => {
    pickTarget = {si, key};
    status('click on either image to set ' + key);
    document.querySelectorAll('#imgs img').forEach(i => i.style.cursor = 'crosshair');
  };
  const mv = document.createElement('button');
  mv.textContent = 'Move / resize';
  mv.style.fontSize = '11px';
  mv.title = 'select this step, then drag on the right image to move it and scroll to resize it (Shift+scroll = fine steps)';
  mv.onclick = () => {
    hlIndex = si;
    updateHighlight();
  };
  row.appendChild(document.createElement('label'));
  const wrap = document.createElement('span');
  wrap.style.display = 'inline-flex'; wrap.style.gap = '4px';
  wrap.appendChild(btn); wrap.appendChild(mv);
  row.appendChild(wrap);
  return row;
}

['ref', 'ours'].forEach(id => {
  document.getElementById(id).addEventListener('click', e => {
    if (!pickTarget) return;
    const r = e.target.getBoundingClientRect();
    const x = Math.round((e.clientX - r.left) / r.width * 1000) / 1000;
    const y = Math.round((e.clientY - r.top) / r.height * 1000) / 1000;
    pushUndo();
    profile.filters[pickTarget.si][pickTarget.key] = [y, x];
    pickTarget = null;
    document.querySelectorAll('#imgs img').forEach(i => i.style.cursor = 'auto');
    document.getElementById('status').textContent = 'center set to y=' + y + ' x=' + x;
    buildUI();
    requestRender();
  });
});

// -- sketch overlay: draw where the effect should (not) be; render-inert ----
let drawMode = false, sketchPaths = [], curPath = null;
const canvases = ['refC', 'oursC'].map(id => document.getElementById(id));
function syncCanvases(){
  ['ref', 'ours'].forEach((id, i) => {
    const img = document.getElementById(id);
    canvases[i].width = img.clientWidth;
    canvases[i].height = img.clientHeight;
  });
  redrawSketch();
}
function redrawSketch(){
  canvases.forEach(c => {
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.strokeStyle = 'rgba(255,0,60,0.9)';
    ctx.lineWidth = 2;
    sketchPaths.forEach(p => {
      ctx.beginPath();
      p.forEach(([x, y], i) => i ? ctx.lineTo(x * c.width, y * c.height)
                                 : ctx.moveTo(x * c.width, y * c.height));
      ctx.stroke();
    });
  });
}
let eraseMode = false;
function setDrawButtons(){
  document.getElementById('draw').classList.toggle('active', drawMode && !eraseMode);
  document.getElementById('erase').classList.toggle('active', eraseMode);
  canvases.forEach(c => {
    c.style.pointerEvents = (drawMode || eraseMode) ? 'auto' : 'none';
    c.style.cursor = drawMode || eraseMode ? 'crosshair' : 'auto';
  });
}
document.getElementById('draw').onclick = () => {
  drawMode = !drawMode; if (drawMode) eraseMode = false; setDrawButtons();
};
document.getElementById('erase').onclick = () => {
  eraseMode = !eraseMode; if (eraseMode) drawMode = false; setDrawButtons();
};
document.getElementById('undoDraw').onclick = () => { sketchPaths.pop(); redrawSketch(); };
document.getElementById('clearDraw').onclick = () => { sketchPaths = []; redrawSketch(); };
function eraseAt(c, e){
  const x = e.offsetX / c.width, y = e.offsetY / c.height, r = 12 / c.width;
  sketchPaths = sketchPaths.filter(p =>
    !p.some(([px, py]) => (px - x) ** 2 + (py - y) ** 2 < r * r));
  redrawSketch();
}
canvases.forEach(c => {
  c.addEventListener('pointerdown', e => {
    if (eraseMode) { c.setPointerCapture(e.pointerId); curPath = 'erasing'; eraseAt(c, e); return; }
    if (!drawMode) return;
    c.setPointerCapture(e.pointerId);
    curPath = [[e.offsetX / c.width, e.offsetY / c.height]];
    sketchPaths.push(curPath);
  });
  c.addEventListener('pointermove', e => {
    if (curPath === 'erasing') { eraseAt(c, e); return; }
    if (!drawMode || !curPath) return;
    curPath.push([e.offsetX / c.width, e.offsetY / c.height]);
    redrawSketch();
  });
  c.addEventListener('pointerup', () => { curPath = null; });
});
document.getElementById('ref').addEventListener('load', syncCanvases);
document.getElementById('ours').addEventListener('load', syncCanvases);
window.addEventListener('resize', syncCanvases);

function requestRender(){
  clearTimeout(renderTimer);
  renderTimer = setTimeout(doRender, 250);
}

const status = t => document.getElementById('status').textContent = t;

// -- undo (filter values only; the sketch has its own undo) -------------------
let undoStack = [], undoTick = 0;
function pushUndo(){
  if (!profile) return;
  undoStack.push(JSON.stringify(profile.filters));
  if (undoStack.length > 80) undoStack.shift();
}
function pushUndoOnce(){            // throttled variant for key-repeat sources
  const now = Date.now();
  if (now - undoTick > 1000){ undoTick = now; pushUndo(); }
}
function doUndo(){
  if (!undoStack.length){ status('nothing to undo'); return; }
  profile.filters = JSON.parse(undoStack.pop());
  hlIndex = null; buildUI(); requestRender();
  status('undo ✓ (' + undoStack.length + ' left)');
}
document.getElementById('undoBtn').onclick = doUndo;
window.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z'
      && !e.target.matches('input,select,textarea')){
    e.preventDefault(); doUndo();
  }
});

// -- hold B / R: flash the base image or the ref in the OURS slot -------------
let lastRender = '', flipHeld = null;
window.addEventListener('keydown', e => {
  if (e.repeat || !profile || e.target.matches('input,select,textarea')) return;
  const k = e.key.toLowerCase();
  const ours = document.getElementById('ours');
  if (k === 'b'){ flipHeld = 'b'; ours.src = '/base?id=' + profile.id; }
  if (k === 'r'){ flipHeld = 'r'; ours.src = document.getElementById('ref').src; }
});
window.addEventListener('keyup', e => {
  const k = e.key.toLowerCase();
  if (flipHeld && (k === 'b' || k === 'r')){
    flipHeld = null;
    if (lastRender) document.getElementById('ours').src = lastRender;
  }
});

// -- score deltas: how did the LAST change move each metric? --------------------
let lastVals = null, sessionBest = null;
function deltaHtml(cur, prev, big){
  if (prev === null || prev === undefined) return '';
  const d = cur - prev;
  const size = big ? '' : 'font-size:10px;';
  if (Math.abs(d) < 0.05)
    return ` <span style="${size}color:#9aa0ab">±0.0</span>`;
  const col = d > 0 ? '#15803d' : '#b91c1c';
  return ` <span style="${size}color:${col};font-weight:600">${d > 0 ? '+' : ''}${d.toFixed(1)}</span>`;
}

// -- selected step: drag on OURS moves its center, scroll resizes it ----------
const CENTER_KEYS = ['center', 'mask_center', 'inner_center', 'halo_center'];
const RADIUS_KEYS = ['radius_x', 'radius_y', 'mask_radius', 'vortex_radius',
                     'radius', 'spread_x', 'spread_y'];
let dragCenter = null, uiTimer = null;
const rebuildSoon = () => { clearTimeout(uiTimer); uiTimer = setTimeout(buildUI, 400); };
const oursImg = document.getElementById('ours');
function centerFromEvent(e){
  const r = oursImg.getBoundingClientRect();
  return [Math.round((e.clientY - r.top) / r.height * 1000) / 1000,
          Math.round((e.clientX - r.left) / r.width * 1000) / 1000];
}
oursImg.addEventListener('pointerdown', e => {
  if (pickTarget || drawMode || eraseMode || hlIndex === null || !profile) return;
  const step = profile.filters[hlIndex];
  if (!step) return;
  const key = CENTER_KEYS.find(k => Array.isArray(step[k]) && step[k].length === 2);
  if (!key) return;
  pushUndo();
  dragCenter = {step, key};
  oursImg.setPointerCapture(e.pointerId);
  step[key] = centerFromEvent(e);
  requestRender();
});
oursImg.addEventListener('pointermove', e => {
  if (!dragCenter) return;
  dragCenter.step[dragCenter.key] = centerFromEvent(e);
  requestRender();
});
oursImg.addEventListener('pointerup', () => {
  if (dragCenter){ dragCenter = null; rebuildSoon(); }
});
oursImg.addEventListener('wheel', e => {
  if (hlIndex === null || !profile) return;
  const step = profile.filters[hlIndex];
  if (!step) return;
  const keys = RADIUS_KEYS.filter(k => typeof step[k] === 'number' && step[k] > 0);
  if (!keys.length) return;
  e.preventDefault();
  pushUndoOnce();
  const m = e.deltaY < 0 ? (e.shiftKey ? 1.02 : 1.05) : (e.shiftKey ? 0.98 : 0.95);
  keys.forEach(k => step[k] = Math.round(step[k] * m * 10000) / 10000);
  requestRender(); rebuildSoon();
}, {passive: false});

// -- custom ref / base upload (session-only) -----------------------------------
function hookUpload(btnId, fileId, kind){
  const inp = document.getElementById(fileId);
  document.getElementById(btnId).onclick = () => inp.click();
  inp.onchange = () => {
    const f = inp.files[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = async () => {
      const res = await fetch('/upload', {method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: profile.id, kind, png: rd.result})});
      status(res.ok ? 'custom ' + kind + ' loaded (session only)' : 'upload failed');
      loadProfile(profile.id);
    };
    rd.readAsDataURL(f);
    inp.value = '';
  };
}
hookUpload('upRef', 'fileRef', 'ref');
hookUpload('upBase', 'fileBase', 'base');
document.getElementById('upClear').onclick = async () => {
  for (const kind of ['ref', 'base'])
    await fetch('/upload', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id: profile.id, kind, clear: true})});
  status('back to the paper images');
  loadProfile(profile.id);
};

// -- auto-tune -------------------------------------------------------------------
document.getElementById('autotune').onclick = async () => {
  const btn = document.getElementById('autotune');
  btn.disabled = true;
  status('auto-tuning… (~25 s, coordinate descent)');
  try {
    const res = await fetch('/autotune', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id: profile.id, filters: profile.filters, seconds: 25})});
    const d = await res.json();
    if (d.error){ status('auto-tune failed: ' + d.error); return; }
    pushUndo();
    profile.filters = d.filters;
    hlIndex = null; buildUI(); requestRender();
    status('auto-tune ' + d.before + '% → ' + d.after + '% (' + d.evals
           + ' tries) · Ctrl+Z reverts');
  } finally {
    btn.disabled = false;
  }
};

// -- new-profile wizard ----------------------------------------------------------
const SYMPTOMS = [
  ['Scotoma (blind/dark spot)', [
    {filter: 'foveal_darkness', severity: 0.4, center: [0.5, 0.5], radius_x: 0.15, radius_y: 0.12, edge_width: 0.3},
    {filter: 'field_loss', severity: 0.5, field_type: 'central', mask_center: [0.5, 0.5]}]],
  ['Central vision loss (blurry middle)', [{filter: 'field_loss', severity: 0.5, field_type: 'central'}]],
  ['Tunnel vision (blurry edges)', [{filter: 'field_loss', severity: 0.6, field_type: 'peripheral'}]],
  ['Overall blur (hyperopia)', [{filter: 'blur', severity: 0.05}]],
  ['Detail loss', [{filter: 'detail_loss', severity: 0.3}]],
  ['Mosaic / blocky vision', [{filter: 'pixelation', pixel_size: 80, hard: true, sampling: 'point'}]],
  ['Color deficiency (CVD)', [{filter: 'cvd', severity: 0.5, cvd_type: 'deuteranomaly'}]],
  ['Contrast sensitivity / washout', [{filter: 'bcg', brightness: 1.3, contrast: 0.6, gamma: 1, saturation: 0.7}]],
  ['Cataracts haze', [{filter: 'cataracts', severity: 0.2}]],
  ['Glare / light sensitivity', [{filter: 'bloom', severity: 0.3}]],
  ['Wavy distortion (metamorphopsia)', [{filter: 'distortion', severity: 0.1}]],
  ['Local swirl (metamorphopsia point)', [{filter: 'vortex', severity: 0.4, center: [0.5, 0.5], vortex_radius: 0.2, suction_strength: 1}]],
  ['Double vision', [{filter: 'double_vision', severity: 0.1, axis: 'horizontal'}]],
  ['Nystagmus (shaky vision)', [{filter: 'nystagmus', severity: 0.05, axis: 'horizontal'}]],
  ['Floaters', [{filter: 'floaters', severity: 0.3, n_floaters: 1500, mode: 'light', center: [0.5, 0.5], spread_x: 0.2, spread_y: 0.2, seed: 7}]],
  ['Teichopsia (migraine aura arc)', [{filter: 'teichopsia', severity: 0.3}]],
  ['Flickering stars', [{filter: 'flickering_stars', severity: 0.3}]],
  ['Dark edges (vignette)', [{filter: 'vignette', severity: 0.3, mode: 'dark'}]],
];
document.getElementById('docsBtn').onclick = () => window.open('/docs', '_blank');
document.getElementById('newprof').onclick = () => {
  const list = document.getElementById('wizlist');
  list.innerHTML = SYMPTOMS.map(([label, steps], i) =>
    `<label><input type="checkbox" data-i="${i}"> ${label}
     <span class="hint" style="color:#8a8d98;font-size:11px">(${steps.map(s => s.filter).join(' + ')})</span></label>`).join('');
  document.getElementById('wizard').style.display = 'flex';
};
document.getElementById('wizCancel').onclick = () =>
  document.getElementById('wizard').style.display = 'none';
document.getElementById('wizCreate').onclick = async () => {
  const picked = [...document.querySelectorAll('#wizlist input:checked')]
    .map(cb => SYMPTOMS[+cb.dataset.i][1]).flat();
  if (!picked.length){ status('tick at least one symptom'); return; }
  const newId = (prompt('Name for the new profile (e.g. p8):') || '').trim();
  if (!newId) return;
  const filters = JSON.parse(JSON.stringify(picked));
  const res = await saveAsRequest('', newId, filters);
  if (res){
    document.getElementById('wizard').style.display = 'none';
    await loadList(newId);
    status('created ' + newId + ' ✓, now tune it');
  }
};

// saveas with 409 -> "overwrite?" flow, shared by Save-as and the wizard
async function saveAsRequest(baseId, newId, filters){
  const send = overwrite => fetch('/saveas', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({baseId, newId, filters, overwrite})});
  let res = await send(false);
  if (res.status === 409){
    if (!confirm(newId + ' already exists. Overwrite it?')) return null;
    res = await send(true);
  }
  if (!res.ok){
    const err = await res.json().catch(() => ({}));
    status(err.error || 'save-as failed');
    return null;
  }
  return res;
}

// -- independent zoom for sidebar and image area --------------------------------
function initZoom(key, target, outBtn, inBtn, pctEl, lo, hi){
  let z = parseFloat(localStorage.getItem(key)) || 1;
  const apply = () => {
    z = Math.round(Math.min(hi, Math.max(lo, z)) * 10) / 10;
    document.getElementById(target).style.zoom = z;
    if (pctEl) document.getElementById(pctEl).textContent = Math.round(z * 100) + '%';
    localStorage.setItem(key, z);
    if (typeof syncCanvases === 'function') syncCanvases();
  };
  document.getElementById(outBtn).onclick = () => { z -= 0.1; apply(); };
  document.getElementById(inBtn).onclick = () => { z += 0.1; apply(); };
  apply();
}
initZoom('avp_zoom_panel', 'steps', 'panelZoomOut', 'panelZoomIn', 'panelZoomPct', 0.7, 1.6);
initZoom('avp_zoom_view', 'viewinner', 'viewZoomOut', 'viewZoomIn', 'viewZoomPct', 0.5, 2.0);

// -- layout toggle: side by side <-> stacked -----------------------------------
const layoutBtn = document.getElementById('layoutBtn');
function applyLayout(stacked){
  document.getElementById('imgs').classList.toggle('stacked', stacked);
  layoutBtn.textContent = stacked ? 'Side by side' : 'Stack images';
  localStorage.setItem('avp_layout', stacked ? 'stacked' : 'side');
  syncCanvases();
}
layoutBtn.onclick = () =>
  applyLayout(!document.getElementById('imgs').classList.contains('stacked'));
applyLayout(localStorage.getItem('avp_layout') === 'stacked');

// -- resizable panel --------------------------------------------------------------
(() => {
  const dv = document.getElementById('divider'), panel = document.getElementById('panel');
  let dragging = false;
  dv.addEventListener('pointerdown', e => { dragging = true; dv.setPointerCapture(e.pointerId); });
  window.addEventListener('pointermove', e => {
    if (dragging) panel.style.width = Math.max(300, Math.min(780, e.clientX)) + 'px';
  });
  window.addEventListener('pointerup', () => dragging = false);
})();

// -- similarity vs ref ---------------------------------------------------------
let simSeq = 0;
async function updateSimilarity(){
  const seq = ++simSeq;
  const res = await fetch('/similarity', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: profile.id, filters: profile.filters})});
  const d = await res.json();
  if (seq !== simSeq) return;
  const el = document.getElementById('simline');
  if (d.error){ el.textContent = 'similarity: n/a (' + d.error + ')'; return; }
  const p = lastVals || {};
  el.innerHTML = 'ours is <b>' + d.overall + '%</b>' + deltaHtml(d.overall, p.overall, true)
    + ' similar to the ref &nbsp;·&nbsp; '
    + 'structure (SSIM) ' + d.ssim + '%' + deltaHtml(d.ssim, p.ssim)
    + ' &nbsp;·&nbsp; pixel ' + d.pixel + '%' + deltaHtml(d.pixel, p.pixel)
    + ' &nbsp;·&nbsp; color (ΔE00) ' + d.color + '%' + deltaHtml(d.color, p.color);
  sessionBest = sessionBest === null ? d.overall : Math.max(sessionBest, d.overall);
  document.getElementById('simWrap').title = 'session best: ' + sessionBest.toFixed(1) + '%';
  lastVals = d;
}

// -- step influence highlight -------------------------------------------------
let hlIndex = null, hlSeq = 0;
function hlCanvas(){
  const img = document.getElementById('ours'), c = document.getElementById('oursH');
  c.width = img.clientWidth; c.height = img.clientHeight;
  return c;
}
async function updateHighlight(){
  const c = hlCanvas(), ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  document.querySelectorAll('#steps .step').forEach((el, i) =>
    el.classList.toggle('hl', i === hlIndex));
  const stepSel = hlIndex !== null ? profile.filters[hlIndex] : null;
  const movable = stepSel && CENTER_KEYS.some(k =>
    Array.isArray(stepSel[k]) && stepSel[k].length === 2);
  document.getElementById('ours').style.cursor = movable ? 'move' : 'auto';
  if (hlIndex === null) return;
  if (movable) status('drag on the right image to move this effect · scroll to resize');
  const seq = ++hlSeq;
  document.getElementById('status').textContent = 'locating\u2026';
  const res = await fetch('/influence', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: profile.id, filters: profile.filters, index: hlIndex})});
  const data = await res.json();
  if (seq !== hlSeq || hlIndex === null) return;
  if (data.empty){
    document.getElementById('status').textContent =
      'this step currently changes nothing (severity 0?)';
    return;
  }
  const im = new Image();
  im.onload = () => {
    if (seq !== hlSeq || hlIndex === null) return;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(im, 0, 0, c.width, c.height);
    document.getElementById('status').textContent = 'affected areas in pink'
      + (movable ? ' · drag image = move · scroll = resize' : '');
  };
  im.src = 'data:image/png;base64,' + data.png;
}

async function doRender(){
  const seq = ++renderSeq;
  document.getElementById('status').textContent = 'rendering…';
  const res = await fetch('/render', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: profile.id, filters: profile.filters})});
  const data = await res.json();
  if (seq !== renderSeq) return;  // stale
  const src = 'data:image/png;base64,' + data.png;
  lastRender = src;
  if (!flipHeld) document.getElementById('ours').src = src;  // don't fight B/R flip
  document.getElementById('cmpOurs').src = src;
  document.getElementById('status').textContent = data.ms + ' ms';
  if (hlIndex !== null) updateHighlight();
  updateSimilarity();
}

document.getElementById('save').onclick = async () => {
  if (!confirm('Overwrite ' + profile.id + '.json with the current values?\\n\\n'
               + 'Cancel = save under a different name instead.')){
    document.getElementById('saveas').click();
    return;
  }
  const res = await fetch('/save', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: profile.id, filters: profile.filters})});
  if (res.ok){
    status('saved ✓');
  } else {
    const err = await res.json().catch(() => ({}));
    status('SAVE FAILED: ' + (err.error || res.status));
  }
};
document.getElementById('saveas').onclick = async () => {
  const newId = (prompt('Save current values as new profile.\\nName (e.g. p3c):') || '').trim();
  if (!newId) return;
  const res = await saveAsRequest(profile.id, newId, profile.filters);
  if (res){
    await loadList(newId);
    status('saved as ' + newId + ' ✓');
  }
};
document.getElementById('reload').onclick = () => loadProfile(profile.id);

loadList();
</script></body></html>"""

# Inject the Python-side documentation dicts into the page JS (single source).
PAGE = (PAGE
        .replace("__FILTER_DESC__", json.dumps(FILTER_DESC))
        .replace("__FILTER_MORE__", json.dumps(FILTER_MORE))
        .replace("__PARAM_DESC__", json.dumps(PARAM_DESC))
        .replace("__ENUM_OPTIONS__", json.dumps(ENUM_OPTIONS)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _q(self, key):
        from urllib.parse import parse_qs, urlparse
        return parse_qs(urlparse(self.path).query).get(key, [None])[0]

    def _same_origin(self):
        """Only serve the tuner's own page: any open web page can send
        requests to localhost, and a rebound DNS name arrives with a
        foreign Host header."""
        port = self.server.server_address[1]
        hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{h}" for h in hosts}

    def do_GET(self):
        # Nonbreaking: any error becomes a JSON response, never a dead thread.
        try:
            if not self._same_origin():
                return self._send(403, json.dumps({"error": "forbidden origin"}))
            self._get()
        except Exception as e:  # noqa: BLE001
            try:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
            except Exception:
                pass

    def _get(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/list":
            ids = sorted(p.stem for p in PROFILE_DIR.glob("p*.json") if valid_pid(p.stem))
            self._send(200, json.dumps(ids))
        elif path == "/profile":
            pid = self._q("id")
            f = PROFILE_DIR / f"{pid}.json"
            if not valid_pid(pid) or not f.is_file():
                return self._send(404, "{}")
            self._send(200, f.read_text())
        elif path == "/ref":
            pid = self._q("id") or ""
            img = ref_image(pid)
            if img is None:
                return self._send(404, b"", "image/png")
            self._send(200, png_bytes(img), "image/png")
        elif path == "/base":
            # unfiltered base scene at ref size (for the hold-B flip)
            pid = self._q("id") or ""
            img = ref_image(pid)
            size = img.size if img is not None else (425, 237)
            self._send(200, png_bytes(scene_for(pid, size)), "image/png")
        elif path == "/filters":
            names = sorted(vf._FILTER_DISPATCH.keys())
            self._send(200, json.dumps({
                "names": names,
                "defaults": {n: NEW_STEP_DEFAULTS.get(n, {"severity": 0.3})
                             for n in names},
                "signature": filter_signatures(),
            }))
        elif path == "/docs":
            self._send(200, docs_html(), "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        # Nonbreaking: any error becomes a JSON response, never a dead thread.
        try:
            if not self._same_origin():
                return self._send(403, json.dumps({"error": "forbidden origin"}))
            self._post()
        except Exception as e:  # noqa: BLE001
            try:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
            except Exception:
                pass

    def _post(self):
        import time
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        pid = payload.get("id") or payload.get("baseId", "")
        filters = payload.get("filters", [])
        ref = ref_image(pid)
        size = ref.size if ref is not None else (425, 237)

        if self.path == "/render":
            t0 = time.time()
            with _render_lock:
                img = apply_stack(scene_for(pid, size), filters)
            self._send(200, json.dumps({
                "png": base64.b64encode(png_bytes(img)).decode(),
                "ms": int((time.time() - t0) * 1000),
            }))
        elif self.path == "/similarity":
            # Same metric family as the VipSim shader-equivalence validation:
            # SSIM (structure), mean abs pixel diff, CIEDE2000 color distance.
            import numpy as np
            from skimage.metrics import structural_similarity
            from skimage.color import rgb2lab, deltaE_ciede2000
            if ref is None:
                return self._send(200, json.dumps({"error": "no ref"}))
            with _render_lock:
                ours = apply_stack(scene_for(pid, size), filters)
            a = np.asarray(ref, np.float32)
            b = np.asarray(ours, np.float32)
            gray = np.array([0.299, 0.587, 0.114], np.float32)
            ssim_val = float(structural_similarity(a @ gray, b @ gray,
                                                   data_range=255.0))
            mae = float(np.abs(a - b).mean())
            d_e = float(deltaE_ciede2000(rgb2lab(a / 255.0),
                                         rgb2lab(b / 255.0)).mean())
            pix = max(0.0, (1.0 - mae / 255.0) * 100.0)
            ssim_pct = max(0.0, ssim_val * 100.0)
            color_pct = max(0.0, 100.0 - 2.0 * d_e)
            overall = (pix + ssim_pct + color_pct) / 3.0
            self._send(200, json.dumps({
                "overall": round(overall, 1), "ssim": round(ssim_pct, 1),
                "pixel": round(pix, 1), "color": round(color_pct, 1),
                "mae": round(mae, 2), "deltaE": round(d_e, 2),
                "ssim_raw": round(ssim_val, 3),
            }))
        elif self.path == "/influence":
            # Heat overlay of the pixels a single step actually changes:
            # render with vs without that step and diff.
            import numpy as np
            t0 = time.time()
            idx = int(payload.get("index", -1))
            if not (0 <= idx < len(filters)):
                return self._send(400, json.dumps({"error": "bad index"}))
            rest = [s for i, s in enumerate(filters) if i != idx]
            with _render_lock:
                full = apply_stack(scene_for(pid, size), filters)
                without = apply_stack(scene_for(pid, size), rest)
            diff = np.abs(
                np.asarray(full, np.float32) - np.asarray(without, np.float32)
            ).mean(axis=2)
            peak = float(diff.max())
            if peak < 1.0:
                return self._send(200, json.dumps({"empty": True}))
            norm = np.clip(diff / peak, 0.0, 1.0) ** 0.6
            h, w = norm.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = 255   # magenta-pink heat
            rgba[..., 1] = 45
            rgba[..., 2] = 120
            rgba[..., 3] = (norm * 190).astype(np.uint8)
            self._send(200, json.dumps({
                "png": base64.b64encode(png_bytes(Image.fromarray(rgba))).decode(),
                "ms": int((time.time() - t0) * 1000),
            }))
        elif self.path == "/autotune":
            seconds = min(120.0, float(payload.get("seconds", 25)))
            tuned, before, after, evals = autotune(pid, filters, seconds)
            self._send(200, json.dumps({
                "filters": tuned, "before": before, "after": after,
                "evals": evals,
            }))
        elif self.path == "/upload":
            # Custom ref/base image for this pid, session-only (in memory).
            kind = payload.get("kind")
            if kind not in ("ref", "base"):
                return self._send(400, json.dumps({"error": "kind must be ref|base"}))
            if payload.get("clear"):
                _custom[kind].pop(pid, None)
                _scene_cache.clear()
                return self._send(200, json.dumps({"cleared": True}))
            raw = base64.b64decode(payload["png"].split(",")[-1])
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if kind == "ref" and max(img.size) > 900:  # keep metrics fast
                img.thumbnail((900, 900), Image.LANCZOS)
            _custom[kind][pid] = img
            _scene_cache.clear()
            self._send(200, json.dumps({"size": img.size}))
        elif self.path == "/save":
            if not valid_pid(pid):
                return self._send(400, json.dumps({"error": "invalid profile id"}))
            f = PROFILE_DIR / f"{pid}.json"
            # The file may have been moved/deleted while the tab was open
            # (e.g. sorted into alternative-profiles/): recreate it instead
            # of crashing, keeping whatever documentation still exists.
            prof = {}
            if f.is_file():
                try:
                    prof = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    prof = {}
            prof.setdefault("id", pid)
            prof["filters"] = filters
            f.write_text(json.dumps(prof, indent=2) + "\n")
            self._send(200, "{}")
        elif self.path == "/saveas":
            import shutil
            base_id = payload.get("baseId", "")
            new_id = payload.get("newId", "")
            if not valid_pid(new_id):
                return self._send(400, json.dumps(
                    {"error": "name must start with p<digit>, e.g. p3c"}))
            if base_id and not valid_pid(base_id):
                return self._send(400, json.dumps({"error": "invalid base profile id"}))
            dst = PROFILE_DIR / f"{new_id}.json"
            if dst.exists() and not payload.get("overwrite"):
                return self._send(409, json.dumps({"error": new_id + " already exists"}))
            src = PROFILE_DIR / f"{base_id}.json"
            try:
                prof = json.loads(src.read_text()) if src.is_file() else {}
            except (json.JSONDecodeError, OSError):
                prof = {}
            prof["id"] = new_id
            prof["caveat"] = (f"Saved from the tuner as a variant of {base_id}. "
                              "Merge the winner into the base profile, then delete.")
            prof["filters"] = filters
            dst.write_text(json.dumps(prof, indent=2) + "\n")
            src_ref, dst_ref = REF_DIR / f"{base_id}.png", REF_DIR / f"{new_id}.png"
            if src_ref.is_file() and not dst_ref.exists():
                shutil.copy(src_ref, dst_ref)
            self._send(200, "{}")
        else:
            self._send(404, "not found", "text/plain")


if __name__ == "__main__":
    print(f"Profile tuner: http://localhost:{PORT}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
