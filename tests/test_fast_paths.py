"""Fast paths in the filters must give exactly what the plain code gave.

Each test runs a fast path against the straightforward version it replaced
(kept here as a reference) and requires bit-identical output.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import vipsim_filters as vf  # noqa: E402


def noise_image(h: int, w: int, seed: int = 0) -> Image.Image:
    return Image.fromarray(
        np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


def to_uint8_reference(x, scale=255.0):
    return np.clip(np.rint(np.asarray(x) * scale), 0, 255).astype(np.uint8)


class TestSrgbConversions(unittest.TestCase):
    """Decode table and in-place encode vs the per-pixel formulas."""

    def test_decode_table_matches_formula_for_every_level(self):
        levels = np.arange(256, dtype=np.uint8)
        for dt in (np.float32, np.float64):
            with self.subTest(dtype=dt.__name__):
                want = vf._srgb_to_linear(levels.astype(dt) / 255.0)
                got = vf._image_to_linear(levels, dt)
                self.assertEqual(got.dtype, want.dtype)
                np.testing.assert_array_equal(got, want)

    def test_decode_image_matches_call_site_expression(self):
        img = noise_image(19, 23)
        want = vf._srgb_to_linear(np.asarray(img, dtype=np.float32) / 255.0)
        got = vf._image_to_linear(img)
        self.assertEqual(got.dtype, want.dtype)
        np.testing.assert_array_equal(got, want)

    def test_decode_falls_back_to_formula_beyond_8_bit(self):
        img = Image.fromarray(np.arange(0, 600, 2, dtype=np.uint16).reshape(15, 20))
        want = vf._srgb_to_linear(np.asarray(img, dtype=np.float32) / 255.0)
        np.testing.assert_array_equal(vf._image_to_linear(img), want)

    def test_encode_matches_where_formula(self):
        knee = 0.0031308
        for dt in (np.float32, np.float64):
            with self.subTest(dtype=dt.__name__):
                x = np.concatenate([
                    np.linspace(-0.25, 1.25, 200_001),
                    [knee, np.nextafter(dt(knee), dt(0)), np.nextafter(dt(knee), dt(1))],
                ]).astype(dt)
                xc = np.clip(x, 0.0, 1.0)
                want = np.where(xc <= knee, xc * 12.92,
                                1.055 * np.power(xc, 1.0 / 2.4) - 0.055)
                got = vf._linear_to_srgb(x)
                self.assertEqual(got.dtype, want.dtype)
                np.testing.assert_array_equal(got, want)

    def test_to_uint8_matches_reference(self):
        x = np.linspace(-2.0, 300.0, 100_003)
        for dt in (np.float32, np.float64):
            for scale in (255.0, 1.0):
                with self.subTest(dtype=dt.__name__, scale=scale):
                    xs = x.astype(dt) / (255.0 if scale == 255.0 else 1.0)
                    np.testing.assert_array_equal(vf._to_uint8(xs, scale),
                                                  to_uint8_reference(xs, scale))

    def test_8bit_levels_survive_decode_to_float_and_back(self):
        # apply_vortex copies pixels outside its field straight from the input.
        levels = np.arange(256, dtype=np.uint8)
        for dt in (np.float32, np.float64):
            with self.subTest(dtype=dt.__name__):
                x = levels.astype(np.float32) / 255.0
                np.testing.assert_array_equal(vf._to_uint8(x.astype(dt)), levels)


def paint_blobs_reference(out, xs, ys, sizes, rng, alpha, target_val,
                          overlay_size=1024.0):
    """The per-blob loop apply_floaters ran before _paint_blobs."""
    h, w = out.shape[:2]
    for sx, sy, sz in zip(xs, ys, sizes):
        s_x = max(1, int(round(sz * w / overlay_size)))
        s_y = max(1, int(round(sz * h / overlay_size)))
        hx, hy = s_x // 2, s_y // 2
        x0, x1 = max(0, sx - hx), min(w, sx + hx + 1)
        y0, y1 = max(0, sy - hy), min(h, sy + hy + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = rng.random((y1 - y0, x1 - x0)) > 0.4
        for c in range(3):
            tile = out[y0:y1, x0:x1, c]
            tile[mask] = tile[mask] * (1.0 - alpha) + target_val * alpha
            out[y0:y1, x0:x1, c] = tile


class TestFloaterBlobs(unittest.TestCase):
    def check(self, h, w, n, pool, alpha, target_val, seed=0):
        rng = np.random.default_rng(seed)
        base = (rng.random((h, w, 3)) * 255.0).astype(np.float32)
        xs = rng.integers(0, w, n)
        ys = rng.integers(0, h, n)
        sizes = rng.choice(np.asarray(pool), size=n)
        want, got = base.copy(), base.copy()
        rng_want, rng_got = np.random.default_rng(seed + 1), np.random.default_rng(seed + 1)
        paint_blobs_reference(want, xs, ys, sizes, rng_want, alpha, target_val)
        vf._paint_blobs(got, xs, ys, sizes, rng_got, alpha, target_val)
        np.testing.assert_array_equal(got, want)
        # Same number of draws: the generators end in the same state.
        self.assertEqual(rng_got.bit_generator.state, rng_want.bit_generator.state)

    def test_dense_overlapping_blobs(self):
        # Hundreds of blobs per pixel row, so most pixels are hit many times.
        for alpha, target_val in [(0.24, 255.0), (0.5, 0.0), (1.0, 216.75), (0.3, 255.0)]:
            with self.subTest(alpha=alpha, target_val=target_val):
                self.check(37, 53, 3000, [2, 3, 5, 40, 90], alpha, target_val)

    def test_blobs_clipped_at_frame_edges(self):
        self.check(9, 7, 200, [3, 60, 300], 0.4, 255.0, seed=3)
        self.check(240, 420, 5000, [2, 3, 5, 9], 0.34, 255.0, seed=4)

    def test_float_sizes_and_no_blobs(self):
        self.check(31, 45, 500, [2.5, 3.0, 6.5], 0.7, 0.0, seed=5)
        self.check(31, 45, 0, [2], 0.7, 0.0, seed=6)


def blend_reference(arr, bias, n_levels):
    """The stacked pyramid lookup apply_field_loss ran before _blend_mip_levels."""
    pyramid = vf._build_mip_pyramid(arr, n_levels)
    lo = np.floor(bias).astype(np.int32)
    hi = np.clip(lo + 1, 0, n_levels - 1)
    frac = (bias - lo)[..., None]
    stack = np.stack(pyramid, axis=0)
    iy, ix = np.indices(bias.shape)
    return stack[lo, iy, ix] * (1 - frac) + stack[hi, iy, ix] * frac


class TestFieldLossBlend(unittest.TestCase):
    def test_matches_stacked_lookup(self):
        h, w = 41, 67
        arr = np.asarray(noise_image(h, w), dtype=np.float32) / 255.0
        n_levels = int(2 + np.floor(np.log(max(h, w))))
        top = np.float32(n_levels - 1)
        rng = np.random.default_rng(1)
        radial = vf._radial_weight_mask(h, w, kind="central")
        biases = {
            "random": rng.random((h, w)).astype(np.float32) * top,
            "zero": np.zeros((h, w), np.float32),
            "whole levels": rng.integers(0, n_levels, (h, w)).astype(np.float32),
            "top level": np.full((h, w), top),
            "low severity": radial * np.float32(0.06) * top,
            "high severity": vf._radial_weight_mask(h, w, kind="peripheral") * top,
        }
        for name, bias in biases.items():
            with self.subTest(bias=name):
                want = blend_reference(arr, bias, n_levels)
                got = vf._blend_mip_levels(arr, bias, n_levels)
                self.assertEqual(got.dtype, want.dtype)
                np.testing.assert_array_equal(got, want)


def vortex_reference(img: Image.Image, severity: float = 0.5,
                     center=(0.5, 0.5),
                     vortex_radius: float = None,
                     radius_y: float = None,
                     radius_y_down: float = None,
                     suction_strength: float = None,
                     falloff_exponent: float = 2.0,
                     swirl_strength: float = 0.0,
                     inner_circle_radius: float = 0.01,
                     inner_color: float = 0.5,
                     inner_center=None,
                     inner_fade: float = None,
                     inner_opacity: float = 1.0,
                     halo_radius: float = 0.0,
                     halo_center=None,
                     halo_thickness: float = 0.012,
                     halo_color: float = 1.0,
                     halo_opacity: float = 0.5,
                     noise_amount: float = 0.05,
                     noise_scale: float = 10.0,
                     fade_width: float = 0.05) -> Image.Image:
    """apply_vortex as it was before the bounding box: every pixel, full frame."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0.0:
        return img.copy()
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    cy, cx = center
    if vortex_radius is None:
        vortex_radius = 0.05 + severity * 0.45  # 0.05..0.5
    if suction_strength is None:
        suction_strength = 0.2 + severity * 1.8  # 0.2..2.0

    aspect = w / h

    ys_norm = (np.arange(h, dtype=np.float32) + 0.5) / h
    xs_norm = (np.arange(w, dtype=np.float32) + 0.5) / w
    yy, xx = np.meshgrid(ys_norm, xs_norm, indexing="ij")

    adj_x = xx * aspect
    adj_y = yy
    adj_cx = cx * aspect
    adj_cy = cy
    # Elliptical field: stretch y so the ellipse becomes a circle of radius
    # `vortex_radius` in the working space; sampling positions are unstretched
    # on the way back out. `radius_y_down` allows an asymmetric vertical
    # reach (upper vs lower half of the field).
    ys_up = vortex_radius / max(radius_y, 1e-6) if radius_y else 1.0
    ys_dn = (vortex_radius / max(radius_y_down, 1e-6)
             if radius_y_down else ys_up)
    dx = adj_x - adj_cx
    dy_raw = adj_y - adj_cy
    ys = np.where(dy_raw < 0, ys_up, ys_dn)
    dy = dy_raw * ys
    dist = np.sqrt(dx * dx + dy * dy)

    # Value-noise grid (matches the shader's rand/lerp construction)
    def _value_noise(u, v, scale):
        p_u = np.floor(u * scale)
        p_v = np.floor(v * scale)
        f_u = u * scale - p_u
        f_v = v * scale - p_v

        def _rand(a, b):
            return np.modf(np.sin(a * 12.9898 + b * 78.233) * 43758.5453)[0]

        a = _rand(p_u, p_v)
        b = _rand(p_u + 1, p_v)
        c = _rand(p_u, p_v + 1)
        d = _rand(p_u + 1, p_v + 1)
        su = f_u * f_u * (3.0 - 2.0 * f_u)
        sv = f_v * f_v * (3.0 - 2.0 * f_v)
        return (a + (b - a) * su) + (c - a) * sv * (1.0 - su) + (d - b) * su * sv

    noise_val = _value_noise(xx, yy, noise_scale)
    dist_n = dist + noise_val * noise_amount * vortex_radius

    inside = dist_n < vortex_radius
    out = arr.copy()
    if np.any(inside):
        eps = 1e-8
        angle = np.arctan2(dy, dx)
        # Optional angular noise (small)
        angle_noise = _value_noise(xx, yy, noise_scale)
        angle = angle + angle_noise * noise_amount

        # Radial suction: sample further out
        d_over_R = np.clip(dist_n / max(vortex_radius, eps), 0.0, 1.0)
        r_sample = dist_n * (1.0 + suction_strength
                             * (1.0 - d_over_R) ** falloff_exponent)

        # Swirl: rotate the sample angle, strongest at center, 0 at the rim
        if swirl_strength:
            angle = angle + swirl_strength * (1.0 - d_over_R) ** 2

        sample_x = adj_cx + r_sample * np.cos(angle)
        sin_a = np.sin(angle)
        ys_back = np.where(sin_a < 0, ys_up, ys_dn)
        sample_y = adj_cy + r_sample * sin_a / ys_back
        sample_x = sample_x / aspect  # undo aspect correction
        sample_x = np.clip(sample_x, 0.0, 1.0)
        sample_y = np.clip(sample_y, 0.0, 1.0)
        # Bilinear sampling (Unity tex2D is bilinear; nearest-neighbour
        # shreds strong stretches into aliased filaments)
        fx = sample_x * (w - 1)
        fy = sample_y * (h - 1)
        x0 = np.floor(fx).astype(np.int32)
        y0 = np.floor(fy).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        wx = (fx - x0)[..., None]
        wy = (fy - y0)[..., None]
        top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
        bot = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
        vortexed = top * (1.0 - wy) + bot * wy

        # Outer fade back to source (smoothstep R-fade..R)
        t_out = np.clip((dist_n - (vortex_radius - fade_width))
                        / max(fade_width, eps), 0.0, 1.0)
        s_out = t_out * t_out * (3.0 - 2.0 * t_out)
        blended = vortexed * (1.0 - s_out[..., None]) + arr * s_out[..., None]

        # Inner core dot (grey in Unity; position/brightness/softness/opacity
        # configurable — only rendered inside the vortex field)
        has_halo = halo_radius > 0 and halo_opacity > 0
        if inner_circle_radius > 0 or has_halo:
            if inner_center is not None:
                icy, icx = inner_center
                idx = adj_x - icx * aspect
                idy = adj_y - icy
                idist = np.sqrt(idx * idx + idy * idy)
            else:
                idist = dist_n

        if inner_circle_radius > 0:
            i_fade = fade_width if inner_fade is None else inner_fade
            grey = np.full_like(arr, inner_color)
            t_in = np.clip((idist - (inner_circle_radius - i_fade))
                            / max(i_fade, eps), 0.0, 1.0)
            s_in = t_in * t_in * (3.0 - 2.0 * t_in)
            alpha = (1.0 - s_in)[..., None] * inner_opacity
            alpha = np.where((idist < inner_circle_radius)[..., None], alpha, 0.0)
            blended = grey * alpha + blended * (1.0 - alpha)

        # Halo: luminous orb (solid glow inside halo_radius, Gaussian
        # falloff outward); own center if given
        if has_halo:
            if halo_center is not None:
                hcy, hcx = halo_center
                hdx = adj_x - hcx * aspect
                hdy = adj_y - hcy
                hdist = np.sqrt(hdx * hdx + hdy * hdy)
            else:
                hdist = idist
            band = np.where(
                hdist <= halo_radius, 1.0,
                np.exp(-0.5 * ((hdist - halo_radius)
                               / max(halo_thickness, eps)) ** 2))
            h_alpha = band[..., None] * halo_opacity
            halo_col = np.full_like(arr, halo_color)
            blended = halo_col * h_alpha + blended * (1.0 - h_alpha)

        out = np.where(inside[..., None], blended, arr)

    return Image.fromarray(to_uint8_reference(out))


class TestVortexBoundingBox(unittest.TestCase):
    """apply_vortex only computes inside its field's bounding box."""

    CASES = [
        {},
        {"noise_amount": 0},
        {"noise_amount": 1.5, "noise_scale": 37.0},
        {"radius_y": 0.18},
        {"radius_y": 0.3, "radius_y_down": 0.1, "swirl_strength": 0.8},
        {"center": (0.1, 0.95)},
        {"center": (1.0, 0.0), "vortex_radius": 0.2},
        {"center": (-0.4, 1.6)},
        {"vortex_radius": 1.5},
        {"vortex_radius": 0.0},
        {"inner_circle_radius": 0.05, "inner_color": 1.0, "inner_fade": 0.0,
         "inner_center": (0.45, 0.55), "inner_opacity": 0.6},
        {"halo_radius": 0.02, "halo_opacity": 0.7, "halo_center": (0.52, 0.47)},
        {"center": (0.5, 0.61), "vortex_radius": 0.33, "suction_strength": 2,
         "fade_width": 0, "swirl_strength": -0.05, "radius_y": 0.18,
         "inner_circle_radius": 0.006, "inner_color": 1, "inner_center": (0.494, 0.549),
         "inner_fade": 0, "halo_radius": 0.004, "halo_thickness": 0.03,
         "halo_opacity": 0.68, "halo_center": (0.506, 0.61)},
    ]

    def test_matches_full_frame(self):
        for h, w in [(48, 80), (31, 17), (5, 7)]:
            img = noise_image(h, w, seed=h)
            for kw in self.CASES:
                for sev in (0.05, 0.5, 1.0):
                    with self.subTest(size=(h, w), severity=sev, **kw):
                        want = vortex_reference(img, sev, **kw)
                        got = vf.apply_vortex(img, sev, **kw)
                        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


if __name__ == "__main__":
    unittest.main()
