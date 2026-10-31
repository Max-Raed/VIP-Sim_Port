"""Invalid input must raise, never crash or hang the interpreter.

Pillow 12 segfaults on a NaN, infinite or huge GaussianBlur radius (Pillow
13 raises ValueError), and apply_floaters' rejection samplers loop forever
when no candidate point can be accepted; those parameters are rejected
before anything is drawn.  Both reach the public kwargs, so
the filters check them first.  apply_bcg converts its input to RGB like the
other filters instead of tone-mapping an alpha channel.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import vipsim  # noqa: E402
from vipsim import filters  # noqa: E402

# A hand-drawn cluster (normalised [y, x]) like the floater profiles use.
REGION = [[0.30, 0.35], [0.25, 0.60], [0.45, 0.75], [0.70, 0.62], [0.65, 0.30]]


def noise_image(h=48, w=64, mode="RGB", seed=0):
    a = np.random.default_rng(seed).integers(0, 256, (h, w, 4), dtype=np.uint8)
    return Image.fromarray(a, "RGBA").convert(mode)


class TestBlurRadius(unittest.TestCase):
    # Pillow 13's own checks, so both supported Pillows raise the same error.
    def test_nan_severity(self):
        with self.assertRaisesRegex(ValueError, "radius must be >= 0"):
            vipsim.apply_blur(noise_image(), float("nan"))

    def test_bloom_blur_size(self):
        for blur_size, message in ((float("nan"), "radius must be >= 0"),
                                   (float("inf"), "radius must be >= 0"),
                                   (1e18, "radius too large")):
            with self.subTest(blur_size=blur_size), self.assertRaisesRegex(ValueError, message):
                vipsim.apply_bloom(noise_image(), 0.5, blur_size=blur_size)

    def test_floaters_region_feather(self):
        for feather, message in ((float("nan"), "radius must be >= 0"),
                                 (float("inf"), "radius must be >= 0"),
                                 (1e7, "radius too large")):
            with self.subTest(region_feather=feather), self.assertRaisesRegex(ValueError, message):
                vipsim.apply_floaters(noise_image(), 0.5, region=REGION, region_feather=feather)

    def test_valid_radii_still_blur(self):
        img = noise_image()
        self.assertEqual(vipsim.apply_blur(img, 0.3).size, img.size)
        self.assertEqual(vipsim.apply_bloom(img, 0.5, blur_size=-np.inf).size, img.size)
        self.assertEqual(vipsim.apply_floaters(img, 0.5, region=REGION,
                                               region_feather=-0.01).size, img.size)


class NoDraws:
    """Stands in for numpy's Generator: fails the test if anything is drawn."""
    def __getattr__(self, name):
        raise AssertionError(f"the sampler drew ({name})")


class TestFloatersSampler(unittest.TestCase):
    BAD = ({"circle_radius_uv": float("nan")}, {"circle_radius_uv": -np.inf},
           {"circle_radius_uv": -1e308},             # * 1024 overflows to -inf
           {"distribution": "gaussian", "center": (5.0, 5.0), "spread": 0.01},
           {"distribution": "gaussian", "center": (1.2, 0.5), "spread": 0.01},
           {"distribution": "gaussian", "center": (float("nan"), 0.5)},
           {"distribution": "gaussian", "center": (0.5, np.inf)},
           {"distribution": "gaussian", "angle": float("nan")},
           {"distribution": "gaussian", "angle": np.inf},
           {"distribution": "gaussian", "spread": np.inf},
           {"distribution": "gaussian", "spread": 1e6},
           # A thin ellipse passing outside a corner: both of its x and y
           # marginals overlap the frame, but no draw lands in it.
           {"distribution": "gaussian", "center": (-0.2, -0.2), "spread_x": 2.0,
            "spread_y": 0.001, "angle": -45.0})

    def test_parameters_that_accept_no_point_raise_before_sampling(self):
        # The old loops never ended here; a draw counter took minutes with
        # one floater (2 draws per pass).  Nothing may be drawn at all.
        img = noise_image()
        for kw in self.BAD:
            for n in (1, 200):
                with self.subTest(n_floaters=n, **kw), \
                        mock.patch.object(np.random, "default_rng", lambda seed=None: NoDraws()), \
                        self.assertRaisesRegex(ValueError, "no point would land"):
                    vipsim.apply_floaters(img, 1.0, n_floaters=n, **kw)

    def test_frame_probability_matches_closed_form(self):
        # Axis-aligned (0 and 180 degrees, and 90 with the spreads swapped),
        # the probability is a product of two normal-CDF differences.
        phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))      # noqa: E731
        for mu_x, mu_y, sx, sy in ((512, 512, 164, 164), (1049.6, 512, 10.24, 10.24),
                                   (-300, 900, 400, 50), (1070.0, 512, 10.24, 10.24)):
            exact = ((phi((1024 - mu_x) / sx) - phi(-mu_x / sx))
                     * (phi((1024 - mu_y) / sy) - phi(-mu_y / sy)))
            for angle in (0.0, 90.0, 180.0):
                rad = np.deg2rad(angle)
                sxa, sya = (sx, sy) if angle != 90.0 else (sy, sx)
                with self.subTest(mu=(mu_x, mu_y), sigma=(sx, sy), angle=angle):
                    p = filters._gaussian_frame_probability(mu_x, mu_y, sxa, sya, np.cos(rad),
                                                            np.sin(rad), 1024.0)
                    self.assertAlmostEqual(p / exact, 1.0, delta=1e-3)

    def test_threshold(self):
        # 4.1 and 5.6 sigma off the right edge: in-frame probability about
        # 2e-5 and 1e-8, on either side of _FLOATER_MIN_ACCEPTANCE (1e-7).
        sigma = 10.24
        filters._check_gaussian_reachable(1024 + 4.1 * sigma, 512, sigma, sigma, 1.0, 0.0, 1024.0)
        with self.assertRaisesRegex(ValueError, "no point would land"):
            filters._check_gaussian_reachable(1024 + 5.6 * sigma, 512, sigma, sigma, 1.0, 0.0,
                                              1024.0)

    def test_low_but_nonzero_acceptance_still_finishes(self):
        # A centre 2.5 sigma off-frame accepts ~0.6% of draws: slow, not stuck,
        # and the same image as before (the checks draw nothing).
        out = vipsim.apply_floaters(noise_image(), 0.5, n_floaters=500, seed=3,
                                    distribution="gaussian", center=(0.5, 1.025), spread=0.01)
        self.assertEqual(out.size, (64, 48))


class TestBcgModes(unittest.TestCase):
    def test_non_rgb_input_is_converted_like_other_filters(self):
        # RGBA kept its mode and put alpha through the tone curve; L and P
        # with saturation != 1 raised a matmul ValueError.
        for mode in ("RGBA", "LA", "L", "P"):
            img = noise_image(mode=mode)
            for kw in ({"brightness": 0.5}, {"saturation": 0.5, "contrast": 1.2},
                       {"gamma": 1.4, "color_space": "srgb"}):
                with self.subTest(mode=mode, **kw):
                    out = vipsim.apply_bcg(img, **kw)
                    self.assertEqual(out.mode, "RGB")
                    np.testing.assert_array_equal(
                        np.asarray(out), np.asarray(vipsim.apply_bcg(img.convert("RGB"), **kw)))


if __name__ == "__main__":
    unittest.main()
