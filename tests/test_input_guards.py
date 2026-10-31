"""Invalid input must raise, never crash or hang the interpreter.

Pillow 12 segfaults on a NaN, infinite or huge GaussianBlur radius (Pillow
13 raises ValueError), and apply_floaters' rejection samplers loop forever
when no candidate point can be accepted.  Both reach the public kwargs, so
the filters check them first.  apply_bcg converts its input to RGB like the
other filters instead of tone-mapping an alpha channel.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import vipsim  # noqa: E402

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


class TestFloatersSampler(unittest.TestCase):
    def test_parameters_that_accept_no_point_raise(self):
        img = noise_image()
        for kw in ({"circle_radius_uv": float("nan")}, {"circle_radius_uv": -np.inf},
                   {"distribution": "gaussian", "center": (5.0, 5.0), "spread": 0.01},
                   {"distribution": "gaussian", "center": (1.2, 0.5), "spread": 0.01},
                   {"distribution": "gaussian", "angle": float("nan")},
                   {"distribution": "gaussian", "spread": np.inf}):
            with self.subTest(**kw), self.assertRaisesRegex(ValueError, "no point landed"),                     np.errstate(invalid="ignore"):            # NaN/inf geometry
                vipsim.apply_floaters(img, 0.5, n_floaters=200, **kw)

    def test_low_but_nonzero_acceptance_still_finishes(self):
        # A centre 2.5 sigma off-frame accepts ~0.6% of draws: slow, not stuck.
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
