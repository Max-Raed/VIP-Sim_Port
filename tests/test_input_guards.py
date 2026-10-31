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

    # Rotated: a small round Gaussian tilted by 0.03 degrees and two thin
    # ellipses, where the line through the centre crosses a frame edge inside
    # the integration range.  Parameters as apply_floaters derives them; the
    # values are scipy multivariate_normal.cdf (abseps=1e-14) and quad
    # conditioned on x and on y, which agree to 1e-9.
    ROTATED = ((1042.0152375613443, 576.1103522726089, 3.4689434504226266,
                3.4689434504226266, -0.03088682521694854, 1.033036672e-07),
               (-33.09965573837451, 12.68955623074828, 128.5509272185956,
                1.1663899096118466, 151.74263039481275, 1.138183079e-07),
               (-257.59673335083085, 904.0559432517198, 2739.8007792332273,
                1.0811504701228878, 25.65676268157199, 8.629678332e-08))

    def test_frame_probability_rotated(self):
        for mu_x, mu_y, sx, sy, angle, ref in self.ROTATED:
            rad = np.deg2rad(angle)
            with self.subTest(angle=angle):
                p = filters._gaussian_frame_probability(mu_x, mu_y, sx, sy, np.cos(rad),
                                                        np.sin(rad), 1024.0)
                self.assertAlmostEqual(p / ref, 1.0, delta=1e-6)

    def test_threshold_rotated(self):
        # Within 15% of the threshold on either side, the decision follows
        # the true probability.
        for mu_x, mu_y, sx, sy, angle, ref in self.ROTATED:
            rad = np.deg2rad(angle)
            args = (mu_x, mu_y, sx, sy, np.cos(rad), np.sin(rad), 1024.0)
            with self.subTest(angle=angle):
                if ref >= filters._FLOATER_MIN_ACCEPTANCE:
                    filters._check_gaussian_reachable(*args)
                else:
                    with self.assertRaisesRegex(ValueError, "no point would land"):
                        filters._check_gaussian_reachable(*args)

    def test_threshold(self):
        # 4.1 and 5.6 sigma off the right edge: in-frame probability about
        # 2e-5 and 1e-8, on either side of _FLOATER_MIN_ACCEPTANCE (1e-7).
        sigma = 10.24
        filters._check_gaussian_reachable(1024 + 4.1 * sigma, 512, sigma, sigma, 1.0, 0.0, 1024.0)
        with self.assertRaisesRegex(ValueError, "no point would land"):
            filters._check_gaussian_reachable(1024 + 5.6 * sigma, 512, sigma, sigma, 1.0, 0.0,
                                              1024.0)

    def test_one_element_array_parameters(self):
        # A 1-element array broadcasts like the scalar in the sampler: the
        # image is the scalar's, and the check judges the value.
        img = noise_image()
        args = {"n_floaters": 200, "seed": 1, "distribution": "gaussian"}
        for kw, ref in (({"angle": np.array([30.0])}, {"angle": 30.0}),
                        ({"spread": np.array([0.1])}, {"spread": 0.1}),
                        ({"spread_x": np.array([0.05])}, {"spread_x": 0.05}),
                        ({"angle": np.array([[30.0]])}, {"angle": 30.0}),
                        ({"center": (np.array([0.3]), 0.7)}, {"center": (0.3, 0.7)})):
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                np.testing.assert_array_equal(
                    np.asarray(vipsim.apply_floaters(img, 0.6, **args, **kw)),
                    np.asarray(vipsim.apply_floaters(img, 0.6, **args, **ref)))
        with mock.patch.object(np.random, "default_rng", lambda seed=None: NoDraws()), \
                self.assertRaisesRegex(ValueError, "no point would land"):
            vipsim.apply_floaters(img, 1.0, n_floaters=1, distribution="gaussian",
                                  center=(np.array([5.0]), 5.0), spread=0.01)

    def test_few_floaters_below_min_acceptance_still_sample(self):
        # 5.2 sigma off the right edge (p ~ 9.96e-8): one floater needs ~1e7
        # draws, which the old loop finished in seconds to minutes; ten
        # would need ~1e8 and raise, as does one floater at 5.7 sigma.
        sigma = 10.24
        mu = 1024 + 5.2 * sigma
        filters._check_gaussian_reachable(mu, 512, sigma, sigma, 1.0, 0.0, 1024.0, 1)
        for x, n in ((mu, 10), (1024 + 5.7 * sigma, 1)):
            with self.subTest(sigmas=(x - 1024) / sigma, n_floaters=n), \
                    self.assertRaisesRegex(ValueError, "no point would land"):
                filters._check_gaussian_reachable(x, 512, sigma, sigma, 1.0, 0.0, 1024.0, n)
        img = noise_image()
        out = vipsim.apply_floaters(img, 0.5, n_floaters=1, seed=3, distribution="gaussian",
                                    center=(0.5, 1.052), spread=0.01)
        self.assertEqual((np.asarray(out) != np.asarray(img.convert("RGB"))).any(-1).sum(), 1)

    def test_slow_but_reachable_parameters_warn_before_sampling(self):
        # 5 sigma off the right edge passes the floor, but the default 50000
        # floaters then need ~1.7e11 draws (about 2 h): warn first.
        img = noise_image()
        with mock.patch.object(np.random, "default_rng", lambda seed=None: NoDraws()), \
                self.assertWarnsRegex(RuntimeWarning, r"need about 1\.74e\+11 draws"), \
                self.assertRaisesRegex(AssertionError, "the sampler drew"):
            vipsim.apply_floaters(img, 1.0, distribution="gaussian", center=(0.5, 1.05),
                                  spread=0.01)
        # In-frame and moderately off-frame parameters stay silent.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for n in (1, 50000):
                filters._check_gaussian_reachable(512, 512, 164, 164, 1.0, 0.0, 1024.0, n)
            filters._check_gaussian_reachable(1024 + 2.5 * 10.24, 512, 10.24, 10.24, 1.0, 0.0,
                                              1024.0, 500)

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
