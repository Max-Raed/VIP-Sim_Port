"""The GPU colour filters must reproduce the CPU filters they port.

VipSimGPU.cvd, bcg, detail_loss and vignette are checked against apply_cvd,
apply_bcg, apply_detail_loss and apply_vignette on random and natural frames
at 64x96, 84x84 and an odd size, at severities near 0 and 1, with their
keyword arguments and on a batch of two different frames.

detail_loss, vignette and bcg without saturation are integer masks, float32
products and per-level tables built by the CPU's own code, so they must match
exactly.  cvd and bcg's saturation step also replay the order in which
numpy's BLAS sums 3-element dots; where numpy sums like the twin (checked in
setUpClass) they must match exactly too, elsewhere they must stay within
GpuParityCase.assert_close (max 1 LSB, mean < 0.02).
Runs on CUDA when available, else on the CPU; skipped without torch.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import functools
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vipsim  # noqa: E402
from vipsim import filters  # noqa: E402
from test_gpu_primitives import TRAFFIC, natural_crop  # noqa: E402
from test_rounding import GpuParityCase  # noqa: E402

SCENE = ROOT / "vipsim_assets" / "Trafficscene_2048x1024.png"
SIZES = ((64, 96), (84, 84), (61, 97))              # (height, width)
SEVERITIES = (1e-6, 0.05, 0.37, 0.5, 0.6, 0.95, 1.0)
CVD_TYPES = tuple(vipsim.CVD_MATRICES) + ("monochrome",)
# Identity, the Unity validation point, the presets, gamma both ways (torch
# special-cases 0.5 and 2), saturation to grey and beyond, white-out, flat
# contrast, out-of-range values, and the bcg steps of profiles p3 and p6.
BCG_KWARGS = [{}, {"brightness": 0.75, "contrast": 1.25},
              {"brightness": 0.9, "contrast": 0.8, "gamma": 1.1},
              {"brightness": 0.75, "contrast": 0.5, "gamma": 1.4},
              {"gamma": 0.5}, {"gamma": 2.2}, {"saturation": 0.0}, {"saturation": 0.5},
              {"brightness": 0.8, "contrast": 1.3, "gamma": 1.4, "saturation": 1.8},
              {"brightness": 2.5, "contrast": 0.0},
              {"brightness": -0.5, "contrast": 3.0, "gamma": 3.0, "saturation": 5.0},
              {"brightness": 0.85, "contrast": 0.67, "gamma": 1.06, "saturation": 0.48},
              {"brightness": 3.1395, "contrast": 0.355, "gamma": 0.7875, "saturation": 0.08}]
# Severities giving every bit depth: bits = round(8 - 7 * severity) -> 8..1
# (0.5 -> 4.5 rounds half to even, to 4).
DETAIL_SEVERITIES = (0.001, 0.143, 0.3, 0.43, 0.5, 0.64, 0.72, 0.86, 1.0, 1.4)
VIGNETTE_KWARGS = ([{"severity": s} for s in SEVERITIES]
                   + [{"severity": 0.5, "mode": "light"}, {"severity": 1.0, "mode": "light"},
                      {"strength": 0.8, "radius": 0.2, "softness": 0.1},
                      {"strength": 1.0, "radius": 0.0, "softness": 1.0},
                      {"strength": 0.7, "radius": 0.9, "softness": 0.0, "mode": "light"},
                      {"strength": 0.6, "radius": 1.2},
                      {"severity": 0.2, "strength": 1.5},
                      {"severity": 0.11, "radius": 0.61, "softness": 1.3}])


def left_to_right_dot(a: np.ndarray, m: np.ndarray, sdot: bool = False) -> np.ndarray:
    """a[..., 3] . m[i] for each row i, summed left to right without fused
    multiply-adds, written out in numpy: the order _channel_dot replays.
    sdot=True sums the products in float64 and rounds once."""
    a = a.astype(m.dtype)
    p0, p1, p2 = (a[..., j:j + 1] * m[:, j] for j in range(3))
    if sdot:
        return ((p0.astype(np.float64) + p1) + p2).astype(p0.dtype)
    return (p0 + p1) + p2


def numpy_dots_match_the_twin(vg) -> bool:
    """Does numpy sum its 3-element dots here in the order the twin replays?

    np.dot(lin, mat.T) (ddot) and float32 `arr @ luma` (sgemv) add the
    products left to right; on an image one pixel wide `arr @ luma` (sdot)
    adds the float32 products in float64.  That is numpy 2.5 with OpenBLAS
    0.3 on x86-64; a BLAS that fuses multiply-adds rounds some sums otherwise.
    The answer compares numpy with left_to_right_dot, not with the twin, so
    a twin that sums in another order cannot turn the exact checks off
    (TestGpuColorTables checks the twin against left_to_right_dot).
    """
    rng = np.random.default_rng(8)
    lin = rng.random((64, 7, 3), dtype=np.float32)
    mat, luma = rng.random((3, 3)), vg._BCG_LUMA
    return (np.array_equal(np.dot(lin, mat.T), left_to_right_dot(lin, mat))
            and np.array_equal(lin @ luma[0], left_to_right_dot(lin, luma)[..., 0])
            and np.array_equal(lin[:, :1] @ luma[0],
                               left_to_right_dot(lin[:, :1], luma, True)[..., 0]))


@functools.lru_cache(maxsize=None)
def color_frames() -> dict:
    """Random and natural frames at SIZES, built once for every test class."""
    rng = np.random.default_rng(5)
    frames = {}
    for h, w in SIZES:
        frames[f"random_{h}x{w}"] = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        if TRAFFIC.is_file():
            frames[f"traffic_{h}x{w}"] = natural_crop(h, w)
    return frames


class GpuColorCase(GpuParityCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.exact_dots = numpy_dots_match_the_twin(cls.vg)
        cls.frames = color_frames()

    def to_gpu(self, frame):
        return self.vg.frames_to_gpu(frame, self.sim.device)

    def compare(self, gpu, cpu, exact):
        if exact:
            np.testing.assert_array_equal(gpu, np.asarray(cpu))
        else:
            self.assert_close(gpu.astype(int), cpu)

    def check(self, name, exact, frames=None, **kw):
        """sim.<name>(x, **kw) against apply_<name>(img, **kw) on every frame."""
        cpu_fn = getattr(vipsim, "apply_" + name)
        for fname, frame in (frames or self.frames).items():
            with self.subTest(frame=fname, **kw):
                gpu = self.vg.gpu_to_frames(getattr(self.sim, name)(self.to_gpu(frame), **kw))[0]
                self.compare(gpu, cpu_fn(Image.fromarray(frame), **kw), exact)


class TestGpuCvd(GpuColorCase):
    def test_every_type_and_severity(self):
        self.assertIn("traffic_84x84", self.frames, "Trafficscene.png missing")
        for cvd_type in CVD_TYPES:
            for s in SEVERITIES:
                self.check("cvd", self.exact_dots, cvd_type=cvd_type, severity=s)

    def test_monochrome_colours_where_float32_matters(self):
        # The CPU scales monochrome's linear image in float32; in float64 these
        # colours (about 45 in 2**24) round to another level.
        colours = np.array([[4, 168, 83], [5, 215, 247], [11, 229, 180], [20, 87, 198],
                            [35, 206, 175], [37, 208, 179], [40, 73, 44], [43, 168, 14]],
                           np.uint8)
        self.check("cvd", self.exact_dots, {"column": colours[:, None, :]},
                   cvd_type="monochrome", severity=0.05)

    def test_zero_severity_returns_input_before_the_type_lookup(self):
        x = self.to_gpu(self.frame)
        self.assertIs(self.sim.cvd(x, "no-such-type", 0.0), x)
        self.assertIs(self.sim.cvd(x, "deuteranomaly", -0.5), x)
        with self.assertRaises(KeyError):
            self.sim.cvd(x, "no-such-type", 0.5)
        with self.assertRaises(KeyError):
            vipsim.apply_cvd(Image.fromarray(self.frame), "no-such-type", 0.5)


class TestGpuBcg(GpuColorCase):
    def test_settings_in_both_color_spaces(self):
        for kw in BCG_KWARGS:
            exact = self.exact_dots or kw.get("saturation", 1.0) == 1.0
            for color_space in ("linear", "srgb"):
                self.check("bcg", exact, color_space=color_space, **kw)

    def test_one_pixel_wide_frames_use_the_sdot_sum(self):
        # numpy's `arr @ luma` takes another summation path when W == 1.
        rng = np.random.default_rng(6)
        frames = {"37x1": rng.integers(0, 256, (37, 1, 3), dtype=np.uint8),
                  "1x1": rng.integers(0, 256, (1, 1, 3), dtype=np.uint8),
                  "1x29": rng.integers(0, 256, (1, 29, 3), dtype=np.uint8)}
        for kw in BCG_KWARGS[6:]:
            for color_space in ("linear", "srgb"):
                self.check("bcg", self.exact_dots, frames, color_space=color_space, **kw)

    def test_colours_where_sdot_and_sgemv_round_apart(self):
        # About one colour in a million rounds differently in the two sums;
        # random frames almost never contain one, so these columns do.
        columns = {("linear", 0.5): [[2, 208, 103], [7, 56, 44], [29, 120, 97], [37, 181, 227],
                                     [37, 196, 180], [46, 211, 243], [59, 156, 15], [63, 207, 211]],
                   ("srgb", 0.0): [[1, 69, 13], [1, 178, 69], [1, 179, 253], [2, 125, 134],
                                   [3, 45, 231], [3, 126, 135], [3, 235, 191], [4, 100, 168]]}
        for (color_space, saturation), colours in columns.items():
            column = {"column": np.array(colours, np.uint8)[:, None, :]}
            self.check("bcg", self.exact_dots, column, color_space=color_space,
                       saturation=saturation)

    def test_nan_saturation_matches_cpu(self):
        # NaN reaches the linear -> sRGB step, which gives it the level of
        # numpy's NaN -> uint8 cast (0 here), not the last threshold's.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.check("bcg", True, saturation=float("nan"))

    def test_numpy_scalar_kwargs_promote_like_numpy(self):
        # Under NEP 50 a NumPy float64 kwarg makes the CPU work in float64; the
        # same values as Python floats must still get the float32 results.
        f64 = np.float64
        cases = [({"brightness": f64(0.75), "contrast": f64(1.25), "gamma": f64(1.4)}, True),
                 ({"brightness": 0.75, "contrast": 1.25, "gamma": 1.4}, True),
                 ({"brightness": 0.85, "gamma": 1.06, "saturation": f64(0.48)}, self.exact_dots),
                 ({"brightness": 0.85, "gamma": 1.06, "saturation": 0.48}, self.exact_dots),
                 # float64 tone table: numpy's dgemv fuses the luma dot.
                 ({"brightness": f64(0.85), "gamma": 1.06, "saturation": 0.48}, False),
                 ({"brightness": 0.85, "gamma": 1.06, "saturation": 0.48}, self.exact_dots)]
        for kw, exact in cases:
            self.check("bcg", exact, **kw)


class TestGpuDetailLoss(GpuColorCase):
    def test_every_bit_depth(self):
        for s in DETAIL_SEVERITIES:
            self.check("detail_loss", True, severity=s)

    def test_zero_severity_returns_input(self):
        x = self.to_gpu(self.frame)
        self.assertIs(self.sim.detail_loss(x, 0.0), x)


class TestGpuVignette(GpuColorCase):
    def test_kwargs(self):
        for kw in VIGNETTE_KWARGS:
            self.check("vignette", True, **kw)

    def test_numpy_scalar_kwargs_do_not_share_a_cache_entry(self):
        # Equal values give different ramps: float32, or float64 from the
        # subtraction (NumPy radius) or only from the division (NumPy softness)
        # on.  Sharing one entry flipped 6 levels of the 2048x1024 scene.
        f64 = np.float64
        kwargs = ({"radius": f64(0.3), "softness": 0.37}, {"radius": 0.3, "softness": f64(0.37)},
                  {"radius": 0.3, "softness": 0.37}, {"radius": f64(0.3), "softness": f64(0.37)})
        frames = dict(self.frames)
        if SCENE.is_file():
            frames["scene_2048x1024"] = np.asarray(Image.open(SCENE).convert("RGB"))
        for kw in kwargs:
            self.check("vignette", True, frames, severity=0.4, mode="light", **kw)
        sim = self.vg.VipSimGPU(device=self.sim.device)
        for kw in kwargs:
            sim.vignette(self.to_gpu(self.frame), 0.4, **kw)
        self.assertEqual(len(sim.cache), len(kwargs))

    def test_zero_strength_returns_input_before_the_mode_check(self):
        x = self.to_gpu(self.frame)
        self.assertIs(self.sim.vignette(x, 1.0, strength=0.0, mode="bogus"), x)
        with self.assertRaises(ValueError):
            self.sim.vignette(x, 0.5, mode="bogus")


class TestGpuColorBatchAndChain(GpuColorCase):
    CASES = [("cvd", {"cvd_type": "protanomaly", "severity": 0.6}),
             ("cvd", {"cvd_type": "monochrome", "severity": 0.8}),
             ("bcg", {"brightness": 0.8, "contrast": 1.3, "gamma": 1.4, "saturation": 1.8}),
             ("bcg", {"brightness": 0.75, "contrast": 0.5, "gamma": 1.4, "color_space": "srgb"}),
             ("detail_loss", {"severity": 0.6}),
             ("vignette", {"severity": 0.7, "mode": "light"})]

    def test_batch_of_two_frames(self):
        self.assertIn("traffic_61x97", self.frames, "Trafficscene.png missing")
        for h, w in SIZES:
            pair = np.stack([self.frames[f"random_{h}x{w}"], self.frames[f"traffic_{h}x{w}"]])
            for name, kw in self.CASES:
                with self.subTest(size=(h, w), filter=name, **kw):
                    gpu = self.vg.gpu_to_frames(getattr(self.sim, name)(self.to_gpu(pair), **kw))
                    for n in range(2):
                        cpu = getattr(vipsim, "apply_" + name)(Image.fromarray(pair[n]), **kw)
                        self.compare(gpu[n], cpu, self.exact_dots)

    def test_chain_sees_the_cpu_images(self):
        frame = self.frames.get("traffic_84x84", self.frame)
        x, img = self.to_gpu(frame), Image.fromarray(frame)
        for name, kw in self.CASES:
            x = getattr(self.sim, name)(x, **kw)
            img = getattr(vipsim, "apply_" + name)(img, **kw)
            with self.subTest(after=name, **kw):
                self.compare(self.vg.gpu_to_frames(x)[0], img, self.exact_dots)


class TestGpuColorTables(GpuColorCase):
    def test_channel_dot_sums_left_to_right(self):
        import torch
        lin = np.random.default_rng(8).random((64, 7, 3), dtype=np.float32)
        for m in (np.random.default_rng(9).random((3, 3)), self.vg._BCG_LUMA):
            for sdot in (False, True):
                with self.subTest(dtype=m.dtype.name, sdot=sdot):
                    x = torch.from_numpy(lin.astype(m.dtype)).permute(2, 0, 1)[None]
                    twin = self.vg._channel_dot(x, torch.from_numpy(m), sdot)
                    np.testing.assert_array_equal(twin[0].permute(1, 2, 0).numpy(),
                                                  left_to_right_dot(lin, m, sdot))

    def test_srgb_tables_match_the_cpu_functions(self):
        import torch
        levels = np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1).repeat(3, -1)
        lin = self.sim._srgb_to_linear_u8(self.to_gpu(levels)).cpu().numpy()
        np.testing.assert_array_equal(
            lin[0].transpose(1, 2, 0), filters._srgb_to_linear(levels[0].astype(np.float32) / 255.0))
        rng = np.random.default_rng(7)
        for dtype, tdtype in ((np.float32, torch.float32), (np.float64, torch.float64)):
            with self.subTest(dtype=dtype.__name__):
                steps = self.vg._srgb_u8_steps(dtype)
                v = np.concatenate([
                    steps, np.nextafter(steps, dtype(-1)), np.nextafter(steps, dtype(2)),
                    filters._srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0).astype(dtype),
                    rng.uniform(-0.1, 1.1, 200_000).astype(dtype),
                    np.array([0.0, -0.0, 1.0, 1.5, -2.0, np.nan], dtype)])
                gpu = self.sim._linear_to_srgb_u8(torch.from_numpy(v).to(self.sim.device, tdtype))
                with np.errstate(invalid="ignore"):     # numpy's NaN -> uint8 cast
                    cpu = filters._to_uint8(filters._linear_to_srgb(v))
                np.testing.assert_array_equal(gpu.cpu().numpy(), cpu)


if __name__ == "__main__":
    unittest.main()
