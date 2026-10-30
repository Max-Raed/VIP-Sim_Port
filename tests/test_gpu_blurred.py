"""The GPU twins of apply_cataracts, apply_bloom and apply_field_loss must
reproduce the CPU filters.

Their colour maths runs as lookup tables of the CPU's own numpy code, and the
blurs and resizes replay Pillow's integer arithmetic, so the twins are
bit-exact (max difference 0 on CUDA with Pillow 12.2 and on the CPU with
Pillow 13.0).  The tests therefore assert equality, which is stricter than
GpuParityCase.assert_close.  Every case runs on a batch of two different
frames.  Runs on CUDA when available, else on the CPU; skipped without torch.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vipsim  # noqa: E402
from test_rounding import GpuParityCase  # noqa: E402

TRAFFIC = ROOT / "vipsim_assets" / "Trafficscene.png"
NAMES = ("cataracts", "bloom", "field_loss")
# Pillow 12.2 crashes (segfault) on GaussianBlur(nan or inf); Pillow 13 raises.
PIL13 = int(PIL.__version__.split(".")[0]) >= 13
# (H, W): the RL sizes, an odd size, and a portrait frame, whose /4 bloom
# downsample takes Pillow 13's vertical-first resize.
SIZES = [(64, 96), (84, 84), (61, 97), (96, 40)]
# (H, W) at least 100 times taller than wide: Pillow 12.2 resizes those
# vertically first when the height shrinks (1000x10 is on its boundary).
TALL_SIZES = [(1000, 10), (1001, 10), (202, 2), (2390, 19)]
SEVERITIES = (1e-3, 0.05, 0.3, 0.5, 0.99, 1.0)
KWARGS = {
    "cataracts": [dict(severity=0.6, use_frosting=False),
                  dict(severity=0.6, use_brightness=False,
                       contrast_coeff=(0.5, 0.3, 0.9), gamma=2.2),
                  dict(severity=0.4, use_contrast=False, gamma=0.5),
                  dict(severity=0.8, use_brightness=False, use_contrast=False,
                       use_frosting=False)],
    "bloom": [dict(severity=0.0, intensity=1.5),
              dict(severity=0.5, intensity=2.5, threshold=0.1, blur_size=5.5,
                   iterations=3),
              dict(severity=0.3, threshold=0.2, iterations=0),
              dict(severity=0.3, threshold=0.2, blur_size=-1.0, iterations=2),
              dict(severity=0.9, threshold=0.05, blur_size=0.25, iterations=2),
              # Per-channel numpy arrays broadcast like the CPU's HWC arrays.
              dict(severity=0.5, threshold=np.array([0.1, 0.3, 0.05]),
                   intensity=np.array([2.0, 1.0, 3.0], np.float32))],
    "field_loss": [dict(severity=0.8, field_type="peripheral"),
                   dict(severity=0.8, overlay_scale=1.6),
                   dict(severity=0.7, use_overlay_asset=False),
                   dict(severity=0.7, use_overlay_asset=False,
                        field_type="peripheral", overlay_scale=0.3),
                   dict(severity=1.0, mask_center=(0.3, 0.7), mask_radius=0.2),
                   dict(severity=0.9, use_overlay_asset=False, mask_radius=0.9)],
}
# Bloom (threshold, intensity) pairs where numpy's float64 promotion of a
# np.float64 operand moves output pixels: the first on a 237x425 traffic
# crop (iterations 0 and 1), the second on all frames of the test below.
NP64_PAIRS = [(0.2177126740324215, 2.2230322417726645),
              (0.18511852908422885, 1.3814432440461442)]


def natural_crop(h, w):
    """Native-resolution crop from the centre of the traffic scene."""
    img = Image.open(TRAFFIC).convert("RGB")
    left, top = img.width // 2 - w // 2, img.height // 2 - h // 2
    return np.asarray(img.crop((left, top, left + w, top + h)))


def ramp_batch():
    """Every 8-bit level on every channel (16 x 16), plus its mirror image."""
    ramp = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
    return np.stack([ramp, ramp[::-1, ::-1]])


class GpuBlurredCase(GpuParityCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch
        cls.torch = torch
        rng = np.random.default_rng(10)
        # One batch per size: a random frame and a natural crop (or a second
        # random frame if the asset is missing).
        cls.batches = {}
        for h, w in SIZES:
            other = (natural_crop(h, w) if TRAFFIC.is_file()
                     else rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
            cls.batches[(h, w)] = np.stack(
                [rng.integers(0, 256, (h, w, 3), dtype=np.uint8), other])

    def cpu(self, name, batch, **kw):
        fn = getattr(vipsim, f"apply_{name}")
        return np.stack([np.asarray(fn(Image.fromarray(f), **kw)) for f in batch])

    def assert_matches_cpu(self, name, batch, sim=None, **kw):
        sim = sim or self.sim
        x = self.vg.frames_to_gpu(batch, sim.device)
        out = self.vg.gpu_to_frames(getattr(sim, name)(x, **kw))
        np.testing.assert_array_equal(out, self.cpu(name, batch, **kw),
                                      err_msg=f"{name} {kw}")

    def check_filter(self, name):
        self.assertTrue(TRAFFIC.is_file(), "Trafficscene.png missing")
        for hw, batch in self.batches.items():
            for kw in [dict(severity=s) for s in SEVERITIES] + KWARGS[name]:
                with self.subTest(size=hw, **kw):
                    self.assert_matches_cpu(name, batch, **kw)


class TestGpuBlurredMatchesCpu(GpuBlurredCase):
    def test_cataracts(self):
        self.check_filter("cataracts")

    def test_bloom(self):
        self.check_filter("bloom")

    def test_field_loss(self):
        self.check_filter("field_loss")

    def test_bloom_numpy_scalar_kwargs(self):
        # NEP 50: apply_bloom's `clip(down - threshold, 0) * intensity` turns
        # float64 at the step whose operand is a np.float64, where torch would
        # read the np.float64 as a Python float and stay in float32.  A dark
        # left half puts the moved levels next to unsaturated output pixels.
        edge = np.zeros((2, 64, 96, 3), np.uint8)
        edge[0, :, 48:], edge[1, :, 48:] = 223, 161
        batches = [edge, natural_crop(237, 425)[None], self.batches[(84, 84)]]
        f64 = np.float64
        for t, i in NP64_PAIRS:
            for kinds in [(f64, f64), (f64, float), (float, f64)]:
                for it in (0, 1, 2):
                    kw = dict(severity=0.5, threshold=kinds[0](t),
                              intensity=kinds[1](i), iterations=it)
                    for batch in batches:
                        with self.subTest(size=batch.shape[1:3], **kw):
                            self.assert_matches_cpu("bloom", batch, **kw)
        # The edge frames do exercise each promotion: the CPU output moves.
        t, i = NP64_PAIRS[1]
        ref = self.cpu("bloom", edge, threshold=t, intensity=i)
        for kw in (dict(threshold=f64(t), intensity=i),
                   dict(threshold=t, intensity=f64(i))):
            self.assertFalse(np.array_equal(self.cpu("bloom", edge, **kw), ref), kw)
        # Other numpy scalars and 0-d arrays, as numpy promotes them.
        for kw in (dict(threshold=np.float32(0.3), intensity=np.int64(2)),
                   dict(threshold=np.array(0.2), intensity=np.float16(1.5),
                        blur_size=np.float32(2.7), iterations=np.int64(2))):
            with self.subTest(**kw):
                self.assert_matches_cpu("bloom", self.batches[(64, 96)],
                                        severity=0.5, **kw)

    def test_bloom_bool_kwargs(self):
        # numpy computes float32 - bool in float32; torch has no tensor - bool.
        batches = [self.batches[(84, 84)], natural_crop(237, 425)[None]]
        for kw in (dict(threshold=False), dict(threshold=True),
                   dict(threshold=np.bool_(False), intensity=True),
                   dict(threshold=np.array(False), intensity=np.bool_(True)),
                   dict(threshold=np.array([True, False, False]))):
            for batch in batches:
                with self.subTest(size=batch.shape[1:3], **kw):
                    self.assert_matches_cpu("bloom", batch, severity=0.5, **kw)

    def test_nan_kwargs_the_cpu_accepts(self):
        # A NaN severity only reaches the BCG table when frosting is off, and
        # the table is the CPU's own (NaN -> uint8, with a RuntimeWarning).
        with np.errstate(invalid="ignore"):
            for kw in (dict(severity=float("nan"), use_frosting=False),
                       dict(gamma=float("nan"))):
                with self.subTest(**kw):
                    self.assert_matches_cpu("cataracts", self.batches[(64, 96)], **kw)

    def test_numpy_scalar_kwargs_get_their_own_fields(self):
        # gamma=np.float64(g) runs the cataract BCG in float64 and moves one
        # level of this table; the cache must not hand back the float32 one.
        sim = self.vg.VipSimGPU(device=self.sim.device)
        s, g = 0.37239417054968194, 2.410801328336279
        batch = ramp_batch()
        for gamma in (g, np.float64(g)):
            with self.subTest(gamma=type(gamma).__name__):
                self.assert_matches_cpu("cataracts", batch, sim, severity=s,
                                        gamma=gamma, use_frosting=False)
        self.assertFalse(np.array_equal(
            self.cpu("cataracts", batch, severity=s, gamma=g, use_frosting=False),
            self.cpu("cataracts", batch, severity=s, gamma=np.float64(g),
                     use_frosting=False)))
        for kw in (dict(mask_radius=0.33), dict(mask_radius=np.float64(0.33)),
                   dict(mask_center=(np.float64(0.4), 0.55))):
            with self.subTest(**kw):
                self.assert_matches_cpu("field_loss", self.batches[(64, 96)], sim,
                                        severity=0.9, use_overlay_asset=False, **kw)

    def test_bloom_python_int_kwargs(self):
        # numpy converts a Python int through a C double, beyond int64 too
        # (float32 - 2**70 is float32), where torch cannot hold the int.
        batch = self.batches[(84, 84)]
        with np.errstate(over="ignore"):            # 10**40 is inf in float32
            for kw in (dict(threshold=0, intensity=2), dict(threshold=2**70),
                       dict(threshold=0.0, intensity=-(10**20)),
                       dict(threshold=0.0, intensity=2**60 + 2**36 + 1),
                       dict(threshold=10**40), dict(threshold=-(10**40)),
                       dict(threshold=np.float64(0.1), intensity=2**70)):
                with self.subTest(**kw):
                    self.assert_matches_cpu("bloom", batch, severity=0.5, **kw)

    def test_tall_frames(self):
        # Bloom's /4 downsample goes vertical first on all four on Pillow 13,
        # and on all but 1000x10 on Pillow 12.2.
        rng = np.random.default_rng(12)
        bloom_kwargs = [dict(severity=0.5),
                        dict(severity=0.5, threshold=0.1, intensity=2.0,
                             iterations=0),
                        dict(severity=0.9, threshold=0.0, intensity=1.0,
                             blur_size=-1.0)]
        for h, w in TALL_SIZES:
            batch = rng.integers(0, 256, (2, h, w, 3), dtype=np.uint8)
            for name in NAMES:
                for kw in bloom_kwargs if name == "bloom" else [dict(severity=0.7)]:
                    with self.subTest(size=(h, w), filter=name, **kw):
                        self.assert_matches_cpu(name, batch, **kw)
        # The pass order shows on these frames: horizontal first misses the
        # CPU wherever Pillow goes vertical first.
        if self.vg._pil_vertical_first(10, 1001, 2, 250):
            batch = rng.integers(0, 256, (2, 1001, 10, 3), dtype=np.uint8)
            x = self.vg.frames_to_gpu(batch, self.sim.device)
            with mock.patch.object(self.vg, "_pil_vertical_first", return_value=False):
                out = self.vg.gpu_to_frames(self.sim.bloom(x, **bloom_kwargs[1]))
            self.assertFalse(np.array_equal(
                out, self.cpu("bloom", batch, **bloom_kwargs[1])))

    def test_tiny_frames(self):
        # Bloom's /4 downsample clamps to 1 px; one pyramid level of field loss.
        rng = np.random.default_rng(11)
        for h, w in [(1, 1), (3, 5), (7, 2)]:
            batch = rng.integers(0, 256, (2, h, w, 3), dtype=np.uint8)
            for name in NAMES:
                with self.subTest(size=(h, w), filter=name):
                    self.assert_matches_cpu(name, batch, severity=0.7)

    def test_field_loss_without_overlay_asset_falls_back_to_radial(self):
        sim = self.vg.VipSimGPU(device=self.sim.device)     # empty cache
        missing = ROOT / "vipsim_assets" / "no-such-overlay.jpg"
        with mock.patch.object(vipsim.filters, "_MACULAR_OVERLAY_PATH", missing):
            self.assert_matches_cpu("field_loss", self.batches[(64, 96)], sim,
                                    severity=0.8)


class TestGpuBlurredBehaviour(GpuBlurredCase):
    def test_zero_severity_is_identity(self):
        x = self.vg.frames_to_gpu(self.batches[(64, 96)], self.sim.device)
        for name in NAMES:
            for s in (0.0, -1.0):
                with self.subTest(filter=name, severity=s):
                    self.assertIs(getattr(self.sim, name)(x, severity=s), x)

    def test_input_is_read_as_8_bit_levels(self):
        # The CPU filters take a uint8 image, so off-grid floats (e.g. from
        # an unquantised GPU filter earlier in a chain) and float64, float16
        # or bfloat16 input give what the CPU gives for the rounded image;
        # only the output takes x's dtype.  The 4x ramp hands bloom's /4
        # downsample every linear level, where a half-precision `down` would
        # move pixels across the threshold's rounding boundaries.
        ramp4 = ramp_batch().repeat(4, axis=1).repeat(4, axis=2)
        for batch in (self.batches[(84, 84)], ramp4):
            x = self.vg.frames_to_gpu(batch, self.sim.device)
            off_grid = x + (self.torch.rand_like(x) - 0.5) * (0.9 / 255)
            for name in NAMES:
                ref = self.cpu(name, batch, severity=0.6)
                for inp in (off_grid, x.double(), x.half(), x.bfloat16()):
                    with self.subTest(size=batch.shape[1:3], filter=name,
                                      dtype=str(inp.dtype)):
                        out = getattr(self.sim, name)(inp, severity=0.6)
                        self.assertEqual(out.dtype, inp.dtype)
                        np.testing.assert_array_equal(
                            self.vg.gpu_to_frames(out.float()), ref)

    def test_invalid_kwargs_fail_like_cpu(self):
        # NaN truncates to -2**31 in the CPU's astype(np.int32), and its fancy
        # indexing raises IndexError.  A NaN or inf blur radius reaches PIL,
        # which raises ValueError on Pillow 13 (on 12.2 it crashes, so the CPU
        # is not run there).  Bloom operands follow numpy's broadcasting, and
        # a 4-D result fails in Image.fromarray.
        nan, inf = float("nan"), float("inf")
        cases = [("cataracts", dict(severity=nan), IndexError, True),
                 ("field_loss", dict(severity=nan), IndexError, True),
                 ("field_loss", dict(use_overlay_asset=False, mask_radius=nan),
                  IndexError, True),
                 ("field_loss", dict(mask_center=(nan, 0.5)), IndexError, True),
                 ("bloom", dict(severity=nan), ValueError, PIL13),
                 ("bloom", dict(blur_size=nan, iterations=2), ValueError, PIL13),
                 ("bloom", dict(blur_size=inf), ValueError, PIL13),
                 ("bloom", dict(threshold=np.array([0.1, 0.2])), ValueError, True),
                 ("bloom", dict(intensity=np.ones((2, 1, 1))), ValueError, True),
                 ("bloom", dict(threshold=np.zeros((1, 1, 1, 3))), TypeError, True),
                 # Beyond a C double: 'int too large to convert to float'.
                 ("bloom", dict(threshold=10**400), OverflowError, True)]
        batch = self.batches[(64, 96)]
        x = self.vg.frames_to_gpu(batch, self.sim.device)
        with np.errstate(invalid="ignore"):
            for name, kw, error, run_cpu in cases:
                with self.subTest(filter=name, **kw):
                    with self.assertRaises(error):
                        getattr(self.sim, name)(x, **kw)
                    if run_cpu:
                        with self.assertRaises(error):
                            self.cpu(name, batch, **kw)
        # A radius <= 0, -inf included, skips the blur on both.
        self.assert_matches_cpu("bloom", batch, blur_size=-inf, iterations=2)

    def test_fields_are_cached(self):
        sim = self.vg.VipSimGPU(device=self.sim.device)
        x = self.vg.frames_to_gpu(self.batches[(64, 96)], sim.device)
        for name in NAMES:
            getattr(sim, name)(x, severity=0.6)
        n = len(sim.cache)
        for name in NAMES:
            getattr(sim, name)(x, severity=0.6)
        self.assertEqual(len(sim.cache), n)

    def test_unknown_field_type_is_rejected_like_cpu(self):
        x = self.vg.frames_to_gpu(self.batches[(64, 96)], self.sim.device)
        for use_overlay_asset in (True, False):
            with self.subTest(use_overlay_asset=use_overlay_asset):
                with self.assertRaises(ValueError):
                    self.sim.field_loss(x, field_type="diagonal",
                                        use_overlay_asset=use_overlay_asset)


if __name__ == "__main__":
    unittest.main()
