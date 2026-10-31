"""The shared GPU primitives must reproduce the CPU operations they replace.

The NumPy/PIL filters are the reference, so the blur and resize twins are
checked against PIL itself and the gathers against numpy fancy indexing.
Runs on CUDA when available, else on the CPU; skipped without torch.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vipsim  # noqa: E402
from test_rounding import GpuParityCase  # noqa: E402

TRAFFIC = ROOT / "vipsim_assets" / "Trafficscene.png"
BLUR_SEVERITIES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.2, 0.4, 1.0)
# Severities where Pillow's float32 box radius or weights (r, ww, fw) round
# to other pixels than float64 ones would (0.07: the radius; the rest: ww).
BOX_WEIGHT_SEVERITIES = (0.01, 0.02, 0.025, 0.04, 0.07, 0.225, 0.433, 0.517)
# (width, height) -> (width, height), PIL order: bloom's /4 and back,
# pixelation-like odd ratios, one axis only, and degenerate sizes.  Then the
# pass order (see _pil_vertical_first): three pairs that shrink the height by
# more than twice as many pixels as the width, where Pillow 13 resamples
# vertically first (bloom's /4 of a portrait frame), and four frames at least
# 100 times taller than wide, where Pillow 12.2 goes vertical first when the
# height shrinks: 10x1000 is on its boundary (horizontal first), and the
# last two pairs go vertical first on 12.2 only.
RESIZES = [((96, 64), (24, 16)), ((24, 16), (96, 64)), ((84, 84), (21, 21)),
           ((21, 21), (84, 84)), ((97, 61), (20, 20)), ((20, 20), (97, 61)),
           ((96, 64), (96, 16)), ((96, 64), (31, 64)), ((5, 7), (3, 11)),
           ((96, 64), (990, 990)), ((990, 990), (96, 64)), ((1, 1), (5, 4)),
           ((7, 3), (1, 1)), ((40, 96), (10, 24)), ((5, 30), (3, 22)),
           ((2, 7), (1, 1)), ((10, 1000), (2, 250)), ((10, 1001), (2, 250)),
           ((10, 1001), (9, 1000)), ((3, 400), (2, 399))]


def natural_crop(h, w):
    """Native-resolution crop from the centre of the traffic scene."""
    img = Image.open(TRAFFIC).convert("RGB")
    left, top = img.width // 2 - w // 2, img.height // 2 - h // 2
    return np.asarray(img.crop((left, top, left + w, top + h)))


class GpuPrimitiveCase(GpuParityCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch
        cls.torch = torch
        rng = np.random.default_rng(1)
        cls.frames = {"random_64x96": cls.frame,
                      "random_84x84": rng.integers(0, 256, (84, 84, 3), dtype=np.uint8)}
        if TRAFFIC.is_file():
            cls.frames["traffic_64x96"] = natural_crop(64, 96)
            cls.frames["traffic_84x84"] = natural_crop(84, 84)

    def to_gpu(self, frame):
        return self.vg.frames_to_gpu(frame, self.sim.device)

    def to_frame(self, x):
        return self.vg.gpu_to_frames(x)[0]


class TestGpuBlurMatchesPil(GpuPrimitiveCase):
    def test_blur_is_bit_exact_with_apply_blur(self):
        self.assertIn("traffic_84x84", self.frames, "Trafficscene.png missing")
        for name, frame in self.frames.items():
            for s in BLUR_SEVERITIES:
                with self.subTest(frame=name, severity=s):
                    cpu = np.asarray(vipsim.apply_blur(Image.fromarray(frame), s))
                    gpu = self.to_frame(self.sim.blur(self.to_gpu(frame), s))
                    np.testing.assert_array_equal(gpu, cpu)

    def test_blur_edge_cases_are_bit_exact(self):
        # Box radius beyond the image, single rows/columns, single channel.
        rng = np.random.default_rng(2)
        for shape, radius in [((5, 3, 3), 40.0), ((1, 7, 3), 2.0), ((7, 1, 3), 2.0),
                              ((33, 47), 2.3), ((130, 33), 0.001)]:
            with self.subTest(shape=shape, radius=radius):
                a = rng.integers(0, 256, shape, dtype=np.uint8)
                cpu = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius)))
                x = self.torch.from_numpy(a.reshape(a.shape[:2] + (-1,))).permute(2, 0, 1)
                x = self.vg.exact_div(x.to(self.sim.device).float(), 255.0)[None]
                gpu = self.vg.gpu_to_frames(self.vg.pil_gaussian_blur(x, radius))[0]
                np.testing.assert_array_equal(gpu.reshape(a.shape), cpu)

    def test_window_sums_beyond_int32_are_bit_exact(self):
        # Pillow sums a window of up to 2r + 1 levels in a UINT32: 255s
        # overflow an int32 from a box radius of ~4.2M px (bloom's blur_size
        # around 5.6e6).  Measured: 0 where PIL gives 255 before the fix.
        # Both window-sum paths: int64 (small tensors) and int32 (large).
        for numel in (self.vg._BOX_BLUR_INT64_NUMEL, 0):
            for shape, radius in [((1, 1, 3), 4.5e6), ((1, 3, 3), 6e6), ((1, 2, 3), 8e6)]:
                with self.subTest(int64_numel=numel, shape=shape, radius=radius),                         mock.patch.object(self.vg, "_BOX_BLUR_INT64_NUMEL", numel):
                    a = np.full(shape, 255, np.uint8)
                    cpu = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius)))
                    x = self.vg.frames_to_gpu(a, self.sim.device)
                    gpu = self.vg.gpu_to_frames(self.vg.pil_gaussian_blur(x, radius))[0]
                    np.testing.assert_array_equal(gpu, cpu)

    def test_int32_window_sums_are_bit_exact(self):
        # Tensors above _BOX_BLUR_INT64_NUMEL sum in int32; force that path.
        with mock.patch.object(self.vg, "_BOX_BLUR_INT64_NUMEL", 0):
            for name, frame in self.frames.items():
                for s in BLUR_SEVERITIES:
                    with self.subTest(frame=name, severity=s):
                        cpu = np.asarray(vipsim.apply_blur(Image.fromarray(frame), s))
                        gpu = self.to_frame(self.sim.blur(self.to_gpu(frame), s))
                        np.testing.assert_array_equal(gpu, cpu)

    def test_radii_beyond_the_frame_are_counted_not_padded(self):
        # r >= n: every window is the whole row plus edge copies.  A padded
        # index would need O(r) memory (35 GiB for bloom's blur_size 1e8).
        rng = np.random.default_rng(14)
        for shape, radius in [((5, 3, 3), 1e8), ((1, 7, 3), 1e9), ((7, 1, 3), 2147483583.0),
                              ((33, 47, 3), 3e7)]:
            with self.subTest(shape=shape, radius=radius):
                a = rng.integers(0, 256, shape, dtype=np.uint8)
                cpu = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius)))
                gpu = self.vg.gpu_to_frames(self.vg.pil_gaussian_blur(
                    self.vg.frames_to_gpu(a, self.sim.device), radius))[0]
                np.testing.assert_array_equal(gpu, cpu)
        frame = self.frames["random_84x84"]
        for blur_size in (3e7, 1e9):
            with self.subTest(bloom_blur_size=blur_size):
                cpu = np.asarray(vipsim.apply_bloom(Image.fromarray(frame), 0.5, blur_size=blur_size))
                gpu = self.to_frame(self.sim.bloom(self.to_gpu(frame), 0.5, blur_size=blur_size))
                np.testing.assert_array_equal(gpu, cpu)

    def test_box_weights_are_computed_in_float32(self):
        rng = np.random.default_rng(13)
        for s in BOX_WEIGHT_SEVERITIES:
            for seed in range(8):
                frame = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
                with self.subTest(severity=s, frame=seed):
                    cpu = np.asarray(vipsim.apply_blur(Image.fromarray(frame), s))
                    gpu = self.to_frame(self.sim.blur(self.to_gpu(frame), s))
                    np.testing.assert_array_equal(gpu, cpu)

    def test_invalid_radii_raise_like_pillow_13(self):
        # Pillow 13 raises these; Pillow 12 segfaults on the same radii.
        x = self.to_gpu(self.frame)
        for radius, message in [(np.nan, "radius must be >= 0"), (np.inf, "radius must be >= 0"),
                                (-np.inf, "radius must be >= 0"),
                                (2147483584.0, "radius too large"), (-3e9, "radius too large")]:
            for quantize in (True, False):
                with self.subTest(radius=radius, quantize=quantize),                         self.assertRaisesRegex(ValueError, message):
                    self.vg.pil_gaussian_blur(x, radius, quantize=quantize)
        with self.assertRaisesRegex(ValueError, "radius must be >= 0"):
            self.sim.blur(x, np.nan)

    def test_batch_is_blurred_per_frame(self):
        a, b = self.frames["random_84x84"], self.frames["traffic_84x84"]
        x = self.vg.frames_to_gpu(np.stack([a, b]), self.sim.device)
        out = self.vg.gpu_to_frames(self.sim.blur(x, 0.2))
        for i, frame in enumerate((a, b)):
            np.testing.assert_array_equal(
                out[i], np.asarray(vipsim.apply_blur(Image.fromarray(frame), 0.2)))

    def test_float_variant_is_within_pil_rounding(self):
        # No per-pass 8-bit rounding: within 2 LSB from 1.2 px (mostly 1 LSB
        # above ~2.25 px, 2 at e.g. 2.55 px), up to 3 LSB on noise at
        # sub-pixel radii.  Radii every 0.05 px from 1.5 to 4 px.
        dense = tuple(r / 15 for r in np.arange(1.5, 4.0, 0.05))
        for name, frame in self.frames.items():
            for s in BLUR_SEVERITIES + dense:
                with self.subTest(frame=name, severity=s):
                    cpu = np.asarray(vipsim.apply_blur(Image.fromarray(frame), s), int)
                    gpu = self.to_frame(self.vg.pil_gaussian_blur(
                        self.to_gpu(frame), s * 15.0, quantize=False)).astype(int)
                    self.assertLessEqual(np.abs(gpu - cpu).max(), 2 if s * 15.0 >= 1.2 else 3)

    def test_zero_severity_is_identity(self):
        x = self.to_gpu(self.frame)
        for s in (0.0, -1.0):
            self.assertIs(self.sim.blur(x, s), x)

    def test_box_radius_matches_pillow_formula(self):
        # 3 passes (Gwosdek et al. 2011): sigma=1 -> box radius 1/4 exactly in
        # real arithmetic; sigma=15 -> 14 + 29/60.  Tiny sigmas stay > 0, so
        # even severity 0.005 (radius 0.075 px) changes the image.
        self.assertAlmostEqual(self.vg.pil_box_radius(1.0), 0.25, places=6)
        self.assertAlmostEqual(self.vg.pil_box_radius(15.0), 14 + 29 / 60, places=5)
        self.assertGreater(self.vg.pil_box_radius(0.075), 0.0)


class TestGpuResizeMatchesPil(GpuPrimitiveCase):
    def test_resize_is_bit_exact(self):
        rng = np.random.default_rng(3)
        for (w, h), (w2, h2) in RESIZES:
            a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            for mode, pil_mode in (("bilinear", Image.BILINEAR), ("nearest", Image.NEAREST)):
                with self.subTest(src=(w, h), dst=(w2, h2), mode=mode):
                    cpu = np.asarray(Image.fromarray(a).resize((w2, h2), pil_mode))
                    gpu = self.to_frame(self.sim.pil_resize(self.to_gpu(a), w2, h2, mode))
                    np.testing.assert_array_equal(gpu, cpu)

    def test_tap_by_tap_resample_is_bit_exact(self):
        # Passes gathering more than _RESAMPLE_GATHER_NUMEL taps add one tap
        # at a time in int32; force that path on every size pair.
        rng = np.random.default_rng(3)
        with mock.patch.object(self.vg, "_RESAMPLE_GATHER_NUMEL", 0):
            for (w, h), (w2, h2) in RESIZES:
                a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                with self.subTest(src=(w, h), dst=(w2, h2)):
                    cpu = np.asarray(Image.fromarray(a).resize((w2, h2), Image.BILINEAR))
                    gpu = self.to_frame(self.sim.pil_resize(self.to_gpu(a), w2, h2))
                    np.testing.assert_array_equal(gpu, cpu)

    def test_bloom_downsample_of_natural_crop_is_bit_exact(self):
        frame = self.frames["traffic_64x96"]
        cpu = np.asarray(Image.fromarray(frame).resize((24, 16), Image.BILINEAR))
        np.testing.assert_array_equal(
            self.to_frame(self.sim.pil_resize(self.to_gpu(frame), 24, 16)), cpu)

    def test_float_bilinear_residual(self):
        # F.interpolate(antialias=True): Pillow's filter without the
        # intermediate 8-bit rounding (measured max 1, mean <= 0.20 LSB).
        rng = np.random.default_rng(3)
        for (w, h), (w2, h2) in RESIZES:
            a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            with self.subTest(src=(w, h), dst=(w2, h2)):
                cpu = np.asarray(Image.fromarray(a).resize((w2, h2), Image.BILINEAR), int)
                gpu = self.to_frame(self.sim.pil_resize(self.to_gpu(a), w2, h2,
                                                        quantize=False)).astype(int)
                d = np.abs(gpu - cpu)
                self.assertLessEqual(d.max(), 1)
                if w2 * h2 > 1:                # one pixel has no meaningful mean
                    self.assertLess(d.mean(), 0.25)

    def test_same_size_is_a_copy(self):
        x = self.to_gpu(self.frame)
        h, w = self.frame.shape[:2]
        np.testing.assert_array_equal(self.to_frame(self.sim.pil_resize(x, w, h)), self.frame)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.sim.pil_resize(self.to_gpu(self.frame), 10, 10, "bicubic")


class TestGpuGathersMatchNumpy(GpuPrimitiveCase):
    def setUp(self):
        self.x = self.to_gpu(self.frame)
        self.arr = self.x[0].permute(1, 2, 0).cpu().numpy()   # same floats, HWC
        self.rng = np.random.default_rng(4)

    def as_hwc(self, out):
        return out[0].permute(1, 2, 0).cpu().numpy()

    def test_nearest_shared_map(self):
        h, w = self.arr.shape[:2]
        sy = self.rng.integers(0, h, (40, 50)).astype(np.int32)
        sx = self.rng.integers(0, w, (40, 50)).astype(np.int32)
        t = self.torch.from_numpy
        out = self.vg.gather_nearest(self.x, t(sy).to(self.sim.device), t(sx).to(self.sim.device))
        np.testing.assert_array_equal(self.as_hwc(out), self.arr[sy, sx])

    def test_nearest_broadcast_rows_and_columns(self):
        # apply_led style: arr[sy[:, None], sx[None, :]]
        h, w = self.arr.shape[:2]
        sy = self.sim.to_device(self.rng.integers(0, h, h))
        sx = self.sim.to_device(self.rng.integers(0, w, w))
        out = self.vg.gather_nearest(self.x, sy[:, None], sx[None, :])
        expect = self.arr[sy.cpu().numpy()[:, None], sx.cpu().numpy()[None, :]]
        np.testing.assert_array_equal(self.as_hwc(out), expect)

    def test_nearest_per_sample_maps(self):
        h, w = self.arr.shape[:2]
        x = self.torch.cat([self.x, self.x.flip(-1)])
        sy = self.rng.integers(0, h, (2, 20, 30))
        sx = self.rng.integers(0, w, (2, 20, 30))
        out = self.vg.gather_nearest(x, self.sim.to_device(sy), self.sim.to_device(sx))
        for i in range(2):
            arr = x[i].permute(1, 2, 0).cpu().numpy()
            np.testing.assert_array_equal(out[i].permute(1, 2, 0).cpu().numpy(),
                                          arr[sy[i], sx[i]])

    def test_nearest_clamps_to_edge(self):
        h, w = self.arr.shape[:2]
        sy = np.array([[-5, 0, h - 1, h + 3]])
        sx = np.array([[-1, w, 2, w + 9]])
        out = self.vg.gather_nearest(self.x, self.sim.to_device(sy), self.sim.to_device(sx))
        np.testing.assert_array_equal(self.as_hwc(out),
                                      self.arr[np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)])

    def numpy_bilinear(self, fy, fx, weight_dtype=None):
        """The sampling block of apply_vortex.  weight_dtype=None keeps its
        promotion (float32 - int32 -> float64 weights)."""
        arr = self.arr
        h, w = arr.shape[:2]
        x0 = np.floor(fx).astype(np.int32)
        y0 = np.floor(fy).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        wx = (fx - (x0 if weight_dtype is None else x0.astype(weight_dtype)))[..., None]
        wy = (fy - (y0 if weight_dtype is None else y0.astype(weight_dtype)))[..., None]
        top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
        bot = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
        return top * (1.0 - wy) + bot * wy

    def test_bilinear_matches_vortex_sampling_exactly(self):
        h, w = self.arr.shape[:2]
        # Same expression as apply_vortex: clipped uv * (size - 1), float32.
        uy = np.clip(self.rng.random((40, 50)) * 1.2 - 0.1, 0.0, 1.0).astype(np.float32)
        ux = np.clip(self.rng.random((40, 50)) * 1.2 - 0.1, 0.0, 1.0).astype(np.float32)
        fy, fx = uy * (h - 1), ux * (w - 1)
        for dtype, weight_dtype in ((self.torch.float64, None),
                                    (self.torch.float32, np.float32)):
            with self.subTest(coords=dtype):
                expect = self.numpy_bilinear(fy, fx, weight_dtype)
                out = self.vg.gather_bilinear(self.x, self.sim.to_device(fy, dtype),
                                              self.sim.to_device(fx, dtype))
                self.assertEqual(out.dtype, dtype)
                np.testing.assert_array_equal(self.as_hwc(out), expect)

    def test_bilinear_on_integer_coords_is_nearest(self):
        h, w = self.arr.shape[:2]
        sy = self.rng.integers(0, h, (10, 12))
        sx = self.rng.integers(0, w, (10, 12))
        out = self.vg.gather_bilinear(self.x, self.sim.to_device(sy, self.torch.float32),
                                      self.sim.to_device(sx, self.torch.float32))
        np.testing.assert_array_equal(self.as_hwc(out), self.arr[sy, sx])


class TestGpuCache(GpuPrimitiveCase):
    def test_lru_evicts_oldest_and_keeps_recent(self):
        cache = self.vg.TensorCache(maxsize=2)
        calls = []

        def build(v):
            return lambda: calls.append(v) or v
        cache.get("a", build(1))
        cache.get("b", build(2))
        cache.get("a", build(99))           # hit: refreshes "a"
        cache.get("c", build(3))            # evicts "b", the least recent
        self.assertEqual((len(cache), "a" in cache, "b" in cache), (2, True, False))
        self.assertEqual(calls, [1, 2, 3])

    def test_key_uses_exact_params_hw_and_dtype(self):
        sim = self.vg.VipSimGPU(device=self.sim.device, cache_size=8)
        n = []

        def field():
            n.append(1)
            return np.arange(6, dtype=np.float64).reshape(2, 3)
        a = sim.numpy_field("f", {"severity": 0.5}, (2, 3), field)
        self.assertIs(sim.numpy_field("f", {"severity": 0.5}, (2, 3), field), a)
        sim.numpy_field("f", {"severity": 0.5004}, (2, 3), field)   # not rounded
        sim.numpy_field("f", {"severity": 0.5}, (3, 2), field)
        b = sim.numpy_field("f", {"severity": 0.5}, (2, 3), field, self.torch.float64)
        self.assertEqual(len(n), 4)
        self.assertEqual((a.dtype, a.device.type), (self.torch.float32, self.sim.device.type))
        self.assertEqual(b.dtype, self.torch.float64)
        np.testing.assert_array_equal(b.cpu().numpy(), field())

    def test_key_tells_equal_values_of_other_types_apart(self):
        # NEP 50 makes np.float64(0.5) and 0.5 build different CPU fields,
        # and numpy rejects some equal values (a float seed, a Decimal):
        # each type gets its own entry, and what _freeze cannot key exactly
        # is built on every call, never cached.
        from decimal import Decimal
        sim = self.vg.VipSimGPU(device=self.sim.device, cache_size=16)
        n = []

        def field():
            n.append(1)
            return np.zeros(1)
        params = (0.5, np.float64(0.5), np.float32(0.5), np.array(0.5), 1, 1.0,
                  True, -0.0, 0.0, Decimal("0.5"), Decimal("0.5"))
        for p in params:
            sim.numpy_field("f", {"severity": p}, (1, 1), field)
        self.assertEqual((len(n), len(sim.cache)), (len(params), len(params) - 2))
        for p in params[:-2]:                   # all hits now
            sim.numpy_field("f", {"severity": p}, (1, 1), field)
        self.assertEqual(len(n), len(params))

    def test_numpy_field_tuple_and_dtypes(self):
        sy, sx, m = self.sim.numpy_field(
            "maps", (1,), (2, 2), lambda: (np.zeros((2, 2), np.int32),
                                           np.ones((2, 2), np.int32),
                                           np.eye(2, dtype=bool)))
        self.assertEqual((sy.dtype, sx.dtype, m.dtype),
                         (self.torch.long, self.torch.long, self.torch.bool))

    def test_cache_is_bounded(self):
        sim = self.vg.VipSimGPU(device=self.sim.device, cache_size=3)
        for s in np.linspace(0.1, 0.9, 10):
            sim.vortex(self.to_gpu(self.frame), float(s))
        self.assertEqual(len(sim.cache), 3)


class TestGpuHelpers(GpuPrimitiveCase):
    def test_clamp_severity_matches_cpu_clip(self):
        for s in (-1, 0, 0.25, np.float32(0.3), 1, 7.5):
            out = self.vg.clamp_severity(s)
            self.assertIsInstance(out, float)
            self.assertEqual(out, float(np.clip(s, 0.0, 1.0)))

    def test_quantize_matches_to_uint8(self):
        vals = np.array([0.0, 0.5 / 255, 1.5 / 255, 0.2, 0.99, 1.2, -0.1], np.float32)
        x = self.sim.to_device(vals)
        expect = vipsim.filters._to_uint8(vals)
        np.testing.assert_array_equal((self.vg.quantize_u8(x) * 255).round().cpu().numpy(), expect)
        np.testing.assert_array_equal(
            (self.vg.quantize_u8(x * 255, scale=1.0) * 255).round().cpu().numpy(), expect)

    def test_frames_to_gpu_matches_numpy_division_without_warning(self):
        levels = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, -1)
        readonly = np.asarray(Image.fromarray(levels))
        self.assertFalse(readonly.flags.writeable)
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # from_numpy's read-only UserWarning
            x = self.vg.frames_to_gpu(readonly, self.sim.device)
        np.testing.assert_array_equal(x[0].permute(1, 2, 0).cpu().numpy(),
                                      levels.astype(np.float32) / 255.0)

    def test_cvd_tables_are_the_cpu_tables(self):
        self.assertEqual(set(self.sim.cvd_tables), set(vipsim.CVD_MATRICES))
        for name, table in self.sim.cvd_tables.items():
            np.testing.assert_array_equal(table.cpu().numpy(),
                                          vipsim.CVD_MATRICES[name].astype(np.float32))
        self.assertIs(self.sim.cvd_deuter, self.sim.cvd_tables["deuteranomaly"])


if __name__ == "__main__":
    unittest.main()
