"""The GPU pixelation and LED twins must reproduce apply_pixelation and apply_led.

The NumPy/PIL filters are the reference.  Both twins are bit-exact (max
difference 0), stricter than GpuParityCase.assert_close (max 1 LSB, mean
< 0.02): pixelation is Pillow's fixed-point resampler, run in the installed
Pillow's pass order, plus index gathers; LED replaces the device's pow with
numpy-built tables.  Runs on CUDA when available, else on the CPU; skipped
without torch.

Run from the repo root:  python -m unittest discover -s tests -v
"""
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
from test_gpu_primitives import TRAFFIC, GpuPrimitiveCase, natural_crop  # noqa: E402

# Near 0 (pixelation upscales to ~1000 cells), mid, and the range where a
# non-square frame takes Pillow 13's vertical-first pass, up to 1.
SEVERITIES = (0.001, 0.5, 0.75, 0.9, 0.95, 0.999, 1.0)
PIXELATION_MODES = [dict(), dict(hard=True), dict(sampling="point"),
                    dict(hard=True, sampling="point")]
# Fractions of a cell; negative and > 1 wrap with % 1.0 as on the CPU.
GRID_OFFSETS = [(0.5, 0.0), (0.0, 0.37), (0.3, 0.8), (-0.25, 1.6)]
# Overrides with severity 0: below the minimum of 2, Python's round() of
# x.4 and x.5, more cells than pixels.
PIXEL_SIZES = (1.0, 2.4, 7, 50.5, 85, 400)
# Default float kwargs, then numpy scalars, which NumPy 2 promotes
# differently (a np.float64 makes apply_led compute in float64; 1.7 and
# np.float64(1.7) give different CPU outputs, so they must not share a
# cache entry), then non-finite values, which the twin follows through
# numpy's NaN/inf arithmetic.
LED_KWARGS = [dict(scale=80), dict(scale=33.3), dict(scale=7), dict(scale=500),
              dict(brightness=1.7), dict(brightness=0.4), dict(shape=0.7),
              dict(shape=3.0), dict(margin=0.2), dict(automatic_ratio=False),
              dict(scale=7, brightness=1.3, shape=2.2, margin=0.2, automatic_ratio=False),
              dict(brightness=np.float64(1.7)), dict(shape=np.float64(2.2)),
              dict(scale=np.float64(33.3)), dict(margin=np.float64(0.1)),
              dict(brightness=np.float32(1.3)),
              dict(severity=float("nan")), dict(shape=np.inf), dict(shape=1e20),
              dict(brightness=np.inf), dict(brightness=float("nan")),
              # Per channel, as apply_led multiplies its (h, w, 3) cells:
              # a list is float64, a float32 array stays float32.
              dict(brightness=[0.5, 1.0, 1.0]), dict(brightness=np.array([1.0, 1.0, 1.0])),
              dict(brightness=np.array([0.3, 1.7, 0.9], np.float32)),
              dict(brightness=np.array([[1.3]])), dict(brightness=np.array([[[0.6, 1.2, 2.0]]])),
              dict(brightness=[float("nan"), np.inf, 0.5])]
# (width, height) -> (width, height) where the pass order shows: Pillow 13
# goes vertical first for the first two (height shrinks by more than twice
# the width change), Pillow 12.2 for the last two (over 100x taller than
# wide), Pillow 12.0/12.1 never.
ORDERED_RESIZES = [((85, 85), (96, 64)), ((61, 97), (69, 69)),
                   ((2, 281), (284, 73)), ((3, 450), (40, 265))]
# Frames over 100x taller than wide: Pillow 12.2 resizes them height first.
TALL_FRAMES = ((300, 2), (450, 3))


class GpuResampleCase(GpuPrimitiveCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        rng = np.random.default_rng(5)
        cls.frames["random_97x61"] = rng.integers(0, 256, (97, 61, 3), dtype=np.uint8)
        if TRAFFIC.is_file():
            cls.frames["traffic_97x61"] = natural_crop(97, 61)

    def assert_twin(self, name, frame, **kw):
        """GPU output equals the CPU filter's, pixel for pixel."""
        cpu_fn = getattr(vipsim, f"apply_{name}")
        cpu = np.asarray(cpu_fn(Image.fromarray(frame), **kw))
        gpu = self.to_frame(getattr(self.sim, name)(self.to_gpu(frame), **kw))
        np.testing.assert_array_equal(gpu, cpu)


class TestGpuPixelationMatchesCpu(GpuResampleCase):
    def test_severities_and_modes(self):
        self.assertIn("traffic_97x61", self.frames, "Trafficscene.png missing")
        for name, frame in self.frames.items():
            for s in SEVERITIES:
                for mode in PIXELATION_MODES:
                    with self.subTest(frame=name, severity=s, **mode):
                        self.assert_twin("pixelation", frame, severity=s, **mode)

    def test_grid_offsets(self):
        for name in ("random_64x96", "traffic_97x61"):
            for ox, oy in GRID_OFFSETS:
                for s in (0.5, 0.95):
                    for hard in (False, True):
                        with self.subTest(frame=name, offset=(ox, oy), severity=s, hard=hard):
                            self.assert_twin("pixelation", self.frames[name], severity=s,
                                             hard=hard, grid_offset_x=ox, grid_offset_y=oy)

    def test_pixel_size_overrides(self):
        for name in ("random_84x84", "traffic_97x61"):
            for p in PIXEL_SIZES:
                for mode in PIXELATION_MODES:
                    with self.subTest(frame=name, pixel_size=p, **mode):
                        self.assert_twin("pixelation", self.frames[name], severity=0.0,
                                         pixel_size=p, **mode)

    def test_tall_frames(self):
        rng = np.random.default_rng(6)
        for h, w in TALL_FRAMES:
            frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            for kw in (dict(severity=0.75), dict(severity=0.0, pixel_size=160),
                       dict(severity=0.0, pixel_size=7, hard=True)):
                with self.subTest(shape=(h, w), **kw):
                    self.assert_twin("pixelation", frame, **kw)

    def test_resize_follows_installed_pillow_pass_order(self):
        rng = np.random.default_rng(7)
        for (w, h), (w2, h2) in ORDERED_RESIZES:
            with self.subTest(src=(w, h), dst=(w2, h2)):
                a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                cpu = np.asarray(Image.fromarray(a).resize((w2, h2), Image.BILINEAR))
                gpu = self.to_frame(self.sim.pil_resize(self.to_gpu(a), w2, h2))
                np.testing.assert_array_equal(gpu, cpu)

    def test_batch_of_two_frames(self):
        a, b = self.frames["random_84x84"], self.frames["traffic_84x84"]
        x = self.vg.frames_to_gpu(np.stack([a, b]), self.sim.device)
        for kw in (dict(severity=0.9, grid_offset_x=0.3), dict(severity=0.95, hard=True),
                   dict(severity=0.5, sampling="point")):
            out = self.vg.gpu_to_frames(self.sim.pixelation(x, **kw))
            for i, frame in enumerate((a, b)):
                with self.subTest(frame=i, **kw):
                    np.testing.assert_array_equal(
                        out[i], np.asarray(vipsim.apply_pixelation(Image.fromarray(frame), **kw)))

    def test_zero_severity_is_identity(self):
        x = self.to_gpu(self.frame)
        for s in (0.0, -1.0):
            self.assertIs(self.sim.pixelation(x, s), x)


class TestGpuLedMatchesCpu(GpuResampleCase):
    def test_severities(self):
        for name, frame in self.frames.items():
            for s in SEVERITIES:
                with self.subTest(frame=name, severity=s):
                    self.assert_twin("led", frame, severity=s)

    def test_kwargs(self):
        for name in ("random_64x96", "traffic_84x84", "traffic_97x61"):
            for kw in LED_KWARGS:
                kw = {"severity": 0.6, **kw}
                with self.subTest(frame=name, **kw), warnings.catch_warnings():
                    # numpy's RuntimeWarnings from the non-finite kwargs
                    warnings.simplefilter("ignore", RuntimeWarning)
                    self.assert_twin("led", self.frames[name], **kw)

    def test_batch_of_two_frames(self):
        a, b = self.frames["random_84x84"], self.frames["traffic_84x84"]
        x = self.vg.frames_to_gpu(np.stack([a, b]), self.sim.device)
        kw = dict(severity=0.7, brightness=1.3, margin=0.1)
        out = self.vg.gpu_to_frames(self.sim.led(x, **kw))
        for i, frame in enumerate((a, b)):
            np.testing.assert_array_equal(
                out[i], np.asarray(vipsim.apply_led(Image.fromarray(frame), **kw)))

    def test_zero_severity_is_identity(self):
        x = self.to_gpu(self.frame)
        for s in (0.0, -1.0):
            self.assertIs(self.sim.led(x, s), x)

    def test_per_channel_brightness_batch_and_bad_shapes(self):
        a, b = self.frames["random_84x84"], self.frames["traffic_84x84"]
        x = self.vg.frames_to_gpu(np.stack([a, b]), self.sim.device)
        kw = dict(severity=0.7, brightness=[0.4, 1.0, 1.9], margin=0.1)
        out = self.vg.gpu_to_frames(self.sim.led(x, **kw))
        for i, frame in enumerate((a, b)):
            np.testing.assert_array_equal(
                out[i], np.asarray(vipsim.apply_led(Image.fromarray(frame), **kw)))
        img = Image.fromarray(a)
        for bad in ([0.5, 1.0], [0.5, 1.0, 1.0, 1.0], np.ones((2, 3))):
            with self.subTest(brightness=bad):
                with self.assertRaises(ValueError):
                    vipsim.apply_led(img, 0.6, brightness=bad)
                with self.assertRaises(ValueError):
                    self.sim.led(x, 0.6, brightness=bad)
        # Per-pixel brightness: the CPU takes it, no level table can hold it.
        per_pixel = np.linspace(0.5, 1.5, 84)[:, None, None]
        vipsim.apply_led(img, 0.6, brightness=per_pixel)
        with self.assertRaises(NotImplementedError):
            self.sim.led(x, 0.6, brightness=per_pixel)


if __name__ == "__main__":
    unittest.main()
