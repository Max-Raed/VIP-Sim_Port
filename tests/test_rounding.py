"""float -> uint8 conversions must round, not truncate.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vipsim import filters as vf  # noqa: E402


def ramp_image() -> Image.Image:
    """Every 8-bit level on every channel, 4 rows tall."""
    ramp = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    return Image.fromarray(np.stack([ramp, ramp[:, ::-1], ramp], axis=-1))


class TestUint8Rounding(unittest.TestCase):
    """Truncation biases output darker and the bias accumulates over a chain."""

    def test_identity_bcg_is_identity(self):
        img = ramp_image()
        out = vf.apply_bcg(img, brightness=1.0, contrast=1.0, gamma=1.0)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(img))

    def test_chained_identity_bcg_does_not_drift(self):
        img = ramp_image()
        out = img
        for _ in range(5):
            out = vf.apply_bcg(out, 1.0, 1.0, 1.0)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(img))

    def test_near_zero_severity_is_identity(self):
        img = ramp_image()
        for name, fn in [
            ("cvd", lambda im: vf.apply_cvd(im, "deuteranomaly", 1e-9)),
            ("vignette", lambda im: vf.apply_vignette(im, strength=1e-9)),
            ("foveal_darkness", lambda im: vf.apply_foveal_darkness(im, 1e-9)),
        ]:
            with self.subTest(filter=name):
                np.testing.assert_array_equal(np.asarray(fn(img)), np.asarray(img))


class GpuParityCase(unittest.TestCase):
    """Shared setup for GPU-vs-CPU comparisons; skipped without torch."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from vipsim import gpu as vg
        except ImportError as e:
            raise unittest.SkipTest(f"torch not available: {e}")
        cls.vg = vg
        cls.sim = vg.VipSimGPU(device="cuda" if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(0)
        cls.frame = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)

    def gpu(self, name, **kw):
        x = self.vg.frames_to_gpu(self.frame, self.sim.device)
        return self.vg.gpu_to_frames(getattr(self.sim, name)(x, **kw))[0].astype(int)

    def assert_close(self, gpu_out, cpu_img):
        d = np.abs(gpu_out - np.asarray(cpu_img).astype(int))
        self.assertLessEqual(d.max(), 1)
        self.assertLess(d.mean(), 0.02)


class TestGpuBcgMatchesCpu(GpuParityCase):
    def test_bcg(self):
        self.assert_close(self.gpu("bcg", brightness=0.75, contrast=1.25),
                          vf.apply_bcg(Image.fromarray(self.frame), 0.75, 1.25, 1.0))


if __name__ == "__main__":
    unittest.main()
