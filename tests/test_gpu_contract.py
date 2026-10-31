"""Promises every GPU twin makes, checked across all of them at once.

Each twin clamps severity like its CPU filter, keys its cached fields by
every keyword it reads, and returns level/255 in x's dtype; the conversion
helpers take only what the CPU filters take and hand back what they return;
profile_chain rejects what render_profiles_v2.apply_stack rejects.
Runs on CUDA when available, else on the CPU; skipped without torch.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vipsim  # noqa: E402
from vipsim import filters  # noqa: E402
from test_rounding import GpuParityCase  # noqa: E402

HAVE_SKIMAGE = importlib.util.find_spec("skimage") is not None


def gradient_frame(h=61, w=97):
    """Gradients plus noise: bright regions for bloom's threshold, and every
    level band for the tables."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 // (w - 1), yy * 255 // (h - 1), (xx + yy) * 7 % 256], -1)
    noise = np.random.default_rng(2).integers(-40, 41, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


class GpuContractCase(GpuParityCase):
    def names(self):
        """Every ported filter, without teichopsia when scikit-image is missing."""
        return [n for n in filters._FILTER_DISPATCH if HAVE_SKIMAGE or n != "teichopsia"]

    def gpu_frame(self, name, frame, sim=None, **kw):
        sim = sim or self.sim
        x = self.vg.frames_to_gpu(frame, sim.device)
        return self.vg.gpu_to_frames(getattr(sim, name)(x, **kw))[0]

    def assert_matches_cpu(self, name, frame, sim=None, **kw):
        cpu = np.asarray(filters._FILTER_DISPATCH[name](Image.fromarray(frame), **kw))
        np.testing.assert_array_equal(self.gpu_frame(name, frame, sim, **kw), cpu)


class TestEveryTwin(GpuContractCase):
    def test_out_of_range_severity_is_clamped_like_the_cpu(self):
        frame = gradient_frame()
        for name in self.names():
            if "severity" not in inspect.signature(getattr(self.sim, name)).parameters:
                continue                                 # bcg
            for s in (-0.5, 1.5):
                with self.subTest(filter=name, severity=s):
                    self.assert_matches_cpu(name, frame, severity=s)
        # At severity >= 1 the default table makes the image constant and
        # hides the frost.
        self.assert_matches_cpu("cataracts", frame, severity=1.5,
                                use_brightness=False, use_contrast=False)

    def test_output_is_on_the_8_bit_grid(self):
        # Every parity test goes through gpu_to_frames, which rounds; the
        # float output itself must be level/255 too.
        import torch
        g = torch.Generator().manual_seed(0)
        x = torch.rand(1, 3, 64, 96, generator=g).to(self.sim.device)
        for name in self.names():
            kw = ({"severity": 0.5} if "severity" in
                  inspect.signature(getattr(self.sim, name)).parameters else {})
            with self.subTest(filter=name):
                out = getattr(self.sim, name)(x, **kw)
                self.assertTrue(torch.equal(out, self.vg.quantize_u8(out)))

    def test_cache_keys_include_every_keyword(self):
        # One instance, called again with one keyword changed: a key that
        # dropped that keyword would hand back the first call's field.
        cases = [("vortex", {"severity": 0.6, "halo_radius": 0.05}, {"halo_center": (0.3, 0.6)}),
                 ("foveal_darkness", {"severity": 0.6, "invert": True}, {"band_width": 0.2}),
                 ("cataracts", {"severity": 0.5}, {"contrast_coeff": (0.3, 0.5, 0.9)}),
                 ("floaters", {"severity": 0.5, "distribution": "gaussian",
                               "spread_x": 0.3, "spread_y": 0.05}, {"angle": 45.0})]
        if HAVE_SKIMAGE:
            cases.append(("teichopsia", {}, {"gaze": (0.3, 0.7)}))
        frame = np.random.default_rng(1).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        for name, kw, change in cases:
            with self.subTest(filter=name, change=change):
                sim = self.vg.VipSimGPU(device=self.sim.device)
                self.assert_matches_cpu(name, frame, sim, **kw)
                self.assert_matches_cpu(name, frame, sim, **kw, **change)

    def test_tolerances_are_the_documented_ones(self):
        # test_gpu_profiles and the validator read their bounds from here.
        for name, tolerance in self.vg.TWIN_TOLERANCE.items():
            with self.subTest(filter=name):
                self.assertEqual(tolerance, (1, 0.02) if name in ("cvd", "bcg", "foveal_darkness")
                                 else (0, 0.0))
        self.assertEqual(set(self.vg.TWIN_TOLERANCE), set(filters._FILTER_DISPATCH))


class TestConversionHelpers(GpuContractCase):
    def test_frames_to_gpu_takes_only_uint8_rgb_frames(self):
        frame = np.zeros((24, 32, 3), np.uint8)
        for bad in (frame.astype(np.float32) / 255, frame.astype(np.uint16), frame.astype(bool)):
            with self.subTest(dtype=bad.dtype.name), self.assertRaises(TypeError):
                self.vg.frames_to_gpu(bad, self.sim.device)
        for bad in (frame.transpose(2, 0, 1), frame[..., 0], np.zeros((24, 32, 4), np.uint8),
                    frame[None, None]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.vg.frames_to_gpu(bad, self.sim.device)
        self.assertEqual(tuple(self.vg.frames_to_gpu(frame, self.sim.device).shape),
                         (1, 3, 24, 32))

    def test_gpu_to_frames_returns_c_contiguous_frames(self):
        # Like the CPU filters' np.asarray(img): memoryview casts and
        # torch.from_numpy(...).view(-1) need a contiguous buffer.
        out = self.vg.gpu_to_frames(self.vg.frames_to_gpu(self.frame, self.sim.device))
        self.assertTrue(out.flags.c_contiguous)
        np.testing.assert_array_equal(out[0], self.frame)
        memoryview(out[0]).cast("B")


class TestProfileChain(GpuContractCase):
    def test_unknown_and_non_filter_names_raise_key_error(self):
        # apply_stack looks the name up in _FILTER_DISPATCH; VipSimGPU has
        # other methods (pil_resize, apply_preset, cached) that are not filters.
        for name in ("blurr", "pil_resize", "apply_preset", "cached"):
            with self.subTest(filter=name), self.assertRaises(KeyError):
                self.vg.profile_chain([{"filter": name, "severity": 0.5}])

    def test_disabled_steps_and_notes_are_dropped(self):
        chain = self.vg.profile_chain({"filters": [
            {"filter": "blur", "severity": 0.3, "_note": "x"},
            {"filter": "nonexistent", "_disabled": True},
            {"filter": "bcg", "severity": 0.5, "brightness": 0.8}]})
        self.assertEqual(chain, [("blur", {"severity": 0.3}), ("bcg", {"brightness": 0.8})])


if __name__ == "__main__":
    unittest.main()
