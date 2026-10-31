"""Promises every GPU twin makes, checked across all of them at once.

Each twin clamps severity like its CPU filter, keys its cached fields by
the keywords it reads (one changed keyword per caching twin here; the
per-filter files change the others), and returns level/255 in x's dtype; the
conversion helpers take only what the CPU filters take and hand back what
they return; profile_chain rejects what render_profiles_v2.apply_stack
rejects; the validator fails on a skip unless told otherwise.
Runs on CUDA when available, else on the CPU; skipped without torch.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
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
                               "spread_x": 0.3, "spread_y": 0.05}, {"angle": 45.0}),
                 ("field_loss", {"severity": 0.6, "overlay_scale": 0.75}, {"overlay_scale": 1.8}),
                 ("led", {"severity": 0.6}, {"margin": 0.2}),
                 ("wiggle", {"severity": 0.8}, {"timer": 1.5}),
                 ("glitch", {"severity": 0.7, "seed": 3}, {"max_displace": 0.3}),
                 ("vignette", {"severity": 0.6}, {"softness": 0.1}),
                 ("bcg", {"brightness": 0.8, "contrast": 1.2}, {"color_space": "srgb"}),
                 ("distortion", {"severity": 0.8},
                  {"warp_centers": [(0.3, 0.6, 0.2, 0.9)]}),
                 ("flickering_stars", {"severity": 0.8, "seed": 2}, {"n_stars": 40}),
                 ("pixelation", {"severity": 0.9}, {"grid_offset_x": 0.3})]
        if HAVE_SKIMAGE:
            cases.append(("teichopsia", {}, {"gaze": (0.3, 0.7)}))
        frame = np.random.default_rng(1).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        for name, kw, change in cases:
            with self.subTest(filter=name, change=change):
                sim = self.vg.VipSimGPU(device=self.sim.device)
                self.assert_matches_cpu(name, frame, sim, **kw)
                self.assert_matches_cpu(name, frame, sim, **{**kw, **change})

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
        # torch.from_numpy(...).view(-1) need a contiguous buffer.  A
        # filter's output is NCHW-contiguous (frames_to_gpu's own output is
        # channels-last, which would come back contiguous anyway).
        x = self.vg.frames_to_gpu(self.frame, self.sim.device)
        out = self.vg.gpu_to_frames(self.sim.blur(x, 0.3))
        self.assertTrue(out.flags.c_contiguous)
        np.testing.assert_array_equal(
            out[0], np.asarray(vipsim.apply_blur(Image.fromarray(self.frame), 0.3)))
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


class TestValidatorGate(GpuContractCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        spec = importlib.util.spec_from_file_location(
            "validate_gpu_vs_numpy", ROOT / "validation_scripts" / "validate_gpu_vs_numpy.py")
        cls.validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.validator)

    def test_only_a_missing_scikit_image_counts_as_optional(self):
        missing = ModuleNotFoundError("No module named 'skimage'", name="skimage.draw")
        chained = ImportError("teichopsia needs scikit-image")
        chained.__cause__ = missing
        for e, optional in ((missing, True), (chained, True),
                            (ImportError("cannot import name '_x' from 'vipsim.filters'"), False),
                            (ModuleNotFoundError("No module named 'scipy'", name="scipy"), False)):
            with self.subTest(error=repr(e)):
                self.assertIs(self.validator._missing_optional_dep(e), optional)

    def run_validator(self, extra, failing):
        """main() on a small frame with the CPU filters in `failing` replaced."""
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "frame.png"
            Image.fromarray(gradient_frame(24, 32)).save(frame)
            argv = ["validate_gpu_vs_numpy.py", "--input", str(frame), "--out", tmp,
                    "--device", str(self.sim.device)] + extra
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(filters._FILTER_DISPATCH, failing), \
                    mock.patch("builtins.print"):
                self.validator.main()

    def test_a_skip_fails_the_run_unless_allowed(self):
        def no_skimage(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'skimage'", name="skimage")
        for extra, code in (([], 1), (["--allow-skip"], 0)):
            with self.subTest(argv=extra), self.assertRaises(SystemExit) as done:
                self.run_validator(extra, {"teichopsia": no_skimage})
            self.assertEqual(done.exception.code, code)

    def test_other_import_errors_fail_the_run(self):
        # A broken import inside a twin or filter is a bug, not a skip.
        def broken(*args, **kwargs):
            raise ImportError("cannot import name '_x' from 'vipsim.filters'")
        with self.assertRaisesRegex(ImportError, "cannot import name"):
            self.run_validator(["--allow-skip"], {"blur": broken})


if __name__ == "__main__":
    unittest.main()
