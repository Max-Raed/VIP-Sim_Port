"""GPU twins of the shape filters (vortex, teichopsia, foveal_darkness) must
reproduce the CPU filters.

The NumPy filters are the reference.  All three twins are bit-exact (their
fields come from the CPU's own numpy code and the per-frame blend repeats
the CPU's float ops in the same dtypes), so the tests assert equality, which
is stricter than GpuParityCase.assert_close.  Runs on CUDA when available,
else on the CPU; skipped without torch.  teichopsia also needs scikit-image.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import functools
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vipsim  # noqa: E402
from test_gpu_primitives import TRAFFIC, natural_crop  # noqa: E402
from test_rounding import GpuParityCase  # noqa: E402

SIZES = [(64, 96), (84, 84), (61, 97)]          # (height, width); the last is odd
HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None

# The vortex of validation profile p1 (every extension keyword at once).
VORTEX_P1 = dict(severity=0.55, center=[0.5, 0.61], vortex_radius=0.33,
                 suction_strength=2, fade_width=0, swirl_strength=-0.05,
                 radius_y=0.18, inner_circle_radius=0.006, inner_color=1,
                 inner_center=[0.494, 0.549], inner_fade=0, inner_opacity=1,
                 halo_radius=0.004, halo_thickness=0.03, halo_color=1,
                 halo_opacity=0.68, halo_center=[0.506, 0.61])
VORTEX_KWARGS = [
    VORTEX_P1,
    # Profile tuner preset: fixed radius and suction.
    dict(severity=0.4, center=[0.5, 0.5], vortex_radius=0.2, suction_strength=1),
    # Asymmetric ellipse, strong swirl, soft translucent RGB dot, halo on it.
    dict(severity=0.7, center=(0.45, 0.55), radius_y=0.3, radius_y_down=0.1,
         swirl_strength=1.5, falloff_exponent=1.0, inner_circle_radius=0.08,
         inner_color=(1.0, 0.2, 0.0), inner_fade=0.03, inner_opacity=0.6,
         halo_radius=0.05, halo_thickness=0.02, halo_color=0.9, halo_opacity=0.8),
    # No noise, no dot, wide fade, field cut by the image edge.
    dict(severity=0.9, center=(0.2, 0.9), noise_amount=0.0, noise_scale=3.0,
         inner_circle_radius=0.0, fade_width=0.2),
    # Coarse heavy noise, halo with its own centre, dark dot.
    dict(severity=0.6, noise_amount=0.5, noise_scale=25.0, inner_color=0.0,
         halo_radius=0.02, halo_center=(0.6, 0.4)),
]
FOVEAL_KWARGS = [
    dict(severity=0.5, mode="desat"),
    dict(severity=0.8, mode="light", radius=0.3),
    dict(severity=0.49, center=[0.52, 0.69], radius_x=0.19425,
         radius_y=0.12319125, edge_width=0.23775),
    dict(severity=0.15, center=[0.75, 0.35], radius_x=0.95, radius_y=1.05,
         edge_width=0.25, mode="dark", invert=True),
    dict(severity=0.07, center=[0.75, 0.35], radius_x=0.88, radius_y=0.98,
         edge_width=0.15, band_width=0.25, mode="light", invert=True),
    dict(severity=0.9, center=(0.3, 0.6), radius_x=0.2, radius_y=0.35,
         edge_width=0.0, mode="desat", invert=True, band_width=0.5),
]
TEICHOPSIA_KWARGS = [
    dict(severity=0.3),
    dict(severity=1.0, seed=3, gaze=(0.3, 0.6), sat_boost=3.0, lum_contribution=0.2),
    dict(severity=0.75, seed=11, gaze=[0.8, 0.25], sat_boost=1.0, lum_contribution=1.0),
]
f64 = np.float64


@functools.lru_cache(maxsize=None)
def large_frames():
    """512x1024 random and traffic frames.  The 1-LSB cases below are rare
    (1-5 px per frame), so they need this many pixels to show."""
    return {"random_512x1024": np.random.default_rng(1).integers(
                0, 256, (512, 1024, 3), dtype=np.uint8),
            "traffic_512x1024": natural_crop(512, 1024)}


class GpuShapesCase(GpuParityCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        rng = np.random.default_rng(3)
        cls.frames = {}
        for h, w in SIZES:
            cls.frames[f"random_{h}x{w}"] = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            if TRAFFIC.is_file():
                cls.frames[f"traffic_{h}x{w}"] = natural_crop(h, w)

    def run_both(self, name, frame, sim=None, **kw):
        """(GPU, CPU) outputs of filter `name` as uint8 HxWx3 arrays."""
        sim = sim or self.sim
        cpu = np.asarray(getattr(vipsim, f"apply_{name}")(Image.fromarray(frame), **kw))
        x = self.vg.frames_to_gpu(frame, sim.device)
        gpu = self.vg.gpu_to_frames(getattr(sim, name)(x, **kw))[0]
        return gpu, cpu

    def check(self, name, kwargs_list, frames=None, sim=None):
        if frames is None:
            self.assertIn("traffic_84x84", self.frames, "Trafficscene.png missing")
            frames = self.frames
        for fname, frame in frames.items():
            for kw in kwargs_list:
                with self.subTest(frame=fname, **{k: repr(v) for k, v in kw.items()}):
                    gpu, cpu = self.run_both(name, frame, sim=sim, **kw)
                    np.testing.assert_array_equal(gpu, cpu)

    def check_numpy_scalar_cache(self, name, kw):
        """kw as Python floats, then as np.float64 on the same instance: the
        second call must build its own field, not reuse the first one
        (_freeze alone keys np.float64(v) like v)."""
        sim = self.vg.VipSimGPU(device=self.sim.device)
        kw64 = {k: f64(v) for k, v in kw.items() if k != "severity"}
        self.check(name, [kw, dict(kw, **kw64)], large_frames(), sim)
        self.assertEqual(len(sim.cache), 2)

    def check_batch(self, name, **kw):
        """N=2 different frames in one call match two CPU calls."""
        h, w = SIZES[0]
        frames = np.stack([self.frames[f"random_{h}x{w}"], natural_crop(h, w)])
        out = self.vg.gpu_to_frames(
            getattr(self.sim, name)(self.vg.frames_to_gpu(frames, self.sim.device), **kw))
        for i, frame in enumerate(frames):
            with self.subTest(sample=i):
                cpu = np.asarray(getattr(vipsim, f"apply_{name}")(Image.fromarray(frame), **kw))
                np.testing.assert_array_equal(out[i], cpu)

    def check_identity(self, name, **kw):
        """Severity <= 0 (and fields that touch no pixel) leave the frame as is."""
        frame = self.frames["random_64x96"]
        gpu, cpu = self.run_both(name, frame, **kw)
        np.testing.assert_array_equal(cpu, frame)
        np.testing.assert_array_equal(gpu, frame)

    def check_reads_8_bit_levels(self, name, **kw):
        """Input off its 8-bit levels by up to 0.49 LSB (e.g. a float output
        of an earlier GPU filter) gives the output of the rounded image, as
        the CPU only ever sees a uint8 image."""
        import torch
        frame = self.frames["traffic_84x84"]
        x = self.vg.frames_to_gpu(frame, self.sim.device)
        off = (torch.rand(x.shape, generator=torch.Generator().manual_seed(0)) - 0.5) * (0.98 / 255)
        out = getattr(self.sim, name)(x + off.to(x.device), **kw)
        cpu = np.asarray(getattr(vipsim, f"apply_{name}")(Image.fromarray(frame), **kw))
        np.testing.assert_array_equal(self.vg.gpu_to_frames(out)[0], cpu)


class TestGpuVortex(GpuShapesCase):
    def test_severities_match_cpu(self):
        self.check("vortex", [dict(severity=s) for s in
                              (0.005, 0.02, 0.3, 0.5, 0.75, 0.999, 1.0, 1.7)])

    def test_keywords_match_cpu(self):
        self.check("vortex", VORTEX_KWARGS)

    def test_batch_of_two_frames(self):
        self.check_batch("vortex", **VORTEX_P1)

    def test_identity_cases(self):
        for kw in (dict(severity=0.0), dict(severity=-0.3),
                   dict(severity=0.5, center=(4.0, 4.0))):     # field off-screen
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                self.check_identity("vortex", **kw)

    def test_reads_8_bit_levels(self):
        self.check_reads_8_bit_levels("vortex", **VORTEX_KWARGS[2])

    def test_fields_are_cached_per_parameters(self):
        sim = self.vg.VipSimGPU(device=self.sim.device)
        x = self.vg.frames_to_gpu(self.frame, sim.device)
        sim.vortex(x, 0.5)
        sim.vortex(x, 0.5)
        self.assertEqual(len(sim.cache), 1)
        sim.vortex(x, 0.5, swirl_strength=0.2)
        sim.vortex(x[..., :32, :48], 0.5)
        self.assertEqual(len(sim.cache), 3)

    def test_numpy_float64_keywords_get_their_own_field(self):
        # Was 2 px off by 1 LSB on the random frame when the np.float64 call
        # reused the field of the Python-float call.
        self.check_numpy_scalar_cache(
            "vortex", dict(severity=0.5, noise_amount=0.05, vortex_radius=0.3))

    def test_numpy_float32_ellipse_matches_cpu(self):
        # A numpy float32 radius_y makes the CPU's whole field float32, so its
        # arr * s_out is a float32 product.  A float64 product was 1 LSB off
        # on 1 px of this random frame.
        frame = np.random.default_rng(0).integers(0, 256, (512, 1024, 3), dtype=np.uint8)
        self.check("vortex", [dict(severity=0.9, vortex_radius=np.float32(0.45),
                                   radius_y=np.float32(0.4), fade_width=0.4,
                                   inner_circle_radius=0.0)], {"random_512x1024": frame})


class TestGpuFovealDarkness(GpuShapesCase):
    def test_severities_and_modes_match_cpu(self):
        self.check("foveal_darkness", [dict(severity=s, mode=m)
                                       for s in (0.005, 0.25, 0.5, 1.0, 2.0)
                                       for m in ("dark", "light", "desat")])

    def test_keywords_match_cpu(self):
        self.check("foveal_darkness", FOVEAL_KWARGS)

    def test_batch_of_two_frames(self):
        self.check_batch("foveal_darkness", **FOVEAL_KWARGS[0])

    def test_identity_cases(self):
        for s in (0.0, -1.0, 1e-9):
            with self.subTest(severity=s):
                self.check_identity("foveal_darkness", severity=s)

    def test_reads_8_bit_levels(self):
        self.check_reads_8_bit_levels("foveal_darkness", **FOVEAL_KWARGS[0])

    def test_desat_on_width_one_frames(self):
        # For W == 1, numpy's `arr @ w` runs BLAS sdot instead of gemv and
        # rounds the luminance differently on 22% of colours.  The column
        # holds exactly those colours; with the gemv sum the twin was 1 LSB
        # off on 1151 of them in the first case.
        k = np.arange(1 << 24, dtype=np.uint32)
        rgb = np.stack([k >> 16, (k >> 8) & 255, k & 255], -1).astype(np.uint8)
        a = rgb.astype(np.float32)
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gemv = (a[:, 0] * w[0] + a[:, 1] * w[1]) + a[:, 2] * w[2]
        column = rgb[(a[:, None] @ w)[:, 0] != gemv][:, None]
        self.check("foveal_darkness", [dict(severity=1.0, mode="desat", radius_x=50.0,
                                            radius_y=50.0, edge_width=1e-3)],
                   {"column": column})
        rng = np.random.default_rng(5)
        frames = {f"random_{h}x1": rng.integers(0, 256, (h, 1, 3), dtype=np.uint8)
                  for h in (1, 17, 240)}
        self.check("foveal_darkness", [dict(severity=s, mode="desat") for s in (0.5, 1.0)],
                   frames)

    def test_numpy_float64_keywords_match_cpu(self):
        # A numpy float64 keyword makes the CPU's darken float64, and its
        # blend too.  In float32 these were 1 LSB off on 1-4 px per frame.
        frame = np.random.default_rng(84093).integers(0, 256, (84, 84, 3), dtype=np.uint8)
        self.check("foveal_darkness", [dict(severity=0.5, center=np.array([0.4, 0.6])),
                                       dict(severity=f64(0.5), center=(f64(0.4), f64(0.6)))],
                   {"random_84x84": frame})
        self.check("foveal_darkness", [dict(severity=0.5, mode="desat", radius_x=f64(0.3),
                                            radius_y=f64(0.25)),
                                       dict(severity=0.5, mode="light", edge_width=f64(0.3),
                                            radius=f64(0.31))],
                   large_frames())

    def test_numpy_float64_keywords_get_their_own_field(self):
        # Was 5 px (traffic) and 2 px (random) off by 1 LSB.
        self.check_numpy_scalar_cache(
            "foveal_darkness", dict(severity=0.5, radius_x=0.3, radius_y=0.25))


class TestGpuTeichopsiaExtra(GpuShapesCase):
    def test_missing_teichopsia_extra_names_it(self):
        from unittest import mock
        sim = self.vg.VipSimGPU(device=self.sim.device)      # nothing cached yet
        x = self.vg.frames_to_gpu(self.frame, sim.device)
        blocked = dict.fromkeys(["skimage", "skimage.color", "skimage.draw"])
        with mock.patch.dict(sys.modules, blocked):
            with self.assertRaisesRegex(ImportError, r"vipsim\[teichopsia\]"):
                sim.teichopsia(x, 0.5)


@unittest.skipUnless(HAS_SKIMAGE, "teichopsia needs scikit-image (vipsim[teichopsia])")
class TestGpuTeichopsia(GpuShapesCase):
    def test_severities_match_cpu(self):
        self.check("teichopsia", [dict(severity=s) for s in (0.005, 0.5, 0.75, 1.0)])

    def test_keywords_match_cpu(self):
        self.check("teichopsia", TEICHOPSIA_KWARGS)

    def test_batch_of_two_frames(self):
        self.check_batch("teichopsia", **TEICHOPSIA_KWARGS[1])

    def test_identity_cases(self):
        for kw in (dict(severity=0.0), dict(severity=0.5, gaze=(5.0, 5.0))):  # ring off-screen
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                self.check_identity("teichopsia", **kw)

    def test_reads_8_bit_levels(self):
        self.check_reads_8_bit_levels("teichopsia", **TEICHOPSIA_KWARGS[1])

    def test_numpy_scalar_keywords_match_cpu(self):
        # A numpy float64 (or int64) sat_boost or lum_contribution makes the
        # CPU's product float64 before it is stored as float32.  In float32
        # these were 1 LSB off on 1-2 px per frame.  A numpy float32 keeps
        # the float32 product.
        self.check("teichopsia", [dict(severity=0.8, lum_contribution=f64(0.75)),
                                  dict(severity=0.8, sat_boost=f64(1.7)),
                                  dict(severity=0.8, sat_boost=np.int64(2),
                                       lum_contribution=np.float32(0.6))],
                   large_frames())

    def test_hsv_round_trip_matches_skimage(self):
        # 2**20 random colours plus every colour with two equal channels (the
        # ties where skimage's hue overwrite order matters).  All 2**24
        # colours also match, but take ~20 s.
        from skimage.color import hsv2rgb, rgb2hsv
        k = np.arange(256, dtype=np.uint8)
        a, b = (m.ravel() for m in np.meshgrid(k, k, indexing="ij"))
        ties = [np.stack(c, -1) for c in ((a, a, b), (a, b, a), (b, a, a))]
        rand = np.random.default_rng(4).integers(0, 256, (1 << 20, 3), dtype=np.uint8)
        rgb = np.concatenate([rand] + ties)[None]
        arr = rgb.astype(np.float32) / 255.0
        hsv = rgb2hsv(arr)
        x = self.vg.frames_to_gpu(rgb, self.sim.device)
        t_hsv = self.vg.skimage_rgb2hsv(x)
        np.testing.assert_array_equal(t_hsv[0].permute(1, 2, 0).cpu().numpy(), hsv)
        hsv[..., 1] = np.clip(hsv[..., 1] * 2.0, 0.0, 1.0)
        t_hsv[:, 1] = (t_hsv[:, 1] * 2.0).clamp(0.0, 1.0)
        np.testing.assert_array_equal(
            self.vg.skimage_hsv2rgb(t_hsv)[0].permute(1, 2, 0).cpu().numpy(), hsv2rgb(hsv))


if __name__ == "__main__":
    unittest.main()
