"""The GPU twins of the random-field filters must reproduce the CPU filters.

noise, glitch, flickering_stars and floaters build their random fields with
the CPU's own numpy code, and the per-frame work is a float32 multiply-add,
a gather or an integer lookup, so they are compared for exact equality
(stricter than GpuParityCase.assert_close's 1 LSB budget).
Runs on CUDA when available, else on the CPU; skipped without torch.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vipsim  # noqa: E402
from test_gpu_primitives import TRAFFIC, GpuPrimitiveCase, natural_crop  # noqa: E402

FILTERS = ("noise", "glitch", "flickering_stars", "floaters")
SEVERITIES = (0.001, 0.3, 0.999, 1.0)
# A hand-drawn cluster (normalised [y, x]) like the floater profiles use.
REGION = [[0.30, 0.35], [0.25, 0.60], [0.45, 0.75], [0.70, 0.62], [0.65, 0.30]]


class TestGpuStochasticMatchesCpu(GpuPrimitiveCase):
    # One class: setUpClass decodes the 16 MB traffic scene for every crop.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # One size odd in both axes (glitch bands and the 1024-px overlay
        # mapping do not divide evenly).
        rng = np.random.default_rng(5)
        cls.frames["random_47x71"] = rng.integers(0, 256, (47, 71, 3), dtype=np.uint8)
        if TRAFFIC.is_file():
            cls.frames["traffic_47x71"] = natural_crop(47, 71)

    def check(self, name, severities=SEVERITIES, **kw):
        """GPU method == CPU apply_<name> on every frame and severity."""
        self.assertIn("traffic_47x71", self.frames, "Trafficscene.png missing")
        cpu_fn = getattr(vipsim, "apply_" + name)
        for frame_name, frame in self.frames.items():
            for s in severities:
                with self.subTest(frame=frame_name, severity=s, **kw):
                    cpu = np.asarray(cpu_fn(Image.fromarray(frame), s, **kw))
                    gpu = self.to_frame(getattr(self.sim, name)(self.to_gpu(frame), s, **kw))
                    np.testing.assert_array_equal(gpu, cpu)

    def batch(self):
        """Two different 84x84 frames and their [2, 3, H, W] tensor."""
        frames = (self.frames["random_84x84"], self.frames["traffic_84x84"])
        return frames, self.vg.frames_to_gpu(np.stack(frames), self.sim.device)

    # ---- noise ----
    def test_noise_default(self):
        self.check("noise")

    def test_noise_seed(self):
        self.check("noise", (0.5,), seed=7)

    # ---- glitch ----
    def test_glitch_default(self):
        self.check("glitch")

    def test_glitch_max_displace_and_seed(self):
        # 1.7 shifts rows by more than the width: np.roll wraps around.
        for md in (0.3, 1.7):
            self.check("glitch", (0.5, 1.0), max_displace=md, seed=3)

    # ---- flickering_stars ----
    def test_stars_default(self):
        self.check("flickering_stars")

    def test_stars_n_stars_and_seed(self):
        # Many overlapping sprites, many clipped at the border; and none.
        self.check("flickering_stars", (0.5, 1.0), n_stars=400, seed=2)
        self.check("flickering_stars", (0.5,), n_stars=0)

    # ---- floaters ----
    def test_floaters_default_disc(self):
        self.check("floaters")

    def test_floaters_distributions(self):
        for kw in ({"distribution": "uniform"},
                   {"distribution": "gaussian", "center": (0.4, 0.6), "spread": 0.1},
                   {"distribution": "gaussian", "spread_x": 0.05, "spread_y": 0.25,
                    "angle": 30.0},
                   {"distribution": "disc", "circle_radius_uv": 0.3, "seed": 4}):
            self.check("floaters", (0.3, 1.0), **kw)

    def test_floaters_mode_and_brightness(self):
        self.check("floaters", (0.3, 1.0), mode="light")
        self.check("floaters", (0.3, 1.0), mode="light", dot_brightness=0.85)
        self.check("floaters", (0.6,), dot_brightness=0.2)

    def test_floaters_blobs(self):
        # Blobs compound where they overlap; floater_size 40 makes them several
        # pixels wide even at 47 px, size 3 sends 1-px dots down the blob path.
        self.check("floaters", (0.3, 1.0), size_variance=True, n_floaters=3000)
        self.check("floaters", (0.5,), size_variance=True, floater_size=40,
                   n_floaters=3000)
        self.check("floaters", (0.8,), floater_size=3, n_floaters=3000)

    def test_floaters_blobs_at_full_density(self):
        # The default count at severity 1 stacks dozens of blob blends per pixel.
        frame = self.frames["traffic_84x84"]
        cpu = np.asarray(vipsim.apply_floaters(Image.fromarray(frame), 1.0,
                                               size_variance=True))
        gpu = self.to_frame(self.sim.floaters(self.to_gpu(frame), 1.0, size_variance=True))
        np.testing.assert_array_equal(gpu, cpu)

    def test_floaters_region(self):
        self.check("floaters", (0.3, 1.0), region=REGION)
        self.check("floaters", (0.5,), region=REGION, region_center_bias=0.6,
                   size_variance=True, n_floaters=2000)
        self.check("floaters", (0.5,), region=REGION, region_feather=0.0, mode="light")

    def test_floaters_degenerate_region_falls_back(self):
        # Fewer than 3 points, or a polygon off the overlay: distribution is used.
        self.check("floaters", (0.5,), region=[[0.2, 0.2], [0.8, 0.8]])
        self.check("floaters", (0.5,), region=[[2.0, 2.0], [2.5, 2.0], [2.0, 2.5]],
                   region_feather=0.0, distribution="uniform")

    def test_floaters_n_floaters(self):
        self.check("floaters", (0.5,), n_floaters=25)
        self.check("floaters", (0.5,), n_floaters=0)

    # ---- all four ----
    def test_batch_of_two_frames(self):
        frames, x = self.batch()
        for name, kw in (("noise", {}), ("glitch", {"max_displace": 0.3}),
                         ("flickering_stars", {}),
                         ("floaters", {"size_variance": True, "n_floaters": 6000})):
            out = self.vg.gpu_to_frames(getattr(self.sim, name)(x, 0.6, **kw))
            for i, frame in enumerate(frames):
                with self.subTest(filter=name, frame=i):
                    cpu = getattr(vipsim, "apply_" + name)(Image.fromarray(frame), 0.6, **kw)
                    np.testing.assert_array_equal(out[i], np.asarray(cpu))

    def test_generator_seed_draws_per_image(self):
        # A Generator advances on every CPU call, so image i of the batch
        # must match the i-th CPU call sharing an equal Generator.
        frames, x = self.batch()
        for name, kw in (("noise", {}), ("glitch", {"max_displace": 0.3}),
                         ("flickering_stars", {}),
                         ("floaters", {"size_variance": True, "n_floaters": 3000})):
            gpu_rng, cpu_rng = np.random.default_rng(11), np.random.default_rng(11)
            out = self.vg.gpu_to_frames(getattr(self.sim, name)(x, 0.6, seed=gpu_rng, **kw))
            for i, frame in enumerate(frames):
                with self.subTest(filter=name, frame=i):
                    cpu = getattr(vipsim, "apply_" + name)(Image.fromarray(frame), 0.6,
                                                           seed=cpu_rng, **kw)
                    np.testing.assert_array_equal(out[i], np.asarray(cpu))

    def test_seed_none_is_fresh_per_image_and_call(self):
        sim = self.vg.VipSimGPU(device=self.sim.device)
        frame = self.frames["traffic_84x84"]
        x = self.vg.frames_to_gpu(np.stack([frame, frame]), sim.device)
        for name, kw in (("noise", {}), ("glitch", {"max_displace": 0.3}),
                         ("flickering_stars", {}), ("floaters", {})):
            with self.subTest(filter=name):
                a = self.vg.gpu_to_frames(getattr(sim, name)(x, 0.6, seed=None, **kw))
                b = self.vg.gpu_to_frames(getattr(sim, name)(x, 0.6, seed=None, **kw))
                self.assertFalse(np.array_equal(a[0], a[1]))
                self.assertFalse(np.array_equal(a, b))
        # Nothing random is cached, only the floaters' level table.
        self.assertEqual(len(sim.cache), 1)

    def test_float64_sim(self):
        sim = self.vg.VipSimGPU(device=self.sim.device, dtype=self.torch.float64)
        frame = self.frames["traffic_47x71"]
        x = self.to_gpu(frame).double()
        for name in FILTERS:
            with self.subTest(filter=name):
                out = getattr(sim, name)(x, 0.7)
                self.assertEqual(out.dtype, self.torch.float64)
                cpu = getattr(vipsim, "apply_" + name)(Image.fromarray(frame), 0.7)
                np.testing.assert_array_equal(self.to_frame(out), np.asarray(cpu))

    def test_off_grid_float64_input_is_rounded_in_its_own_dtype(self):
        # The CPU gets the uint8 frame gpu_to_frames makes, which rounds x in
        # its own dtype.  Values within a few float32 ulps of every level's
        # .5 boundary round otherwise through float32 (about half of these).
        sim = self.vg.VipSimGPU(device=self.sim.device, dtype=self.torch.float64)
        k = np.arange(255.0)[:, None] + 0.5 + np.linspace(-4e-7, 4e-7, 64)
        ramp = self.torch.from_numpy(k / 255).expand(3, -1, -1)
        x = self.torch.stack([ramp, ramp.flip(-1)]).to(sim.device)
        frames = self.vg.gpu_to_frames(x)
        for name in FILTERS:
            out = self.vg.gpu_to_frames(getattr(sim, name)(x, 0.7))
            for i, frame in enumerate(frames):
                with self.subTest(filter=name, frame=i):
                    cpu = getattr(vipsim, "apply_" + name)(Image.fromarray(frame), 0.7)
                    np.testing.assert_array_equal(out[i], np.asarray(cpu))

    def test_float_seed_fails_like_cpu_after_the_equal_int_seed(self):
        # 1.0 == 1 with an equal hash, but default_rng(1.0) raises TypeError:
        # the field cached for seed=1 must not answer for seed=1.0.
        x = self.to_gpu(self.frame)
        for name in FILTERS:
            with self.subTest(filter=name):
                getattr(self.sim, name)(x, 0.5, seed=1)
                with self.assertRaises(TypeError):
                    getattr(vipsim, "apply_" + name)(Image.fromarray(self.frame), 0.5, seed=1.0)
                with self.assertRaises(TypeError):
                    getattr(self.sim, name)(x, 0.5, seed=1.0)

    def test_zero_severity_is_identity(self):
        x = self.to_gpu(self.frame)
        for name in FILTERS:
            for s in (0.0, -1.0):
                with self.subTest(filter=name, severity=s):
                    self.assertIs(getattr(self.sim, name)(x, s), x)
        self.assertIs(self.sim.floaters(x, 0.5, n_floaters=0), x)

    def test_fields_are_cached(self):
        sim = self.vg.VipSimGPU(device=self.sim.device)
        x = self.to_gpu(self.frame)
        for _ in range(2):
            for name in FILTERS:
                getattr(sim, name)(x, 0.5)
            # One field each, plus the floaters' level table.
            self.assertEqual(len(sim.cache), 5)
        sim.noise(x, 0.9)                   # the noise field ignores severity
        self.assertEqual(len(sim.cache), 5)


if __name__ == "__main__":
    unittest.main()
