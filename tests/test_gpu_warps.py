"""The GPU warp filters must reproduce the CPU filters they port.

distortion, double_vision, wiggle and nystagmus only move whole pixels or
average integer shifts of them, so the twins are bit-exact with
apply_distortion, apply_double_vision, apply_wiggle and apply_nystagmus
(assert_array_equal, stricter than GpuParityCase.assert_close).
Runs on CUDA when available, else on the CPU; skipped without torch.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import array
import collections
import gc
import sys
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vipsim  # noqa: E402
from test_gpu_primitives import GpuPrimitiveCase, natural_crop  # noqa: E402

SEVERITIES = (1e-6, 0.01, 0.13, 0.5, 0.77, 1.0)
# Main non-default keyword arguments of each filter ({} = CPU defaults).
CASES = {
    "distortion": [
        {},
        {"warp_centers": [(0.2, 0.8, 0.25, -0.6), (0.7, 0.3, 0.1, 0.8)]},
        # radius 0 and magnitude 0 centres are skipped by the CPU
        {"warp_centers": [(0.5, 0.5, 0.3, 1.0), (0.1, 0.1, 0.0, 1.0), (0.4, 0.4, 0.2, 0.0)]},
        {"warp_centers": np.array([[0.3, 0.6, 0.2, 0.9], [0.6, 0.3, 0.3, -0.7]], np.float32)},
    ],
    "double_vision": [{}, {"axis": "vertical"}, {"axis": "diagonal"}],
    "wiggle": [
        {},
        {"mode": "simple"},
        {"freq": 5.0, "amplitude": 0.05, "timer": 0.5},
        {"mode": "simple", "freq": 30, "amplitude": 0.03, "timer": 2},
    ],
    "nystagmus": [{}, {"axis": "vertical"}, {"axis": "both"}],
}


class GpuWarpCase(GpuPrimitiveCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Odd size: double_vision shifts up to 4 px, nystagmus up to 6 px.
        rng = np.random.default_rng(3)
        cls.frames["random_97x161"] = rng.integers(0, 256, (97, 161, 3), dtype=np.uint8)
        if "traffic_84x84" in cls.frames:
            cls.frames["traffic_97x161"] = natural_crop(97, 161)

    def cpu(self, name, frame, severity, **kw):
        fn = getattr(vipsim, "apply_" + name)
        return np.asarray(fn(Image.fromarray(frame), severity, **kw))

    def gpu_batch(self, name, frames, severity, **kw):
        x = self.vg.frames_to_gpu(frames, self.sim.device)
        return self.vg.gpu_to_frames(getattr(self.sim, name)(x, severity, **kw))

    def assert_matches_cpu(self, name):
        self.assertIn("traffic_84x84", self.frames, "Trafficscene.png missing")
        for fname, frame in self.frames.items():
            for kw in CASES[name]:
                for s in SEVERITIES:
                    with self.subTest(frame=fname, severity=s, **kw):
                        np.testing.assert_array_equal(
                            self.gpu_batch(name, frame, s, **kw)[0],
                            self.cpu(name, frame, s, **kw))

    def assert_same_outcome(self, name, frame, severity, make_kw):
        """The GPU returns the CPU's image or raises the CPU's exception.
        make_kw() gives each side its own kwargs (a generator is one-shot)."""
        try:
            expected = self.cpu(name, frame, severity, **make_kw())
        except Exception as cpu:
            with self.assertRaises(type(cpu)) as gpu:
                self.gpu_batch(name, frame, severity, **make_kw())
            self.assertEqual(str(gpu.exception), str(cpu))
        else:
            np.testing.assert_array_equal(
                self.gpu_batch(name, frame, severity, **make_kw())[0], expected)

    def assert_batch_matches_cpu(self, name):
        pair = (self.frames["random_84x84"], self.frames["traffic_84x84"])
        for kw in CASES[name]:
            with self.subTest(**kw):
                out = self.gpu_batch(name, np.stack(pair), 0.8, **kw)
                for i, frame in enumerate(pair):
                    np.testing.assert_array_equal(out[i], self.cpu(name, frame, 0.8, **kw))


class TestGpuWarpsMatchCpu(GpuWarpCase):
    def test_distortion(self):
        self.assert_matches_cpu("distortion")

    def test_double_vision(self):
        self.assert_matches_cpu("double_vision")

    def test_wiggle(self):
        self.assert_matches_cpu("wiggle")

    def test_nystagmus(self):
        self.assert_matches_cpu("nystagmus")

    def test_nystagmus_reads_any_float_dtype_as_8_bit_levels(self):
        # The CPU sums float32 levels/255; half-precision input must too.
        frame = self.frames["random_97x161"]
        x = self.vg.frames_to_gpu(frame, self.sim.device)
        for kw in CASES["nystagmus"]:
            ref = self.cpu("nystagmus", frame, 0.5, **kw)
            for inp in (x.double(), x.half(), x.bfloat16()):
                with self.subTest(dtype=str(inp.dtype), **kw):
                    out = self.sim.nystagmus(inp, 0.5, **kw)
                    self.assertEqual(out.dtype, inp.dtype)
                    np.testing.assert_array_equal(
                        self.vg.gpu_to_frames(out.float())[0], ref)

    def test_batch_of_two_different_frames(self):
        for name in CASES:
            with self.subTest(filter=name):
                self.assert_batch_matches_cpu(name)


class TestGpuWarpsEdgeCases(GpuWarpCase):
    def test_out_of_range_severity_and_thin_frames(self):
        rng = np.random.default_rng(4)
        frames = [rng.integers(0, 256, shape, dtype=np.uint8)
                  for shape in ((1, 9, 3), (9, 1, 3), (2, 2, 3))]
        frames.append(self.frames["random_84x84"])
        for name in CASES:
            for frame in frames:
                for s in (-0.5, 1.5):
                    with self.subTest(filter=name, shape=frame.shape, severity=s):
                        np.testing.assert_array_equal(self.gpu_batch(name, frame, s)[0],
                                                      self.cpu(name, frame, s))

    def test_double_vision_shifts_of_exactly_half_a_pixel(self):
        # severity * 0.025 * (W - 1) is an exact .5 at these sizes, where the
        # CPU's round() goes to the even pixel, not up.
        rng = np.random.default_rng(12)
        for h, w, s, axis in ((64, 101, 1.0, "horizontal"), (64, 41, 0.5, "horizontal"),
                              (64, 21, 1.0, "horizontal"), (41, 64, 0.5, "vertical")):
            frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            with self.subTest(shape=frame.shape, severity=s, axis=axis):
                np.testing.assert_array_equal(
                    self.gpu_batch("double_vision", frame, s, axis=axis)[0],
                    self.cpu("double_vision", frame, s, axis=axis))

    def test_zero_severity_returns_input(self):
        x = self.to_gpu(self.frame)
        for name in CASES:
            with self.subTest(filter=name):
                self.assertIs(getattr(self.sim, name)(x, 0.0), x)

    def test_unknown_axis(self):
        # double_vision rejects it like the CPU; nystagmus, like the CPU,
        # treats any other axis as "both".
        x = self.to_gpu(self.frame)
        with self.assertRaises(ValueError):
            self.sim.double_vision(x, 0.5, axis="sideways")
        np.testing.assert_array_equal(
            self.gpu_batch("nystagmus", self.frame, 0.5, axis="sideways")[0],
            self.cpu("nystagmus", self.frame, 0.5, axis="sideways"))

    def test_input_is_rounded_like_the_cpu_uint8_image(self):
        # After a twin that returns unrounded floats the CPU chain would have
        # rounded to uint8; an off-grid input must give the on-grid output.
        x = self.to_gpu(self.frames["random_84x84"])
        off = (x + 0.45 / 255).clamp(0, 1)
        for name in CASES:
            with self.subTest(filter=name):
                fn = getattr(self.sim, name)
                np.testing.assert_array_equal(self.to_frame(fn(off, 0.9)),
                                              self.to_frame(fn(x, 0.9)))

    def test_numpy_scalar_kwargs_get_their_own_maps(self):
        # np.float64 / np.int64 kwargs make the CPU maps float64 (NEP 50), so
        # they must not reuse the maps cached for the equal Python number.
        frame = self.frames["random_97x161"]
        pairs = [("wiggle", {"timer": 0.5}, {"timer": np.float64(0.5)}),
                 ("wiggle", {"amplitude": 0.03}, {"amplitude": np.float64(0.03)}),
                 ("distortion", {"warp_centers": [(0.3, 0.6, 0.2, 0.9)]},
                  {"warp_centers": [tuple(np.float64(v) for v in (0.3, 0.6, 0.2, 0.9))]})]
        for name, py_kw, np_kw in pairs:
            for first, second in ((py_kw, np_kw), (np_kw, py_kw)):
                with self.subTest(filter=name, first=first):
                    self.sim.cache.clear()
                    for kw in (first, second):
                        np.testing.assert_array_equal(self.gpu_batch(name, frame, 0.5, **kw)[0],
                                                      self.cpu(name, frame, 0.5, **kw))
        self.assertGreater(
            np.count_nonzero(self.cpu("wiggle", frame, 0.5, timer=0.5)
                             != self.cpu("wiggle", frame, 0.5, timer=np.float64(0.5))), 0)

    def test_object_and_structured_kwargs_get_their_own_maps(self):
        # The shared _freeze keys an ndarray by (dtype.str, tobytes()).  An
        # object array's bytes are PyObject pointers: a freed float's address
        # comes back holding another value, and a list row mutated in place
        # keeps its pointer.  Every 32-byte structured dtype is "|V32", so the
        # same bytes read through two field layouts must not share maps.
        frame = self.frames["random_97x161"]

        def check(warp_centers):
            np.testing.assert_array_equal(
                self.gpu_batch("distortion", frame, 0.5, warp_centers=warp_centers)[0],
                self.cpu("distortion", frame, 0.5, warp_centers=warp_centers))

        self.sim.cache.clear()
        for m in np.linspace(-0.9, 0.9, 10):
            with self.subTest(case="object array of floats", m=m):
                centres = np.empty((1, 4), dtype=object)
                centres[0] = [0.3, 0.6, 0.25, float(m)]
                check(centres)
                del centres
                gc.collect()
        # distortion keys the numbers it reads from the centres; a wiggle
        # amplitude reaches the key as the object array itself.
        for a in np.linspace(0.005, 0.05, 10):
            with self.subTest(case="object array amplitude", a=a):
                amplitude = np.empty(1, dtype=object)
                amplitude[0] = float(a)
                np.testing.assert_array_equal(
                    self.gpu_batch("wiggle", frame, 0.5, amplitude=amplitude)[0],
                    self.cpu("wiggle", frame, 0.5, amplitude=amplitude))
                del amplitude
                gc.collect()
        row = np.empty(1, dtype=object)
        row[0] = [0.3, 0.6, 0.25, 0.9]
        for m in (0.9, -0.9):
            with self.subTest(case="object array of a list, mutated", m=m):
                row[0][3] = m
                check(row)
        layout = {"names": ["x", "y", "r", "m"], "formats": ["<f8"] * 4}
        xy = np.array([(0.3, 0.6, 0.25, 0.9)],
                      np.dtype({**layout, "offsets": [0, 8, 16, 24]}))
        yx = xy.view(np.dtype({**layout, "offsets": [8, 0, 16, 24]}))
        for centres in (xy, yx):
            with self.subTest(case="structured array", x=centres[0][0]):
                check(centres)

        # np.void records of the same float32-exact values, with f8 and f4
        # fields swapped: both are "|V24" and item() gives equal floats, but
        # f4 and f8 centres promote differently (NEP 50) and move one pixel.
        values = (0.390625, 0.328125, 0.453125, 0.8125)
        f8_first = np.array([values], [("x", "<f8"), ("y", "<f8"), ("r", "<f4"), ("m", "<f4")])
        f4_first = np.array([values], [("x", "<f4"), ("y", "<f4"), ("r", "<f8"), ("m", "<f8")])
        records = ([f8_first[0]], [f4_first[0]])
        for order in (records, records[::-1]):
            self.sim.cache.clear()
            for centres in order:
                with self.subTest(case="np.void records", first=order[0][0].dtype.descr):
                    check(centres)
        self.assertGreater(np.count_nonzero(
            self.cpu("distortion", frame, 0.5, warp_centers=records[0])
            != self.cpu("distortion", frame, 0.5, warp_centers=records[1])), 0)

    def test_other_containers_and_number_types_match_the_cpu(self):
        # The CPU iterates any iterable of indexable centres and lets numpy
        # take (or reject) any number type.  Unhashable kwargs used to crash
        # the cache-key lookup, and Decimal("0.5") == 0.5 with an equal hash
        # hit the maps cached for 0.5 instead of raising the CPU's TypeError.
        # Each case runs on a cold cache and after an equal value of another
        # type (Fraction(1, 2) works where Decimal("0.5") raises).
        frame = self.frames["random_97x161"]
        centres = [(0.3, 0.6, 0.25, 0.9), (0.6, 0.3, 0.2, -0.7)]
        cases = [
            ("distortion", {"warp_centers": centres},
             lambda: {"warp_centers": collections.deque(centres)}),
            ("distortion", {"warp_centers": centres},
             lambda: {"warp_centers": [array.array("d", c) for c in centres]}),
            # A generator is used up by the CPU's first pass (x), so it
            # gives no centres at all.
            ("distortion", {"warp_centers": centres},
             lambda: {"warp_centers": (c for c in centres)}),
            ("distortion", {"warp_centers": centres},
             lambda: {"warp_centers": [tuple(Fraction(v) for v in c) for c in centres]}),
            ("wiggle", {"freq": 12.0}, lambda: {"freq": array.array("d", [12.0])}),
            ("wiggle", {"timer": 0.5}, lambda: {"timer": Decimal("0.5")}),
            ("wiggle", {"timer": Fraction(1, 2)}, lambda: {"timer": Decimal("0.5")}),
            ("wiggle", {"timer": 0.5}, lambda: {"timer": Fraction(1, 2)}),
            ("wiggle", {"mode": "simple", "timer": 0.5},
             lambda: {"mode": "simple", "timer": Fraction(1, 2)}),
        ]
        for name, warm_kw, make_kw in cases:
            for warm in (False, True):
                with self.subTest(filter=name, kw=make_kw(), warm=warm):
                    self.sim.cache.clear()
                    if warm:
                        self.gpu_batch(name, frame, 0.5, **warm_kw)
                    self.assert_same_outcome(name, frame, 0.5, make_kw)

    def test_nan_parameters_raise_like_the_cpu(self):
        # NaN passes np.clip and astype(np.int32) makes it -2**31, so the
        # CPU's arr[sy, sx] raises IndexError (freq=inf too: 0 * inf on the
        # first row and column); a gather that clamps would return an image.
        # double_vision and nystagmus raise ValueError from int(round(nan)).
        nan = float("nan")
        frame = self.frames["random_84x84"]
        cases = [("distortion", nan, {}),
                 ("distortion", 0.5, {"warp_centers": [(nan, 0.6, 0.25, 0.9)]}),
                 ("distortion", 0.5, {"warp_centers": [(0.3, nan, 0.25, 0.9)]}),
                 ("distortion", 0.5, {"warp_centers": [(0.3, 0.6, nan, 0.9)]}),
                 ("distortion", 0.5, {"warp_centers": [(0.3, 0.6, 0.25, nan)]}),
                 ("wiggle", nan, {}),
                 ("wiggle", 0.5, {"timer": nan}),
                 ("wiggle", 0.5, {"amplitude": nan}),
                 ("wiggle", 0.5, {"freq": float("inf")}),
                 ("wiggle", 0.5, {"mode": "simple", "timer": nan}),
                 ("double_vision", nan, {}),
                 ("nystagmus", nan, {})]
        # vortex's arr[y0, x0] covers pixels outside the field too: suction
        # inf * 0 at the rim is NaN there while every inside pixel is finite.
        inf = float("inf")
        cases += [("vortex", 0.5, kw) for kw in (
            {"vortex_radius": inf}, {"noise_amount": inf}, {"noise_amount": -inf},
            {"suction_strength": inf}, {"suction_strength": -inf},
            {"suction_strength": nan}, {"swirl_strength": inf},
            {"swirl_strength": nan}, {"falloff_exponent": nan})]
        for name, s, kw in cases:
            with self.subTest(filter=name, severity=s, **kw), np.errstate(invalid="ignore"):
                with self.assertRaises(Exception) as cpu:
                    self.cpu(name, frame, s, **kw)
                with self.assertRaises(type(cpu.exception)) as gpu:
                    self.gpu_batch(name, frame, s, **kw)
                self.assertEqual(str(gpu.exception), str(cpu.exception))

    def test_maps_are_cached_per_parameters_and_size(self):
        self.sim.cache.clear()
        x = self.to_gpu(self.frame)
        self.sim.wiggle(x, 0.5)
        self.sim.wiggle(x, 0.5)
        self.sim.distortion(x, 0.5)
        self.assertEqual(len(self.sim.cache), 2)
        self.sim.wiggle(x, 0.5004)
        self.sim.wiggle(self.to_gpu(self.frames["random_84x84"]), 0.5)
        self.assertEqual(len(self.sim.cache), 4)


if __name__ == "__main__":
    unittest.main()
