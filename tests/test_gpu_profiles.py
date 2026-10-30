"""Profiles and presets run on the GPU must reproduce their CPU renders.

validation_scripts/render_profiles_v2.py renders each vipsim_profiles_v2
profile (its "filters" stack, honouring "_disabled") on the Trafficscene,
framed and resized to the paper's panel.  VipSimGPU.apply_profile runs the
same stack on the GPU.  Each step, fed the CPU's image before it, must stay
within its filter's tolerance (vipsim.gpu.TWIN_TOLERANCE), and the whole GPU
chain within the loosest tolerance of its filters; every twin is bit-exact
where tested, so in practice both are equal.  The 24 presets
(VipSimGPU.apply_preset) are checked the same way end to end.
Runs on CUDA when available, else on the CPU; skipped without torch.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vipsim  # noqa: E402
from test_rounding import GpuParityCase  # noqa: E402
from validation_scripts.render_profiles_v2 import (  # noqa: E402
    PROFILE_DIR, REF_DIR, SCENE, apply_stack, scene_ref_framing)

PROFILES = [f"p{i}" for i in range(1, 8)]
HAVE_SKIMAGE = importlib.util.find_spec("skimage") is not None


class GpuChainCase(GpuParityCase):
    def assert_within(self, gpu, cpu, tolerance, msg):
        d = np.abs(gpu.astype(int) - np.asarray(cpu, int))
        self.assertLessEqual(d.max(), tolerance[0], msg)
        self.assertLessEqual(d.mean(), tolerance[1], msg)

    def chain_tolerance(self, chain):
        """The loosest (max, mean) tolerance of the filters in a chain."""
        tols = [self.vg.TWIN_TOLERANCE[name] for name, _ in chain]
        return max(t[0] for t in tols), max(t[1] for t in tols)

    def run_gpu(self, method, frame, *args):
        x = self.vg.frames_to_gpu(np.asarray(frame), self.sim.device)
        return self.vg.gpu_to_frames(getattr(self.sim, method)(x, *args))[0]


class TestGpuProfilesMatchCpuRenders(GpuChainCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not SCENE.is_file():
            raise unittest.SkipTest("Trafficscene.png missing")
        cls.scene = Image.open(SCENE).convert("RGB")

    def panel_input(self, pid):
        """render_profiles_v2's input: the scene framed like the paper's
        panel and resized to it."""
        with Image.open(REF_DIR / f"{pid}.png") as ref:
            size = ref.size
        return scene_ref_framing(self.scene, pid).resize(size, Image.LANCZOS)

    def test_every_profile(self):
        for pid in PROFILES:
            with self.subTest(profile=pid):
                profile = json.loads((PROFILE_DIR / f"{pid}.json").read_text())
                chain = self.vg.profile_chain(profile)
                steps = [s for s in profile["filters"] if not s.get("_disabled")]
                self.assertEqual([name for name, _ in chain], [s["filter"] for s in steps])
                img = self.panel_input(pid)
                # Step by step: each GPU filter on the CPU's image before it.
                cpu = img
                for (name, kwargs), step in zip(chain, steps):
                    nxt = apply_stack(cpu, [step])
                    gpu = self.run_gpu("apply_chain", cpu, [(name, kwargs)])
                    self.assert_within(gpu, nxt, self.vg.TWIN_TOLERANCE[name],
                                       f"{pid} step {name} {kwargs}")
                    cpu = nxt
                # End to end: the whole stack on the GPU.
                self.assert_within(self.run_gpu("apply_profile", img, profile), cpu,
                                   self.chain_tolerance(chain), f"{pid} end to end")

    def test_disabled_steps_are_skipped(self):
        steps = [{"filter": "blur", "severity": 0.3, "_disabled": True},
                 {"filter": "bcg", "severity": 0.5, "brightness": 0.8, "_note": "x"}]
        self.assertEqual(self.vg.profile_chain(steps), [("bcg", {"brightness": 0.8})])


class TestGpuPresetsMatchCpu(GpuChainCase):
    def test_every_preset(self):
        rng = np.random.default_rng(9)
        frame = rng.integers(0, 256, (61, 97, 3), dtype=np.uint8)
        for name, preset in vipsim.PRESETS.items():
            chain = preset["filters"]
            if not HAVE_SKIMAGE and any(f == "teichopsia" for f, _ in chain):
                continue                    # needs the teichopsia extra
            with self.subTest(preset=name):
                cpu = vipsim.apply_preset(Image.fromarray(frame), name)
                self.assert_within(self.run_gpu("apply_preset", frame, name), cpu,
                                   self.chain_tolerance(chain), name)


if __name__ == "__main__":
    unittest.main()
