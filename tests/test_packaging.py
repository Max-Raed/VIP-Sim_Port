"""The importable package, its bundled asset, and the in-repo aliases."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import vipsim  # noqa: E402
import vipsim_filters  # noqa: E402
from vipsim import filters  # noqa: E402


class TestPackage(unittest.TestCase):
    def test_alias_is_the_package_module(self):
        self.assertIs(vipsim_filters, filters)

    def test_public_api_covers_every_filter(self):
        for name in filters._FILTER_DISPATCH:
            with self.subTest(filter=name):
                self.assertIs(getattr(vipsim, f"apply_{name}"), filters._FILTER_DISPATCH[name])

    def test_overlay_asset_ships_inside_the_package(self):
        path = filters._MACULAR_OVERLAY_PATH
        self.assertTrue(path.is_file(), path)
        self.assertEqual(path.parent.parent, Path(filters.__file__).parent)
        self.assertIsNotNone(filters._load_macular_overlay(32, 64))

    def test_import_does_not_pull_torch(self):
        code = "import sys, vipsim; print('torch' in sys.modules)"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=ROOT)
        self.assertEqual(r.stdout.strip(), "False", r.stderr)

    def test_module_cli(self):
        r = subprocess.run([sys.executable, "-m", "vipsim", "--list-presets"],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(r.stdout.strip().splitlines()), len(vipsim.PRESETS))

    def test_blur_passes_a_plain_float_radius(self):
        # Pillow 13 compares the radius with a tuple, which fails for numpy
        # scalars ("truth value of an array ... is ambiguous").
        from unittest import mock
        seen = []
        real = filters.ImageFilter.GaussianBlur

        def spy(radius=2):
            seen.append(type(radius))
            return real(radius=radius)

        with mock.patch.object(filters.ImageFilter, "GaussianBlur", spy):
            vipsim.apply_blur(Image.new("RGB", (8, 8)), 0.3)
        self.assertEqual(seen, [float])

    def test_missing_teichopsia_extra_names_it(self):
        from unittest import mock
        blocked = dict.fromkeys(["skimage", "skimage.color", "skimage.draw"])
        with mock.patch.dict(sys.modules, blocked):
            with self.assertRaisesRegex(ImportError, r"vipsim\[teichopsia\]"):
                vipsim.apply_teichopsia(Image.new("RGB", (8, 8)), 0.5)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_missing_gym_extra_names_it(self):
        # Without gymnasium/gym the wrapper's base is `object`, which used to
        # fail with "object.__init__() takes exactly one argument".
        from unittest import mock
        from vipsim import gpu
        with mock.patch.object(gpu, "_Wrapper", object):
            with self.assertRaisesRegex(ImportError, r"vipsim\[gym\]"):
                gpu.VisualImpairmentWrapper(object(), [], device="cpu")

    def test_field_loss_uses_overlay(self):
        img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 128, 3), np.uint8))
        with_overlay = np.asarray(vipsim.apply_field_loss(img, 0.8))
        radial = np.asarray(vipsim.apply_field_loss(img, 0.8, use_overlay_asset=False))
        self.assertFalse(np.array_equal(with_overlay, radial))


if __name__ == "__main__":
    unittest.main()
