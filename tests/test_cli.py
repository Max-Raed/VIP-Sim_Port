"""Regression tests for the vipsim_filters command line."""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "vipsim_filters.py"
sys.path.insert(0, str(ROOT))
import vipsim_filters as vf  # noqa: E402


def _run(*args):
    return subprocess.run([sys.executable, str(CLI), *map(str, args)],
                          capture_output=True, text=True)


class TestCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = Path(cls.tmp.name) / "in.png"
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(cls.src)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_list_presets_needs_no_paths(self):
        r = _run("--list-presets")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(r.stdout.strip().splitlines()), len(vf.PRESETS))

    def test_every_filter_runs(self):
        for name in vf._FILTER_DISPATCH:
            with self.subTest(filter=name):
                if name == "teichopsia" and importlib.util.find_spec("skimage") is None:
                    self.skipTest("teichopsia needs scikit-image (vipsim[teichopsia])")
                dst = Path(self.tmp.name) / f"{name}.png"
                r = _run(self.src, dst, "--filter", name)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertTrue(dst.is_file())

    def test_missing_paths_is_a_usage_error(self):
        r = _run("--filter", "blur")
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("Traceback", r.stderr)

    def test_unsupported_axis_is_a_usage_error(self):
        r = _run(self.src, Path(self.tmp.name) / "dv.png",
                 "--filter", "double_vision", "--axis", "both")
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
