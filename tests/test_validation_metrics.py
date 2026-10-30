"""The SSIM_blur pre-blur must be one shared value, and the docs must quote it."""
import importlib
import importlib.util
import inspect
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation_scripts"))

from validation_scripts.metric_constants import SSIM_BLUR_SIGMA  # noqa: E402

BURST_VALIDATORS = ("validate_flickering_stars_burst", "validate_nystagmus_burst",
                    "validate_wiggle_burst")


class TestSsimBlurSigma(unittest.TestCase):
    def test_abs_diff_uses_shared_sigma(self):
        src = inspect.getsource(importlib.import_module("validation_scripts.abs_diff"))
        self.assertIn("SSIM_BLUR_SIGMA", src)
        self.assertNotRegex(src, r"gaussian_filter\([^)]*sigma=\(\s*\d")

    @unittest.skipIf(importlib.util.find_spec("scipy") is None
                     or importlib.util.find_spec("skimage") is None,
                     "burst validators need scipy and scikit-image")
    def test_burst_validators_use_shared_sigma(self):
        for name in BURST_VALIDATORS:
            with self.subTest(module=name):
                mod = importlib.import_module(name)
                default = inspect.signature(mod.ssim_blur).parameters["sigma"].default
                self.assertEqual(default, SSIM_BLUR_SIGMA)

    def test_docs_quote_the_sigma_in_use(self):
        pattern = re.compile(r"SSIM_blur[^\n]*?σ\s*=\s*([0-9.]+)")
        for doc in ("readme.md", "vipsim_equivalence_report.md"):
            with self.subTest(doc=doc):
                found = pattern.findall((ROOT / doc).read_text(encoding="utf-8"))
                self.assertTrue(found, f"{doc} does not state the SSIM_blur sigma")
                for value in found:
                    self.assertEqual(float(value), SSIM_BLUR_SIGMA)

    def test_readme_commands_point_at_existing_scripts(self):
        text = (ROOT / "readme.md").read_text(encoding="utf-8")
        for script in re.findall(r"python\s+(\S+\.py)", text):
            with self.subTest(script=script):
                self.assertTrue((ROOT / script).is_file(), script)


if __name__ == "__main__":
    unittest.main()
