"""CVD must run in linear light, like upstream VIP-Sim.

Upstream is a Linear-colour-space Unity project (m_ActiveColorSpace: 1), so
myRecolour.shader applies the Machado matrices to linear RGB.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vipsim_filters as vf  # noqa: E402
from test_rounding import GpuParityCase  # noqa: E402

UNITY_REFS = ROOT / "vipsim_assets" / "unity_refs"


class TestCvdLinearLight(unittest.TestCase):
    def _px(self, rgb, cvd_type, severity=1.0):
        img = Image.new("RGB", (1, 1), rgb)
        return vf.apply_cvd(img, cvd_type, severity).getpixel((0, 0))

    def test_pure_red_matches_linear_space_transform(self):
        for cvd_type, expected in [("protanomaly", (109, 95, 0)),
                                   ("deuteranomaly", (163, 144, 0))]:
            with self.subTest(cvd_type=cvd_type):
                got = self._px((255, 0, 0), cvd_type)
                self.assertTrue(all(abs(g - e) <= 1 for g, e in zip(got, expected)),
                                f"{cvd_type}: got {got}, expected {expected}")

    def test_matches_unity_capture(self):
        base = UNITY_REFS / "baseline.png"
        ref = UNITY_REFS / "cvd_deutan_sev06.png"
        if not (base.is_file() and ref.is_file()):
            self.skipTest("Unity reference captures not present")
        # Feed the image Unity actually rendered (its input texture is
        # block-compressed), so only the filter math is compared.
        out = vf.apply_cvd(Image.open(base).convert("RGB"), "deuteranomaly", 0.6)
        mae = np.abs(np.asarray(out, np.float32)
                     - np.asarray(Image.open(ref).convert("RGB"), np.float32)).mean()
        self.assertLess(mae, 0.2)

    def test_accepts_non_rgb_input(self):
        img = Image.fromarray(np.full((5, 8), 128, np.uint8), "L")
        out = vf.apply_cvd(img, "deuteranomaly", 0.5)
        self.assertEqual((out.size, out.mode), ((8, 5), "RGB"))


class TestGpuCvdMatchesCpu(GpuParityCase):
    def test_cvd(self):
        self.assert_close(self.gpu("cvd", severity=0.6),
                          vf.apply_cvd(Image.fromarray(self.frame), "deuteranomaly", 0.6))


if __name__ == "__main__":
    unittest.main()
