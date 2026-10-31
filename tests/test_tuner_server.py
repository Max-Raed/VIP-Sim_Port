"""Regression tests for the profile tuner's HTTP endpoints (path and origin checks)."""
import http.client
import json
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation_scripts"))
import profile_tuner as pt  # noqa: E402


class TestTunerServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), pt.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.outside = Path(cls.tmp.name)
        (cls.outside / "secret.json").write_text('{"secret": 1}')
        Image.new("RGB", (4, 4), (255, 0, 0)).save(cls.outside / "secret.png")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def _request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        hdrs = {"Content-Type": "application/json"}
        hdrs.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        out = resp.status, resp.read()
        conn.close()
        return out

    def test_profile_read_cannot_leave_profile_dir(self):
        status, body = self._request("GET", f"/profile?id={self.outside / 'secret'}")
        self.assertNotEqual(status, 200)
        self.assertNotIn(b"secret", body)
        status, _ = self._request("GET", "/profile?id=../vipsim_profiles_v2/p1")
        self.assertNotEqual(status, 200)

    def test_ref_read_cannot_leave_ref_dir(self):
        status, _ = self._request("GET", f"/ref?id={self.outside / 'secret'}")
        self.assertNotEqual(status, 200)

    def test_save_cannot_leave_profile_dir(self):
        target = self.outside / "evil"
        status, _ = self._request("POST", "/save", {"id": str(target), "filters": []})
        self.assertEqual(status, 400)
        self.assertFalse(target.with_suffix(".json").exists())

    def test_saveas_rejects_bad_base_id(self):
        status, _ = self._request("POST", "/saveas", {
            "baseId": str(self.outside / "secret"), "newId": "p9zz_audit_test",
            "filters": []})
        self.assertEqual(status, 400)
        self.assertFalse((pt.PROFILE_DIR / "p9zz_audit_test.json").exists())

    def test_cross_origin_post_is_rejected(self):
        status, _ = self._request("POST", "/render", {"id": "p1", "filters": []},
                                  {"Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_foreign_host_is_rejected(self):
        status, _ = self._request("GET", "/list", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)

    def test_same_origin_requests_still_work(self):
        status, body = self._request("GET", "/list")
        self.assertEqual(status, 200)
        self.assertIn("p1", json.loads(body))
        status, _ = self._request("GET", "/profile?id=p1")
        self.assertEqual(status, 200)
        status, _ = self._request("POST", "/render", {"id": "p1", "filters": []},
                                  {"Origin": f"http://localhost:{self.port}"})
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
