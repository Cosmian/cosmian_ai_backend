import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))
from app import app


class ApiTest(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.app = app.test_client()
        self.header = {"Authorization": "Bearer JWT_TOKEN"}

    def test_health(self):
        res = self.app.get("/health")
        self.assertEqual(res.status_code, 200)

    def test_summarize_no_doc(self):
        res = self.app.post("/summarize", headers=self.header)
        self.assertEqual(res.status_code, 400)

    def test_summarize_short_text(self):
        res = self.app.post("/summarize", data={"doc": "hello"}, headers=self.header)
        self.assertEqual(res.status_code, 400)

    def test_summarize(self):
        res = self.app.post(
            "/summarize",
            data={"doc": "Hello " * 50},
            headers=self.header,
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("summary", res.json)

    def test_translate_no_doc(self):
        res = self.app.post("/translate", headers=self.header)
        self.assertEqual(res.status_code, 400)

    def test_translate_no_lang(self):
        res = self.app.post(
            "/translate",
            data={"doc": "Hello"},
            headers=self.header,
        )
        self.assertEqual(res.status_code, 400)

    def test_translate_wrong_lang(self):
        res = self.app.post(
            "/translate",
            data={"doc": "Hello", "src_lang": "Wrong", "tgt_lang": "Missing"},
            headers=self.header,
        )
        self.assertEqual(res.status_code, 400)

    def test_translate(self):
        res = self.app.post(
            "/translate",
            data={"doc": "Hello", "src_lang": "English", "tgt_lang": "French"},
            headers=self.header,
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("translation", res.json)


if __name__ == "__main__":
    unittest.main()
