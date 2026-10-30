"""Basic API tests. Run after training:  python -m unittest test_app -v"""
import unittest

from app import app


class SpamShieldTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_spam(self):
        r = self.client.post("/predict", json={
            "message": "WINNER!! You have won a free prize. Call 09061701461 now to claim your cash reward"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["label"], "spam")

    def test_ham(self):
        r = self.client.post("/predict", json={"message": "Hey, are we still meeting for lunch tomorrow?"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["label"], "ham")

    def test_bad_input(self):
        self.assertEqual(self.client.post("/predict", json={}).status_code, 400)

    def test_batch(self):
        r = self.client.post("/predict/batch", json={"messages": ["hi there", "Free entry! Text WIN to 80086 now"]})
        self.assertEqual(len(r.get_json()["results"]), 2)


if __name__ == "__main__":
    unittest.main()
