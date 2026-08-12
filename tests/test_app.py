"""Hermetic API tests: create a throwaway model file, then exercise the endpoints.

No trained weights needed - we save a fresh (random) ECG1DCNN state_dict so the
test runs anywhere (CI included).
"""

import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "src"))
from model import ECG1DCNN

# write a dummy model file and point the app at it BEFORE importing the app
_tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
torch.save(ECG1DCNN(n_classes=5).state_dict(), _tmp.name)
os.environ["MODEL_PATH"] = _tmp.name

sys.path.append(os.path.join(ROOT, "app"))
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok():
    r = client.post("/predict", json={"signal": [0.0] * 187})
    assert r.status_code == 200
    body = r.json()
    assert body["class_id"] in range(5)
    assert body["class_code"] in ["N", "S", "V", "F", "Q"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_wrong_length():
    r = client.post("/predict", json={"signal": [0.0] * 50})
    assert r.status_code == 422
