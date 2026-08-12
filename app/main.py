"""FastAPI serving for the ECG 1D-CNN.

Mirrors the k8s-fastapi-starter pattern: load the state_dict once, accept a
187-sample beat, return the predicted class. Model path is configurable via
the MODEL_PATH env var so tests/containers can point at their own weights.
"""

import os
import sys

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# make src/ importable whether run from repo root or inside the container
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ECG1DCNN

MODEL_PATH = os.getenv("MODEL_PATH", "ecg_1dcnn.pth")
CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_NAMES = {
    "N": "normal", "S": "supraventricular", "V": "ventricular",
    "F": "fusion", "Q": "unknown",
}

app = FastAPI(title="ecg-1dcnn")
_model = None


def get_model():
    """Load the model once and cache it (lazy, so the app starts without weights)."""
    global _model
    if _model is None:
        m = ECG1DCNN(n_classes=5)
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        _model = m
    return _model


class Beat(BaseModel):
    signal: list[float]  # one heartbeat, length 187


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(beat: Beat):
    if len(beat.signal) != 187:
        raise HTTPException(status_code=422, detail=f"signal must be length 187, got {len(beat.signal)}")
    try:
        model = get_model()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"model weights not found at {MODEL_PATH}")

    x = torch.tensor(beat.signal, dtype=torch.float32).view(1, 1, 187)
    with torch.no_grad():                      # inference: no gradient tracking
        probs = torch.softmax(model(x), dim=1)
        cls = int(probs.argmax(1))
        confidence = float(probs[0, cls])

    code = CLASSES[cls]
    return {"class_id": cls, "class_code": code, "class_name": CLASS_NAMES[code],
            "confidence": round(confidence, 4)}
