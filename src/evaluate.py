"""Evaluate the trained 1D-CNN: per-class precision/recall/F1 + confusion matrix.

Accuracy alone is misleading on this imbalanced dataset, so report per class.

Run from the repo root:
    python src/evaluate.py
"""

import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset import ECGBeatDataset
from model import ECG1DCNN

CLASSES = ["N (normal)", "S (supravent.)", "V (ventric.)", "F (fusion)", "Q (unknown)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="data/mitbih_test.csv")
    ap.add_argument("--weights", default="ecg_1dcnn.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ECGBeatDataset(args.test_csv)
    dl = DataLoader(ds, batch_size=512)

    model = ECG1DCNN(n_classes=5).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    preds, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            out = model(x.to(device))
            preds += out.argmax(1).cpu().tolist()
            ys += list(y)

    print(classification_report(ys, preds, target_names=CLASSES, digits=3))
    print("confusion matrix (rows = true, cols = pred):")
    print(confusion_matrix(ys, preds))


if __name__ == "__main__":
    main()
