"""Train the 1D-CNN on the MIT-BIH heartbeat CSV, logging to MLflow.

Run from the repo root:
    python src/train.py --epochs 10
"""

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import mlflow
import numpy as np

from dataset import ECGBeatDataset   # src/ is on sys.path when run as python src/train.py
from model import ECG1DCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/mitbih_train.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="ecg_1dcnn.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ECGBeatDataset(args.train_csv)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = ECG1DCNN(n_classes=5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    #Dataset is imbalanced; "fixing" here
    counts = np.bincount(ds.y, minlength=5)
    weights = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    mlflow.set_experiment("ecg-1dcnn")
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)
            avg = running / len(ds)
            mlflow.log_metric("train_loss", avg, step=epoch)
            print(f"epoch {epoch:2d}  train_loss={avg:.4f}")

        #State Dict, not whole model
        torch.save(model.state_dict(), args.out)
        mlflow.log_artifact(args.out)
        print(f"saved weights -> {args.out}")


if __name__ == "__main__":
    main()
