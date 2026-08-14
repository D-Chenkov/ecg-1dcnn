"""Train the 1D-CNN on the MIT-BIH heartbeat CSV, logging to MLflow.

Controlled runs: fixed seed + a train/val split so weighting schemes can be
compared on VALIDATION (the test set is touched only once, in evaluate.py).
Class weights are computed from the TRAIN split only (no val/test leakage).

Run (compare schemes with the same seed, changing only --weighting):
    python src/train.py --epochs 10 --weighting none
    python src/train.py --epochs 10 --weighting inverse
    python src/train.py --epochs 10 --weighting sqrt_inverse
    python src/train.py --epochs 10 --weighting class_balanced
Pick the winner by best_val_macro_f1 in MLflow, THEN run evaluate.py once on test.
"""

import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import mlflow

from dataset import ECGBeatDataset   # src/ is on sys.path when run as python src/train.py
from model import ECG1DCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y, scheme, n_classes=5, beta=0.999):
    """Compute per-class loss weights. Normalized so schemes are comparable
    (a global constant cancels in weighted-mean CE anyway - that is why raw
    inverse-freq 1/n and N/n are the SAME objective)."""
    counts = np.bincount(y, minlength=n_classes).astype("float64")
    if scheme == "none":
        w = np.ones(n_classes)
    elif scheme == "inverse":
        w = counts.sum() / counts                    # proportional to 1/n_c
    elif scheme == "sqrt_inverse":
        w = np.sqrt(counts.sum() / counts)           # softer than inverse
    elif scheme == "class_balanced":                 # Cui et al. 2019 (arXiv 1901.05555)
        eff_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / eff_num
    else:
        raise ValueError(f"unknown weighting scheme: {scheme}")
    w = w / w.sum() * n_classes                      # normalize for comparability
    return torch.tensor(w, dtype=torch.float32)


def val_macro_f1(model, dl, device):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
            ys += list(y)
    return f1_score(ys, preds, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/mitbih_train.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weighting", default="class_balanced",
                    choices=["none", "inverse", "sqrt_inverse", "class_balanced"])
    ap.add_argument("--beta", type=float, default=0.999)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ecg_1dcnn.pth")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = ECGBeatDataset(args.train_csv)
    n_val = int(args.val_frac * len(full))
    train_ds, val_ds = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512)

    # weights from the TRAIN split only
    train_y = full.y[train_ds.indices]
    weights = class_weights(train_y, args.weighting, beta=args.beta).to(device)

    model = ECG1DCNN(n_classes=5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    mlflow.set_experiment("ecg-1dcnn")
    with mlflow.start_run(run_name=args.weighting):
        mlflow.log_params(vars(args))
        best_val, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)
            train_loss = running / len(train_ds)
            vf1 = val_macro_f1(model, val_dl, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_macro_f1", vf1, step=epoch)
            print(f"epoch {epoch:2d}  loss={train_loss:.4f}  val_macro_f1={vf1:.4f}")
            if vf1 > best_val:                          # keep best-on-val checkpoint
                best_val = vf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        mlflow.log_metric("best_val_macro_f1", best_val)
        torch.save(best_state, args.out)               # save state_dict (best on val)
        mlflow.log_artifact(args.out)
        print(f"[{args.weighting}] best val macro-F1 = {best_val:.4f}  ->  saved {args.out}")


if __name__ == "__main__":
    main()
