"""Train the 1D-CNN on the MIT-BIH heartbeat CSV, logging to MLflow.

Controlled runs: fixed seed + a train/val split so weighting schemes can be
compared on VALIDATION (test is touched only once, in evaluate.py). Class
weights are computed from the TRAIN split only (no val/test leakage).

Training loop features (PC-friendly):
  - early stopping on val macro-F1 (so you set an epoch CAP, not a guess)
  - mixed precision / AMP on CUDA (autocast + GradScaler) - big GPU speedup
  - ReduceLROnPlateau scheduler on val macro-F1
  - larger default batch size for the GPU

Run (example, on the PC):
    python src/train.py --epochs 40 --batch_size 512 --weighting class_balanced
    # early stopping usually stops well before 40; AMP is on by default on CUDA.
Pick the winner by best_val_macro_f1 in MLflow, THEN run evaluate.py once on test.
"""

import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
import mlflow

from dataset import ECGBeatDataset   # src/ is on sys.path when run as python src/train.py
from model import ECG1DCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(y, scheme, n_classes=5, beta=0.999):
    """Per-class loss weights, normalized (a global constant cancels in
    weighted-mean CE, so only the ratios matter)."""
    counts = np.bincount(y, minlength=n_classes).astype("float64")
    if scheme == "none":
        w = np.ones(n_classes)
    elif scheme == "inverse":
        w = counts.sum() / counts
    elif scheme == "sqrt_inverse":
        w = np.sqrt(counts.sum() / counts)
    elif scheme == "class_balanced":                 # Cui et al. 2019 (arXiv 1901.05555)
        eff_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / eff_num
    else:
        raise ValueError(f"unknown weighting scheme: {scheme}")
    w = w / w.sum() * n_classes
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
    ap.add_argument("--epochs", type=int, default=40, help="EPOCH CAP; early stopping usually ends sooner")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weighting", default="class_balanced",
                    choices=["none", "inverse", "sqrt_inverse", "class_balanced"])
    ap.add_argument("--beta", type=float, default=0.999)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=7, help="early-stop patience on val macro-F1")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="mixed precision on CUDA (--no-amp to disable)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ecg_1dcnn.pth")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"           # AMP only helps on GPU
    print(f"device={device}  amp={use_amp}")

    full = ECGBeatDataset(args.train_csv)
    n_val = int(args.val_frac * len(full))
    train_ds, val_ds = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=(device == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=1024, pin_memory=(device == "cuda"))

    train_y = full.y[train_ds.indices]                # weights from TRAIN split only
    weights = class_weights(train_y, args.weighting, beta=args.beta).to(device)

    model = ECG1DCNN(n_classes=5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    scaler = GradScaler(device, enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    mlflow.set_experiment("ecg-1dcnn")
    with mlflow.start_run(run_name=args.weighting):
        mlflow.log_params(vars(args))
        best_val, best_state, since_improve = -1.0, None, 0
        epochs_run = 0
        for epoch in range(args.epochs):
            epochs_run = epoch + 1
            model.train()
            running = 0.0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with autocast(device_type="cuda", enabled=use_amp):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                running += loss.item() * x.size(0)
            train_loss = running / len(train_ds)
            vf1 = val_macro_f1(model, val_dl, device)
            sched.step(vf1)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_macro_f1", vf1, step=epoch)
            mlflow.log_metric("lr", opt.param_groups[0]["lr"], step=epoch)
            print(f"epoch {epoch:2d}  loss={train_loss:.4f}  val_macro_f1={vf1:.4f}  lr={opt.param_groups[0]['lr']:.2e}")

            if vf1 > best_val:
                best_val = vf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= args.patience:
                    print(f"early stop @ epoch {epoch} (no val gain in {args.patience})")
                    break

        mlflow.log_metric("best_val_macro_f1", best_val)
        mlflow.log_metric("epochs_run", epochs_run)
        torch.save(best_state, args.out)               # save state_dict (best on val)
        mlflow.log_artifact(args.out)
        print(f"[{args.weighting}] best val macro-F1={best_val:.4f} in {epochs_run} epochs -> {args.out}")


if __name__ == "__main__":
    main()
