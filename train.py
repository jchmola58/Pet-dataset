"""Tréning baseline CNN na Oxford-IIIT Pet.

Spustenie (Windows):
    python train.py --img-dir "C:\\Users\\jchmo\\Desktop\\archive (2)\\images" --epochs 15
"""
import argparse
import json
import random
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import get_dataloaders  # noqa: E402
from model import SimpleCNN  # noqa: E402

DEFAULT_DIR = r"C:\Users\jchmo\Desktop\archive (2)\images"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epocha")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epocha")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", default=DEFAULT_DIR)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zariadenie: {device}")

    train_dl, val_dl, test_dl, classes = get_dataloaders(
        args.img_dir, args.img_size, args.batch_size, args.seed)
    print(f"Tried: {len(classes)} | train {len(train_dl.dataset)} | "
          f"val {len(val_dl.dataset)} | test {len(test_dl.dataset)}")

    model = SimpleCNN(num_classes=len(classes)).to(device)
    print(f"Parametre: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step()

        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(v)

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), "best_baseline.pt")

        print(f"Epocha {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - t0:.0f}s")

    # finálne vyhodnotenie najlepšieho modelu na teste (použi len raz!)
    model.load_state_dict(torch.load("best_baseline.pt", map_location=device))
    te_loss, te_acc = run_epoch(model, test_dl, criterion, device)
    print(f"\nTEST | loss {te_loss:.3f} | accuracy {te_acc:.3f}")

    history["test_acc"] = te_acc
    with open("history_baseline.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, "curves_baseline.png")
    print("Uložené: best_baseline.pt, history_baseline.json, curves_baseline.png")


if __name__ == "__main__":
    main()
