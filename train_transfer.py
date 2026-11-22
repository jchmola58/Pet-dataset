"""Transfer learning s ResNet18 (predtrénovaný na ImageNet).

Dva režimy:
  1) frozen   - backbone zamrazený, trénuje sa len nová klasifikačná vrstva (rýchle, ide aj na CPU)
  2) finetune - trénuje sa celá sieť s nízkym learning rate (lepšie na GPU)

Odporúčaný postup:
    python train_transfer.py --mode frozen --epochs 5
    python train_transfer.py --mode finetune --epochs 5 --init best_frozen.pt
"""
import argparse
import json
import time

import torch
import torch.nn as nn
from torchvision import models

from data import get_dataloaders
from train import DEFAULT_DIR, plot_history, run_epoch, set_seed


class TransferResNet(nn.Module):
    def __init__(self, num_classes, freeze_backbone):
        super().__init__()
        self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for name, p in self.net.named_parameters():
                if not name.startswith("fc"):
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # pri zamrazenom backbone držíme aj BatchNorm vrstvy v eval móde,
        # inak by sa im menili štatistiky, aj keď sa váhy netrénujú
        if self.freeze_backbone and mode:
            for name, module in self.net.named_children():
                if name != "fc":
                    module.eval()
        return self

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", default=DEFAULT_DIR)
    p.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None,
                   help="predvolene 1e-3 pre frozen, 1e-4 pre finetune")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=160,
                   help="160 na CPU, 224 na GPU")
    p.add_argument("--init", default=None,
                   help="cesta k checkpointu (napr. best_frozen.pt) pre finetune")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    lr = args.lr or (1e-3 if args.mode == "frozen" else 1e-4)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zariadenie: {device} | režim: {args.mode} | lr: {lr}")

    train_dl, val_dl, test_dl, classes = get_dataloaders(
        args.img_dir, args.img_size, args.batch_size, args.seed)

    model = TransferResNet(len(classes), freeze_backbone=(args.mode == "frozen"))
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        print(f"Načítané váhy z {args.init}")
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trénovateľné parametre: {sum(p.numel() for p in trainable):,} / "
          f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val = 0.0
    ckpt = f"best_{args.mode}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step()

        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(v)

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), ckpt)

        print(f"Epocha {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - t0:.0f}s")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    te_loss, te_acc = run_epoch(model, test_dl, criterion, device)
    print(f"\nTEST | loss {te_loss:.3f} | accuracy {te_acc:.3f}")

    history["test_acc"] = te_acc
    with open(f"history_{args.mode}.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, f"curves_{args.mode}.png")
    print(f"Uložené: {ckpt}, history_{args.mode}.json, curves_{args.mode}.png")


if __name__ == "__main__":
    main()
