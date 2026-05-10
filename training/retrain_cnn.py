"""Retrain MRI CNN with transfer learning and class balancing.
Saves best checkpoint and metadata for backend consumption.

Fixes applied (v2):
  1. Sampler-only balancing  — removed weight= from CrossEntropyLoss to avoid
     double-correcting class imbalance.
  2. Deeper unfreezing       — layer3 + layer4 + fc are now trainable; only the
     very early layers (conv1, bn1, layer1, layer2) stay frozen so the backbone
     can adapt to the MRI domain.
  3. Dual-output architecture — training now uses MRIResNet18 (returns logits AND
     an embedding vector) so the saved checkpoint matches the backend exactly and
     strict=False is no longer papering over a shape mismatch.
  4. Longer schedule          — default epochs raised to 25; CosineAnnealingLR
     (stepped once per epoch) replaces the step-count-sensitive OneCycleLR;
     differential learning rates per layer group.

Usage:
    python training/retrain_cnn.py \
        --data-dir data/raw/MRI/Alzhiemer/combined_images \
        --epochs 25 --batch 32 --lr 3e-4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=str, default="data/raw/MRI/Alzhiemer/combined_images")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch",  type=int, default=16,
                   help="Batch size. 16 is a good default for CPU; raise to 32+ if using GPU.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", type=str, default="notebooks/models/saved/cnn_best_retrained.pth")
    p.add_argument("--save-copy", action="store_true", help="Also copy checkpoint to models/saved/")
    default_workers = 0 if sys.platform == "win32" else 2
    p.add_argument("--workers", type=int, default=default_workers,
                   help="DataLoader workers. Defaults to 0 on Windows to avoid multiprocessing issues.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Cap per-class sample count (0 = use all). Set ~800 if one class still dominates.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Architecture — matches backend MRIResNet18 exactly (logits, embedding)
# ---------------------------------------------------------------------------

class MRIResNet18(nn.Module):
    """ResNet-18 that returns (logits, embedding) — identical to the backend."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            print("Warning: could not load pretrained weights — using random init.")
            backbone = models.resnet18(weights=None)

        # Re-expose named layer groups so we can freeze/unfreeze them selectively.
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        self.embedding_dim = backbone.fc.in_features   # 512 for ResNet-18
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        embedding = torch.flatten(x, 1)
        logits = self.fc(embedding)
        return logits, embedding


def build_model(num_classes: int, device: torch.device) -> MRIResNet18:
    """
    Freeze only the very early layers (conv1, bn1, layer1, layer2).
    layer3, layer4, and fc are trainable — enough capacity for MRI domain shift.
    """
    model = MRIResNet18(num_classes=num_classes, pretrained=True).to(device)

    frozen_prefixes = ("conv1", "bn1", "layer1", "layer2")
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in frozen_prefixes):
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_datasets(
    data_dir: str,
    val_split: float = 0.2,
    img_size: int = 160,
    max_samples_per_class: int = 0,
):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(data_dir)
    targets = np.array([s[1] for s in full.samples])
    num_classes = len(full.classes)

    train_idx: list[int] = []
    val_idx:   list[int] = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        if max_samples_per_class and len(idx) > max_samples_per_class:
            idx = idx[:max_samples_per_class]
        n_val = max(1, int(len(idx) * val_split))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    train_dataset = Subset(datasets.ImageFolder(data_dir, transform=train_tf), train_idx)
    val_dataset   = Subset(datasets.ImageFolder(data_dir, transform=val_tf),   val_idx)

    # FIX 1: WeightedRandomSampler for class balance — NO weighted loss alongside it.
    train_targets = [full.targets[i] for i in train_idx]
    class_counts  = np.array([train_targets.count(i) for i in range(num_classes)])
    sample_weights = [1.0 / class_counts[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    print("Class distribution in training split:")
    for i, cls in enumerate(full.classes):
        print(f"  {cls}: {class_counts[i]} samples")

    return train_dataset, val_dataset, sampler, full.classes


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: MRIResNet18,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0
    n_batches = len(loader)
    print_every = max(1, n_batches // 10)   # print ~10 times per epoch

    for batch_idx, (xb, yb) in enumerate(loader, 1):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        correct      += (logits.argmax(dim=1) == yb).sum().item()
        total        += xb.size(0)

        if batch_idx % print_every == 0 or batch_idx == n_batches:
            avg_loss = running_loss / total
            avg_acc  = correct / total
            print(
                f"  Epoch {epoch}/{total_epochs} | "
                f"batch {batch_idx}/{n_batches} | "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f}",
                flush=True,
            )

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: MRIResNet18,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list, list]:
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits, _ = model(xb)
        ps.extend(logits.argmax(dim=1).cpu().tolist())
        ys.extend(yb.tolist())
    return ys, ps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    assert os.path.exists(args.data_dir), f"Data directory not found: {args.data_dir}"

    train_ds, val_ds, sampler, classes = make_datasets(
        args.data_dir,
        val_split=args.val_split,
        img_size=args.img_size,
        max_samples_per_class=args.max_samples_per_class,
    )
    print("Classes:", classes)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,   num_workers=args.workers, pin_memory=pin)

    model = build_model(len(classes), device)

    # FIX 1 (cont.): no weight= argument — sampler already balances batches.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # FIX 4: differential LRs per group; CosineAnnealingLR stepped once per epoch.
    optimizer = optim.AdamW(
        [
            {"params": model.layer3.parameters(), "lr": args.lr * 0.1},
            {"params": model.layer4.parameters(), "lr": args.lr * 0.3},
            {"params": model.fc.parameters(),     "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_acc  = 0.0
    best_path = Path(args.out)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)

        ys, ps = validate(model, val_loader, device)
        overall_acc = float(np.mean(np.array(ys) == np.array(ps))) if ys else 0.0

        # Step scheduler once per epoch (CosineAnnealingLR)
        scheduler.step()

        report = classification_report(ys, ps, target_names=classes, zero_division=0, output_dict=True)
        elapsed = time.time() - t0
        lrs = [pg["lr"] for pg in optimizer.param_groups]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_acc={overall_acc:.4f} | "
            f"lr={lrs[-1]:.2e} | {elapsed:.1f}s"
        )
        print("Per-class F1:")
        for cls in classes:
            print(f"  {cls}: precision={report[cls]['precision']:.3f}  recall={report[cls]['recall']:.3f}  f1={report[cls]['f1-score']:.3f}")

        if overall_acc > best_acc:
            best_acc = overall_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes":          classes,
                    "class_names":      classes,   # alias for backend compatibility
                    "overall_acc":      overall_acc,
                    "epoch":            epoch,
                    "args":             vars(args),
                },
                best_path,
            )
            print(f"  ✓ Saved new best to {best_path} (val_acc={overall_acc:.4f})")

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    ys, ps = validate(model, val_loader, device)
    final_acc = float(np.mean(np.array(ys) == np.array(ps))) if ys else 0.0
    print("\n=== Final evaluation ===")
    print(f"Overall accuracy: {final_acc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(ys, ps))
    print(classification_report(ys, ps, target_names=classes, zero_division=0))

    # ------------------------------------------------------------------
    # Optional copy to models/saved/ for backend pick-up
    # ------------------------------------------------------------------
    if args.save_copy:
        dest = Path("models/saved/cnn_best_retrained.pth")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_path, dest)
        print(f"Copied best checkpoint to {dest}")

    # Summary JSON next to checkpoint
    summary = {
        "best_path":  str(best_path),
        "best_acc":   float(best_acc),
        "final_acc":  float(final_acc),
        "classes":    classes,
        "epochs":     args.epochs,
    }
    summary_path = best_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()