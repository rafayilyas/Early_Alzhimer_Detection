"""
Improved Training Script for Alzheimer's MRI Classification
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    DataLoader,
    Subset,
    WeightedRandomSampler,
)

from torchvision import datasets, transforms

from models.cnn_classifier import AlzheimerCNN


# =========================================================
# CONFIG
# =========================================================

SEED = 42
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 5

FULL_DATA_DIR = Path("data/raw/MRI/Alzhiemer")

SAVE_DIR = Path("models/saved")
PLOTS_DIR = Path("evaluation/plots")

HISTORY_PATH = PLOTS_DIR / "cnn_history.json"
BEST_MODEL_PATH = SAVE_DIR / "cnn_best.pth"

CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
]


# =========================================================
# SEED
# =========================================================

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# TRANSFORMS
# =========================================================

def build_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05)
        ),

        transforms.Grayscale(num_output_channels=1),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.Grayscale(num_output_channels=1),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        ),
    ])

    return train_transform, eval_transform


# =========================================================
# STRATIFIED SPLIT
# =========================================================

def stratified_split(dataset):

    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        stratify=targets,
        random_state=SEED,
    )

    temp_targets = targets[temp_indices]

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        stratify=temp_targets,
        random_state=SEED,
    )

    return train_indices, val_indices, test_indices


# =========================================================
# SAMPLER
# =========================================================

def make_weighted_sampler(targets):

    class_counts = np.bincount(
        targets,
        minlength=NUM_CLASSES
    )

    class_weights = 1.0 / class_counts

    sample_weights = class_weights[targets]

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


# =========================================================
# CLASS WEIGHTS
# =========================================================

def compute_class_weights(targets, device):

    class_counts = np.bincount(
        targets,
        minlength=NUM_CLASSES
    )

    class_weights = (
        len(targets)
        / (NUM_CLASSES * class_counts)
    )

    return torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device
    )


# =========================================================
# DATALOADERS
# =========================================================

def build_dataloaders():

    if not FULL_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {FULL_DATA_DIR}"
        )

    train_transform, eval_transform = build_transforms()

    base_dataset = datasets.ImageFolder(
        FULL_DATA_DIR
    )

    print("\nDetected Classes:")
    print(base_dataset.classes)

    if len(base_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes "
            f"but found {len(base_dataset.classes)}"
        )

    train_idx, val_idx, test_idx = stratified_split(
        base_dataset
    )

    targets = np.array(base_dataset.targets)

    # Print distributions
    print("\nClass Distribution:")

    for split_name, split_indices in [
        ("TRAIN", train_idx),
        ("VAL", val_idx),
        ("TEST", test_idx),
    ]:

        counts = np.bincount(
            targets[split_indices],
            minlength=NUM_CLASSES
        )

        print(f"\n{split_name}")

        for class_name, count in zip(CLASS_NAMES, counts):
            print(f"{class_name}: {count}")

    # Datasets
    train_dataset = datasets.ImageFolder(
        FULL_DATA_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        FULL_DATA_DIR,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        FULL_DATA_DIR,
        transform=eval_transform
    )

    train_subset = Subset(
        train_dataset,
        train_idx.tolist()
    )

    val_subset = Subset(
        val_dataset,
        val_idx.tolist()
    )

    test_subset = Subset(
        test_dataset,
        test_idx.tolist()
    )

    train_targets = targets[train_idx]

    train_sampler = make_weighted_sampler(
        train_targets
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader, train_targets


# =========================================================
# TRAIN
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    running_loss = 0.0

    all_preds = []
    all_targets = []

    total_samples = 0

    for images, targets in loader:

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():

            logits, _ = model(images)

            loss = criterion(
                logits,
                targets
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item() * images.size(0)

        total_samples += images.size(0)

        preds = logits.argmax(dim=1)

        all_preds.extend(
            preds.detach().cpu().numpy()
        )

        all_targets.extend(
            targets.detach().cpu().numpy()
        )

    epoch_loss = running_loss / total_samples

    epoch_acc = accuracy_score(
        all_targets,
        all_preds
    )

    return epoch_loss, epoch_acc


# =========================================================
# EVALUATE
# =========================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_probs = []
    all_preds = []
    all_targets = []

    total_samples = 0

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(device)
            targets = targets.to(device)

            logits, _ = model(images)

            loss = criterion(
                logits,
                targets
            )

            probs = torch.softmax(
                logits,
                dim=1
            )

            running_loss += loss.item() * images.size(0)

            total_samples += images.size(0)

            all_probs.append(
                probs.cpu().numpy()
            )

            preds = logits.argmax(dim=1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

    epoch_loss = running_loss / total_samples

    epoch_acc = accuracy_score(
        all_targets,
        all_preds
    )

    probs_array = np.concatenate(
        all_probs,
        axis=0
    )

    return (
        epoch_loss,
        epoch_acc,
        np.array(all_targets),
        np.array(all_preds),
        probs_array,
    )


# =========================================================
# PLOTS
# =========================================================

def plot_history(history):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Val Loss"
    )

    plt.legend()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")

    # Accuracy
    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train Acc"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Val Acc"
    )

    plt.legend()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "training_curves.png",
        dpi=300
    )

    plt.close()


# =========================================================
# CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()


# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(history):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)


# =========================================================
# SAVE MODEL
# =========================================================

def save_best_model(
    model,
    optimizer,
    epoch,
    val_acc
):

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_acc": val_acc,
        },
        BEST_MODEL_PATH
    )


# =========================================================
# FINAL METRICS
# =========================================================

def print_final_metrics(
    y_true,
    y_pred,
    y_prob
):

    acc = accuracy_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("\n========== TEST RESULTS ==========")

    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:\n")

    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    ))

    print("\nAUC ROC Scores:")

    y_true_onehot = np.eye(NUM_CLASSES)[y_true]

    for i, class_name in enumerate(CLASS_NAMES):

        try:

            auc = roc_auc_score(
                y_true_onehot[:, i],
                y_prob[:, i]
            )

            print(f"{class_name}: {auc:.4f}")

        except:
            print(f"{class_name}: Undefined")


# =========================================================
# MAIN
# =========================================================

def main():

    seed_everything()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing Device: {device}")

    train_loader, val_loader, test_loader, train_targets = \
        build_dataloaders()

    model = AlzheimerCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    class_weights = compute_class_weights(
        train_targets,
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=3,
        factor=0.5
    )

    scaler = torch.cuda.amp.GradScaler()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device
        )

        val_loss, val_acc, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            patience_counter = 0

            save_best_model(
                model,
                optimizer,
                epoch,
                val_acc
            )

            print("Best model saved.")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")

            break

    # Save training outputs
    save_history(history)

    plot_history(history)

    # Load best model
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Test evaluation
    test_loss, test_acc, y_true, y_pred, y_prob = \
        evaluate(
            model,
            test_loader,
            criterion,
            device
        )

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    print(f"Test Loss: {test_loss:.4f}")

    print_final_metrics(
        y_true,
        y_pred,
        y_prob
    )

    save_confusion_matrix(
        y_true,
        y_pred
    )

    print("\nTraining Complete.")


if __name__ == "__main__":
    main()