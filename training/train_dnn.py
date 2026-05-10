"""
Training script for Alzheimer's clinical data DNN classification.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from imblearn.over_sampling import SMOTE
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The imbalanced-learn package is required for SMOTE oversampling. "
        "Install it with `pip install imbalanced-learn`."
    ) from exc

from models.dnn_tabular import AlzheimerDNN, FeatureImportanceAnalyzer, FEATURE_NAMES


SEED = 42
NUM_CLASSES = 4
N_SPLITS = 5
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
T_MAX = 30
DATA_PATH = Path("data/raw/clinical/alzheimers_dataset.csv")
SAVE_PATH = Path("models/saved/dnn_best.pth")
PLOT_PATH = Path("evaluation/plots/feature_importance.png")
CLASS_NAMES = ["NonDemented", "VeryMild", "Mild", "Moderate"]

NUMERIC_FEATURES = [
    "age",
    "education_years",
    "MMSE_score",
    "CDR_score",
    "eTIV",
    "nWBV",
    "ASF",
    "BMI",
    "depression_score",
    "sleep_hours",
    "physical_activity",
    "cholesterol",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
]

CATEGORICAL_FEATURES = ["gender", "smoking_history", "family_history", "diabetes"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClinicalTabularPreprocessor:
    """Impute missing values, encode categoricals, and scale numeric features."""

    def __init__(self, numeric_features: Sequence[str], categorical_features: Sequence[str]):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.feature_order = list(self.numeric_features) + list(self.categorical_features)
        self.numeric_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, str] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    @staticmethod
    def _normalize_category(value) -> str:
        return str(value)

    def fit(self, frame: pd.DataFrame) -> "ClinicalTabularPreprocessor":
        working = frame.copy()

        for column in self.numeric_features:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            self.numeric_medians[column] = float(working[column].median())
            working[column] = working[column].fillna(self.numeric_medians[column])

        for column in self.categorical_features:
            normalized = working[column].map(self._normalize_category)
            mode_value = normalized.mode(dropna=True).iloc[0]
            self.categorical_modes[column] = mode_value
            filled = normalized.fillna(mode_value)
            encoder = LabelEncoder()
            encoder.fit(filled)
            self.label_encoders[column] = encoder
            working[column] = encoder.transform(filled)

        self.scaler.fit(working[self.numeric_features].to_numpy(dtype=np.float32))
        self.is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ClinicalTabularPreprocessor must be fitted before transform().")

        working = frame.copy()

        for column in self.numeric_features:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            working[column] = working[column].fillna(self.numeric_medians[column])

        for column in self.categorical_features:
            normalized = working[column].map(self._normalize_category)
            filled = normalized.fillna(self.categorical_modes[column])
            working[column] = self.label_encoders[column].transform(filled)

        working[self.numeric_features] = self.scaler.transform(working[self.numeric_features].to_numpy(dtype=np.float32))
        return working[self.feature_order].to_numpy(dtype=np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def to_torch(self, x: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        frame = pd.DataFrame(x, columns=self.feature_order)
        transformed = self.transform(frame)
        tensor = torch.tensor(transformed, dtype=torch.float32)
        return tensor.to(device) if device is not None else tensor


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    frame = pd.read_csv(DATA_PATH)
    expected_columns = FEATURE_NAMES + ["diagnosis"]
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    frame = frame[expected_columns].copy()
    frame["diagnosis"] = frame["diagnosis"].astype(int)
    return frame


def make_loaders(features: np.ndarray, targets: np.ndarray, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    dataset = TensorDataset(feature_tensor, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_smote(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    class_counts = np.bincount(y, minlength=NUM_CLASSES)
    minority_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 1
    k_neighbors = max(1, min(5, minority_count - 1))
    smote = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
    return smote.fit_resample(x, y)


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    average_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)
    return average_loss, accuracy


def evaluate(model, loader, criterion, device) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[np.ndarray] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits, _ = model(features)
            loss = criterion(logits, targets)
            probabilities = torch.softmax(logits, dim=1)

            running_loss += loss.item() * features.size(0)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            all_probabilities.append(probabilities.detach().cpu().numpy())

    average_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_prob = np.concatenate(all_probabilities, axis=0)
    return average_loss, accuracy, y_true, y_pred, y_prob


def save_checkpoint(payload: Dict) -> None:
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, SAVE_PATH)


def save_feature_importance_plot(analyzer: FeatureImportanceAnalyzer, x: np.ndarray, y: np.ndarray) -> None:
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    analyzer.compute_importance(x, y)
    analyzer.plot_importance(save_path=str(PLOT_PATH))


def main() -> None:
    seed_everything(SEED)
    frame = load_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = frame[FEATURE_NAMES]
    targets = frame["diagnosis"].to_numpy(dtype=np.int64)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_accuracies: List[float] = []
    oof_true: List[int] = []
    oof_pred: List[int] = []
    best_global_val_acc = -1.0
    best_fold_bundle: Optional[Dict] = None
    best_fold_val_x: Optional[np.ndarray] = None
    best_fold_val_y: Optional[np.ndarray] = None

    for fold_idx, (train_val_indices, test_indices) in enumerate(skf.split(features, targets), start=1):
        fold_train_features = features.iloc[train_val_indices].reset_index(drop=True)
        fold_train_targets = targets[train_val_indices]

        fold_train_indices, fold_val_indices = train_test_split(
            np.arange(len(fold_train_features)),
            test_size=0.2,
            stratify=fold_train_targets,
            random_state=SEED,
        )

        train_frame = fold_train_features.iloc[fold_train_indices].reset_index(drop=True)
        val_frame = fold_train_features.iloc[fold_val_indices].reset_index(drop=True)
        y_train = fold_train_targets[fold_train_indices]
        y_val = fold_train_targets[fold_val_indices]

        preprocessor = ClinicalTabularPreprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        x_train = preprocessor.fit_transform(train_frame)
        x_val = preprocessor.transform(val_frame)

        x_train_resampled, y_train_resampled = make_smote(x_train, y_train)

        train_loader = make_loaders(x_train_resampled, y_train_resampled, shuffle=True)
        val_loader = make_loaders(x_val, y_val, shuffle=False)

        model = AlzheimerDNN(input_features=len(FEATURE_NAMES), num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)

        best_fold_acc = -1.0
        best_fold_state = None

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                best_fold_state = copy.deepcopy(model.state_dict())

            print(
                f"Fold {fold_idx}/{N_SPLITS} | Epoch {epoch:03d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        if best_fold_state is None:
            raise RuntimeError(f"Fold {fold_idx} did not produce a valid checkpoint.")

        model.load_state_dict(best_fold_state)
        val_loss, val_acc, y_true, y_pred, y_prob = evaluate(model, val_loader, criterion, device)

        fold_accuracies.append(val_acc)
        oof_true.extend(y_true.tolist())
        oof_pred.extend(y_pred.tolist())

        if val_acc > best_global_val_acc:
            best_global_val_acc = val_acc
            best_fold_bundle = {
                "fold": fold_idx,
                "model_state_dict": copy.deepcopy(best_fold_state),
                "preprocessor": copy.deepcopy(preprocessor),
                "feature_names": FEATURE_NAMES,
                "class_names": CLASS_NAMES,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }
            best_fold_val_x = val_frame[FEATURE_NAMES].to_numpy(dtype=object)
            best_fold_val_y = y_val.copy()

    mean_accuracy = float(np.mean(fold_accuracies))
    std_accuracy = float(np.std(fold_accuracies))
    per_class_f1 = f1_score(np.array(oof_true), np.array(oof_pred), average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    print(f"Mean Accuracy across 5 folds: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print("Per-class F1:")
    for class_name, score in zip(CLASS_NAMES, per_class_f1):
        print(f"  {class_name}: {score:.4f}")

    if best_fold_bundle is None or best_fold_val_x is None or best_fold_val_y is None:
        raise RuntimeError("No best fold checkpoint was produced.")

    save_checkpoint(best_fold_bundle)

    best_model = AlzheimerDNN(input_features=len(FEATURE_NAMES), num_classes=NUM_CLASSES).to(device)
    best_model.load_state_dict(best_fold_bundle["model_state_dict"])
    best_model.eval()

    analyzer = FeatureImportanceAnalyzer(
        model=best_model,
        preprocessing_pipeline=best_fold_bundle["preprocessor"],
        feature_names=FEATURE_NAMES,
        device=device,
    )
    save_feature_importance_plot(analyzer, best_fold_val_x, best_fold_val_y)


if __name__ == "__main__":
    main()