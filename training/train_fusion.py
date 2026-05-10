"""
Training script for Alzheimer's MRI + clinical fusion classification.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.cnn_classifier import AlzheimerCNN
from models.dnn_tabular import AlzheimerDNN, FEATURE_NAMES


SEED = 42
NUM_CLASSES = 4
DATA_ROOT = Path("data/raw/MRI")
CLINICAL_CSV = Path("data/raw/clinical/alzheimers_dataset.csv")
CNN_CHECKPOINT = Path("models/saved/cnn_best.pth")
DNN_CHECKPOINT = Path("models/saved/dnn_best.pth")
SAVE_PATH = Path("models/saved/fusion_best.pth")
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MISSING_VALUE_COLUMNS = set(FEATURE_NAMES)

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


def build_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


class ClinicalPreprocessor:
    """Impute, encode, and scale the clinical features for the frozen DNN."""

    def __init__(self):
        self.numeric_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, str] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    @staticmethod
    def _as_string(value) -> str:
        return str(value)

    def fit(self, frame: pd.DataFrame) -> "ClinicalPreprocessor":
        working = frame.copy()

        for column in NUMERIC_FEATURES:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            median_value = float(working[column].median())
            self.numeric_medians[column] = median_value
            working[column] = working[column].fillna(median_value)

        for column in CATEGORICAL_FEATURES:
            normalized = working[column].map(self._as_string)
            mode_value = normalized.mode(dropna=True).iloc[0]
            self.categorical_modes[column] = mode_value
            filled = normalized.fillna(mode_value)
            encoder = LabelEncoder()
            encoder.fit(filled)
            self.encoders[column] = encoder
            working[column] = encoder.transform(filled)

        self.scaler.fit(working[FEATURE_NAMES].to_numpy(dtype=np.float32))
        self.is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ClinicalPreprocessor must be fitted before transform().")

        working = frame.copy()

        for column in NUMERIC_FEATURES:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            working[column] = working[column].fillna(self.numeric_medians[column])

        for column in CATEGORICAL_FEATURES:
            normalized = working[column].map(self._as_string)
            filled = normalized.fillna(self.categorical_modes[column])
            working[column] = self.encoders[column].transform(filled)

        working[FEATURE_NAMES] = self.scaler.transform(working[FEATURE_NAMES].to_numpy(dtype=np.float32))
        return working[FEATURE_NAMES].to_numpy(dtype=np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


def load_model_checkpoint(model_path: Path, model: nn.Module, device: torch.device) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def find_images(root: Path) -> pd.DataFrame:
    records: List[Dict[str, str]] = []
    for class_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        label_name = class_dir.name
        if label_name not in CLASSES:
            continue
        label = CLASSES.index(label_name)
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "image_path": str(image_path),
                        "image_stem": image_path.stem,
                        "image_key": image_path.stem,
                        "diagnosis": label,
                        "label_name": label_name,
                    }
                )
    if not records:
        raise FileNotFoundError(f"No MRI images found under {root}")
    return pd.DataFrame.from_records(records)


def build_clinical_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {path}")

    frame = pd.read_csv(path)
    required = set(FEATURE_NAMES) | {"diagnosis"}
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Clinical CSV is missing required columns: {missing}")

    key_candidates = ["patient_id", "subject_id", "id", "filename", "image_name", "image_key", "image_stem"]
    join_key = next((column for column in key_candidates if column in frame.columns), None)
    if join_key is None:
        raise ValueError(
            "Clinical CSV must include a patient/image key column such as patient_id, filename, or image_key."
        )

    clinical = frame.copy()
    clinical["image_key"] = clinical[join_key].astype(str)
    clinical["diagnosis"] = clinical["diagnosis"].astype(int)
    return clinical


def merge_modalities(mri_frame: pd.DataFrame, clinical_frame: pd.DataFrame) -> pd.DataFrame:
    merged = mri_frame.merge(clinical_frame, on="image_key", how="left", suffixes=("_mri", "_clinical"))

    if "diagnosis_clinical" in merged.columns:
        merged["diagnosis"] = merged["diagnosis_clinical"].fillna(merged["diagnosis_mri"]).astype(int)
    elif "diagnosis" not in merged.columns:
        merged["diagnosis"] = merged["diagnosis_mri"].astype(int)

    for column in FEATURE_NAMES:
        if column not in merged.columns:
            merged[column] = np.nan

    merged[FEATURE_NAMES] = merged[FEATURE_NAMES].replace({"": np.nan})
    return merged


def split_dataframe(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        frame,
        test_size=0.15,
        stratify=frame["diagnosis"],
        random_state=SEED,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.17647058823529413,
        stratify=train_val_df["diagnosis"],
        random_state=SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class FusionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, clinical_preprocessor: ClinicalPreprocessor, image_transform=None):
        self.frame = frame.reset_index(drop=True)
        self.preprocessor = clinical_preprocessor
        self.image_transform = image_transform or build_image_transform()

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.image_transform(image)

    def _load_clinical(self, row: pd.Series) -> torch.Tensor:
        clinical_values = row.reindex(FEATURE_NAMES)
        if clinical_values.isna().all():
            return torch.zeros(len(FEATURE_NAMES), dtype=torch.float32)
        clinical_frame = pd.DataFrame([clinical_values], columns=FEATURE_NAMES)
        transformed = self.preprocessor.transform(clinical_frame)
        return torch.tensor(transformed[0], dtype=torch.float32)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self._load_image(row["image_path"])
        clinical = self._load_clinical(row)
        label = torch.tensor(int(row["diagnosis"]), dtype=torch.long)
        return image, clinical, label


class AlzheimerFusionModel(nn.Module):
    def __init__(self, cnn_model: AlzheimerCNN, dnn_model: AlzheimerDNN, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.cnn = cnn_model
        self.dnn = dnn_model
        self.cnn.eval()
        self.dnn.eval()
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.dnn.parameters():
            param.requires_grad = False

        self.dnn_to_cnn = nn.Linear(128, 512)
        self.attention_gate = nn.Linear(640, 2)
        self.classifier = nn.Sequential(
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, mri_data: torch.Tensor, clinical_data: Optional[torch.Tensor] = None):
        if clinical_data is None:
            clinical_data = torch.zeros(mri_data.size(0), len(FEATURE_NAMES), device=mri_data.device, dtype=mri_data.dtype)

        with torch.no_grad():
            _, cnn_emb = self.cnn(mri_data)
            _, dnn_emb = self.dnn(clinical_data)

        combined = torch.cat([cnn_emb, dnn_emb], dim=1)
        attention = torch.softmax(torch.sigmoid(self.attention_gate(combined)), dim=1)

        weighted_cnn = attention[:, 0:1] * cnn_emb
        weighted_dnn = attention[:, 1:2] * self.dnn_to_cnn(dnn_emb)
        weighted_concat = weighted_cnn + weighted_dnn
        fusion_input = torch.cat([weighted_concat, dnn_emb], dim=1)
        logits = self.classifier(fusion_input)
        return logits, attention


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    model.cnn.eval()
    model.dnn.eval()
    total_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    for mri, clinical, targets in loader:
        mri = mri.to(device)
        clinical = clinical.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, attention = model(mri, clinical)
        ce_loss = criterion(logits, targets)
        uniform = torch.full_like(attention, 0.5)
        kl_loss = F.kl_div(torch.log(attention + 1e-8), uniform, reduction="batchmean")
        loss = ce_loss + 0.1 * kl_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mri.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(all_targets, all_predictions)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for mri, clinical, targets in loader:
            mri = mri.to(device)
            clinical = clinical.to(device)
            targets = targets.to(device)

            logits, _ = model(mri, clinical)
            loss = criterion(logits, targets)
            total_loss += loss.item() * mri.size(0)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(all_targets, all_predictions), np.array(all_targets), np.array(all_predictions)


def evaluate_baseline_cnn(cnn_model, loader, device) -> float:
    cnn_model.eval()
    all_targets: List[int] = []
    all_predictions: List[int] = []
    with torch.no_grad():
        for mri, _, targets in loader:
            mri = mri.to(device)
            targets = targets.to(device)
            logits, _ = cnn_model(mri)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return accuracy_score(all_targets, all_predictions)


def evaluate_baseline_dnn(dnn_model, loader, device) -> float:
    dnn_model.eval()
    all_targets: List[int] = []
    all_predictions: List[int] = []
    with torch.no_grad():
        for _, clinical, targets in loader:
            clinical = clinical.to(device)
            targets = targets.to(device)
            logits, _ = dnn_model(clinical)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return accuracy_score(all_targets, all_predictions)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, val_acc: float) -> None:
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_acc,
            "classes": CLASSES,
        },
        SAVE_PATH,
    )


def print_accuracy_table(cnn_acc: float, dnn_acc: float, fusion_acc: float) -> None:
    print("\nFinal Accuracy Comparison")
    print("+---------+----------+")
    print("| Model   | Accuracy |")
    print("+---------+----------+")
    print(f"| CNN     | {cnn_acc:0.4f}   |")
    print(f"| DNN     | {dnn_acc:0.4f}   |")
    print(f"| Fusion  | {fusion_acc:0.4f}   |")
    print("+---------+----------+")


def main() -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"MRI root not found: {DATA_ROOT}")
    if not CLINICAL_CSV.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {CLINICAL_CSV}")

    mri_frame = find_images(DATA_ROOT)
    clinical_frame = build_clinical_frame(CLINICAL_CSV)
    merged_frame = merge_modalities(mri_frame, clinical_frame)

    train_df, val_df, test_df = split_dataframe(merged_frame)

    preprocessor = ClinicalPreprocessor()
    preprocessor.fit(train_df[FEATURE_NAMES])

    train_dataset = FusionDataset(train_df, preprocessor)
    val_dataset = FusionDataset(val_df, preprocessor)
    test_dataset = FusionDataset(test_df, preprocessor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    cnn_model = load_model_checkpoint(CNN_CHECKPOINT, AlzheimerCNN(num_classes=NUM_CLASSES), device)
    dnn_model = load_model_checkpoint(DNN_CHECKPOINT, AlzheimerDNN(input_features=len(FEATURE_NAMES), num_classes=NUM_CLASSES), device)

    fusion_model = AlzheimerFusionModel(cnn_model, dnn_model, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.0005)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, 31):
        train_loss, train_acc = train_one_epoch(fusion_model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate_model(fusion_model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/30 | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(fusion_model.state_dict())
            save_checkpoint(fusion_model, optimizer, val_acc)

    if best_state is None:
        raise RuntimeError("Fusion training did not produce a valid best checkpoint.")

    fusion_model.load_state_dict(best_state)
    _, fusion_test_acc, _, _ = evaluate_model(fusion_model, test_loader, criterion, device)
    cnn_test_acc = evaluate_baseline_cnn(cnn_model, test_loader, device)
    dnn_test_acc = evaluate_baseline_dnn(dnn_model, test_loader, device)

    print_accuracy_table(cnn_test_acc, dnn_test_acc, fusion_test_acc)


if __name__ == "__main__":
    main()