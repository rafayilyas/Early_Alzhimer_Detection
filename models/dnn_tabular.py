"""
DNN for Alzheimer's disease detection from tabular clinical data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES: List[str] = [
    "age",
    "gender",
    "education_years",
    "MMSE_score",
    "CDR_score",
    "eTIV",
    "nWBV",
    "ASF",
    "BMI",
    "smoking_history",
    "family_history",
    "depression_score",
    "sleep_hours",
    "physical_activity",
    "cholesterol",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "diabetes",
]


class ClinicalPreprocessingPipeline:
    """Standardize clinical features before feeding them into the DNN."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, x: np.ndarray) -> "ClinicalPreprocessingPipeline":
        self.scaler.fit(x)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ClinicalPreprocessingPipeline must be fitted before transform().")
        return self.scaler.transform(x)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def to_torch(self, x: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        transformed = self.transform(x)
        tensor = torch.tensor(transformed, dtype=torch.float32)
        return tensor.to(device) if device is not None else tensor


class AlzheimerDNN(nn.Module):
    """Skip-connected DNN for 4-class Alzheimer's disease classification."""

    def __init__(self, input_features: int = 18, num_classes: int = 4):
        super().__init__()
        if input_features != 18:
            raise ValueError("AlzheimerDNN expects exactly 18 input features.")

        self.input_projection = nn.Linear(18, 256)

        self.input_block = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.hidden3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.output_layer = nn.Linear(128, num_classes)

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

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128-dimensional feature embedding."""
        x = x.float()
        projected = self.input_projection(x)
        x = self.input_block(projected)
        x = self.hidden1(x) + projected
        x = self.hidden2(x)
        embedding_128 = self.hidden3(x)
        return embedding_128

    def forward(self, x: torch.Tensor):
        embedding_128 = self.get_embedding(x)
        logits = self.output_layer(embedding_128)
        return logits, embedding_128


class FeatureImportanceAnalyzer:
    """Permutation importance analyzer for the 18 clinical features."""

    def __init__(
        self,
        model: AlzheimerDNN,
        preprocessing_pipeline: ClinicalPreprocessingPipeline,
        feature_names: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.preprocessing_pipeline = preprocessing_pipeline
        self.feature_names = list(feature_names) if feature_names is not None else FEATURE_NAMES.copy()
        self.device = device if device is not None else next(model.parameters()).device
        self.feature_importances_: List[Tuple[str, float]] = []

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_tensor = self.preprocessing_pipeline.to_torch(x, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(x_tensor)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    def compute_importance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> List[Tuple[str, float]]:
        if x.shape[1] != 18:
            raise ValueError("Expected input with exactly 18 features.")

        rng = np.random.default_rng(random_state)
        baseline_probs = self._predict_proba(x)
        baseline_pred = baseline_probs.argmax(axis=1)
        baseline_accuracy = float((baseline_pred == y).mean())

        importances: List[Tuple[str, float]] = []
        for feature_index, feature_name in enumerate(self.feature_names):
            scores: List[float] = []
            for _ in range(n_repeats):
                shuffled = x.copy()
                shuffled[:, feature_index] = rng.permutation(shuffled[:, feature_index])
                permuted_probs = self._predict_proba(shuffled)
                permuted_pred = permuted_probs.argmax(axis=1)
                permuted_accuracy = float((permuted_pred == y).mean())
                scores.append(baseline_accuracy - permuted_accuracy)
            importances.append((feature_name, float(np.mean(scores))))

        importances.sort(key=lambda item: item[1], reverse=True)
        self.feature_importances_ = importances
        return importances

    def plot_importance(self, save_path: str, top_k: Optional[int] = None) -> None:
        if not self.feature_importances_:
            raise RuntimeError("Run compute_importance() before plot_importance().")

        scores = list(self.feature_importances_[:top_k] if top_k is not None else self.feature_importances_)
        features = [item[0] for item in scores][::-1]
        values = [item[1] for item in scores][::-1]

        plt.figure(figsize=(10, max(6, len(features) * 0.4)))
        plt.barh(features, values, color="#2a6fdb")
        plt.xlabel("Permutation Importance")
        plt.title("Clinical Feature Importance")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


class DNNTabular(AlzheimerDNN):
    """Backward-compatible alias for existing training code."""

    pass