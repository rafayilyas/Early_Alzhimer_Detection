"""
CNN classifier for Alzheimer's disease detection from grayscale MRI images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlzheimerCNN(nn.Module):
    """Custom 2D CNN for 4-class Alzheimer's disease classification."""

    def __init__(self, num_classes: int = 4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(256 * 4 * 4, 512)
        self.classifier_hidden = nn.Linear(512, 256)
        self.classifier_out = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-d feature embedding for an input batch."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        return F.relu(self.embedding(x), inplace=True)

    def forward(self, x: torch.Tensor):
        """Return logits and the 512-d embedding."""
        embedding_512 = self.get_embedding(x)
        x = self.dropout(embedding_512)
        x = F.relu(self.classifier_hidden(x), inplace=True)
        logits = self.classifier_out(x)
        return logits, embedding_512


class CNNClassifier(AlzheimerCNN):
    """Backward-compatible alias for existing training code."""

    pass
