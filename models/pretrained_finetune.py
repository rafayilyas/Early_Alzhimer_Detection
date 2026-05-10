"""
Transfer learning models for Alzheimer's disease classification from MRI images.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _load_weights(model_name: str):
    """Load torchvision pretrained ImageNet weights with compatibility fallback."""
    try:
        if model_name == "resnet50":
            return models.ResNet50_Weights.IMAGENET1K_V2
        if model_name == "efficientnet_b3":
            return models.EfficientNet_B3_Weights.IMAGENET1K_V1
        if model_name == "densenet121":
            return models.DenseNet121_Weights.IMAGENET1K_V1
    except AttributeError:
        return None
    return None


class AlzheimerTransferModel(nn.Module):
    """Transfer learning model for 4-class Alzheimer's MRI classification."""

    def __init__(self, backbone: str = "resnet50", num_classes: int = 4):
        super().__init__()

        if backbone not in {"resnet50", "efficientnet_b3", "densenet121"}:
            raise ValueError(
                "backbone must be one of: 'resnet50', 'efficientnet_b3', 'densenet121'"
            )

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.backbone, self.feature_dim = self._build_backbone(backbone)

        self.embedding_layer = nn.Linear(self.feature_dim, 512)
        self.embedding_bn = nn.BatchNorm1d(512)
        self.embedding_dropout = nn.Dropout(0.4)
        self.hidden_layer = nn.Linear(512, 256)
        self.hidden_dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(256, num_classes)

        self._initialize_classifier()
        self.freeze_backbone()

    def _build_backbone(self, backbone: str):
        weights = _load_weights(backbone)

        if backbone == "resnet50":
            backbone_model = models.resnet50(weights=weights)
            feature_dim = backbone_model.fc.in_features
            backbone_model.fc = nn.Identity()
            return backbone_model, feature_dim

        if backbone == "efficientnet_b3":
            backbone_model = models.efficientnet_b3(weights=weights)
            feature_dim = backbone_model.classifier[1].in_features
            backbone_model.classifier = nn.Identity()
            return backbone_model, feature_dim

        backbone_model = models.densenet121(weights=weights)
        feature_dim = backbone_model.classifier.in_features
        backbone_model.classifier = nn.Identity()
        return backbone_model, feature_dim

    def _initialize_classifier(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _replicate_grayscale(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Expected input tensor with shape (batch, channels, height, width)")
        if x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        if x.size(1) == 3:
            return x
        raise ValueError("Expected grayscale input with 1 channel or RGB input with 3 channels")

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._replicate_grayscale(x)

        if self.backbone_name == "resnet50":
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)

        if self.backbone_name == "efficientnet_b3":
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)

        x = self.backbone.features(x)
        x = F.relu(x, inplace=True)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def freeze_backbone(self) -> None:
        """Freeze the backbone and keep the last two blocks trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.backbone_name == "resnet50":
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        elif self.backbone_name == "efficientnet_b3":
            for param in self.backbone.features[-2:].parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.features.denseblock3.parameters():
                param.requires_grad = True
            for param in self.backbone.features.denseblock4.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone and classifier parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-dimensional feature embedding."""
        features = self._extract_features(x)
        embedding = self.embedding_layer(features)
        embedding = self.embedding_bn(embedding)
        return F.relu(embedding, inplace=True)

    def forward(self, x: torch.Tensor):
        embedding_512 = self.get_embedding(x)
        x = self.embedding_dropout(embedding_512)
        x = self.hidden_layer(x)
        x = F.relu(x, inplace=True)
        x = self.hidden_dropout(x)
        logits = self.output_layer(x)
        return logits, embedding_512


class EnsembleModel(nn.Module):
    """Average the predictions of three AlzheimerTransferModel instances."""

    def __init__(
        self,
        models_list: Sequence[AlzheimerTransferModel],
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()

        if len(models_list) != 3:
            raise ValueError("EnsembleModel expects exactly 3 models")

        self.models = nn.ModuleList(models_list)
        self.weights = torch.tensor(weights if weights is not None else [1.0, 1.0, 1.0], dtype=torch.float32)

    def forward(self, x: torch.Tensor, weights: Optional[Sequence[float]] = None) -> torch.Tensor:
        probs = []
        for model in self.models:
            logits, _ = model(x)
            probs.append(F.softmax(logits, dim=1))

        stacked = torch.stack(probs, dim=0)

        current_weights = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device) if weights is not None else self.weights.to(stacked.device, dtype=stacked.dtype)
        current_weights = current_weights / current_weights.sum()
        view_shape = (current_weights.size(0),) + (1,) * (stacked.dim() - 1)
        weighted = stacked * current_weights.view(view_shape)
        return weighted.sum(dim=0)


class PretrainedFinetune(AlzheimerTransferModel):
    """Backward-compatible alias for older imports."""

    pass