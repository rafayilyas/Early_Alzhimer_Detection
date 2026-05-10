from __future__ import annotations

import torch

import models.pretrained_finetune as pretrained_finetune
from models.cnn_classifier import AlzheimerCNN
from models.pretrained_finetune import AlzheimerTransferModel, EnsembleModel


def test_cnn_forward_pass():
    model = AlzheimerCNN(num_classes=4)
    dummy = torch.randn(1, 1, 224, 224)
    logits, embedding = model(dummy)

    assert logits.shape == (1, 4)
    assert embedding.shape[0] == 1


def test_transfer_model_forward(monkeypatch):
    monkeypatch.setattr(pretrained_finetune, "_load_weights", lambda backbone: None)
    model = AlzheimerTransferModel("resnet50", num_classes=4)
    dummy = torch.randn(1, 3, 224, 224)
    logits, embedding = model(dummy)

    assert logits.shape == (1, 4)
    assert embedding.shape == (1, 512)


def test_ensemble_forward(monkeypatch):
    monkeypatch.setattr(pretrained_finetune, "_load_weights", lambda backbone: None)
    models = [AlzheimerTransferModel("resnet50", num_classes=4) for _ in range(3)]
    ensemble = EnsembleModel(models)
    dummy = torch.randn(1, 3, 224, 224)
    output = ensemble(dummy)

    assert output.shape == (1, 4)
