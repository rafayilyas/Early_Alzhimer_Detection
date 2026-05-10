from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms


def _build_train_transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def test_augmentation_output_shape():
    image = Image.fromarray(np.random.randint(0, 255, size=(256, 256), dtype=np.uint8), mode="L")
    transformed = _build_train_transforms()(image)

    assert transformed.shape == (3, 224, 224)


def test_normalization_range():
    image = Image.fromarray(np.full((224, 224), 127, dtype=np.uint8), mode="L")
    tensor = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(image)

    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0


def test_grayscale_to_rgb():
    image = Image.fromarray(np.random.randint(0, 255, size=(64, 64), dtype=np.uint8), mode="L")
    tensor = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )(image)

    assert tensor.shape[0] == 3
