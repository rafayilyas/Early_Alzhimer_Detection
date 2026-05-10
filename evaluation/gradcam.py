"""
Grad-CAM explainability utilities for Alzheimer's MRI CNN models.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _resolve_target_layer(model: nn.Module, target_layer: Union[str, nn.Module]):
    if isinstance(target_layer, nn.Module):
        return target_layer

    if target_layer == "layer4" and hasattr(model, "layer4"):
        layer4 = getattr(model, "layer4")
        if isinstance(layer4, nn.Sequential) and len(layer4) > 0:
            return layer4[-1]
        return layer4

    if target_layer == "block4" and hasattr(model, "block4"):
        return getattr(model, "block4")

    current = model
    for part in target_layer.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _to_numpy_image(image: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.array(image)
    elif torch.is_tensor(image):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    if array.ndim == 3 and array.shape[-1] == 3:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR)

    return array.astype(np.float32)


def _normalize_image_for_display(image: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
    array = _to_numpy_image(image)

    if array.ndim == 2:
        normalized = array
    else:
        normalized = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGR2GRAY) if array.shape[-1] == 3 else array

    normalized = normalized.astype(np.float32)
    if normalized.max() > 1.0:
        normalized = normalized / 255.0
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


class GradCAM:
    """Grad-CAM implementation for 2D MRI classification models."""

    def __init__(self, model: nn.Module, target_layer: Union[str, nn.Module] = "layer4"):
        self.model = model
        self.model.eval()
        self.target_layer = _resolve_target_layer(model, target_layer)
        self.gradients = None
        self.activations = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def close(self) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _forward_logits(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(image_tensor)
        if isinstance(output, tuple):
            logits = output[0]
            embedding = output[1] if len(output) > 1 else None
            return logits, embedding
        return output, None

    def generate_cam(self, image_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate a heatmap resized to the input image size."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits, _ = self._forward_logits(image_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.activations[0]
        gradients = self.gradients[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], device=activations.device, dtype=activations.dtype)

        for channel_index, weight in enumerate(weights):
            cam += weight * activations[channel_index]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.detach().cpu().numpy()

        height, width = image_tensor.shape[-2], image_tensor.shape[-1]
        cam_np = cv2.resize(cam_np, (width, height), interpolation=cv2.INTER_LINEAR)
        cam_np = np.clip(cam_np, 0.0, 1.0)
        return cam_np

    @staticmethod
    def overlay_heatmap(original_image: Union[np.ndarray, torch.Tensor, Image.Image], heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Blend a heatmap on top of the original MRI image using a jet colormap."""
        base = _normalize_image_for_display(original_image)
        if base.ndim == 2:
            base_rgb = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = (base * 255).astype(np.uint8)

        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(base_rgb, 1.0 - alpha, colored_heatmap, alpha, 0)
        return blended


def _infer_class_names(dataloader) -> List[str]:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None and hasattr(dataset, "classes"):
        return list(dataset.classes)
    return ["Class 0", "Class 1", "Class 2", "Class 3"]


def _predict(model: nn.Module, image_tensor: torch.Tensor, clinical_tensor: Optional[torch.Tensor] = None):
    output = model(image_tensor, clinical_tensor) if clinical_tensor is not None else model(image_tensor)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    probs = torch.softmax(logits, dim=1)
    confidence, pred = probs.max(dim=1)
    return int(pred.item()), float(confidence.item()), logits


def _extract_sample(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            images, labels = batch
            clinical = None
        elif len(batch) >= 3:
            images, clinical, labels = batch[:3]
        else:
            raise ValueError("Unsupported batch structure")
    else:
        raise ValueError("Unsupported batch structure")
    return images, clinical if "clinical" in locals() else None, labels


def visualize_predictions(model, dataloader, num_samples: int = 16, save_path: Union[str, Path] = "predictions_gradcam.png"):
    """Save a 2x8 grid showing original MRI, Grad-CAM overlay, predicted label, and confidence."""
    class_names = _infer_class_names(dataloader)
    device = next(model.parameters()).device
    gradcam = GradCAM(model, target_layer="layer4" if hasattr(model, "layer4") else "block4")
    model.eval()

    samples_collected = 0
    fig, axes = plt.subplots(2, 8, figsize=(24, 6), dpi=200)
    axes = axes.flatten()

    for batch in dataloader:
        images, clinical, labels = _extract_sample(batch)
        for idx in range(images.size(0)):
            if samples_collected >= num_samples:
                break

            image_tensor = images[idx].unsqueeze(0).to(device)
            clinical_tensor = clinical[idx].unsqueeze(0).to(device) if clinical is not None else None
            label = int(labels[idx].item())

            pred_class, confidence, _ = _predict(model, image_tensor, clinical_tensor)
            heatmap = gradcam.generate_cam(image_tensor, target_class=pred_class)
            original = image_tensor[0, 0].detach().cpu().numpy()
            overlay = gradcam.overlay_heatmap(original, heatmap, alpha=0.4)

            composite = np.concatenate(
                [
                    np.stack([original, original, original], axis=-1) if original.ndim == 2 else original,
                    overlay,
                ],
                axis=1,
            )

            axis = axes[samples_collected]
            axis.imshow(composite, cmap=None)
            axis.set_title(
                f"True: {class_names[label]}\nPred: {class_names[pred_class]} ({confidence:.2f})",
                fontsize=9,
            )
            axis.axis("off")
            samples_collected += 1

        if samples_collected >= num_samples:
            break

    for axis in axes[samples_collected:]:
        axis.axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    gradcam.close()


def find_misclassified(model, dataloader, save_path: Union[str, Path] = "misclassified_gradcam.png"):
    """Save Grad-CAM visualizations for misclassified samples only."""
    class_names = _infer_class_names(dataloader)
    device = next(model.parameters()).device
    gradcam = GradCAM(model, target_layer="layer4" if hasattr(model, "layer4") else "block4")
    model.eval()

    selected: List[Tuple[np.ndarray, np.ndarray, int, int, float]] = []

    for batch in dataloader:
        images, clinical, labels = _extract_sample(batch)
        for idx in range(images.size(0)):
            image_tensor = images[idx].unsqueeze(0).to(device)
            clinical_tensor = clinical[idx].unsqueeze(0).to(device) if clinical is not None else None
            true_label = int(labels[idx].item())
            pred_class, confidence, _ = _predict(model, image_tensor, clinical_tensor)

            if pred_class != true_label:
                heatmap = gradcam.generate_cam(image_tensor, target_class=pred_class)
                original = image_tensor[0, 0].detach().cpu().numpy()
                selected.append((original, heatmap, true_label, pred_class, confidence))

    if not selected:
        gradcam.close()
        return

    cols = 4
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), dpi=200)
    axes = np.array(axes).reshape(-1)

    for axis, (original, heatmap, true_label, pred_class, confidence) in zip(axes, selected):
        overlay = gradcam.overlay_heatmap(original, heatmap, alpha=0.4)
        axis.imshow(overlay)
        axis.set_title(
            f"True: {class_names[true_label]} | Pred: {class_names[pred_class]} | Conf: {confidence:.2f}",
            fontsize=10,
        )
        axis.axis("off")

    for axis in axes[len(selected):]:
        axis.axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    gradcam.close()


def batch_explain(model, image_folder: Union[str, Path], output_folder: Union[str, Path]):
    """Generate and save Grad-CAM overlays for every image in a folder."""
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    gradcam = GradCAM(model, target_layer="layer4" if hasattr(model, "layer4") else "block4")
    preprocess = transforms = None

    for image_path in image_folder.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        resized = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_AREA)
        grayscale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        tensor = torch.tensor(grayscale, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        pred_class, confidence, _ = _predict(model, tensor)
        heatmap = gradcam.generate_cam(tensor, target_class=pred_class)
        overlay = gradcam.overlay_heatmap(grayscale, heatmap, alpha=0.4)

        stem = image_path.stem
        overlay_path = output_folder / f"{stem}_gradcam_overlay.png"
        heatmap_path = output_folder / f"{stem}_gradcam_heatmap.png"

        Image.fromarray(overlay).save(overlay_path)
        heatmap_image = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        heatmap_image = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_JET)
        heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
        Image.fromarray(heatmap_image).save(heatmap_path)

    gradcam.close()
