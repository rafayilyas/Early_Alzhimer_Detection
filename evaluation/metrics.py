"""
Evaluation and reporting utilities for 4-class Alzheimer's classification.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from fpdf import FPDF
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


CLASS_NAMES = ["Non-Demented", "Very Mild", "Mild", "Moderate"]


def _ensure_path(save_path: Union[str, Path]) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_model_outputs(outputs):
    if isinstance(outputs, Mapping):
        probabilities = outputs.get("probabilities") or outputs.get("y_prob") or outputs.get("probs")
        logits = outputs.get("logits")
        predicted = outputs.get("predictions") or outputs.get("y_pred")

        if isinstance(probabilities, Mapping):
            normalized_map = {str(key).replace("-", "").replace(" ", "").lower(): float(value) for key, value in probabilities.items()}
            ordered = []
            for class_name in CLASS_NAMES:
                normalized_key = class_name.replace("-", "").replace(" ", "").lower()
                ordered.append(normalized_map.get(normalized_key, 0.0))
            probabilities = np.asarray([ordered], dtype=float)
        return probabilities, logits, predicted
    return outputs, None, None


def evaluate_model(model, dataloader, device, class_names: Sequence[str] = CLASS_NAMES):
    """Evaluate a model and return aggregate plus per-class metrics."""
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                inputs, targets = batch
                clinical = None
            elif len(batch) >= 3:
                inputs, clinical, targets = batch[:3]
            else:
                raise ValueError("Unsupported batch structure for evaluation.")

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs, clinical.to(device) if clinical is not None and torch.is_tensor(clinical) else clinical)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_true.extend(targets.detach().cpu().tolist())
            all_pred.extend(predictions.detach().cpu().tolist())
            all_prob.append(probabilities.detach().cpu().numpy())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    y_prob = np.concatenate(all_prob, axis=0) if all_prob else np.empty((0, len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)

    per_class: Dict[str, Dict[str, float]] = {}
    class_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    for class_name in class_names:
        class_report_entry = class_report[class_name]
        per_class[class_name] = {
            "precision": float(class_report_entry["precision"]),
            "recall": float(class_report_entry["recall"]),
            "f1": float(class_report_entry["f1-score"]),
            "support": float(class_report_entry["support"]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "cohen_kappa": float(cohen_kappa),
        "per_class": per_class,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_confusion_matrix(y_true, y_pred, class_names: Sequence[str] = CLASS_NAMES, save_path: Union[str, Path] = "confusion_matrix.png"):
    """Plot a normalized confusion matrix with counts and percentages."""
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    annotations = np.empty_like(cm).astype(object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"

    path = _ensure_path(save_path)
    plt.figure(figsize=(8, 6), dpi=180)
    sns.set_style("white")
    ax = sns.heatmap(
        cm_norm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Normalized frequency"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_roc_auc(y_true, y_prob, class_names: Sequence[str] = CLASS_NAMES, save_path: Union[str, Path] = "roc_auc.png"):
    """Plot one-vs-rest ROC curves with micro and macro averages."""
    y_true = _to_numpy(y_true).astype(int)
    y_prob = _to_numpy(y_prob)
    n_classes = len(class_names)
    y_true_onehot = np.eye(n_classes)[y_true]

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    path = _ensure_path(save_path)
    plt.figure(figsize=(8, 7), dpi=180)
    plt.plot(fpr["micro"], tpr["micro"], linestyle="--", linewidth=2.5, label=f"Micro-average (AUC = {roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], linestyle="--", linewidth=2.5, label=f"Macro-average (AUC = {roc_auc['macro']:.3f})")

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_training_history(history_json_path: Union[str, Path], save_path: Union[str, Path] = "training_history.png"):
    """Plot train/validation loss and accuracy curves from a JSON history file."""
    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    epochs = range(1, max(len(train_loss), len(val_loss), len(train_acc), len(val_acc)) + 1)

    path = _ensure_path(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)

    axes[0].plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(range(1, len(val_loss) + 1), val_loss, label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(range(1, len(train_acc) + 1), train_acc, label="Train Accuracy", linewidth=2)
    axes[1].plot(range(1, len(val_acc) + 1), val_acc, label="Val Accuracy", linewidth=2)
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def compare_models(results_dict: Mapping[str, Mapping[str, float]], save_path: Union[str, Path] = "model_comparison.png"):
    """Bar chart comparing multiple models on accuracy, F1, and AUC."""
    model_names = list(results_dict.keys())
    accuracy = [results_dict[name].get("accuracy", np.nan) for name in model_names]
    f1_scores = [results_dict[name].get("weighted_f1", results_dict[name].get("macro_f1", np.nan)) for name in model_names]
    auc_scores = [results_dict[name].get("auc", results_dict[name].get("roc_auc", np.nan)) for name in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    path = _ensure_path(save_path)
    plt.figure(figsize=(10, 6), dpi=180)
    plt.bar(x - width, accuracy, width, label="Accuracy")
    plt.bar(x, f1_scores, width, label="F1")
    plt.bar(x + width, auc_scores, width, label="AUC")
    plt.xticks(x, model_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def _extract_patient_text(clinical_data) -> Dict[str, str]:
    if clinical_data is None:
        return {}
    if isinstance(clinical_data, Mapping):
        return {str(k): str(v) for k, v in clinical_data.items()}
    if hasattr(clinical_data, "to_dict"):
        return {str(k): str(v) for k, v in clinical_data.to_dict().items()}
    return {"clinical_data": str(clinical_data)}


def _probabilities_from_model_outputs(model_outputs) -> Tuple[np.ndarray, Optional[int], Optional[float]]:
    probabilities, logits, predicted = _extract_model_outputs(model_outputs)

    if probabilities is None and logits is not None:
        logits = _to_numpy(logits)
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    if probabilities is None:
        raise ValueError("model_outputs must include probabilities or logits.")

    probabilities = _to_numpy(probabilities)
    if probabilities.ndim == 1:
        probabilities = probabilities[None, :]

    if predicted is not None:
        predicted_index = int(np.asarray(predicted).reshape(-1)[0])
    else:
        predicted_index = int(probabilities.argmax(axis=1)[0])

    confidence = float(probabilities[0, predicted_index])
    return probabilities, predicted_index, confidence


def _build_risk_gauge(risk_score: float, save_path: Path) -> Path:
    risk_score = float(np.clip(risk_score, 0, 100))
    fig, ax = plt.subplots(figsize=(5, 3), dpi=180)
    ax.axis("off")
    theta = np.linspace(np.pi, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#d0d7de", linewidth=18, solid_capstyle="round")

    filled_theta = np.linspace(np.pi, np.pi + (risk_score / 100.0) * np.pi, 200)
    risk_color = "#2ca02c" if risk_score < 33 else "#ffbf00" if risk_score < 66 else "#d62728"
    ax.plot(np.cos(filled_theta), np.sin(filled_theta), color=risk_color, linewidth=18, solid_capstyle="round")
    ax.text(0, -0.1, f"{risk_score:.0f}/100", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(0, -0.55, "Alzheimer's Risk", ha="center", va="center", fontsize=12)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 1.2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return save_path


def _build_probability_chart(probabilities: np.ndarray, class_names: Sequence[str], save_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=180)
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, probabilities, color="#2a6fdb")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Per-stage Probability")
    for idx, value in enumerate(probabilities):
        ax.text(min(value + 0.02, 0.98), idx, f"{value:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return save_path


def _recommendation_text(risk_score: float, predicted_class: str) -> str:
    if risk_score < 33:
        base = "Low immediate concern, but routine cognitive follow-up is recommended if symptoms persist."
    elif risk_score < 66:
        base = "Moderate concern. Consider formal neurocognitive assessment and closer clinical monitoring."
    else:
        base = "High concern. Recommend prompt specialist review, confirmatory testing, and care planning."
    return f"Prediction: {predicted_class}. {base}"


def generate_patient_report(
    patient_id,
    mri_path,
    clinical_data,
    model_outputs,
    gradcam_image,
    save_path,
):
    """Generate a PDF report using fpdf2 for a single patient."""
    save_path = _ensure_path(save_path)
    class_names = CLASS_NAMES
    probabilities, predicted_index, confidence = _probabilities_from_model_outputs(model_outputs)
    probabilities = probabilities[0]
    risk_score = float(np.clip((probabilities[1] * 35) + (probabilities[2] * 70) + (probabilities[3] * 100), 0, 100))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mri_tmp = tmpdir / "mri.png"
        gradcam_tmp = tmpdir / "gradcam.png"
        risk_tmp = tmpdir / "risk.png"
        probs_tmp = tmpdir / "probs.png"

        Image.open(mri_path).convert("RGB").save(mri_tmp)
        if isinstance(gradcam_image, (str, Path)):
            Image.open(gradcam_image).convert("RGB").save(gradcam_tmp)
        elif isinstance(gradcam_image, Image.Image):
            gradcam_image.convert("RGB").save(gradcam_tmp)
        else:
            Image.fromarray(_to_numpy(gradcam_image).astype(np.uint8)).save(gradcam_tmp)

        _build_risk_gauge(risk_score, risk_tmp)
        _build_probability_chart(probabilities, class_names, probs_tmp)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Alzheimer's MRI Clinical Report", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Patient ID: {patient_id}", ln=True)
        pdf.cell(0, 8, f"Predicted stage: {class_names[predicted_index]}", ln=True)
        pdf.cell(0, 8, f"Confidence: {confidence * 100:.1f}%", ln=True)
        pdf.cell(0, 8, f"Risk score: {risk_score:.1f}/100", ln=True)

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Clinical Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        patient_text = _extract_patient_text(clinical_data)
        if patient_text:
            for key, value in patient_text.items():
                pdf.multi_cell(0, 6, f"{key}: {value}")
        else:
            pdf.multi_cell(0, 6, "No structured clinical data available.")

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "MRI and Grad-CAM", ln=True)
        page_width = pdf.w - 2 * pdf.l_margin
        image_width = (page_width - 4) / 2
        y_before = pdf.get_y()
        pdf.image(str(mri_tmp), x=pdf.l_margin, y=y_before, w=image_width)
        pdf.image(str(gradcam_tmp), x=pdf.l_margin + image_width + 4, y=y_before, w=image_width)
        pdf.ln(image_width * 0.75 + 4)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Risk Gauge", ln=True)
        pdf.image(str(risk_tmp), x=pdf.l_margin + 10, w=page_width - 20)
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Per-stage Probabilities", ln=True)
        pdf.image(str(probs_tmp), x=pdf.l_margin, w=page_width)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Clinical Recommendations", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _recommendation_text(risk_score, class_names[predicted_index]))

        pdf.ln(6)
        pdf.cell(0, 8, "Doctor Signature: ____________________________", ln=True)

        pdf.output(str(save_path))

    return save_path


class ModelMetrics:
    """Backward-compatible metric helper wrapper."""

    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba=None):
        metrics = evaluate_model.__defaults__  # pragma: no cover - compatibility shim
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }
