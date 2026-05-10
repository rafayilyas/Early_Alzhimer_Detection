from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "models" / "saved" / "cnn_best.pth"
DATA_DIR = ROOT / "data" / "raw" / "MRI"
METRICS_PATH = ROOT / "evaluation" / "metrics.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_api_main_module():
    main_path = ROOT / "app" / "api" / "main.py"
    spec = importlib.util.spec_from_file_location("alz_main", main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load backend module from {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MRIResNet18 = _load_api_main_module().MRIResNet18


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=DEVICE, weights_only=False)


def build_model(num_classes: int) -> MRIResNet18:
    model = MRIResNet18(num_classes=num_classes).to(DEVICE)
    return model


def resolve_dataset_root() -> Path:
    preferred = DATA_DIR / "Alzhiemer" / "combined_images"
    if preferred.exists():
        return preferred

    for candidate in DATA_DIR.rglob("combined_images"):
        if candidate.is_dir():
            return candidate

    for candidate in DATA_DIR.rglob("*"):
        if candidate.is_dir():
            child_dirs = [item for item in candidate.iterdir() if item.is_dir()]
            if len(child_dirs) >= 4:
                return candidate

    raise FileNotFoundError(f"Could not find a class-folder dataset under: {DATA_DIR}")


def build_dataloader(data_dir: Path):
    transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.CenterCrop(160),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader, dataset.classes


def evaluate(model: torch.nn.Module, loader: DataLoader):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            logits, _ = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy().tolist()
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(predictions)

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, y_true, y_pred


def main() -> int:
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    dataset_root = resolve_dataset_root()
    loader, classes = build_dataloader(dataset_root)
    model = build_model(num_classes=len(classes))

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    accuracy, y_true, y_pred = evaluate(model, loader)
    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
        "classes": classes,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if accuracy < 0.85:
        print("VALIDATION FAILED")
        print(f"Accuracy: {accuracy:.4f}")
        return 1

    print("VALIDATION PASSED")
    print(f"Accuracy: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
