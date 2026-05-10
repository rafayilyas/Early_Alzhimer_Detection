"""Debug script to identify CNN training issues."""

import sys
import traceback
from pathlib import Path
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from models.cnn_classifier import AlzheimerCNN

SEED = 42
NUM_CLASSES = 4
BATCH_SIZE = 32
DATA_DIR = Path("data/raw/MRI")
FULL_DATA_DIR = Path("data/raw/MRI/Alzhiemer")

def seed_everything(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return train_transform, eval_transform

def main():
    try:
        print("Step 1: Setting seed...")
        seed_everything(SEED)
        
        print("Step 2: Getting device...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")
        
        print("Step 3: Building transforms...")
        train_transform, eval_transform = build_transforms()
        
        print("Step 4: Loading dataset...")
        dataset_root = FULL_DATA_DIR if FULL_DATA_DIR.exists() else DATA_DIR
        print(f"  Dataset root: {dataset_root}")
        print(f"  Exists: {dataset_root.exists()}")
        
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
        
        print("Step 5: Creating ImageFolder...")
        full_dataset = datasets.ImageFolder(dataset_root, transform=train_transform)
        print(f"  Loaded {len(full_dataset)} images")
        print(f"  Classes: {full_dataset.classes}")
        
        print("Step 6: Creating model...")
        model = AlzheimerCNN(num_classes=NUM_CLASSES).to(device)
        print(f"  Model created on {device}")
        
        print("Step 7: Testing forward pass...")
        dummy_batch = torch.randn(2, 1, 224, 224).to(device)
        logits, embeddings = model(dummy_batch)
        print(f"  Logits shape: {logits.shape}")
        print(f"  Embeddings shape: {embeddings.shape}")
        
        print("\n✓ All steps completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error occurred:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
