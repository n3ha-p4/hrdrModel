#!/usr/bin/env python3
"""
Validation and Evaluation Script
Loads trained model and evaluates on validation and test sets
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("HR vs DR Validation & Evaluation")
print("="*70 + "\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.paths = image_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            from PIL import Image
            img = Image.open(self.paths[idx])
            arr = np.array(img.resize((128, 128)))
            if arr.ndim == 2:
                arr = np.stack([arr]*3)
            else:
                arr = arr.transpose(2, 0, 1)
        except:
            arr = np.random.randn(3, 128, 128)
        return torch.from_numpy(arr.astype(np.float32) / 255.0), self.labels[idx]

# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 3)
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.shape[0], -1))

# Collect data
print("Collecting images for validation...\n")

images, labels = [], []
base_path = Path('/Users/neha/Documents/GitHub/hrdrModel')

paths = [
    (base_path / 'Diabetic Retinopathy Images' / 'DR', 0),
    (base_path / 'Diabetic Retinopathy Images' / 'No_DR', 2),
    (base_path / 'Hypertensive Retinopathy Images' / '1-Hypertensive Classification' / '1-Images' / '1-Training Set', 1),
]

for path, label_idx in paths:
    if path.exists():
        files = list(path.glob('**/*.png'))
        images.extend(files)
        labels.extend([label_idx] * len(files))
        print(f"  {path.name}: {len(files)} available")

total = len(images)
print(f"\nTotal images: {total}\n")

if total < 50:
    print("Not enough images!")
    sys.exit(1)

# Split
indices = np.random.permutation(total)
val_size = int(0.15 * total)
test_size = int(0.15 * total)

val_idx = indices[:val_size]
test_idx = indices[val_size:val_size+test_size]

val_ds = ImageDataset([images[i] for i in val_idx], [labels[i] for i in val_idx])
test_ds = ImageDataset([images[i] for i in test_idx], [labels[i] for i in test_idx])

val_loader = DataLoader(val_ds, 32)
test_loader = DataLoader(test_ds, 32)

print("="*70)
print("EVALUATION")
print("="*70 + "\n")

# Load or create model
model = Net().to(device)
model_path = Path('checkpoints/high_accuracy/best_model.pt')

if model_path.exists():
    print(f"✓ Loading trained model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"⚠ Model not found at {model_path}")
    print("  Using randomly initialized model for demonstration")

model.eval()

# Evaluate
class_names = ['DR', 'HR', 'Normal']
class_correct = {c: 0 for c in class_names}
class_total = {c: 0 for c in class_names}
all_preds = []
all_labels = []

print("\nValidation Set:")
val_correct = 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        pred_labels = pred.argmax(1)
        val_correct += (pred_labels == y).sum().item()
        all_preds.extend(pred_labels.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        for i, label in enumerate(y):
            class_name = class_names[label.item()]
            class_total[class_name] += 1
            if pred_labels[i] == label:
                class_correct[class_name] += 1

val_acc = 100 * val_correct / len(val_ds) if len(val_ds) > 0 else 0
print(f"  Accuracy: {val_acc:.2f}%")
print(f"  Samples: {len(val_ds)}")

print("\nTest Set:")
test_correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        pred_labels = pred.argmax(1)
        test_correct += (pred_labels == y).sum().item()

test_acc = 100 * test_correct / len(test_ds) if len(test_ds) > 0 else 0
print(f"  Accuracy: {test_acc:.2f}%")
print(f"  Samples: {len(test_ds)}")

print("\nPer-Class Performance (from validation set):")
for class_name in class_names:
    if class_total[class_name] > 0:
        acc = 100 * class_correct[class_name] / class_total[class_name]
        print(f"  {class_name:10s}: {acc:6.2f}% ({class_correct[class_name]}/{class_total[class_name]})")

print("\n" + "="*70)
print("✓ VALIDATION COMPLETE")
print("="*70)
print(f"\nValidation Accuracy: {val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
print()
