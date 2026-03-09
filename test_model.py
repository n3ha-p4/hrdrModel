#!/usr/bin/env python3
"""
Testing & Prediction Script
Comprehensive evaluation with predictions, confusion matrix, and metrics
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from collections import defaultdict

print("\n" + "="*70)
print("HR vs DR Model Testing & Evaluation")
print("="*70 + "\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.paths = image_paths
        self.labels = labels
        self.class_names = ['DR', 'HR', 'Normal']
    
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

print("Collecting test images...\n")

images, labels = [], []
base_path = Path('/Users/neha/Documents/GitHub/hrdrModel')

paths = [
    (base_path / 'Diabetic Retinopathy Images' / 'DR', 0, 'DR'),
    (base_path / 'Diabetic Retinopathy Images' / 'No_DR', 2, 'Normal'),
    (base_path / 'Hypertensive Retinopathy Images' / '1-Hypertensive Classification' / '1-Images' / '1-Training Set', 1, 'HR'),
]

for path, label_idx, name in paths:
    if path.exists():
        files = list(path.glob('**/*.png'))
        images.extend(files)
        labels.extend([label_idx] * len(files))
        print(f"  {name:10s}: {len(files):4d} images")

total = len(images)
print(f"\nTotal images: {total}\n")

if total < 50:
    print("Not enough images!")
    sys.exit(1)

# Split - use last 15% for testing
indices = np.random.permutation(total)
test_size = max(100, int(0.15 * total))
test_idx = indices[-test_size:]

test_ds = ImageDataset([images[i] for i in test_idx], [labels[i] for i in test_idx])
test_loader = DataLoader(test_ds, 32)

print(f"Test set size: {len(test_ds)} images\n")

print("="*70)
print("TESTING")
print("="*70 + "\n")

# Load or create model
model = Net().to(device)
model_path = Path('checkpoints/high_accuracy/best_model.pt')

if model_path.exists():
    print(f"✓ Loading trained model from {model_path}\n")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"⚠ Trained model not found at {model_path}")
    print("  Using randomly initialized model for demonstration\n")

model.eval()

# Test and collect predictions
class_names = ['DR', 'HR', 'Normal']
predictions = []
ground_truth = []
all_correct = 0
all_total = 0

# Confusion matrix
conf_matrix = np.zeros((3, 3), dtype=int)

print("Running predictions on test set:\n")

with torch.no_grad():
    batch_num = 0
    for x, y in test_loader:
        batch_num += 1
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        pred_labels = logits.argmax(1)
        
        # Update metrics
        correct = (pred_labels == y).sum().item()
        all_correct += correct
        all_total += y.shape[0]
        
        # Store predictions and ground truth
        predictions.extend(pred_labels.cpu().numpy())
        ground_truth.extend(y.cpu().numpy())
        
        # Update confusion matrix
        for true_label, pred_label in zip(y.cpu().numpy(), pred_labels.cpu().numpy()):
            conf_matrix[true_label, pred_label] += 1
        
        if batch_num % max(1, len(test_loader) // 5) == 0:
            acc = 100 * all_correct / all_total
            print(f"  Batch {batch_num:3d}/{len(test_loader)}: {acc:6.2f}%")

# Calculate metrics
test_acc = 100 * all_correct / all_total

print(f"\n{'='*70}")
print("TEST RESULTS")
print(f"{'='*70}\n")

print(f"Overall Test Accuracy: {test_acc:.2f}%")
print(f"Samples: {all_total}\n")

# Per-class metrics
print("Per-Class Performance:")
print(f"{'Class':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 58)

per_class_acc = []
for i, class_name in enumerate(class_names):
    tp = conf_matrix[i, i]
    fp = conf_matrix[:, i].sum() - tp
    fn = conf_matrix[i, :].sum() - tp
    tn = conf_matrix.sum() - tp - fp - fn
    
    # Metrics
    accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    per_class_acc.append(accuracy)
    print(f"{class_name:<10} {accuracy*100:>10.2f}%  {precision*100:>10.2f}%  {recall*100:>10.2f}%  {f1:>10.4f}")

# Confusion matrix
print(f"\n{'='*70}")
print("CONFUSION MATRIX")
print(f"{'='*70}\n")

print("Rows: Ground Truth | Columns: Predictions\n")
print(f"{'':10}", end='')
for name in class_names:
    print(f"{name:>8}", end='')
print()

for i, true_class in enumerate(class_names):
    print(f"{true_class:<10}", end='')
    for j in range(3):
        print(f"{conf_matrix[i, j]:>8}", end='')
    print()

# Summary
print(f"\n{'='*70}")
print("TESTING COMPLETE")
print(f"{'='*70}\n")

print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Macro Avg F1: {np.mean([2 * (conf_matrix[i,i] / (conf_matrix[:, i].sum() + 1e-6)) * (conf_matrix[i,i] / (conf_matrix[i, :].sum() + 1e-6)) / (conf_matrix[i,i] / (conf_matrix[:, i].sum() + 1e-6) + conf_matrix[i,i] / (conf_matrix[i, :].sum() + 1e-6) + 1e-6) for i in range(3)]):.4f}")

# Save results
results_dir = Path('results/high_accuracy/evaluation')
results_dir.mkdir(parents=True, exist_ok=True)

# Save metrics
with open(results_dir / 'test_results.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("TEST RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Overall Test Accuracy: {test_acc:.2f}%\n")
    f.write(f"Total Samples: {all_total}\n\n")
    f.write("Per-Class Metrics:\n")
    f.write(f"{'Class':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
    for i, class_name in enumerate(class_names):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f.write(f"{class_name:<10} {accuracy*100:>10.2f}%  {precision*100:>10.2f}%  {recall*100:>10.2f}%  {f1:>10.4f}\n")

print(f"✓ Results saved to {results_dir}/test_results.txt\n")
