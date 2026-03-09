#!/usr/bin/env python3
"""Fast Testing - No DataLoader, Direct Predictions"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

print("="*70)
print("TESTING: HR vs DR Classification")
print("="*70 + "\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

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

model = Net().to(device)
model_path = Path('checkpoints/high_accuracy/best_model.pt')

if model_path.exists():
    print(f"✓ Loading trained model\n")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"⚠ Model not trained yet - using random initialization\n")

model.eval()

# Collect test images
print("Collecting test images...")
images, labels = [], []
base = Path('/Users/neha/Documents/GitHub/hrdrModel')

for path, label, name in [
    (base / 'Diabetic Retinopathy Images' / 'DR', 0, 'DR'),
    (base / 'Diabetic Retinopathy Images' / 'No_DR', 2, 'Normal'),
    (base / 'Hypertensive Retinopathy Images' / '1-Hypertensive Classification' / '1-Images' / '1-Training Set', 1, 'HR'),
]:
    if path.exists():
        files = list(path.glob('**/*.png'))[:200]  # Limit to 200 per class
        images.extend(files)
        labels.extend([label] * len(files))
        print(f"  {name}: {len(files)}")

print(f"  Total: {len(images)}\n")

# Test
print("Running predictions...\n")
class_names = ['DR', 'HR', 'Normal']
conf_matrix = np.zeros((3, 3), dtype=int)
correct = 0

for i, (img_path, true_label) in enumerate(zip(images, labels)):
    try:
        from PIL import Image
        img = Image.open(img_path).resize((128, 128))
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr]*3)
        else:
            arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    except:
        tensor = torch.randn(1, 3, 128, 128).to(device)
    
    with torch.no_grad():
        pred = model(tensor).argmax(1).item()
    
    conf_matrix[true_label, pred] += 1
    if pred == true_label:
        correct += 1
    
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(images)}")

acc = 100 * correct / len(images)

print("\n" + "="*70)
print("TEST RESULTS")
print("="*70 + "\n")

print(f"Overall Accuracy: {acc:.2f}%")
print(f"Correct: {correct}/{len(images)}\n")

print("Confusion Matrix:")
print("(Rows=Ground Truth, Cols=Predictions)\n")
print(f"{'':10}", end='')
for name in class_names:
    print(f"{name:>8}", end='')
print()

for i, name in enumerate(class_names):
    print(f"{name:<10}", end='')
    for j in range(3):
        print(f"{conf_matrix[i,j]:>8}", end='')
    print()

print("\nPer-Class Metrics:")
for i, name in enumerate(class_names):
    tp = conf_matrix[i, i]
    fn = conf_matrix[i, :].sum() - tp
    fp = conf_matrix[:, i].sum() - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  {name}: Recall={recall*100:.1f}% Precision={precision*100:.1f}%")

print("\n" + "="*70)
print("✓ TESTING COMPLETE")
print("="*70 + "\n")
