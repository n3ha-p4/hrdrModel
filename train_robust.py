#!/usr/bin/env python3
"""
Robust Training with Error Handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import json
import os
import random

print("\n" + "="*70)
print("HR vs DR Training - Robust Version")
print("="*70 + "\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch {torch.__version__} | Device: {device}\n")

class SafeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.images = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
        except:
            # Return blank image if loading fails
            img = Image.new('RGB', (128, 128), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Transforms
train_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(0.3),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

print("Collecting valid images...\n")

images, labels = [], []

# DR
dr_dir = Path('Diabetic Retinopathy Images/DR')
if dr_dir.exists():
    for f in sorted(dr_dir.glob('*.png'))[:1000]:  # Limit
        if f.stat().st_size > 100:
            images.append(f)
            labels.append(0)
    print(f"  DR: {len([l for l in labels if l == 0])} images")

# Normal
normal_dir = Path('Diabetic Retinopathy Images/No_DR')
if normal_dir.exists():
    for f in sorted(normal_dir.glob('*.png'))[:1000]:
        if f.stat().st_size > 100:
            images.append(f)
            labels.append(2)
    print(f"  Normal: {len([l for l in labels if l == 2])} images")

# HR
hr_dir = Path('Hypertensive Retinopathy Images/1-Hypertensive Classification/1-Images/1-Training Set')
if hr_dir.exists():
    for f in sorted(hr_dir.glob('**/*.png'))[:1000]:
        if f.stat().st_size > 100:
            images.append(f)
            labels.append(1)
    print(f"  HR: {len([l for l in labels if l == 1])} images")

total = len(images)
print(f"\nTotal: {total}")

if total < 100:
    print("Not enough images!")
    exit(1)

# Split
combined = list(zip(images, labels))
random.shuffle(combined)
images, labels = zip(*combined)

train_sz = int(0.70 * total)
val_sz = int(0.15 * total)

train_ds = SafeDataset(list(images[:train_sz]), list(labels[:train_sz]), train_tf)
val_ds = SafeDataset(list(images[train_sz:train_sz+val_sz]), list(labels[train_sz:train_sz+val_sz]), val_tf)
test_ds = SafeDataset(list(images[train_sz+val_sz:]), list(labels[train_sz+val_sz:]), val_tf)

print(f"Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}\n")

train_loader = DataLoader(train_ds, 32, shuffle=True)
val_loader = DataLoader(val_ds, 32)
test_loader = DataLoader(test_ds, 32)

# Simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 3)
        )
    
    def forward(self, x):
        return self.net(x)

model = SimpleNet().to(device)
opt = optim.Adam(model.parameters(), 0.001)
loss_fn = nn.CrossEntropyLoss()

print("="*70)
print("TRAINING")
print("="*70 + "\n")

best_acc = 0
for ep in range(25):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        opt.zero_grad()
        out = model(x.to(device))
        loss = loss_fn(out, y.to(device))
        loss.backward()
        opt.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    model.eval()
    val_acc = 0
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x.to(device))
            val_acc += (out.argmax(1) == y.to(device)).float().mean()
    val_acc = 100 * val_acc / len(val_loader)
    
    mark = "✓" if val_acc > best_acc else " "
    print(f"Ep {ep+1:2d} | Loss: {train_loss:.4f} | Val: {val_acc:.1f}% {mark}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs('checkpoints/high_accuracy', exist_ok=True)
        torch.save(model.state_dict(), 'checkpoints/high_accuracy/best_model.pt')

model.eval()
test_acc = 0
with torch.no_grad():
    for x, y in test_loader:
        out = model(x.to(device))
        test_acc += (out.argmax(1) == y.to(device)).float().mean()
test_acc = 100 * test_acc / len(test_loader)

print("\n" + "="*70)
print(f"✓ COMPLETE | Best Val: {best_acc:.1f}% | Test: {test_acc:.1f}%")
print("="*70)
print(f"Model: checkpoints/high_accuracy/best_model.pt\n")
