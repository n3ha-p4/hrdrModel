#!/usr/bin/env python3
"""
Direct Training from Source Data
Reads from original data directories, bypassing copy failures
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
print("HR vs DR Training - Reading from Source Data")
print("="*70 + "\n")

print("✓ PyTorch", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Device: {device}")

# Collect all images from source
class DirectDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.images = image_list
        self.transform = transform
        self.class_map = {'DR': 0, 'HR': 1, 'Normal': 2}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        path, label = self.images[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Transforms
train_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

print("\nCollecting images from source...")

# Collect images
all_images = []

# HR images
hr_dir = Path('Hypertensive Retinopathy Images/1-Hypertensive Classification/1-Images/1-Training Set')
if hr_dir.exists():
    hr_files = list(hr_dir.glob('**/*.png'))
    for f in hr_files:
        all_images.append((f, 1))  # HR = 1
    print(f"  Found {len(hr_files)} HR images")

# DR images
dr_dir = Path('Diabetic Retinopathy Images/DR')
if dr_dir.exists():
    dr_files = list(dr_dir.glob('*.png'))
    for f in dr_files:
        all_images.append((f, 0))  # DR = 0
    print(f"  Found {len(dr_files)} DR images")

# Normal (No DR) images
normal_dir = Path('Diabetic Retinopathy Images/No_DR')
if normal_dir.exists():
    normal_files = list(normal_dir.glob('*.png'))
    for f in normal_files:
        all_images.append((f, 2))  # Normal = 2
    print(f"  Found {len(normal_files)} Normal images")

total = len(all_images)
print(f"\nTotal images: {total}")

if total < 100:
    print("\n✗ Not enough images found!")
    exit(1)

# Split into train/val/test (70/15/15)
random.shuffle(all_images)
train_size = int(0.70 * total)
val_size = int(0.15 * total)

train_imgs = all_images[:train_size]
val_imgs = all_images[train_size:train_size+val_size]
test_imgs = all_images[train_size+val_size:]

print(f"Split: train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

# Create datasets
train_ds = DirectDataset(train_imgs, train_tf)
val_ds = DirectDataset(val_imgs, val_tf)
test_ds = DirectDataset(test_imgs, val_tf)

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
test_loader = DataLoader(test_ds, batch_size)

print(f"Batch size: {batch_size}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*16*16, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 3)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n" + "="*70)
print("TRAINING")
print("="*70 + "\n")

best_val_acc = 0
for epoch in range(30):
    # Train
    model.train()
    train_loss = 0
    for imgs, lbls in train_loader:
        optimizer.zero_grad()
        out = model(imgs.to(device))
        loss = criterion(out, lbls.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            out = model(imgs.to(device))
            val_correct += (out.argmax(1) == lbls.to(device)).sum().item()
            val_total += lbls.size(0)
    
    val_acc = 100 * val_correct / val_total if val_total > 0 else 0
    train_loss /= len(train_loader)
    
    marker = "✓" if val_acc > best_val_acc else " "
    print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.1f}% {marker}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs('checkpoints/high_accuracy', exist_ok=True)
        torch.save(model.state_dict(), 'checkpoints/high_accuracy/best_model.pt')

# Test
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for imgs, lbls in test_loader:
        out = model(imgs.to(device))
        test_correct += (out.argmax(1) == lbls.to(device)).sum().item()
        test_total += lbls.size(0)

test_acc = 100 * test_correct / test_total if test_total > 0 else 0

print("\n" + "="*70)
print("✓ TRAINING COMPLETE")
print("="*70)
print(f"Best validation accuracy: {best_val_acc:.1f}%")
print(f"Test accuracy: {test_acc:.1f}%")
print(f"\n✓ Model saved to: checkpoints/high_accuracy/best_model.pt\n")
