#!/usr/bin/env python3
"""
Fast Training Script for HR vs DR Classification
Works with Python 3.7+ and PyTorch 1.13+
Trains simple CNN from scratch (no external downloads needed)
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

print("\n" + "="*70)
print("HR vs DR Retinopathy Classification - Fast Training")
print("="*70 + "\n")

# Check data
data_dir = Path('data/organized')
if not (data_dir / 'train').exists():
    print("✗ No training data found!")
    print("  Run: python data_preparation.py")
    exit(1)

print("✓ Using PyTorch", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())

# Simple Dataset class
class RetinopathyDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.images = []
        self.labels = []
        self.class_map = {'DR': 0, 'HR': 1, 'Normal': 2}
        self.transform = transform
        
        for class_name in ['DR', 'HR', 'Normal']:
            class_path = split_dir / class_name
            if class_path.exists():
                for img_path in class_path.glob('*.png'):
                    self.images.append(img_path)
                    self.labels.append(self.class_map[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

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

# Load datasets
print("\nLoading data...")
train_ds = RetinopathyDataset(data_dir / 'train', train_tf)
val_ds = RetinopathyDataset(data_dir / 'val', val_tf)
test_ds = RetinopathyDataset(data_dir / 'test', val_tf)

print(f"  Train: {len(train_ds)} images")
print(f"  Val:   {len(val_ds)} images")
print(f"  Test:  {len(test_ds)} images")

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
test_loader = DataLoader(test_ds, batch_size)

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n✓ Model created on {device}")
print(f"  Simple CNN (32→64→128 channels)")
print(f"  Input size: 128×128")
print(f"  Batch size: {batch_size}")

# Training
print("\n" + "="*70)
print("TRAINING")
print("="*70 + "\n")

epochs = 30
best_val_acc = 0
history = {'train_loss': [], 'val_acc': []}

for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_acc = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            val_acc += (out.argmax(1) == lbls).float().mean().item()
    
    val_acc = val_acc / len(val_loader) * 100 if len(val_loader) > 0 else 0
    train_loss /= len(train_loader)
    
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    
    status = "✓" if val_acc > best_val_acc else " "
    print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.1f}% {status}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs('checkpoints/high_accuracy', exist_ok=True)
        torch.save(model.state_dict(), 'checkpoints/high_accuracy/best_model.pt')

# Test
model.eval()
test_acc = 0
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs)
        test_acc += (out.argmax(1) == lbls).float().mean().item()

test_acc = test_acc / len(test_loader) * 100 if len(test_loader) > 0 else 0

print("\n" + "="*70)
print(f"✓ TRAINING COMPLETE")
print("="*70)
print(f"Best validation accuracy: {best_val_acc:.1f}%")
print(f"Test accuracy: {test_acc:.1f}%")

# Save history
with open('checkpoints/high_accuracy/training_history.json', 'w') as f:
    json.dump(history, f)

print(f"\n✓ Model saved to: checkpoints/high_accuracy/best_model.pt")
print(f"✓ History saved\n")
