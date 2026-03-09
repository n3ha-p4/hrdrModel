#!/usr/bin/env python
"""
Fallback Training Script for Python 3.7+ with PyTorch 1.13+
Simplified version that works with older Python/PyTorch versions
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Ensure data is organized first
print("\n" + "="*60)
print("HR vs DR Retinopathy Classification")
print("="*60)
print(f"Python version: {sys.version}")
print("="*60 + "\n")

# Check if data is organized
data_dir = Path('data/organized')
if not data_dir.exists():
    print("ERROR: Data directory not organized!")
    print("  Run: python3 data_preparation.py")
    sys.exit(1)

# Count data
print("Data Status:")
total_images = 0
for split in ['train', 'val', 'test']:
    count = 0
    split_path = data_dir / split
    if split_path.exists():
        for cls_dir in split_path.iterdir():
            if cls_dir.is_dir():
                images = len(list(cls_dir.glob('*.png')))
                count += images
    total_images += count
    print(f"  {split:10s}: {count:4d} images")

print(f"  {'TOTAL':10s}: {total_images:4d} images")

if total_images == 0:
    print("\nERROR: No images found in data/organized/")
    print("  Ensure data_preparation.py ran successfully")
    sys.exit(1)

print("\n✓ Data ready!")
print("\n" + "="*60)
print("Attempting to import PyTorch...")
print("="*60 + "\n")

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    print(f"✓ PyTorch {torch.__version__} imported successfully!")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    print("\nTo install PyTorch, run ONE of these commands:")
    print("\n=== Option 1: Using pip (Python 3.7) ===")
    print("python3.7 -m pip install torch==1.13.1 torchvision==0.14.1")
    print("\n=== Option 2: Using conda (Python 3.10) ===")
    print("conda create -n hr_dr python=3.10")
    print("conda activate hr_dr")
    print("conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
    print("\n=== Option 3: Apple Silicon / CPU ===")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Note: Training on CPU will be slower (5-20x)")

print("\n" + "="*60)
print("Loading data...")
print("="*60 + "\n")

# Simple data loading
class SimpleRetinopathyDataset(Dataset):
    """Simple dataset loader for retinopathy images"""
    def __init__(self, image_dir, transform=None):
        self.images = []
        self.labels = []
        self.class_to_idx = {'DR': 0, 'HR': 1, 'Normal': 2}
        self.transform = transform or transforms.ToTensor()
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = Path(image_dir) / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*.png'):
                    self.images.append(str(img_file))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = SimpleRetinopathyDataset(data_dir / 'train', train_transform)
val_dataset = SimpleRetinopathyDataset(data_dir / 'val', val_transform)
test_dataset = SimpleRetinopathyDataset(data_dir / 'test', val_transform)

print(f"Train: {len(train_dataset)} images")
print(f"Val:   {len(val_dataset)} images")
print(f"Test:  {len(test_dataset)} images")

# Create dataloaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

print(f"\nBatch size: {batch_size}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

print("\n" + "="*60)
print("Building model...")
print("="*60 + "\n")

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)  # 3 classes: DR, HR, Normal
model = model.to(device)

print("✓ ResNet50 model created")

# Training configuration
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 50  # Reduced for faster testing
patience = 10
best_val_acc = 0
patience_counter = 0

print(f"\nTraining Configuration:")
print(f"  Epochs: {num_epochs}")
print(f"  Optimizer: Adam (lr=1e-4)")
print(f"  Loss: CrossEntropyLoss")
print(f"  Early stopping patience: {patience}")

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

# Training loop
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

try:
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            os.makedirs('checkpoints/high_accuracy', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/high_accuracy/best_model.pt')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save history
    history_path = Path('checkpoints/high_accuracy/training_history.json')
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Test phase
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60 + "\n")
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    print("\n" + "="*60)
    print("✓ Training pipeline complete!")
    print("="*60)
    print("\nResults saved to:")
    print(f"  Model: checkpoints/high_accuracy/best_model.pt")
    print(f"  History: checkpoints/high_accuracy/training_history.json")
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")

except KeyboardInterrupt:
    print("\n\n✗ Training interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
