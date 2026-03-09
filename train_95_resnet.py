#!/usr/bin/env python3.7
"""
Fast ResNet50 Training - 95%+ Accuracy
Uses PyTorch only (no sklearn dependency)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("RESNET50 TRAINING - 95%+ ACCURACY")
print("="*80)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

EPOCHS = 50  # Reduced for faster training on CPU
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 3
IMG_SIZE = 224

class_names = ['DR', 'HR', 'Normal']

print(f"\nConfiguration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Classes: {class_names}")


# ============================================================================
# DATASET
# ============================================================================

class RetinopathyDataset(Dataset):
    """Dataset with basic augmentation"""
    
    def __init__(self, data_dir, split='train', augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.images = []
        self.labels = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.data_dir / split / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.png')):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} {split} images")
        
        # Augmentation
        if augment and split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = self.transforms(img)
        return img, self.labels[idx]


# ============================================================================
# MODEL
# ============================================================================

class RetinopathyClassifier(nn.Module):
    """ResNet50-based classifier"""
    
    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        try:
            # Try loading pre-trained (may fail due to SSL)
            self.backbone = models.resnet50(pretrained=pretrained)
        except:
            print("⚠️  Using ResNet50 without pre-training (SSL issue)")
            # Load without pre-training
            self.backbone = models.resnet50(pretrained=False)
        
        in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# METRICS (PyTorch only - no sklearn)
# ============================================================================

def accuracy(pred, target):
    """Calculate accuracy"""
    return (pred == target).float().mean().item()

def confusion_matrix_np(pred, target, num_classes):
    """Calculate confusion matrix using numpy"""
    pred = np.array(pred)
    target = np.array(target)
    cm = np.zeros((num_classes, num_classes))
    for i in range(len(pred)):
        cm[target[i], pred[i]] += 1
    return cm


# ============================================================================
# TRAINING
# ============================================================================

def main():
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_dataset = RetinopathyDataset('data/organized', split='train', augment=True)
    val_dataset = RetinopathyDataset('data/organized', split='val', augment=False)
    test_dataset = RetinopathyDataset('data/organized', split='test', augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    model = RetinopathyClassifier(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    checkpoint_dir = Path('checkpoints/advanced')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        
        # TRAIN
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_acc = accuracy(torch.tensor(train_preds), torch.tensor(train_labels))
        
        # VALIDATE
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy(torch.tensor(val_preds), torch.tensor(val_labels))
        
        scheduler.step()
        
        print(f"Loss - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        print(f"Acc  - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n✓ Training complete. Best accuracy: {best_val_acc:.4f}")
    
    # ========================================================================
    # TESTING
    # ========================================================================
    
    print("\n" + "="*80)
    print("EVALUATION ON TEST SET")
    print("="*80)
    
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())
            test_probs.extend(probs.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)
    
    # Calculate metrics
    test_acc = accuracy(torch.tensor(test_preds), torch.tensor(test_labels))
    cm = confusion_matrix_np(test_preds, test_labels, NUM_CLASSES)
    
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(cm.astype(int))
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        if (tp + fp) > 0:
            prec = tp / (tp + fp)
        else:
            prec = 0
        
        if (tp + fn) > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0
        
        if (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0
        
        print(f"  {class_name}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    # Save results
    results_dir = Path('results/advanced_95')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'accuracy': float(test_acc),
        'accuracy_percent': float(test_acc * 100),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'test_samples': len(test_labels),
        'status': '✓ COMPLETE'
    }
    
    results_path = results_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Model: {checkpoint_path}")
    print(f"Final Accuracy: {test_acc*100:.2f}%")
    print(f"Target: 95%+ {'✓ ACHIEVED!' if test_acc >= 0.95 else f'(current: {test_acc*100:.1f}%)'}")
    print(f"\nNext step: Run evaluate_95_advanced.py for detailed metrics")


if __name__ == "__main__":
    main()
