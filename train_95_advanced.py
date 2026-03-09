#!/usr/bin/env python3.7
"""
Advanced Training for 95%+ Accuracy
- ResNet50 pre-trained backbone
- Advanced data augmentation
- Scikit-image enhancement techniques
- Extended training (100 epochs)
- Learning rate scheduling
- Class weighting
- Comprehensive metrics
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from skimage import exposure, restoration, filters, morphology, util
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("="*80)
print("ADVANCED TRAINING: 95%+ ACCURACY FOR RETINOPATHY CLASSIFICATION")
print("="*80)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 3
IMG_SIZE = 224  # ResNet50 expects 224x224

class_names = ['DR', 'HR', 'Normal']
class_weights = torch.tensor([1.0, 2.0, 1.0]).to(DEVICE)  # Balance HR (minority class)

print(f"\nConfiguration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Classes: {class_names}")


# ============================================================================
# ADVANCED IMAGE PREPROCESSING WITH SCIKIT-IMAGE
# ============================================================================

class RetinalImageProcessor:
    """Advanced preprocessing for retinal images using scikit-image"""
    
    @staticmethod
    def enhance_contrast(image):
        """CLAHE-like contrast enhancement"""
        try:
            # Convert to LAB for better contrast adjustment
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Equalize histogram adaptively
            img_enhanced = exposure.equalize_adapthist(img_array, clip_limit=0.03)
            
            return (img_enhanced * 255).astype(np.uint8)
        except:
            return image
    
    @staticmethod
    def remove_noise(image):
        """Denoise using non-local means"""
        try:
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Apply bilateral filter (better edge preservation)
            img_denoised = filters.median(img_array)
            
            return (img_denoised * 255).astype(np.uint8)
        except:
            return image
    
    @staticmethod
    def enhance_vessels(image):
        """Enhance blood vessel visibility using Frangi filter"""
        try:
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                img_gray = (0.299 * img_array[:,:,0] + 
                           0.587 * img_array[:,:,1] + 
                           0.114 * img_array[:,:,2])
            else:
                img_gray = img_array
            
            # Apply morphological operations to enhance vessels
            img_enhanced = morphology.black_tophat(img_gray, morphology.disk(5))
            
            # Stack back to 3 channels
            if len(np.array(image).shape) == 3:
                return np.stack([img_enhanced]*3, axis=-1)
            return img_enhanced
        except:
            return image


# ============================================================================
# ADVANCED DATA AUGMENTATION
# ============================================================================

class AdvancedAugmentation:
    """Advanced augmentation transformations"""
    
    def __init__(self, p=0.5):
        self.p = p
        self.processor = RetinalImageProcessor()
    
    def __call__(self, img):
        """Apply random augmentations"""
        
        # Preprocessing enhancement (always apply)
        img_array = np.array(img)
        
        # Random application of enhancements (p probability)
        if np.random.random() < self.p:
            img_array = self.processor.enhance_contrast(img_array)
        
        if np.random.random() < self.p:
            img_array = self.processor.remove_noise(img_array)
        
        # Convert back to PIL
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Apply geometric augmentations
        aug_transforms = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
        ])
        
        return aug_transforms(img)


# ============================================================================
# CUSTOM DATASET WITH ADVANCED AUGMENTATION
# ============================================================================

class RetinopathyDataset(Dataset):
    """Dataset with advanced augmentation"""
    
    def __init__(self, data_dir, split='train', augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.images = []
        self.labels = []
        
        # Load image paths and labels
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.data_dir / split / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images for {split}")
        
        # Base transforms
        self.transforms_base = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Augmentation
        self.augmentation = None
        if augment and split == 'train':
            self.augmentation = AdvancedAugmentation(p=0.7)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        
        # Apply augmentation if available
        if self.augmentation is not None:
            img = self.augmentation(img)
        
        # Apply base transforms
        img = self.transforms_base(img)
        
        return img, self.labels[idx]


# ============================================================================
# RESNET50 MODEL
# ============================================================================

class RetinopathyClassifier(nn.Module):
    """ResNet50-based classifier"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        
        # Add dropout and additional layers for regularization
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
# TRAINING FUNCTION
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    # Create datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    train_dataset = RetinopathyDataset('data/organized', split='train', augment=True)
    val_dataset = RetinopathyDataset('data/organized', split='val', augment=False)
    test_dataset = RetinopathyDataset('data/organized', split='test', augment=False)
    
    # Calculate class weights for imbalanced data
    label_counts = np.bincount(train_dataset.labels)
    class_weights_calc = 1.0 / label_counts
    class_weights_calc = class_weights_calc / class_weights_calc.sum()
    
    print(f"\nClass distribution:")
    for i, count in enumerate(label_counts):
        print(f"  {class_names[i]}: {count} samples (weight: {class_weights_calc[i]:.3f})")
    
    # Create data loaders with weighted sampling
    sampler = WeightedRandomSampler(
        weights=[class_weights_calc[label] for label in train_dataset.labels],
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    model = RetinopathyClassifier(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints/advanced')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
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
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Saved final model to {final_path}")
    
    # Save history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to {history_path}")
    
    # ========================================================================
    # TESTING
    # ========================================================================
    
    print("\n" + "="*80)
    print("EVALUATION ON TEST SET")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nTest Results:")
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (weighted): {precision_weighted:.4f}")
    print(f"  Recall (weighted): {recall_weighted:.4f}")
    print(f"  F1 Score (weighted): {f1_weighted:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(
                np.array(all_labels)[class_mask],
                np.array(all_preds)[class_mask]
            )
            class_prec = precision_score(
                np.array(all_labels)[class_mask],
                np.array(all_preds)[class_mask],
                average='binary' if i > 1 else 'weighted',
                zero_division=0
            )
            class_rec = recall_score(
                np.array(all_labels)[class_mask],
                np.array(all_preds)[class_mask],
                average='binary' if i > 1 else 'weighted',
                zero_division=0
            )
            print(f"  {class_name}: Acc={class_acc:.4f}, Prec={class_prec:.4f}, Rec={class_rec:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Save results
    results_dir = Path('results/advanced_95')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'overall_accuracy': float(accuracy),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'test_samples': len(all_labels)
    }
    
    results_path = results_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {results_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = results_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrix plot to {cm_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Model: {checkpoint_path}")
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(f"Target: 95%+ {'✓ ACHIEVED!' if accuracy >= 0.95 else '(aim for 95% or higher)'}")


if __name__ == "__main__":
    main()
