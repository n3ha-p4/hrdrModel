"""
High-Accuracy Training Script for HR vs DR Classification
Optimized to achieve 95%+ accuracy with advanced techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings

from model import create_model
from data_loader import get_dataloaders, get_transforms
from config_high_accuracy import HighAccuracyConfig

warnings.filterwarnings('ignore')


class HighAccuracyTrainer:
    """Advanced trainer for high accuracy model training"""
    
    def __init__(self, model, device, config):
        """
        Args:
            model: PyTorch model
            device: cuda or cpu
            config: Configuration object
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        
        # Optimizer - AdamW for better generalization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.WARMUP_EPOCHS
        )
        
        # Main scheduler - Cosine annealing with warmup
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.T_MAX,
            eta_min=1e-6
        )
        
        # Combine schedulers
        class WarmupCosineScheduler:
            def __init__(self, warmup_sched, main_sched, warmup_epochs):
                self.warmup_sched = warmup_sched
                self.main_sched = main_sched
                self.warmup_epochs = warmup_epochs
                self.epoch = 0
            
            def step(self):
                if self.epoch < self.warmup_epochs:
                    self.warmup_sched.step()
                else:
                    self.main_sched.step()
                self.epoch += 1
            
            def get_last_lr(self):
                if self.epoch < self.warmup_epochs:
                    return self.warmup_sched.get_last_lr()
                return self.main_sched.get_last_lr()
        
        self.scheduler = WarmupCosineScheduler(
            warmup_scheduler, main_scheduler, config.WARMUP_EPOCHS
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader, class_names):
        """Train for one epoch with mixed precision"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device)
            
            # Convert labels
            label_to_idx = {name: idx for idx, name in enumerate(class_names)}
            numeric_labels = torch.tensor(
                [label_to_idx[label] if isinstance(label, str) else label for label in labels]
            ).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, numeric_labels)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == numeric_labels.data).item()
            total += images.size(0)
            
            current_acc = running_corrects / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, class_names):
        """Validate model with detailed metrics"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating', leave=False)
            for images, labels, _ in pbar:
                images = images.to(self.device)
                
                # Convert labels
                label_to_idx = {name: idx for idx, name in enumerate(class_names)}
                numeric_labels = torch.tensor(
                    [label_to_idx[label] if isinstance(label, str) else label for label in labels]
                ).to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, numeric_labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == numeric_labels.data).item()
                total += images.size(0)
                
                current_acc = running_corrects / total
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, class_names, save_dir='checkpoints'):
        """Main training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_path = save_dir / 'best_model.pt'
        
        print("\n" + "="*70)
        print("HIGH ACCURACY TRAINING - TARGET: 95%+")
        print("="*70)
        self.config.print_config()
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, class_names)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, class_names)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Print results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                pct = self.best_val_acc * 100
                print(f"✓ Best model saved! Validation Accuracy: {pct:.2f}%")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠ Early stopping triggered (no improvement for {self.patience_counter} epochs)")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_CHECKPOINT_INTERVAL == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                    'best_val_acc': self.best_val_acc
                }, checkpoint_path)
        
        print("\n" + "="*70)
        print(f"TRAINING COMPLETE!")
        print(f"Best Validation Accuracy: {self.best_val_acc*100:.2f}%")
        print("="*70)
        
        return self.history


def train_for_high_accuracy(data_dir='data/organized', save_dir='checkpoints/high_accuracy'):
    """
    Main training function optimized for 95%+ accuracy
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = HighAccuracyConfig()
    
    # Load data with larger image size
    print("\nLoading data...")
    
    # Create custom transforms for larger images
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=config.AUGMENTATION['horizontal_flip']),
        transforms.RandomVerticalFlip(p=config.AUGMENTATION['vertical_flip']),
        transforms.RandomRotation(config.AUGMENTATION['rotation']),
        transforms.ColorJitter(
            brightness=config.AUGMENTATION['color_brightness'],
            contrast=config.AUGMENTATION['color_contrast'],
            saturation=config.AUGMENTATION['color_saturation']
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=config.AUGMENTATION['affine_translate']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Monkey-patch transforms for data loading
    import data_loader as dl
    original_get_transforms = dl.get_transforms
    
    def custom_get_transforms(image_size=config.IMAGE_SIZE, augment=False):
        if augment:
            return {'train': train_transform, 'val': val_transform, 'test': val_transform}
        return {'train': val_transform, 'val': val_transform, 'test': val_transform}
    
    dl.get_transforms = custom_get_transforms
    
    dataloaders, dataset_sizes, class_names = dl.get_dataloaders(
        data_dir=data_dir,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        augment=config.USE_AUGMENTATION,
        image_size=config.IMAGE_SIZE
    )
    
    print(f"Class names: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    
    # Create model
    print(f"\nCreating model: {config.MODEL_NAME}")
    model = create_model(
        num_classes=len(class_names),
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        device=device
    )
    
    # Create trainer
    trainer = HighAccuracyTrainer(model, device, config)
    
    # Train
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        class_names=class_names,
        save_dir=save_dir
    )
    
    # Save configuration and history
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    config_path = save_dir_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    history_path = save_dir_path / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Checkpoints saved to: {save_dir}")
    print(f"✓ Best model: {save_dir}/best_model.pt")
    
    return model, history, class_names


if __name__ == "__main__":
    train_for_high_accuracy()
