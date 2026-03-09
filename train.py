"""
Training script for HR vs DR classification model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

from model import create_model
from data_loader import get_dataloaders


class Trainer:
    """Trainer class for model training and validation"""
    
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-4):
        """
        Args:
            model: PyTorch model
            device: cuda or cpu
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
        """
        self.model = model
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader, class_names):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels, _ in pbar:
            images = images.to(self.device)
            
            # Create label mapping
            label_to_idx = {name: idx for idx, name in enumerate(class_names)}
            numeric_labels = torch.tensor(
                [label_to_idx[label] if isinstance(label, str) else label for label in labels]
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, numeric_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == numeric_labels.data).item()
            total += images.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(running_corrects / total):.4f}'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, class_names):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for images, labels, _ in pbar:
                images = images.to(self.device)
                
                # Create label mapping
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
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(running_corrects / total):.4f}'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, class_names, num_epochs=50, 
              save_dir='checkpoints', save_interval=5):
        """
        Train model
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            class_names: List of class names
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N epochs
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_acc = 0.0
        best_model_path = save_dir / f'best_model.pt'
        
        print("=" * 60)
        print(f"Starting training for {num_epochs} epochs")
        print(f"Saving to: {save_dir}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, class_names)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, class_names)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Best model saved! (Acc: {best_acc:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best validation accuracy: {best_acc:.4f}")
        print("=" * 60)
        
        return self.history


def train_model(data_dir='data/organized', model_name='efficientnet_b0', 
                batch_size=32, num_epochs=50, learning_rate=1e-3, 
                save_dir='checkpoints', augment=True):
    """
    Main training function
    
    Args:
        data_dir: Path to organized data
        model_name: Model architecture to use
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Initial learning rate
        save_dir: Directory to save checkpoints
        augment: Whether to use data augmentation
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        augment=augment
    )
    
    print(f"Class names: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    
    # Create model
    print(f"\nCreating model: {model_name}")
    model = create_model(
        num_classes=len(class_names),
        model_name=model_name,
        pretrained=True,
        device=device
    )
    
    # Create trainer
    trainer = Trainer(model, device, learning_rate=learning_rate)
    
    # Train
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        class_names=class_names,
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    # Save training history
    history_path = Path(save_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    return model, history, class_names


if __name__ == "__main__":
    # Train model
    model, history, class_names = train_model(
        data_dir='data/organized',
        model_name='efficientnet_b0',
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-3,
        save_dir='checkpoints',
        augment=True
    )
