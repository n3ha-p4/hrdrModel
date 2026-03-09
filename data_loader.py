"""
PyTorch Data Loaders for HR vs DR Images
Handles image loading, preprocessing, and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd


class RetinopathyDataset(Dataset):
    """Dataset class for retinopathy images"""
    
    def __init__(self, img_dir, transform=None, file_ext='*.png'):
        """
        Args:
            img_dir: Path to image directory
            transform: Transforms to apply to images
            file_ext: Image file extension
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Get all image files
        self.img_paths = sorted(list(self.img_dir.glob(file_ext)))
        
        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")
        
        print(f"Loaded {len(self.img_paths)} images from {img_dir}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """Get image and label based on directory structure"""
        img_path = self.img_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224))
        
        # Get label from parent directory name
        label = img_path.parent.name
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path)


class RetinopathyDatasetBalanced(Dataset):
    """Dataset class that handles balanced sampling"""
    
    def __init__(self, img_dir, transform=None, file_ext='*.png'):
        """
        Args:
            img_dir: Path to root directory containing subdirectories for each class
            transform: Transforms to apply
            file_ext: Image file extension
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Dictionary to store images by class
        self.class_images = {}
        self.class_to_idx = {}
        
        # Get all subdirectories (classes)
        class_dirs = sorted([d for d in self.img_dir.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            
            # Get all images in this class
            img_paths = sorted(list(class_dir.glob(file_ext)))
            self.class_images[class_name] = img_paths
            print(f"Class '{class_name}': {len(img_paths)} images")
        
        # Create a flat list of all images
        self.all_images = []
        for class_name, img_paths in self.class_images.items():
            for img_path in img_paths:
                self.all_images.append((img_path, class_name))
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path, class_name = self.all_images[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
        
        # Get label index
        label = self.class_to_idx[class_name]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path)


def get_transforms(image_size=224, augment=False):
    """
    Get data transforms
    
    Args:
        image_size: Size of output images
        augment: Whether to apply augmentation
    
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # Val/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def get_dataloaders(data_dir, batch_size=32, num_workers=4, augment=False, 
                    image_size=224, prefetch_factor=2):
    """
    Create DataLoaders for train/val/test
    
    Args:
        data_dir: Root data directory with train/val/test subdirectories
        batch_size: Batch size
        num_workers: Number of workers for data loading
        augment: Whether to use data augmentation
        image_size: Image size
        prefetch_factor: Prefetch factor for DataLoader
    
    Returns:
        Dictionary with dataloaders
    """
    
    transforms_dict = get_transforms(image_size=image_size, augment=augment)
    
    data_dir = Path(data_dir)
    dataloaders = {}
    dataset_sizes = {}
    class_names = None
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist")
            continue
        
        # Create dataset
        dataset = RetinopathyDatasetBalanced(
            img_dir=split_dir,
            transform=transforms_dict[split]
        )
        
        # Get class names from first dataset
        if class_names is None:
            class_names = sorted(list(dataset.class_to_idx.keys()))
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            persistent_workers=(num_workers > 0)
        )
        
        dataloaders[split] = dataloader
        dataset_sizes[split] = len(dataset)
    
    return dataloaders, dataset_sizes, class_names


if __name__ == "__main__":
    # Test data loading
    from pathlib import Path
    
    data_dir = Path("data/organized")
    
    if data_dir.exists():
        dataloaders, dataset_sizes, class_names = get_dataloaders(
            data_dir,
            batch_size=16,
            num_workers=0,
            augment=True
        )
        
        print(f"\nClass names: {class_names}")
        print(f"Dataset sizes: {dataset_sizes}")
        
        # Test batch loading
        if 'train' in dataloaders:
            for images, labels, paths in dataloaders['train']:
                print(f"Batch shapes - Images: {images.shape}, Labels: {labels}")
                break
    else:
        print(f"Data directory not found: {data_dir}")
