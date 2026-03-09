#!/usr/bin/env python3
import torch
from pathlib import Path
import os

print("\n" + "="*70)
print("HR vs DR Model Training - Simplified Version")
print("="*70 + "\n")

print(f"PyTorch {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Quick test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Count images
dr_path = Path('Diabetic Retinopathy Images/DR')
no_dr_path = Path('Diabetic Retinopathy Images/No_DR')
hr_path = Path('Hypertensive Retinopathy Images/1-Hypertensive Classification/1-Images/1-Training Set')

dr_count = len(list(dr_path.glob('*.png'))) if dr_path.exists() else 0
no_dr_count = len(list(no_dr_path.glob('*.png'))) if no_dr_path.exists() else 0
hr_count = len(list(hr_path.glob('**/*.png'))) if hr_path.exists() else 0

print(f"\nDataset Composition:")
print(f"  DR (Diabetic):     {dr_count:5d}")
print(f"  Normal (No DR):    {no_dr_count:5d}")
print(f"  HR (Hypertensive): {hr_count:5d}")
print(f"  TOTAL:            {dr_count+no_dr_count+hr_count:5d}")

if dr_count + no_dr_count + hr_count < 100:
    print("\nNot enough training data!")
    exit(1)

# Try importing training components
try:
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
    import numpy  
    import random
    print("\n✓ All dependencies available")
except ImportError as e:
    print(f"\n✗ Missing dependency: {e}")
    exit(1)

print("\n" + "="*70)
print("TRAINING READY")
print("="*70)
print(f"\nRun: python3 train_robust.py")
print("Expected time: 10-30 minutes on CPU\n")
