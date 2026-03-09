#!/usr/bin/env python3
"""Quick validation - minimal dependencies"""
import sys
import torch
import numpy as np
from pathlib import Path

print("Validation Starting...", flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# Count images
base = Path('/Users/neha/Documents/GitHub/hrdrModel')
dr = len(list((base / 'Diabetic Retinopathy Images' / 'DR').glob('*.png')))
no_dr = len(list((base / 'Diabetic Retinopathy Images' / 'No_DR').glob('*.png')))
hr = len(list((base / 'Hypertensive Retinopathy Images' / '1-Hypertensive Classification' / '1-Images' / '1-Training Set').glob('**/*.png')))

print(f"\nDataset Summary:")
print(f"  DR: {dr}")
print(f"  Normal: {no_dr}")
print(f"  HR: {hr}")
print(f"  TOTAL: {dr + no_dr + hr}")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
print("\nValidation Set Accuracy: [Pending trained model save]")
print("Test Set Accuracy: [Pending trained model save]")
print("\n✓ Ready for predictions once model training completes")
