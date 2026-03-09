#!/usr/bin/env python3.7
"""
Monitor training and run evaluation when model is ready
"""

import time
import os
from pathlib import Path

model_path = Path('checkpoints/advanced/best_model.pt')

print("Monitoring for trained ResNet50 model...")
print("Will automatically run evaluation when model is saved.")
print("Press Ctrl+C to stop monitoring.\n")

while True:
    if model_path.exists():
        print(f"✓ Model found at {model_path}")
        print("Starting evaluation...")
        os.system('/usr/local/bin/python3.7 evaluate_95_advanced.py')
        break
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Waiting for model... (training still running)")
        time.sleep(60)  # Check every minute

print("Evaluation complete!")