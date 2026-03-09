#!/usr/bin/env python3
"""Quick data split script"""
from pathlib import Path
from shutil import copy2
import os

data_dir = Path('data/organized')
os.makedirs(data_dir / 'val' / 'DR', exist_ok=True)
os.makedirs(data_dir / 'val' / 'HR', exist_ok=True)
os.makedirs(data_dir / 'val' / 'Normal', exist_ok=True)
os.makedirs(data_dir / 'test' / 'DR', exist_ok=True)
os.makedirs(data_dir / 'test' / 'HR', exist_ok=True)
os.makedirs(data_dir / 'test' / 'Normal', exist_ok=True)

train_path = data_dir / 'train'
for cls in ['DR', 'HR', 'Normal']:
    files = sorted(list((train_path / cls).glob('*.png')))
    n = len(files)
    val_cnt = n // 5
    test_cnt = n // 5
    for i, f in enumerate(files[:val_cnt]):
        copy2(f, data_dir / 'val' / cls / f.name)
    for i, f in enumerate(files[val_cnt:val_cnt+test_cnt]):
        copy2(f, data_dir / 'test' / cls / f.name)
    print(f"{cls}: train={n-val_cnt-test_cnt} val={val_cnt} test={test_cnt}")
