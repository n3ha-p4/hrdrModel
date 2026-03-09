"""
Data Preparation Script for HR vs DR Detection
Organizes images into train/val/test splits (70/15/15)
"""

import os
import shutil
import pandas as pd
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
random.seed(42)

# Define paths
BASE_DIR = Path(__file__).parent
DR_IMAGES_DIR = BASE_DIR / "Diabetic Retinopathy Images"
HR_IMAGES_DIR = BASE_DIR / "Hypertensive Retinopathy Images"

# Output data directory
DATA_DIR = BASE_DIR / "data" / "organized"

# HR specific paths
HR_TRAINING_SET = HR_IMAGES_DIR / "1-Hypertensive Classification" / "1-Images" / "1-Training Set"
HR_LABELS_CSV = HR_IMAGES_DIR / "1-Hypertensive Classification" / "2-Groundtruths" / "HRDC Hypertensive Classification Training Labels.csv"

# DR specific paths
DR_POSITIVE_DIR = DR_IMAGES_DIR / "DR"
DR_NEGATIVE_DIR = DR_IMAGES_DIR / "No_DR"

def create_directory_structure():
    """Create train/val/test directory structure"""
    splits = ["train", "val", "test"]
    classes = ["HR", "DR", "Normal"]
    
    for split in splits:
        for class_name in classes:
            split_dir = DATA_DIR / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created: {split_dir}")

def load_hr_data():
    """Load HR data with labels"""
    print("\nLoading HR data...")
    hr_labels = pd.read_csv(HR_LABELS_CSV)
    
    hr_images = []
    for idx, row in hr_labels.iterrows():
        img_path = HR_TRAINING_SET / row['Image']
        if img_path.exists():
            hr_images.append({
                'path': str(img_path),
                'label': 'HR' if row['Hypertensive'] == 1 else 'Normal',
                'original_label': row['Hypertensive']
            })
    
    print(f"Found {len(hr_images)} HR images")
    hr_df = pd.DataFrame(hr_images)
    print(hr_df['label'].value_counts())
    return hr_df

def load_dr_data():
    """Load DR data"""
    print("\nLoading DR data...")
    dr_images = []
    
    # Load DR positive cases
    for img_file in DR_POSITIVE_DIR.glob("*.png"):
        dr_images.append({
            'path': str(img_file),
            'label': 'DR',
            'original_label': 1
        })
    
    # Load DR negative cases (Normal fundus)
    for img_file in DR_NEGATIVE_DIR.glob("*.png"):
        dr_images.append({
            'path': str(img_file),
            'label': 'Normal',
            'original_label': 0
        })
    
    print(f"Found {len(dr_images)} DR images")
    dr_df = pd.DataFrame(dr_images)
    print(dr_df['label'].value_counts())
    return dr_df

def split_and_organize(data_df, disease_type="HR"):
    """Split data into train/val/test and copy to organized directories"""
    print(f"\nSplitting {disease_type} data into train/val/test...")
    
    # First split: 70% train, 30% temp (for val and test)
    train_df, temp_df = train_test_split(
        data_df, 
        test_size=0.30, 
        random_state=42,
        stratify=data_df['label'] if disease_type == "HR" else None
    )
    
    # Second split: Split the 30% into 50/50 (15% val, 15% test)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=42,
        stratify=temp_df['label'] if disease_type == "HR" else None
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Copy files to organized directories
    splits_data = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for split_name, split_data in splits_data.items():
        for idx, row in split_data.iterrows():
            src_file = Path(row['path'])
            dst_dir = DATA_DIR / split_name / row['label']
            dst_file = dst_dir / src_file.name
            
            if src_file.exists() and not dst_file.exists():
                shutil.copy2(src_file, dst_file)
    
    print(f"Organized {disease_type} images into train/val/test directories")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

def create_split_manifest(splits_data, filename):
    """Create a manifest file showing which images are in which split"""
    manifest = []
    for split_name, split_data in splits_data.items():
        for idx, row in split_data.iterrows():
            path = Path(row['path'])
            manifest.append({
                'split': split_name,
                'image_name': path.name,
                'label': row['label'],
                'original_path': row['path']
            })
    
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(DATA_DIR / filename, index=False)
    print(f"Created manifest: {DATA_DIR / filename}")

def main():
    print("=" * 60)
    print("HR vs DR Dataset Preparation")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Load and split data
    hr_df = load_hr_data()
    dr_df = load_dr_data()
    
    # Combine both datasets for mixed training
    combined_df = pd.concat([hr_df, dr_df], ignore_index=True)
    print(f"\nTotal combined images: {len(combined_df)}")
    print(combined_df['label'].value_counts())
    
    # Split combined data
    combined_splits = split_and_organize(combined_df, "Combined")
    
    # Create manifests
    create_split_manifest(combined_splits, "split_manifest.csv")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    for split_name in ['train', 'val', 'test']:
        split_dir = DATA_DIR / split_name
        total = 0
        print(f"\n{split_name.upper()}:")
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.png")))
                total += count
                print(f"  {class_dir.name}: {count}")
        print(f"  Total: {total}")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Organized data saved to: {DATA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
