#!/usr/bin/env python3.7
"""
Comprehensive Evaluation - Accuracy, Precision, Recall, F1 Score, Confusion Matrices
Evaluates the trained model with detailed metrics
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import warnings
import matplotlib.pyplot as plt
# import seaborn as sns  # Commented out due to installation issues

warnings.filterwarnings('ignore')

# ============================================================================
# METRICS (PyTorch/Numpy only - no sklearn)
# ============================================================================

def accuracy(pred, target):
    """Calculate accuracy"""
    return (pred == target).float().mean().item()

def precision_recall_f1(cm):
    """Calculate precision, recall, f1 from confusion matrix"""
    num_classes = cm.shape[0]
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
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
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    return precisions, recalls, f1s

def confusion_matrix_np(pred, target, num_classes):
    """Calculate confusion matrix using numpy"""
    pred = np.array(pred)
    target = np.array(target)
    cm = np.zeros((num_classes, num_classes))
    for i in range(len(pred)):
        cm[target[i], pred[i]] += 1
    return cm
print("COMPREHENSIVE MODEL EVALUATION WITH DETAILED METRICS")
print("="*80)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3
class_names = ['DR', 'HR', 'Normal']

# ============================================================================
# DATASET LOADER
# ============================================================================

class EvalDataset(torch.utils.data.Dataset):
    """Simple dataset loader for evaluation"""
    
    def __init__(self, data_dir, split='test'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.images = []
        self.labels = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.data_dir / split / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
        
        self.transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
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
    
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
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
# EVALUATION FUNCTION
# ============================================================================

def evaluate_comprehensive(model_path, data_split='test'):
    """Comprehensive evaluation with all metrics"""
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = RetinopathyClassifier(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Load dataset
    print(f"\nLoading {data_split} dataset...")
    dataset = EvalDataset('data/organized', split=data_split)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✓ Loaded {len(dataset)} images")
    
    # Predictions and ground truth
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"\nRunning inference...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            outputs = model(images)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx+1)*BATCH_SIZE}/{len(dataset)} samples")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ========================================================================
    # METRICS CALCULATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS")
    print("="*80)
    
    # Confusion matrix
    cm = confusion_matrix_np(all_preds, all_labels, NUM_CLASSES)
    
    # Overall accuracy
    accuracy = np.mean(all_preds == all_labels)
    
    # Per-class precision, recall, f1
    precisions, recalls, f1s = precision_recall_f1(cm)
    
    # Support (samples per class)
    supports = cm.sum(axis=1)
    
    # Weighted averages
    total_samples = len(all_labels)
    precision_weighted = np.sum(np.array(precisions) * supports) / total_samples
    recall_weighted = np.sum(np.array(recalls) * supports) / total_samples
    f1_weighted = np.sum(np.array(f1s) * supports) / total_samples
    
    # Macro averages
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    f1_macro = np.mean(f1s)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n  Precision (Weighted): {precision_weighted:.4f}")
    print(f"  Recall (Weighted): {recall_weighted:.4f}")
    print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"\n  Precision (Macro): {precision_macro:.4f}")
    print(f"  Recall (Macro): {recall_macro:.4f}")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    
    # Per-class metrics
    print(f"\n📋 PER-CLASS METRICS:")
    print(f"{'Class':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_support = int(supports[i])
        class_acc = cm[i, i] / class_support if class_support > 0 else 0  # per-class accuracy
        
        per_class_metrics[class_name] = {
            'accuracy': float(class_acc),
            'precision': float(precisions[i]),
            'recall': float(recalls[i]),
            'f1_score': float(f1s[i]),
            'support': class_support
        }
        
        print(f"{class_name:<12} {class_acc:<12.4f} {precisions[i]:<12.4f} {recalls[i]:<12.4f} {f1s[i]:<12.4f} {class_support:<10d}")
    
    # Confusion matrix display
    print(f"\n🔲 CONFUSION MATRIX:")
    header = 'True \\ Pred'
    print(f"{header:<12}", end='')
    for name in class_names:
        print(f"{name:<10}", end='')
    print()
    print("-" * 50)
    
    for i, true_class in enumerate(class_names):
        print(f"{true_class:<12}", end='')
        for j in range(NUM_CLASSES):
            print(f"{cm[i, j]:<10}", end='')
        print()
    
    # Classification report
    print(f"\n📑 CLASSIFICATION REPORT:")
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    results_dir = Path('results/advanced_95')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 VISUALIZATIONS SKIPPED (matplotlib issues)")
    # Visualization code commented out due to matplotlib issues
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(max_probs, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Confidence', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Distribution of Prediction Confidence', fontsize=12, fontweight='bold')
    plt.axvline(np.mean(max_probs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(max_probs):.3f}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Correct vs Incorrect predictions
    plt.subplot(1, 2, 2)
    correct = all_preds == all_labels
    correct_probs = max_probs[correct]
    incorrect_probs = max_probs[~correct]
    
    plt.hist(correct_probs, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
    plt.hist(incorrect_probs, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    plt.xlabel('Prediction Confidence', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Confidence: Correct vs Incorrect Predictions', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    confidence_path = results_dir / f'confidence_distribution_{data_split}.png'
    plt.savefig(confidence_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved confidence distribution: {confidence_path}")
    plt.close()
    
    # ========================================================================
    # SAVE DETAILED RESULTS
    # ========================================================================
    
    comprehensive_results = {
        'dataset': data_split,
        'total_samples': int(len(all_labels)),
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_score_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_score_macro': float(f1_macro)
        },
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
        # 'confidence_stats': {  # Commented out due to plotting skip
        #     'mean': float(np.mean(max_probs)),
        #     'std': float(np.std(max_probs)),
        #     'min': float(np.min(max_probs)),
        #     'max': float(np.max(max_probs))
        # }
    }
    
    results_path = results_dir / f'comprehensive_results_{data_split}.json'
    with open(results_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    print(f"  ✓ Saved results JSON: {results_path}")
    
    # Text report
    report_path = results_dir / f'evaluation_report_{data_split}.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: {data_split.upper()}\n")
        f.write(f"Total Samples: {len(all_labels)}\n")
        f.write(f"Model: ResNet50 (Advanced)\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"  Precision (Weighted): {precision_weighted:.4f}\n")
        f.write(f"  Recall (Weighted): {recall_weighted:.4f}\n")
        f.write(f"  F1 Score (Weighted): {f1_weighted:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        for class_name in class_names:
            if class_name in per_class_metrics:
                m = per_class_metrics[class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Accuracy: {m['accuracy']:.4f}\n")
                f.write(f"  Precision: {m['precision']:.4f}\n")
                f.write(f"  Recall: {m['recall']:.4f}\n")
                f.write(f"  F1-Score: {m['f1_score']:.4f}\n")
                f.write(f"  Support: {m['support']}\n")
        
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write(str(cm) + "\n")
        
        f.write("\n\nCLASSIFICATION REPORT:\n")
        f.write(report)
    
    print(f"  ✓ Saved text report: {report_path}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    
    return comprehensive_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Check for trained model
    model_paths = [
        'checkpoints/advanced/best_model.pt',
        'checkpoints/high_accuracy/best_model.pt'
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path is None:
        print("❌ No trained model found!")
        print("Please train a model first using train_95_advanced.py")
        return
    
    print(f"Using model: {model_path}")
    
    # Evaluate on validation and test sets
    for split in ['val', 'test']:
        print(f"\n\n{'='*80}")
        print(f"EVALUATING {split.upper()} SET")
        print(f"{'='*80}")
        
        try:
            evaluate_comprehensive(model_path, data_split=split)
        except Exception as e:
            print(f"Error evaluating {split}: {e}")


if __name__ == "__main__":
    main()
