"""
Model Evaluation and Testing with GradCAM Visualization
"""

import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import cv2

from model import create_model
from data_loader import get_dataloaders, get_transforms
from gradcam import GradCAM
from PIL import Image


class Evaluator:
    """Evaluator class for model evaluation"""
    
    def __init__(self, model, device, class_names):
        """
        Args:
            model: Trained PyTorch model
            device: cuda or cpu
            class_names: List of class names
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_paths = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluating')
            for images, labels, paths in pbar:
                images = images.to(self.device)
                
                # Create label mapping
                label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
                numeric_labels = torch.tensor(
                    [label_to_idx[label] if isinstance(label, str) else label 
                     for label in labels]
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(numeric_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_paths.extend(list(paths))
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'classification_report': classification_report(
                all_labels, all_preds, target_names=self.class_names
            ),
            'precision_recall_f1': precision_recall_fscore_support(
                all_labels, all_preds, average='weighted'
            )
        }
        
        return metrics, all_preds, all_labels, all_probs, all_paths
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_metrics(self, metrics, save_dir=None):
        """Plot classification metrics"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract per-class metrics
        report_lines = metrics['classification_report'].split('\n')
        precisions = []
        recalls = []
        f1_scores = []
        
        for line in report_lines[2:-3]:  # Skip header and footer
            parts = line.split()
            if len(parts) >= 4:
                precisions.append(float(parts[1]))
                recalls.append(float(parts[2]))
                f1_scores.append(float(parts[3]))
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, precisions[:len(self.class_names)], width, label='Precision')
        ax.bar(x, recalls[:len(self.class_names)], width, label='Recall')
        ax.bar(x + width, f1_scores[:len(self.class_names)], width, label='F1-Score')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'metrics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to {save_path}")
        
        plt.close()


def evaluate_model(model_path, data_dir='data/organized', 
                   model_name='efficientnet_b0', batch_size=32, 
                   output_dir='evaluation'):
    """
    Main evaluation function
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to organized data
        model_name: Model architecture
        batch_size: Batch size
        output_dir: Directory to save evaluation results
    """
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        augment=False
    )
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = create_model(
        num_classes=len(class_names),
        model_name=model_name,
        pretrained=False,
        device=device
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate
    evaluator = Evaluator(model, device, class_names)
    
    print("\nEvaluating on test set...")
    metrics, preds, labels, probs, paths = evaluator.evaluate(
        dataloaders['test']
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_recall_f1'][0]:.4f}")
    print(f"Recall: {metrics['precision_recall_f1'][1]:.4f}")
    print(f"F1-Score: {metrics['precision_recall_f1'][2]:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("=" * 60)
    
    # Plot results
    evaluator.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=output_dir / 'confusion_matrix.png'
    )
    evaluator.plot_metrics(metrics, save_dir=output_dir)
    
    # Save detailed results
    results_text = f"""
HR vs DR Classification - Evaluation Results
{'='*60}

Overall Accuracy: {metrics['accuracy']:.4f}
Precision: {metrics['precision_recall_f1'][0]:.4f}
Recall: {metrics['precision_recall_f1'][1]:.4f}
F1-Score: {metrics['precision_recall_f1'][2]:.4f}

Classification Report:
{metrics['classification_report']}

Confusion Matrix:
{metrics['confusion_matrix']}
"""
    
    with open(output_dir / 'evaluation_results.txt', 'w') as f:
        f.write(results_text)
    
    print(f"\nResults saved to {output_dir}")
    
    return evaluator, metrics, preds, labels, probs, paths


def generate_gradcam_visualizations(model, model_path, test_image_dir, 
                                    class_names, num_samples=10, output_dir='gradcam_viz'):
    """
    Generate GradCAM visualizations for test images
    
    Args:
        model: Model instance
        model_path: Path to trained model
        test_image_dir: Directory containing test images
        class_names: List of class names
        num_samples: Number of samples per class
        output_dir: Directory to save visualizations
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Create GradCAM
    gradcam = GradCAM(model, target_layer='backbone.layer4', device=device)
    
    # Get transforms
    transforms_dict = get_transforms(image_size=224, augment=False)
    val_transform = transforms_dict['val']
    
    test_image_dir = Path(test_image_dir)
    
    # Process each class
    for class_name in class_names:
        class_dir = test_image_dir / class_name
        if not class_dir.exists():
            continue
        
        print(f"\nGenerating GradCAM for {class_name}...")
        
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get sample images
        image_files = list(class_dir.glob('*.png'))[:num_samples]
        
        for idx, img_path in enumerate(tqdm(image_files)):
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                image_tensor = val_transform(image).unsqueeze(0).to(device)
                
                # Generate GradCAM
                overlay, original, cam, pred_class = gradcam.generate_heatmap(
                    image_tensor
                )
                
                # Save visualizations
                # Original
                original_uint8 = np.uint8(original * 255)
                original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(class_output_dir / f'{idx:03d}_01_original.png'),
                    original_bgr
                )
                
                # CAM
                cam_colored = cv2.applyColorMap(
                    np.uint8(255 * cam),
                    cv2.COLORMAP_JET
                )
                cv2.imwrite(
                    str(class_output_dir / f'{idx:03d}_02_cam.png'),
                    cam_colored
                )
                
                # Overlay
                overlay_uint8 = np.uint8(overlay * 255)
                overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(class_output_dir / f'{idx:03d}_03_overlay.png'),
                    overlay_bgr
                )
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"Saved {len(image_files)} GradCAM visualizations for {class_name}")
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    model_path = 'checkpoints/best_model.pt'
    
    if Path(model_path).exists():
        evaluator, metrics, preds, labels, probs, paths = evaluate_model(
            model_path=model_path,
            data_dir='data/organized',
            model_name='efficientnet_b0',
            batch_size=32,
            output_dir='evaluation'
        )
    else:
        print(f"Model not found: {model_path}")
