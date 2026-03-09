"""
Main Execution Script for HR vs DR Detection Model
Orchestrates data preparation, training, evaluation, and visualization
"""

import argparse
import sys
from pathlib import Path
import json

from data_preparation import main as prepare_data
from train import train_model
from evaluate import evaluate_model, generate_gradcam_visualizations, Evaluator
from model import create_model


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='HR vs DR Detection Model Pipeline'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        choices=['prep', 'train', 'eval', 'gradcam', 'all'],
        default='all',
        help='Which stage to execute'
    )
    parser.add_argument('--data-dir', type=str, default='data/organized',
                        help='Data directory')
    parser.add_argument('--model-name', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet50', 'vit_b16'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--augment', type=bool, default=True,
                        help='Use data augmentation')
    parser.add_argument('--num-gradcam-samples', type=int, default=10,
                        help='Number of samples per class for GradCAM')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("HR vs DR CLASSIFICATION MODEL PIPELINE")
    print("=" * 70)
    
    # Stage 1: Data Preparation
    if args.stage in ['prep', 'all']:
        print("\n[STAGE 1] Data Preparation (70% train, 15% val, 15% test)")
        print("-" * 70)
        try:
            prepare_data()
            print("✓ Data preparation completed")
        except Exception as e:
            print(f"✗ Data preparation failed: {e}")
            return
    
    # Stage 2: Training
    if args.stage in ['train', 'all']:
        print("\n[STAGE 2] Model Training")
        print("-" * 70)
        try:
            model, history, class_names = train_model(
                data_dir=args.data_dir,
                model_name=args.model_name,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                save_dir=args.checkpoint_dir,
                augment=args.augment
            )
            print("✓ Training completed")
            
            # Save configuration
            config = {
                'model_name': args.model_name,
                'num_classes': len(class_names),
                'class_names': class_names,
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
                'learning_rate': args.learning_rate
            }
            config_path = Path(args.checkpoint_dir) / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✓ Configuration saved to {config_path}")
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Stage 3: Evaluation
    if args.stage in ['eval', 'all']:
        print("\n[STAGE 3] Model Evaluation")
        print("-" * 70)
        
        checkpoint_dir = Path(args.checkpoint_dir)
        model_path = checkpoint_dir / 'best_model.pt'
        
        if not model_path.exists():
            print(f"✗ Model checkpoint not found: {model_path}")
            return
        
        try:
            output_dir = Path(args.output_dir) / 'evaluation'
            evaluator, metrics, preds, labels, probs, paths = evaluate_model(
                model_path=str(model_path),
                data_dir=args.data_dir,
                model_name=args.model_name,
                batch_size=args.batch_size,
                output_dir=str(output_dir)
            )
            print("✓ Evaluation completed")
            
        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Stage 4: GradCAM Visualization
    if args.stage in ['gradcam', 'all']:
        print("\n[STAGE 4] GradCAM Visualization")
        print("-" * 70)
        
        checkpoint_dir = Path(args.checkpoint_dir)
        model_path = checkpoint_dir / 'best_model.pt'
        config_path = checkpoint_dir / 'config.json'
        
        if not model_path.exists():
            print(f"✗ Model checkpoint not found: {model_path}")
            return
        
        if not config_path.exists():
            print(f"✗ Configuration not found: {config_path}")
            return
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            class_names = config['class_names']
            
            # Create model
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = create_model(
                num_classes=len(class_names),
                model_name=args.model_name,
                pretrained=False,
                device=device
            )
            
            # Generate visualizations
            test_image_dir = Path(args.data_dir) / 'test'
            gradcam_output_dir = Path(args.output_dir) / 'gradcam'
            
            generate_gradcam_visualizations(
                model=model,
                model_path=str(model_path),
                test_image_dir=str(test_image_dir),
                class_names=class_names,
                num_samples=args.num_gradcam_samples,
                output_dir=str(gradcam_output_dir)
            )
            print("✓ GradCAM visualization completed")
            
        except Exception as e:
            print(f"✗ GradCAM visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nOutput directories:")
    print(f"  - Checkpoints: {args.checkpoint_dir}")
    print(f"  - Results: {args.output_dir}")
    print("\nResults include:")
    print("  - evaluation/: Confusion matrix, metrics plots")
    print("  - gradcam/: GradCAM heatmap visualizations")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
