#!/usr/bin/env python3
"""
Quick Setup and Run Guide for HR vs DR Classification Model
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*70}")
    print(f"📌 {description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=False)
        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully!")
            return True
        else:
            print(f"\n✗ {description} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n✗ Error during {description}: {e}")
        return False


def main():
    """Main function"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  HR vs DR Classification Model - Quick Setup & Run                   ║
    ║  ─────────────────────────────────────────────────────────────────  ║
    ║  Hypertensive Retinopathy vs Diabetic Retinopathy Detection          ║
    ║  with PyTorch and GradCAM Visualization                              ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version}")
    
    # Step 1: Install dependencies
    print("\n" + "="*70)
    print("STEP 1: Installing Dependencies")
    print("="*70)
    
    if not run_command(
        "pip install -r requirements.txt",
        "Installing Python packages"
    ):
        print("\n⚠ Warning: Some packages may have failed to install")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Step 2: Check directories
    print("\n" + "="*70)
    print("STEP 2: Checking Data Directories")
    print("="*70)
    
    dirs_to_check = [
        "Diabetic Retinopathy Images/DR",
        "Diabetic Retinopathy Images/No_DR",
        "Hypertensive Retinopathy Images/1-Hypertensive Classification/1-Images/1-Training Set"
    ]
    
    all_exist = True
    for dir_path in dirs_to_check:
        full_path = Path(dir_path)
        status = "✓" if full_path.exists() else "✗"
        print(f"{status} {dir_path}")
        if not full_path.exists():
            all_exist = False
    
    if not all_exist:
        print("\n⚠ Warning: Some data directories not found")
        print("Make sure all image folders are in the correct locations")
    
    # Step 3: Run complete pipeline
    print("\n" + "="*70)
    print("STEP 3: Running Complete Pipeline")
    print("="*70)
    print("""
    This will run all stages:
    1. Data Preparation (70% train, 15% val, 15% test)
    2. Model Training (EfficientNet-B0 by default)
    3. Model Evaluation
    4. GradCAM Visualization
    
    Note: This may take several hours depending on your hardware.
    """)
    
    response = input("Start training? (y/n): ")
    if response.lower() == 'y':
        # Show options
        print("\nSelect model architecture:")
        print("1. EfficientNet-B0 (default, fast, recommended)")
        print("2. ResNet50 (accurate, slower)")
        print("3. Vision Transformer (state-of-art, slowest)")
        
        model_input = input("\nChoice (1-3) [1]: ").strip() or "1"
        
        model_map = {
            "1": "efficientnet_b0",
            "2": "resnet50",
            "3": "vit_b16"
        }
        
        selected_model = model_map.get(model_input, "efficientnet_b0")
        print(f"\nSelected model: {selected_model}")
        
        # Get batch size
        batch_size_input = input("Enter batch size [32]: ").strip() or "32"
        batch_size = batch_size_input
        
        # Get number of epochs
        epochs_input = input("Enter number of epochs [50]: ").strip() or "50"
        num_epochs = epochs_input
        
        # Run pipeline
        cmd = (
            f"python main.py --stage all "
            f"--model-name {selected_model} "
            f"--batch-size {batch_size} "
            f"--num-epochs {num_epochs} "
            f"--augment True"
        )
        
        run_command(cmd, "Running complete pipeline")
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETED")
        print("="*70)
        print("""
        ✓ All stages completed!
        
        Output directories:
        • checkpoints/        - Model weights and config
        • results/evaluation/ - Metrics and confusion matrix
        • results/gradcam/    - Heatmap visualizations
        
        Next steps:
        1. Review results/evaluation/evaluation_results.txt
        2. View confusion matrix: results/evaluation/confusion_matrix.png
        3. Check GradCAM heatmaps: results/gradcam/**/
        """)
        
    elif response.lower() == 'n':
        print("\nSkipped training.")
    else:
        print("\nInvalid input.")
        return
    
    # Step 4: Run individual stage
    print("\n" + "="*70)
    print("OPTIONAL: Run Individual Stages")
    print("="*70)
    print("""
    You can run individual stages anytime:
    
    python main.py --stage prep      # Data preparation only
    python main.py --stage train     # Training only
    python main.py --stage eval      # Evaluation only
    python main.py --stage gradcam   # GradCAM only
    
    Example:
    python main.py --stage eval --data-dir data/organized
    """)
    
    # Final information
    print("\n" + "="*70)
    print("HELP & DOCUMENTATION")
    print("="*70)
    print("""
    📖 For detailed documentation, see: README.md
    
    📊 Project Structure:
    • data_preparation.py - Organize images into train/val/test
    • model.py            - Neural network architectures
    • data_loader.py      - PyTorch data handling
    • train.py            - Training loop
    • evaluate.py         - Evaluation and metrics
    • gradcam.py          - Heatmap visualization
    • main.py             - Main pipeline orchestrator
    
    🎯 Key Features:
    ✓ Multi-class classification (HR, DR, Normal)
    ✓ Multiple architecture support (EfficientNet, ResNet, ViT)
    ✓ Data augmentation for better generalization
    ✓ GradCAM explainability heatmaps
    ✓ Comprehensive evaluation metrics
    ✓ Automatic best model saving
    
    💡 Tips:
    • Start with EfficientNet-B0 for fast training
    • Use batch size 32-64 for optimal performance
    • Enable augmentation for better generalization
    • Check GradCAM heatmaps to validate model behavior
    """)
    
    print("\n" + "="*70)
    print("Setup complete! You're ready to train. 🚀")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
