"""
Testing and Demo Script
Quick test to verify setup before full training
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

def test_pytorch():
    """Test PyTorch installation and GPU availability"""
    print("\n" + "="*70)
    print("TEST 1: PyTorch Installation")
    print("="*70)
    
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("ℹ CUDA not available - will use CPU (slower)")
    
    # Test tensor operations
    x = torch.randn(2, 3, 224, 224)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    print(f"✓ Tensor operations working on device: {device}")
    
    return True


def test_dependencies():
    """Test all required dependencies"""
    print("\n" + "="*70)
    print("TEST 2: Required Dependencies")
    print("="*70)
    
    dependencies = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('cv2', 'opencv-python'),
        ('tqdm', 'tqdm'),
        ('tensorboard', 'tensorboard'),
    ]
    
    all_ok = True
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - MISSING")
            all_ok = False
    
    return all_ok


def test_data_directories():
    """Test if data directories exist"""
    print("\n" + "="*70)
    print("TEST 3: Data Directories")
    print("="*70)
    
    base_path = Path(__file__).parent
    
    required_dirs = [
        ("DR images", base_path / "Diabetic Retinopathy Images" / "DR"),
        ("No_DR images", base_path / "Diabetic Retinopathy Images" / "No_DR"),
        ("HR training images", base_path / "Hypertensive Retinopathy Images" / 
                               "1-Hypertensive Classification" / "1-Images" / "1-Training Set"),
        ("HR labels CSV", base_path / "Hypertensive Retinopathy Images" / 
                           "1-Hypertensive Classification" / "2-Groundtruths"),
    ]
    
    all_ok = True
    for name, path in required_dirs:
        if path.exists():
            if path.is_dir():
                num_files = len(list(path.glob('*')))
                print(f"✓ {name}: {path.name} ({num_files} items)")
            else:
                print(f"✓ {name}: {path.name}")
        else:
            print(f"✗ {name}: NOT FOUND")
            print(f"  Expected: {path}")
            all_ok = False
    
    return all_ok


def test_model_creation():
    """Test model creation"""
    print("\n" + "="*70)
    print("TEST 4: Model Creation")
    print("="*70)
    
    try:
        from model import create_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        models_to_test = ['efficientnet_b0', 'resnet50']
        
        for model_name in models_to_test:
            model = create_model(
                num_classes=3,
                model_name=model_name,
                pretrained=True,
                device=device
            )
            
            # Test forward pass
            x = torch.randn(2, 3, 224, 224).to(device)
            with torch.no_grad():
                output = model(x)
            
            print(f"✓ {model_name}: model created, output shape {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_data_loading():
    """Test data loading pipeline"""
    print("\n" + "="*70)
    print("TEST 5: Data Loading")
    print("="*70)
    
    try:
        from data_loader import get_transforms
        from PIL import Image
        import numpy as np
        
        # Test transforms
        transforms_dict = get_transforms(image_size=224, augment=True)
        print("✓ Transforms created")
        
        # Create dummy image and test transform
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        for split in ['train', 'val', 'test']:
            transformed = transforms_dict[split](dummy_image)
            assert transformed.shape == (3, 224, 224), f"Invalid shape for {split}"
            print(f"✓ {split} transform: {transformed.shape}")
        
        print("✓ Data loading pipeline working")
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradcam():
    """Test GradCAM functionality"""
    print("\n" + "="*70)
    print("TEST 6: GradCAM Module")
    print("="*70)
    
    try:
        from gradcam import GradCAM
        from model import create_model
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create simple model
        model = create_model(num_classes=3, model_name='efficientnet_b0', 
                           pretrained=False, device=device)
        
        # Create GradCAM
        gradcam = GradCAM(model, target_layer='backbone.layer4', device=device)
        print("✓ GradCAM initialized")
        
        # Test with dummy input
        x = torch.randn(1, 3, 224, 224).to(device)
        cam, pred_class = gradcam.generate_cam(x)
        
        assert cam.shape == (7, 7), f"Invalid CAM shape: {cam.shape}"
        print(f"✓ CAM generation: shape {cam.shape}, predicted class {pred_class}")
        
        return True
        
    except Exception as e:
        print(f"✗ GradCAM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline():
    """Test complete pipeline"""
    print("\n" + "="*70)
    print("TEST 7: Complete Pipeline")
    print("="*70)
    
    try:
        print("Testing imports...")
        from data_preparation import load_hr_data, load_dr_data
        from train import Trainer
        from evaluate import Evaluator
        from gradcam import generate_gradcam_visualization
        
        print("✓ All modules imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  HR vs DR Classification Model - Setup Test Suite                    ║
    ║  ─────────────────────────────────────────────────────────────────  ║
    ║  Verifying all requirements before training                          ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("PyTorch", test_pytorch),
        ("Dependencies", test_dependencies),
        ("Data Directories", test_data_directories),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("GradCAM", test_gradcam),
        ("Pipeline", test_pipeline),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} test error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nYou're all set to start training. Run:")
        print("  python main.py --stage all")
        print("or")
        print("  python quick_start.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease fix the failed tests before proceeding.")
        print("See error messages above for details.")
        sys.exit(1)
    
    return all_passed


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
