"""
OPTIMIZED CONFIG FOR 95%+ ACCURACY
High-precision training configuration for HR vs DR classification
"""

class HighAccuracyConfig:
    """Configuration optimized for maximum accuracy (95%+)"""
    
    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
    
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    RANDOM_SEED = 42
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    
    # Use ResNet50 for better accuracy than EfficientNet
    MODEL_NAME = 'resnet50'
    NUM_CLASSES = 3
    CLASS_NAMES = ['DR', 'Normal', 'HR']
    IMAGE_SIZE = 256  # Larger image size for better detail
    PRETRAINED = True
    
    # ========================================================================
    # TRAINING CONFIGURATION FOR HIGH ACCURACY
    # ========================================================================
    
    BATCH_SIZE = 16  # Smaller batch for more gradient updates
    NUM_EPOCHS = 100  # More epochs for convergence
    LEARNING_RATE = 5e-4  # Lower LR for fine-tuning
    WEIGHT_DECAY = 1e-4
    
    # Aggressive learning rate scheduling
    LR_SCHEDULER = {
        'mode': 'max',
        'factor': 0.5,
        'patience': 5,  # More patient
        'verbose': True,
        'min_lr': 1e-6
    }
    
    EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs
    MAX_GRAD_NORM = 1.0
    
    # ========================================================================
    # DATA AUGMENTATION FOR HIGH ACCURACY
    # ========================================================================
    
    USE_AUGMENTATION = True
    
    # Moderate augmentation (not too aggressive to preserve medical details)
    AUGMENTATION = {
        'random_crop_scale': (0.9, 1.0),  # Small crops only
        'horizontal_flip': 0.3,
        'vertical_flip': 0.3,
        'rotation': 15,  # Limited rotation
        'color_brightness': 0.1,  # Conservative color changes
        'color_contrast': 0.1,
        'color_saturation': 0.1,
        'affine_translate': (0.05, 0.05)  # Small translation
    }
    
    # ========================================================================
    # REGULARIZATION FOR BETTER GENERALIZATION
    # ========================================================================
    
    DROPOUT_RATE = 0.4  # Higher dropout
    USE_AMP = False  # Disable mixed precision for stability
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    
    # ========================================================================
    # CHECKPOINT CONFIGURATION
    # ========================================================================
    
    CHECKPOINT_DIR = 'checkpoints/high_accuracy'
    SAVE_CHECKPOINT_INTERVAL = 5
    RESULTS_DIR = 'results/high_accuracy'
    
    # ========================================================================
    # EVALUATION CONFIGURATION
    # ========================================================================
    
    GRADCAM_SAMPLES_PER_CLASS = 15
    GRADCAM_TARGET_LAYER = 'backbone.layer4'
    GRADCAM_OVERLAY_ALPHA = 0.4
    
    # ========================================================================
    # DEVICE CONFIGURATION
    # ========================================================================
    
    DEVICE = 'auto'  # Auto-detect GPU/CPU
    
    # ========================================================================
    # TRAINING ENHANCEMENTS
    # ========================================================================
    
    # Enable ensemble predictions for final evaluation
    USE_ENSEMBLE = True
    ENSEMBLE_SIZE = 3
    
    # Warmup learning rate schedule
    WARMUP_EPOCHS = 5
    USE_COSINE_ANNEALING = True
    T_MAX = 100  # Total epochs
    
    # Label smoothing for regularization
    LABEL_SMOOTHING = 0.1
    
    # ========================================================================
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        config_dict = {}
        for key in dir(cls):
            if not key.startswith('_') and key.isupper():
                value = getattr(cls, key)
                from pathlib import Path
                if isinstance(value, Path):
                    value = str(value)
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("\n" + "="*70)
        print("HIGH ACCURACY CONFIG (Target: 95%+)")
        print("="*70)
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Batch Size: {cls.BATCH_SIZE} (smaller = more updates)")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Image Size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Augmentation: {'Enabled (conservative)' if cls.USE_AUGMENTATION else 'Disabled'}")
        print(f"Early Stopping: {cls.EARLY_STOPPING_PATIENCE} epochs")
        print(f"Label Smoothing: {cls.LABEL_SMOOTHING}")
        print(f"Cosine Annealing: {cls.USE_COSINE_ANNEALING}")
        print("="*70 + "\n")


# Export for use in training scripts
__all__ = ['HighAccuracyConfig']
