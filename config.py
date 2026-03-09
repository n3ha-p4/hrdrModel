"""
Configuration File for HR vs DR Classification Model
Edit these settings to customize your training
"""

import json
from pathlib import Path


class Config:
    """Configuration class for the project"""
    
    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
    
    # Data directories
    DATA_DIR = Path("data/organized")
    DIABETIC_RETINOPATHY_DIR = Path("Diabetic Retinopathy Images")
    HYPERTENSIVE_RETINOPATHY_DIR = Path("Hypertensive Retinopathy Images")
    
    # Data split (must sum to 1.0)
    TRAIN_SPLIT = 0.70      # 70% training
    VAL_SPLIT = 0.15        # 15% validation
    TEST_SPLIT = 0.15       # 15% testing
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    
    # Model architecture: 'efficientnet_b0', 'resnet50', or 'vit_b16'
    MODEL_NAME = 'efficientnet_b0'
    
    # Number of output classes
    NUM_CLASSES = 3  # DR, Normal, HR
    
    # Class names
    CLASS_NAMES = ['DR', 'Normal', 'HR']
    
    # Input image size
    IMAGE_SIZE = 224
    
    # Use pretrained weights
    PRETRAINED = True
    
    # ========================================================================
    # TRAINING CONFIGURATION
    # ========================================================================
    
    # Batch size (adjust based on GPU memory)
    # 32-64 recommended for GPUs with 8GB+ memory
    # Reduce to 16 or 8 if out of memory
    BATCH_SIZE = 32
    
    # Number of training epochs
    NUM_EPOCHS = 50
    
    # Learning rate (initial)
    LEARNING_RATE = 1e-3
    
    # Weight decay (L2 regularization)
    WEIGHT_DECAY = 1e-4
    
    # Learning rate scheduler - ReduceLROnPlateau settings
    LR_SCHEDULER = {
        'mode': 'max',           # Maximize validation accuracy
        'factor': 0.5,           # Multiply LR by this when no improvement
        'patience': 3,           # Wait 3 epochs before reducing LR
        'verbose': True
    }
    
    # Gradient clipping
    MAX_GRAD_NORM = 1.0
    
    # ========================================================================
    # DATA AUGMENTATION CONFIGURATION
    # ========================================================================
    
    # Enable/disable data augmentation
    USE_AUGMENTATION = True
    
    # Augmentation parameters
    AUGMENTATION = {
        'random_crop_scale': (0.8, 1.0),      # Random resize crop
        'horizontal_flip': 0.5,                # 50% chance
        'vertical_flip': 0.5,                  # 50% chance
        'rotation': 20,                        # ±20 degrees
        'color_brightness': 0.2,
        'color_contrast': 0.2,
        'color_saturation': 0.2,
        'affine_translate': (0.1, 0.1)        # ±10% translation
    }
    
    # ========================================================================
    # DATA LOADING CONFIGURATION
    # ========================================================================
    
    # Number of workers for data loading
    # Increase for faster data loading on multi-core systems
    NUM_WORKERS = 4
    
    # Pin memory for faster GPU transfer
    PIN_MEMORY = True
    
    # Prefetch factor for DataLoader
    PREFETCH_FACTOR = 2
    
    # ========================================================================
    # CHECKPOINT AND SAVING CONFIGURATION
    # ========================================================================
    
    # Directory to save checkpoints
    CHECKPOINT_DIR = Path('checkpoints')
    
    # Save checkpoint every N epochs (0 to disable)
    SAVE_CHECKPOINT_INTERVAL = 5
    
    # Results output directory
    RESULTS_DIR = Path('results')
    
    # ========================================================================
    # EVALUATION CONFIGURATION
    # ========================================================================
    
    # Number of samples per class for GradCAM visualization
    GRADCAM_SAMPLES_PER_CLASS = 10
    
    # Target layer for GradCAM visualization
    GRADCAM_TARGET_LAYER = 'backbone.layer4'
    
    # GradCAM overlay alpha (transparency)
    GRADCAM_OVERLAY_ALPHA = 0.4
    
    # ========================================================================
    # DEVICE CONFIGURATION
    # ========================================================================
    
    # Device: 'cuda' for GPU, 'cpu' for CPU
    # Leave as 'auto' to automatically detect
    DEVICE = 'auto'  # Will use 'cuda' if available, else 'cpu'
    
    # ========================================================================
    # LOGGING AND VISUALIZATION
    # ========================================================================
    
    # Verbose output during training
    VERBOSE = True
    
    # Save training plots
    SAVE_PLOTS = True
    
    # ========================================================================
    # ADVANCED CONFIGURATION
    # ========================================================================
    
    # Mixed precision training (faster, may reduce accuracy)
    USE_AMP = False
    
    # Gradient accumulation steps (for larger effective batch size)
    ACCUMULATION_STEPS = 1
    
    # Early stopping patience (number of epochs with no improvement)
    # Set to 0 to disable early stopping
    EARLY_STOPPING_PATIENCE = 10
    
    # ========================================================================
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        config_dict = {}
        for key in dir(cls):
            if not key.startswith('_') and key.isupper():
                value = getattr(cls, key)
                # Convert Path objects to strings for JSON serialization
                if isinstance(value, Path):
                    value = str(value)
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def save(cls, path='config.json'):
        """Save configuration to JSON file"""
        config_dict = cls.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path='config.json'):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            setattr(cls, key, value)
        
        print(f"Configuration loaded from {path}")


def print_config():
    """Print current configuration"""
    print("\n" + "="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    
    config_dict = Config.to_dict()
    
    categories = {
        'DATA': [k for k in config_dict.keys() if 'DATA' in k or 'SPLIT' in k or 'SEED' in k],
        'MODEL': [k for k in config_dict.keys() if 'MODEL' in k or 'CLASS' in k or 'IMAGE' in k],
        'TRAINING': [k for k in config_dict.keys() if 'BATCH' in k or 'EPOCH' in k or 'LEARNING' in k or 'WEIGHT' in k],
        'AUGMENTATION': [k for k in config_dict.keys() if 'AUGMENT' in k],
        'DEVICE': [k for k in config_dict.keys() if 'DEVICE' in k],
        'SAVING': [k for k in config_dict.keys() if 'CHECKPOINT' in k or 'RESULTS' in k],
    }
    
    for category, keys in categories.items():
        if keys:
            print(f"\n{category}:")
            for key in sorted(keys):
                value = config_dict[key]
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Print current configuration
    print_config()
    
    # Save to file
    Config.save('config.json')
