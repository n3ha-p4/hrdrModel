# HR vs DR Classification Model - Complete Pipeline

## Project Overview

This project implements a comprehensive AI solution for detecting and distinguishing between **Hypertensive Retinopathy (HR)** and **Diabetic Retinopathy (DR)** using deep learning with PyTorch. The model includes explainability through **GradCAM** heatmap visualization to show which regions of fundus images the model focuses on during classification.

### Research Question
Can an AI model accurately detect and distinguish between Diabetic Retinopathy and Hypertensive Retinopathy using fundus images?

### Goal
Create and test an AI model that can identify the differences between HR and DR with improved accuracy compared to existing real-world models using fundus images.

---

## Dataset Information

The project combines two datasets:

### Diabetic Retinopathy Images
- **DR (Positive cases)**: 1,857 images
- **No_DR (Negative cases)**: 1,805 images
- Total: 3,662 images

### Hypertensive Retinopathy Images
- **HR (Positive cases)**: ~356-400 images (label = 1)
- **Normal**: ~312-356 images (label = 0)
- Total: ~712 images

### Data Split
All data is automatically split into:
- **Training Set**: 70% (used for model training)
- **Validation Set**: 15% (used for hyperparameter tuning)
- **Test Set**: 15% (used for evaluation)

**Classes** (3-way classification):
- `DR`: Diabetic Retinopathy positive
- `Normal`: Normal fundus images
- `HR`: Hypertensive Retinopathy positive

---

## Project Structure

```
hrdrModel/
├── data_preparation.py          # Data organization script
├── model.py                      # PyTorch model architectures
├── data_loader.py                # Data loading and preprocessing
├── train.py                      # Training script
├── evaluate.py                   # Evaluation and visualization
├── gradcam.py                    # GradCAM heatmap generation
├── main.py                       # Main pipeline orchestrator
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── data/                         # Organized data directory (created)
│   └── organized/
│       ├── train/                # Training images
│       ├── val/                  # Validation images
│       └── test/                 # Test images
│
├── checkpoints/                  # Model checkpoints (created)
│   ├── best_model.pt
│   ├── config.json
│   └── training_history.json
│
└── results/                      # Results and visualizations (created)
    ├── evaluation/               # Evaluation metrics and plots
    └── gradcam/                  # GradCAM visualizations
```

---

## Installation

### 1. Clone/Navigate to Project
```bash
cd /Users/neha/Documents/GitHub/hrdrModel
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate

# Or using conda
conda create -n hr_dr_detection python=3.10
conda activate hr_dr_detection
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**GPU Support (Optional):**
If you have CUDA-capable GPU:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)
Runs all stages: data prep → training → evaluation → GradCAM

```bash
python main.py --stage all \
  --model-name efficientnet_b0 \
  --batch-size 32 \
  --num-epochs 50 \
  --learning-rate 1e-3 \
  --augment True
```

### Option 2: Run Individual Stages

**Stage 1: Data Preparation**
```bash
python main.py --stage prep
```

**Stage 2: Training**
```bash
python main.py --stage train \
  --model-name efficientnet_b0 \
  --batch-size 32 \
  --num-epochs 50 \
  --learning-rate 1e-3
```

**Stage 3: Evaluation**
```bash
python main.py --stage eval
```

**Stage 4: GradCAM Visualization**
```bash
python main.py --stage gradcam --num-gradcam-samples 10
```

---

## Usage Examples

### Custom Configuration

```bash
# Using ResNet50 with different parameters
python main.py --stage all \
  --model-name resnet50 \
  --batch-size 64 \
  --num-epochs 100 \
  --learning-rate 5e-4 \
  --output-dir custom_results

# Using Vision Transformer with augmentation
python main.py --stage train \
  --model-name vit_b16 \
  --batch-size 16 \
  --num-epochs 50 \
  --augment True
```

### Run Individual Python Scripts

```bash
# Just prepare data
python data_preparation.py

# Train with custom settings
python train.py

# Evaluate trained model
python evaluate.py

# Generate GradCAM visualizations
python evaluate.py --gradcam
```

---

## Model Architectures

The project supports three state-of-the-art CNN architectures:

### 1. EfficientNet-B0 (Default - Recommended)
- **Pros**: Fast, efficient, good accuracy
- **Parameters**: ~5.3M
- **Image Size**: 224x224
- **Best for**: Balanced performance and speed

### 2. ResNet50
- **Pros**: Well-established, good generalization
- **Parameters**: ~23.5M
- **Image Size**: 224x224
- **Best for**: Maximum accuracy with more computational resources

### 3. Vision Transformer (ViT-B/16)
- **Pros**: State-of-the-art attention mechanism
- **Parameters**: ~86M
- **Image Size**: 224x224
- **Best for**: Highest accuracy with sufficient compute

---

## Training Configuration

### Hyperparameters
- **Batch Size**: 32 (adjustable, 16-64 recommended)
- **Learning Rate**: 1e-3 (initial, with scheduler)
- **Optimizer**: Adam with weight decay (L2 regularization)
- **Scheduler**: ReduceLROnPlateau (reduces LR when validation accuracy plateaus)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 50 (adjustable)

### Data Augmentation
When enabled, applies:
- Random resize crop
- Horizontal/vertical flips
- Random rotation (±20 degrees)
- Color jitter
- Random affine transformations

### Early Stopping
- Monitors validation accuracy
- Saves best model automatically
- Stops if no improvement for 5+ epochs (via scheduler)

---

## Output and Results

### Training Outputs
Located in `checkpoints/`:
- `best_model.pt`: Best model weights
- `checkpoint_epoch_*.pt`: Periodic checkpoints
- `config.json`: Model and training configuration
- `training_history.json`: Loss and accuracy over epochs

### Evaluation Results
Located in `results/evaluation/`:
- `confusion_matrix.png`: Classification confusion matrix
- `metrics.png`: Per-class precision, recall, F1-score
- `evaluation_results.txt`: Summary of metrics

### GradCAM Visualizations
Located in `results/gradcam/`:
- `DR/`: Visualizations for DR predictions
  - `*_01_original.png`: Original fundus image
  - `*_02_cam.png`: Raw heatmap
  - `*_03_overlay.png`: Heatmap overlaid on image
- `Normal/`: Visualizations for Normal predictions
- `HR/`: Visualizations for HR predictions

---

## Understanding GradCAM Heatmaps

**GradCAM (Gradient-weighted Class Activation Mapping)** visualizes which regions of the image most influenced the model's prediction.

### What the Heatmaps Show:
- **Red/Hot colors**: Regions the model focuses on for this prediction
- **Blue/Cool colors**: Regions with less influence
- **Overlay**: Hot regions superimposed on the original image

### Interpretation:
- **Good model**: Heatmaps highlight clinically relevant features (microaneurysms, hemorrhages, hard exudates)
- **Poor model**: Heatmaps may highlight noise or irrelevant regions

---

## Performance Metrics

The model evaluation provides:

### Overall Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Per-Class Metrics
Individual metrics for each class (HR, DR, Normal)

### Confusion Matrix
- Shows where misclassifications occur
- Helps identify which classes are confused with each other

---

## Troubleshooting

### 1. Out of Memory (OOM) Error
```bash
# Reduce batch size
python main.py --batch-size 16

# Or use a lighter model
python main.py --model-name efficientnet_b0
```

### 2. Data Not Found
```bash
# Ensure your data is in the correct paths:
# - Diabetic Retinopathy Images/DR/
# - Diabetic Retinopathy Images/No_DR/
# - Hypertensive Retinopathy Images/1-Hypertensive Classification/1-Images/1-Training Set/

# Re-run data preparation
python main.py --stage prep
```

### 3. CUDA Not Available
```bash
# CPU mode will be automatically used
# But training will be slower. To enable GPU, install CUDA PyTorch version.
```

### 4. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Advanced Usage

### Custom Training Loop
```python
from train import Trainer
from model import create_model
from data_loader import get_dataloaders

# Load data
dataloaders, sizes, class_names = get_dataloaders('data/organized')

# Create model and trainer
model = create_model(num_classes=3, model_name='resnet50')
trainer = Trainer(model, device='cuda')

# Train
history = trainer.train(
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    class_names=class_names,
    num_epochs=100
)
```

### Custom GradCAM Generation
```python
from gradcam import generate_gradcam_visualization
from model import create_model
import torch
from PIL import Image
from data_loader import get_transforms

# Load model and image
model = create_model(num_classes=3)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Load and preprocess image
transform = get_transforms()['val']
image = Image.open('path/to/image.png').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Generate visualizations
result = generate_gradcam_visualization(
    model=model,
    image_tensor=image_tensor,
    class_names=['DR', 'Normal', 'HR'],
    target_layer='backbone.layer4',
    device='cuda',
    save_path='gradcam_output/'
)
```

---

## Expected Performance

Based on typical medical imaging benchmarks:

| Metric | Expected Range |
|--------|-----------------|
| Accuracy | 85-95% |
| Precision (per class) | 80-92% |
| Recall (per class) | 80-92% |
| F1-Score (weighted) | 85-92% |

**Note**: Actual performance depends on dataset quality, model architecture, hyperparameters, and data augmentation.

---

## References

### Papers
- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- GradCAM: [Selvaraju et al., 2019](https://arxiv.org/abs/1610.02055)
- ResNet: [He et al., 2015](https://arxiv.org/abs/1512.03385)

### Key Concepts
- **Diabetic Retinopathy**: Complications from diabetes affecting blood vessels in the retina
- **Hypertensive Retinopathy**: Retinal damage from prolonged high blood pressure
- **Fundus Images**: Photographs of the interior surface of the eye

---

## Contributing

To improve this model:
1. Experiment with different architectures
2. Collect more diverse training data
3. Fine-tune hyperparameters
4. Implement additional augmentation strategies
5. Try ensemble methods

---

## License

This project is for educational and research purposes.

---

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Try with smaller batch sizes or fewer epochs first

---

## Citation

If you use this code in your research, please cite:
```
HR vs DR Classification Model (2024)
Author: [Your Name]
GitHub: [Your Repository]
```

---

**Last Updated**: March 6, 2026
**Version**: 1.0
**Status**: Ready for development and experimentation
