#!/bin/bash
################################################################################
# HR vs DR Classification Model - One-Command Setup for 95%+ Accuracy
# Run this script to automatically set everything up and start training
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Main script
clear

print_header "HR vs DR Classification - 95%+ Accuracy Setup"

echo ""
print_info "This script will automatically:"
echo "  1. Check for Conda installation"
echo "  2. Create Python 3.10 environment"
echo "  3. Install PyTorch and dependencies"
echo "  4. Verify setup"
echo "  5. Start training"
echo ""

# Step 1: Check Conda
print_header "Checking Conda Installation"

if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version)
    print_success "Conda found: $CONDA_VERSION"
    CONDA_AVAILABLE=true
else
    print_error "Conda not found"
    echo ""
    print_warning "Please install Conda first:"
    echo "  Option 1 (Homebrew): brew install miniconda"
    echo "  Option 2 (Direct): curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh && bash ~/miniconda.sh -b -p ~/miniconda"
    echo ""
    print_info "After installing Conda, run this script again."
    exit 1
fi

echo ""

# Step 2: Create Environment
print_header "Setting up Python 3.10 Environment"

ENV_NAME="hr_dr"
PYTHON_VERSION="3.10"

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment '$ENV_NAME' already exists"
    read -p "Use existing environment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_success "Using existing environment"
    else
        print_info "Removing old environment..."
        conda env remove -n $ENV_NAME -y > /dev/null 2>&1 || true
        print_info "Creating new environment..."
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y > /dev/null 2>&1
        print_success "New environment created"
    fi
else
    print_info "Creating new environment: $ENV_NAME"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y > /dev/null 2>&1
    print_success "Environment created: $ENV_NAME"
fi

echo ""

# Step 3: Initialize Conda for this shell
print_header "Initializing Conda"

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

print_success "Activated environment: $(conda info --json | python -c 'import sys, json; print(json.load(sys.stdin)[\"default_prefix\"])')"
print_info "Python version: $(python --version)"

echo ""

# Step 4: Install PyTorch
print_header "Installing PyTorch"

print_info "Installing PyTorch and dependencies (this may take 2-5 minutes)..."

# Install PyTorch from conda
conda install pytorch torchvision torchaudio -c pytorch -y > /dev/null 2>&1 && {
    print_success "PyTorch installed"
} || {
    print_error "Failed to install PyTorch"
    print_warning "Try installing manually:"
    echo "  conda install pytorch torchvision torchaudio -c pytorch -y"
    exit 1
}

echo ""

# Step 5: Install other dependencies
print_header "Installing Other Dependencies"

print_info "Installing scikit-learn, pandas, matplotlib, OpenCV, etc..."

pip install -q \
    scikit-learn>=1.0.0 \
    pandas>=1.3.0 \
    numpy>=1.20.0 \
    matplotlib>=3.3.0 \
    seaborn>=0.11.0 \
    opencv-python>=4.5.0 \
    tqdm>=4.60.0 \
    tensorboard>=2.6.0 \
    Pillow>=8.0.0 \
    2>/dev/null && {
    print_success "All dependencies installed"
} || {
    print_error "Some dependencies failed to install"
    print_warning "Try manually: pip install -r requirements.txt"
}

echo ""

# Step 6: Verify Installation
print_header "Verifying Installation"

python -c "
import torch
import pandas as pd
import numpy as np
import cv2
import sklearn

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ NumPy: {np.__version__}')
print(f'✓ Pandas: {pd.__version__}')
print(f'✓ scikit-learn: {sklearn.__version__}')
print(f'✓ OpenCV: {cv2.__version__}')

if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('✓ CUDA not available (will use CPU)')
" && {
    print_success "All packages verified"
} || {
    print_error "Package verification failed"
    exit 1
}

echo ""

# Step 7: Change to project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

print_header "Project Directory"
print_info "Working directory: $PROJECT_DIR"

echo ""

# Step 8: Check data preparation
print_header "Checking Data"

# Count organized images
if [ -d "data/organized" ]; then
    IMAGE_COUNT=$(find data/organized -name "*.png" 2>/dev/null | wc -l)
    if [ "$IMAGE_COUNT" -gt 3000 ]; then
        print_success "Data already organized: $IMAGE_COUNT images"
    else
        print_warning "Data incomplete: $IMAGE_COUNT images (expected 4374)"
        read -p "Run data preparation? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Organizing data..."
            python data_preparation.py
        fi
    fi
else
    print_warning "Data not organized yet"
    read -p "Run data preparation? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Organizing data..."
        python data_preparation.py
    fi
fi

echo ""

# Step 9: Ready to train
print_header "Training Ready"

echo ""
print_success "All setup complete!"
echo ""

echo "You can now run training with:"
echo ""
echo -e "${GREEN}  python train_high_accuracy.py${NC}"
echo ""
echo "Or the complete pipeline (including evaluation and GradCAM):"
echo ""
echo -e "${GREEN}  python main.py --stage all${NC}"
echo ""

print_info "Expected training time:"
echo "  • GPU (NVIDIA 12GB+): 2-4 hours"
echo "  • CPU: 8-12 hours"
echo "  • Apple Silicon: 4-6 hours"
echo ""

print_info "To deactivate this environment later, run:"
echo "  conda deactivate"
echo ""

print_info "To activate this environment in future, run:"
echo "  conda activate $ENV_NAME"
echo ""

# Ask if user wants to start training now
echo ""
read -p "Start training now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_header "Starting High-Accuracy Training"
    echo ""
    python train_high_accuracy.py
else
    print_success "Setup complete! Run 'python train_high_accuracy.py' when ready."
fi

echo ""
