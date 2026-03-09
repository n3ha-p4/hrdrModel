#!/bin/bash
# HR vs DR Classification Model - Automated Setup Script
# This script sets up the complete environment and starts training

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   HR vs DR Classification Model - Automated Setup              ║"
echo "║   Target: 95%+ Accuracy with GradCAM Visualization            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "📦 Installing Conda..."
    # Try Homebrew first
    if command -v brew &> /dev/null; then
        echo "   Using Homebrew to install Miniconda..."
        brew tap-new local/python 2>/dev/null || true
        # Alternative: download and install directly
        curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
        bash ~/miniconda.sh -b -p ~/miniconda
        rm ~/miniconda.sh
        export PATH="~/miniconda/bin:$PATH"
        conda init bash
        echo "   ✓ Miniconda installed"
    else
        echo "   ✗ Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
fi

echo "✓ Conda found"
echo ""

# Check current Conda version
echo "📌 Conda version: $(conda --version)"
echo ""

# Create environment
ENV_NAME="hr_dr_detection"
PYTHON_VERSION="3.10"

echo "🔨 Creating Conda environment: $ENV_NAME (Python $PYTHON_VERSION)..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "   Environment already exists. Updating..."
    conda env update -n $ENV_NAME -f /dev/null
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

echo "✓ Environment created"
echo ""

# Activate environment
echo "🚀 Activating environment..."
source ~/miniconda/etc/profile.d/conda.sh
conda activate $ENV_NAME
echo "✓ Environment activated: $(python --version)"
echo ""

# Install PyTorch
echo "📦 Installing PyTorch..."
conda install pytorch torchvision torchaudio -c pytorch -y
echo "✓ PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install -q \
    scikit-learn pandas numpy \
    matplotlib seaborn \
    opencv-python tqdm tensorboard \
    Pillow

echo "✓ All dependencies installed"
echo ""

# Verify installation
echo "🧪 Verifying installation..."
python -c "
import torch
import pandas as pd
import numpy as np
import cv2
import sklearn
import matplotlib
print('✓ All packages imported successfully')
print(f'  PyTorch: {torch.__version__}')
print(f'  NumPy: {np.__version__}')
print(f'  Pandas: {pd.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
"
echo ""

# Setup complete
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✓ SETUP COMPLETE                                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To start training, run:"
echo "  python main.py --stage all"
echo ""
echo "Or for quick test:"
echo "  python main.py --stage all --num-epochs 20"
echo ""
