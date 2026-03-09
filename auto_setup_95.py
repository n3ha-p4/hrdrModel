#!/usr/bin/env python3
"""
AUTOMATED SETUP FOR HIGH ACCURACY (95%+) TRAINING
Handles environment setup and begins training
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, shell=False):
    """Run a command and report status"""
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}")
    
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        else:
            result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_conda():
    """Check if Conda is available"""
    result = subprocess.run(
        "conda --version",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def create_conda_env():
    """Create Conda environment with Python 3.10"""
    env_name = "hr_dr_detection"
    
    # Check if environment exists
    result = subprocess.run(
        f"conda info --envs | grep -q {env_name}",
        shell=True,
        capture_output=True
    )
    
    if result.returncode == 0:
        print(f"✓ Conda environment '{env_name}' already exists")
        return env_name
    
    print(f"\n📌 Creating Conda environment: {env_name}")
    
    cmd = f"conda create -n {env_name} python=3.10 -y"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Created Conda environment: {env_name}")
        return env_name
    else:
        print(f"✗ Failed to create environment")
        print(result.stderr)
        return None


def main():
    """Main setup orchestration"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  HIGH ACCURACY TRAINING SETUP (Target: 95%+)                  ║
    ║  HR vs DR Retinopathy Classification                          ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print(f"Working directory: {project_dir}")
    
    # Step 1: Check Python version and Conda
    print("\n" + "="*70)
    print("📌 Environment Detection")
    print("="*70)
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Current Python: {python_version}")
    
    if check_conda():
        print("✓ Conda is installed")
        conda_available = True
    else:
        print("✗ Conda not found - Installing Miniconda...")
        # Installation instructions
        print("""
        Please install Miniconda to continue:
        
        macOS Intel:
          curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
          bash ~/miniconda.sh -b -p ~/miniconda
          rm ~/miniconda.sh
          export PATH="~/miniconda/bin:$PATH"
        
        Or using Homebrew:
          brew install miniconda
        
        After installation, re-run this script.
        """)
        return False
    
    # Step 2: Setup environment
    if conda_available:
        print("\n" + "="*70)
        print("🚀 Setting up Conda environment")
        print("="*70)
        
        env_name = create_conda_env()
        if not env_name:
            return False
        
        print(f"\n📌 Installation commands:")
        print(f"   conda activate {env_name}")
        print(f"   conda install pytorch torchvision torchaudio -c pytorch -y")
        print(f"   pip install -q scikit-learn pandas opencv-python tqdm tensorboard")
        print(f"\nYou can run these manually or I'll try to run them now...")
        
        response = input("\nAutomatically install packages? (y/n): ").lower()
        if response == 'y':
            # Note: We can't activate conda in subprocess, so provide instructions
            print("\nTo complete setup, run in a new terminal:")
            print(f"  conda activate {env_name}")
            print(f"  cd {project_dir}")
            print(f"  conda install pytorch torchvision torchaudio -c pytorch -y")
            print(f"  pip install -q scikit-learn pandas opencv-python tqdm tensorboard matplotlib seaborn")
            print(f"  python train_high_accuracy.py")
            return True
    
    print("\n" + "="*70)
    print("✓ Setup guide complete!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
