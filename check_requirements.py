"""
Requirements checker for PPO Osu! Mania agent.
Verifies all dependencies are installed and compatible.
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check_package(package_name, min_version=None, import_name=None):
    """Check if a package is installed and optionally verify version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if min_version and hasattr(module, '__version__'):
            version = module.__version__
            if version < min_version:
                return False, f"Version {version} < {min_version}"
            return True, f"Version {version}"
        return True, "Installed"
    except ImportError:
        return False, "Not installed"


def check_requirements():
    """Check all required packages."""
    print("Checking PPO Osu! Mania Agent Requirements")
    print("=" * 50)
    
    # Core requirements
    requirements = [
        ("stable-baselines3", "2.1.0", "stable_baselines3"),
        ("torch", "2.0.0", "torch"),
        ("gymnasium", "0.28.0", "gymnasium"),
        ("opencv-python", "4.8.0", "cv2"),
        ("mss", "9.0.1", "mss"),
        ("numpy", "1.24.0", "numpy"),
        ("pydirectinput", "1.0.4", "pydirectinput"),
        ("tensorboard", "2.13.0", "tensorboard"),
        ("psutil", "5.9.0", "psutil"),
        ("requests", "2.28.0", "requests"),
    ]
    
    all_good = True
    
    for package, min_version, import_name in requirements:
        is_installed, message = check_package(package, min_version, import_name)
        status = "OK" if is_installed else "MISSING"
        print(f"{status} {package:<20} {message}")
        
        if not is_installed:
            all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("All requirements satisfied!")
        print("You can now run the PPO Osu! Mania agent.")
    else:
        print("Some requirements are missing or outdated.")
        print("\nTo install missing packages:")
        print("pip install -r requirements.txt")
        print("\nOr install minimal requirements:")
        print("pip install -r requirements-minimal.txt")
    
    return all_good


def check_python_version():
    """Check Python version compatibility."""
    print(f"Python Version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("Python 3.8+ is required")
        return False
    else:
        print("Python version is compatible")
        return True


def check_gpu_support():
    """Check if GPU support is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU Support: {gpu_count} GPU(s) available")
            print(f"   GPU: {gpu_name}")
            return True
        else:
            print("GPU Support: Not available (CPU only)")
            return False
    except ImportError:
        print("PyTorch not installed")
        return False


def main():
    """Main requirements check."""
    print("PPO Osu! Mania AI Agent - Requirements Checker")
    print("=" * 60)
    
    # Check Python version
    python_ok = check_python_version()
    print()
    
    # Check packages
    packages_ok = check_requirements()
    print()
    
    # Check GPU support
    gpu_ok = check_gpu_support()
    print()
    
    # Summary
    if python_ok and packages_ok:
        print("Ready to go!")
        print("\nNext steps:")
        print("1. Run setup: python setup.py")
        print("2. Train agent: python main.py --config mania_4k")
        print("3. Play agent: python play_agent.py --config mania_4k")
    else:
        print("Please install missing requirements first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
