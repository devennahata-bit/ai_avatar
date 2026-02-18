#!/usr/bin/env python3
"""
Quick Start Script for AI Avatar
=================================
Single entry point to run the avatar application.

Usage:
    python run.py              # Run with defaults
    python run.py --setup      # Run setup wizard first
    python run.py --debug      # Run with debug logging
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

def check_dependencies():
    """Check if critical dependencies are installed."""
    missing = []

    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. GPU acceleration disabled.")
    except ImportError:
        missing.append("torch")

    try:
        import pydantic_settings
    except ImportError:
        missing.append("pydantic-settings")

    try:
        from rich.console import Console
    except ImportError:
        missing.append("rich")

    return missing

def install_missing(packages):
    """Install missing packages."""
    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

def main():
    # Quick dependency check
    missing = check_dependencies()
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        install_missing(missing)

    # Check for --setup flag
    if "--setup" in sys.argv:
        print("Running setup wizard...")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "setup_wizard.py")])
        print("\nSetup complete. Run 'python run.py' to start the avatar.")
        return

    # Run main application
    import main as avatar_main
    avatar_main.main()

if __name__ == "__main__":
    main()
