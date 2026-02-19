#!/usr/bin/env python3
"""
Download LiveAvatar Models
==========================
Downloads all required model weights for LiveAvatar lip-sync.

Models downloaded:
1. Wan2.2-S2V-14B base model (~28GB) - from Wan-AI/Wan2.2-S2V-14B
2. Live-Avatar LoRA weights (~2GB) - from Quark-Vision/Live-Avatar
3. wav2vec2 audio encoder (~400MB) - from facebook/wav2vec2-base

Total: ~30GB

Requirements:
- 100GB+ free disk space
- Good internet connection
- Python with huggingface_hub installed

Usage:
    python download_models.py
    python download_models.py --check  # Just check what's installed
"""

import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"

# Model directories
LIVEAVATAR_BASE_DIR = MODELS_DIR / "Wan2.2-S2V-14B"
LIVEAVATAR_LORA_DIR = MODELS_DIR / "Live-Avatar"
WAV2VEC2_DIR = MODELS_DIR / "wav2vec2-base"

# HuggingFace repo IDs
MODELS = [
    {
        "name": "Wan2.2-S2V-14B Base Model",
        "repo_id": "Wan-AI/Wan2.2-S2V-14B",
        "local_dir": LIVEAVATAR_BASE_DIR,
        "size": "~28GB",
        "check_file": ["config.json", "model_index.json"],
    },
    {
        "name": "Live-Avatar LoRA Weights",
        "repo_id": "Quark-Vision/Live-Avatar",
        "local_dir": LIVEAVATAR_LORA_DIR,
        "size": "~2GB",
        "check_file": None,  # Just check if directory has files
    },
    {
        "name": "wav2vec2 Audio Encoder",
        "repo_id": "facebook/wav2vec2-base",
        "local_dir": WAV2VEC2_DIR,
        "size": "~400MB",
        "check_file": "config.json",
    },
]


def check_model_installed(model: dict) -> bool:
    """Check if a model is already downloaded."""
    local_dir = model["local_dir"]
    if not local_dir.exists():
        return False

    if model["check_file"]:
        check_file = model["check_file"]
        if isinstance(check_file, (list, tuple)):
            return any((local_dir / name).exists() for name in check_file)
        return (local_dir / check_file).exists()
    else:
        # Check if directory has any files
        return any(local_dir.iterdir())


def get_dir_size(path: Path) -> str:
    """Get human-readable directory size."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size

    if total >= 1e9:
        return f"{total / 1e9:.1f}GB"
    elif total >= 1e6:
        return f"{total / 1e6:.1f}MB"
    else:
        return f"{total / 1e3:.1f}KB"


def check_status():
    """Check and display model installation status."""
    print("\n" + "=" * 60)
    print("  LiveAvatar Models - Installation Status")
    print("=" * 60 + "\n")

    all_installed = True

    for model in MODELS:
        installed = check_model_installed(model)
        status = "\033[92m[INSTALLED]\033[0m" if installed else "\033[91m[MISSING]\033[0m"

        size_info = ""
        if installed and model["local_dir"].exists():
            size_info = f" ({get_dir_size(model['local_dir'])})"

        print(f"  {status} {model['name']}{size_info}")
        print(f"           Expected size: {model['size']}")
        print(f"           Path: {model['local_dir']}")
        print()

        if not installed:
            all_installed = False

    if all_installed:
        print("\033[92mAll models are installed!\033[0m")
    else:
        print("\033[93mSome models are missing. Run without --check to download.\033[0m")

    return all_installed


def download_models():
    """Download all required models from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("  LiveAvatar Models - Download")
    print("=" * 60)
    print("\n\033[93mThis will download approximately 30GB of model weights.\033[0m")
    print("Make sure you have enough disk space and a stable connection.\n")

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] {model['name']} ({model['size']})")
        print("-" * 50)

        if check_model_installed(model):
            print(f"\033[92mAlready downloaded at {model['local_dir']}\033[0m")
            continue

        print(f"Downloading from: {model['repo_id']}")
        print(f"To: {model['local_dir']}")
        print()

        try:
            model["local_dir"].mkdir(parents=True, exist_ok=True)

            snapshot_download(
                repo_id=model["repo_id"],
                local_dir=model["local_dir"],
                local_dir_use_symlinks=False,
            )

            print(f"\033[92mDownload complete!\033[0m")

        except KeyboardInterrupt:
            print("\n\033[93mDownload interrupted. Run again to resume.\033[0m")
            sys.exit(1)

        except Exception as e:
            print(f"\033[91mDownload failed: {e}\033[0m")
            print(f"You can manually download from: https://huggingface.co/{model['repo_id']}")
            continue

    print("\n" + "=" * 60)
    print("  Download Complete!")
    print("=" * 60 + "\n")

    # Final status check
    check_status()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download LiveAvatar models")
    parser.add_argument("--check", action="store_true", help="Only check installation status")
    args = parser.parse_args()

    if args.check:
        check_status()
    else:
        download_models()


if __name__ == "__main__":
    main()
