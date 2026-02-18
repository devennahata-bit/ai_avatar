#!/usr/bin/env python3
"""
AI Avatar Installation Script
=============================
Run this script on a new machine to set up everything.

Usage:
    python install.py           # Full installation
    python install.py --deps    # Only install dependencies
    python install.py --models  # Only download models
    python install.py --server  # AWS/EC2-friendly headless install
    python install.py --check   # Check installation status

For AWS Linux:
    ./aws_setup.sh  (recommended - handles system deps automatically)
"""

import sys
import os
import subprocess
import shutil
import platform
from pathlib import Path

# Minimum Python version
MIN_PYTHON = (3, 10)

PROJECT_ROOT = Path(__file__).parent.resolve()

# Platform detection
IS_LINUX = platform.system().lower() == "linux"
IS_WINDOWS = platform.system().lower() == "windows"
IS_MACOS = platform.system().lower() == "darwin"

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)

def print_step(text):
    print(f"\n[*] {text}")

def print_ok(text):
    print(f"    [OK] {text}")

def print_warn(text):
    print(f"    [WARN] {text}")

def print_error(text):
    print(f"    [ERROR] {text}")


def run_cmd(cmd: list[str], quiet: bool = False) -> bool:
    """Run command and return success."""
    try:
        if quiet:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


def is_linux() -> bool:
    return IS_LINUX


def is_ec2() -> bool:
    if os.environ.get("AWS_EXECUTION_ENV"):
        return True
    # EC2 exposes this UUID prefix in many Linux distros.
    try:
        uuid_path = Path("/sys/hypervisor/uuid")
        if uuid_path.exists():
            return uuid_path.read_text(encoding="utf-8").lower().startswith("ec2")
    except Exception:
        pass
    return False


def is_headless() -> bool:
    """Check if running in headless mode (no display)."""
    return IS_LINUX and os.environ.get("DISPLAY") is None


def check_system_deps() -> list[str]:
    """Check for required system dependencies on Linux."""
    missing = []

    if not IS_LINUX:
        return missing

    # Check for FFmpeg
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")

    # Check for PortAudio (required for sounddevice)
    portaudio_check = run_cmd(["pkg-config", "--exists", "portaudio-2.0"], quiet=True)
    if not portaudio_check:
        missing.append("portaudio (portaudio19-dev on Ubuntu, portaudio-devel on RHEL)")

    # Check for espeak (required for pyttsx3 on Linux)
    if not shutil.which("espeak") and not shutil.which("espeak-ng"):
        missing.append("espeak (for local TTS fallback)")

    return missing


def install_linux_system_deps():
    """Attempt to install system dependencies on Linux."""
    if not IS_LINUX:
        return True

    print_step("Checking/installing Linux system dependencies...")

    # Detect package manager
    if shutil.which("apt-get"):
        pkg_mgr = "apt"
        cmds = [
            ["sudo", "apt-get", "update", "-qq"],
            ["sudo", "apt-get", "install", "-y", "-qq",
             "portaudio19-dev", "libportaudio2", "ffmpeg",
             "espeak", "libespeak1", "python3-dev",
             "libgl1-mesa-glx", "libglib2.0-0"]
        ]
    elif shutil.which("yum"):
        pkg_mgr = "yum"
        cmds = [
            ["sudo", "yum", "install", "-y",
             "portaudio-devel", "ffmpeg", "espeak",
             "python3-devel", "mesa-libGL"]
        ]
    elif shutil.which("dnf"):
        pkg_mgr = "dnf"
        cmds = [
            ["sudo", "dnf", "install", "-y",
             "portaudio-devel", "ffmpeg", "espeak",
             "python3-devel", "mesa-libGL"]
        ]
    else:
        print_warn("Unknown package manager. Please install manually:")
        print_warn("  portaudio, ffmpeg, espeak, python3-dev")
        return True  # Continue anyway

    print(f"    Using {pkg_mgr} package manager...")

    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print_warn(f"Command failed: {' '.join(cmd)}")
            print_warn("You may need to install these packages manually")
        except FileNotFoundError:
            print_warn("sudo not available, skipping system package installation")
            break

    print_ok("System dependencies checked")
    return True

def check_python_version():
    """Check Python version meets requirements."""
    print_step("Checking Python version...")

    if sys.version_info < MIN_PYTHON:
        print_error(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, got {sys.version_info.major}.{sys.version_info.minor}")
        return False

    print_ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_cuda():
    """Check CUDA availability."""
    print_step("Checking CUDA availability...")

    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print_ok(f"CUDA available: {device_name} ({vram:.1f}GB VRAM)")

            if vram < 48:
                print_warn(f"LiveAvatar requires 48GB+ VRAM. You have {vram:.1f}GB.")
                print_warn("You may need to use FP8 quantization or a different lip-sync model.")
            return True
        else:
            print_warn("CUDA not available. GPU acceleration disabled.")
            return False
    except ImportError:
        print_warn("PyTorch not installed yet. Will check after installation.")
        return None

def install_pytorch():
    """Install PyTorch with CUDA support."""
    print_step("Installing PyTorch with CUDA support...")

    try:
        # Install PyTorch with CUDA 12.1
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "-q"
        ]
        subprocess.check_call(cmd)
        print_ok("PyTorch installed with CUDA 12.1 support")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install PyTorch: {e}")
        return False

def install_dependencies(server_mode: bool = False):
    """Install all Python dependencies."""
    print_step("Installing Python dependencies...")

    requirements_file = PROJECT_ROOT / "requirements.txt"
    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        return False

    try:
        # Keep packaging toolchain modern to avoid build/deprecation issues.
        run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], quiet=True)

        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        if server_mode:
            cmd.append("--prefer-binary")
        subprocess.check_call(cmd)
        print_ok(f"Dependencies installed from {requirements_file.name}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def install_realtimetts(server_mode: bool = False):
    """Install RealtimeTTS with Kokoro support."""
    print_step("Installing RealtimeTTS with Kokoro TTS...")

    if server_mode:
        print_warn("Skipping RealtimeTTS in --server mode (optional on headless EC2).")
        print_warn("Set VOICE_TTS_PROVIDER=elevenlabs or local in .env/config for server usage.")
        return True

    try:
        cmd = [sys.executable, "-m", "pip", "install", "realtimetts[kokoro]", "-q"]
        subprocess.check_call(cmd)
        print_ok("RealtimeTTS with Kokoro installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install RealtimeTTS: {e}")
        return False

def setup_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    print_step("Setting up environment file...")

    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"

    if env_file.exists():
        print_ok(".env file already exists")
        return True

    if not env_example.exists():
        print_error(".env.example not found!")
        return False

    shutil.copy(env_example, env_file)
    print_ok(".env file created from .env.example")
    print_warn("Please edit .env and add your API keys (GROQ_API_KEY required)")
    return True

def create_directories():
    """Create required directories."""
    print_step("Creating required directories...")

    dirs = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "assets",
        PROJECT_ROOT / "logs",
    ]

    for d in dirs:
        d.mkdir(exist_ok=True)

    print_ok("Directories created")
    return True

def download_models():
    """Download required model weights."""
    print_step("Downloading model weights...")
    print("    This may take a while (~30GB for LiveAvatar)...")

    try:
        # Run setup_wizard.py to download models
        setup_wizard = PROJECT_ROOT / "setup_wizard.py"
        if setup_wizard.exists():
            subprocess.check_call([sys.executable, str(setup_wizard)])
            print_ok("Models downloaded")
            return True
        else:
            print_error("setup_wizard.py not found!")
            return False
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to download models: {e}")
        return False
    except KeyboardInterrupt:
        print_warn("Model download interrupted. Run 'python install.py --models' to resume.")
        return False

def verify_installation():
    """Verify the installation works."""
    print_step("Verifying installation...")

    errors = []

    # Check imports
    try:
        from config import voice_settings, liveavatar_config
        print_ok("Config module OK")
    except Exception as e:
        errors.append(f"Config import failed: {e}")

    try:
        from modules import SpeechListener, LLMBrain, FaceRenderer
        print_ok("Core modules OK")
    except Exception as e:
        errors.append(f"Module import failed: {e}")

    try:
        from modules.voice import RealtimeTTSSynthesizer
        print_ok("Voice module OK")
    except Exception as e:
        errors.append(f"Voice module failed: {e}")

    # Check CUDA again after PyTorch installation
    try:
        import torch
        if torch.cuda.is_available():
            print_ok(f"CUDA OK: {torch.cuda.get_device_name(0)}")
        else:
            print_warn("CUDA not available")
    except Exception as e:
        errors.append(f"PyTorch check failed: {e}")

    if errors:
        print("\nErrors found:")
        for err in errors:
            print_error(err)
        return False

    return True

def print_next_steps():
    """Print next steps for the user."""
    print_header("Installation Complete!")
    print("""
Next steps:

1. Edit .env file with your API keys:
   - GROQ_API_KEY (required for LLM)
   - Other keys are optional

2. Add your avatar image:
   - Place a face image at: assets/avatar_face.png
   - Or a video at: assets/avatar.mp4

3. Run the avatar:
   python run.py

Options:
   python run.py --setup     # Download models (if not done)
   python run.py --debug     # Run with debug logging
   python run.py --config    # Show current configuration
   python main.py --no-display  # Recommended for headless EC2
""")

def check_installation_status():
    """Check and report installation status."""
    print_header("AI Avatar Installation Status")

    status = []

    # Python
    py_ok = sys.version_info >= MIN_PYTHON
    status.append(("Python", py_ok, f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))

    # PyTorch/CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        status.append(("PyTorch", True, torch.__version__))
        status.append(("CUDA", cuda_ok, gpu_name if cuda_ok else "Not available"))
    except ImportError:
        status.append(("PyTorch", False, "Not installed"))
        status.append(("CUDA", False, "N/A"))

    # Core modules
    try:
        from config import api_config
        status.append(("Config", True, "OK"))
    except Exception as e:
        status.append(("Config", False, str(e)[:30]))

    try:
        from modules.brain import LLMBrain
        status.append(("LLM Brain", True, "OK"))
    except Exception as e:
        status.append(("LLM Brain", False, str(e)[:30]))

    try:
        from modules.voice import VoiceSynthesizer
        status.append(("Voice", True, "OK"))
    except Exception as e:
        status.append(("Voice", False, str(e)[:30]))

    # FFmpeg
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    status.append(("FFmpeg", ffmpeg_ok, shutil.which("ffmpeg") or "Not found"))

    # Models
    models_dir = PROJECT_ROOT / "models"
    has_models = models_dir.exists() and any(models_dir.iterdir()) if models_dir.exists() else False
    status.append(("Models", has_models, str(models_dir) if has_models else "Not downloaded"))

    # .env
    env_ok = (PROJECT_ROOT / ".env").exists()
    status.append((".env", env_ok, "Configured" if env_ok else "Missing"))

    # Print status
    print("\nComponent Status:")
    for name, ok, detail in status:
        symbol = "[OK]" if ok else "[!!]"
        print(f"  {symbol} {name}: {detail}")

    # Platform info
    print(f"\nPlatform: {platform.system()} ({platform.machine()})")
    print(f"Headless: {is_headless()}")
    print(f"EC2: {is_ec2()}")

    all_ok = all(ok for _, ok, _ in status)
    return all_ok


def main():
    print_header("AI Avatar Installation")

    # Parse args
    args = sys.argv[1:]
    only_deps = "--deps" in args
    only_models = "--models" in args
    server_mode = "--server" in args or is_ec2()
    check_only = "--check" in args

    # Check installation status
    if check_only:
        check_installation_status()
        return

    if server_mode:
        print_step("Detected server/headless environment")
        print_ok("Using EC2-friendly install mode")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Only download models
    if only_models:
        download_models()
        return

    # Install Linux system dependencies if needed
    if IS_LINUX:
        install_linux_system_deps()

    # Create directories
    create_directories()

    # Setup .env file
    setup_env_file()

    # Install PyTorch first (with CUDA)
    install_pytorch()

    # Install other dependencies
    install_dependencies(server_mode=server_mode)

    # Install RealtimeTTS (skip on Linux server mode)
    install_realtimetts(server_mode=server_mode)

    # Install ElevenLabs as recommended TTS for Linux
    if IS_LINUX:
        print_step("Installing ElevenLabs TTS (recommended for Linux)...")
        run_cmd([sys.executable, "-m", "pip", "install", "elevenlabs", "-q"], quiet=True)
        print_ok("ElevenLabs installed")

    # Verify
    if not verify_installation():
        print_error("Installation verification failed. Check errors above.")
        sys.exit(1)

    # Only deps - skip models
    if only_deps:
        print_ok("Dependencies installed. Run 'python install.py --models' to download models.")
        return

    # Ask about models (skip prompt in non-interactive mode)
    print("\n" + "="*60)
    if sys.stdin.isatty():
        response = input("Download model weights now? (~30GB) [y/N]: ").strip().lower()
        if response == 'y':
            download_models()
        else:
            print_warn("Skipping model download. Run 'python install.py --models' later.")
    else:
        print_warn("Non-interactive mode. Run 'python install.py --models' to download models.")

    print_next_steps()

if __name__ == "__main__":
    main()
