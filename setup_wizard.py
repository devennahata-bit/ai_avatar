#!/usr/bin/env python3
"""
Real-Time AI Avatar - Setup Wizard
===================================
Automated environment preparation and model download script.

This wizard handles:
1. CUDA/GPU verification
2. Complex dependency installation (OpenMMLab stack)
3. FFmpeg verification/installation
4. MuseTalk repository cloning
5. All required model weights download

Usage:
    python setup_wizard.py
    python setup_wizard.py --skip-cuda-check
    python setup_wizard.py --weights-only
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Optional
import argparse

# Ensure Unicode-rich console output works on Windows terminals that default to cp1252.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Rich for beautiful console output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
except ImportError:
    print("Installing rich for better output...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint

console = Console()

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
VENDOR_DIR = PROJECT_ROOT / "vendor"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Model subdirectories
LIVEAVATAR_BASE_DIR = MODELS_DIR / "Wan2.2-S2V-14B"
# Keep legacy local fallback support for users who already have older weights.
LIVEAVATAR_BASE_DIR_LEGACY = MODELS_DIR / "Wan2.1-S2V-14B"
LIVEAVATAR_LORA_DIR = MODELS_DIR / "Live-Avatar"
WAV2VEC2_DIR = MODELS_DIR / "wav2vec2-base"


def _has_base_model_files(model_dir: Path) -> bool:
    """Return True when a base model directory contains expected metadata files."""
    return any((model_dir / name).exists() for name in ("config.json", "model_index.json"))


# =============================================================================
# STEP 1: CUDA & HARDWARE CHECK
# =============================================================================

def check_cuda_and_gpu() -> tuple[bool, Optional[str], Optional[str]]:
    """
    Check if CUDA is available and get GPU information.

    Returns:
        (cuda_available, gpu_name, cuda_version)
    """
    console.print("\n[bold blue]Step 1: Checking CUDA & GPU...[/bold blue]")

    try:
        import torch
    except ImportError:
        console.print("[yellow]PyTorch not installed. Installing...[/yellow]")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        import torch

    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        console.print(Panel(
            "[bold red]CUDA IS NOT AVAILABLE![/bold red]\n\n"
            "MuseTalk requires an NVIDIA GPU with CUDA support.\n\n"
            "[yellow]Please ensure you have:[/yellow]\n"
            "1. An NVIDIA GPU (GTX 1060+ or RTX series recommended)\n"
            "2. NVIDIA Drivers installed (version 525+ recommended)\n"
            "3. CUDA Toolkit 11.8+ or 12.x installed\n\n"
            "[cyan]Download CUDA Toolkit from:[/cyan]\n"
            "https://developer.nvidia.com/cuda-downloads\n\n"
            "[cyan]After installation, restart your terminal and run this wizard again.[/cyan]",
            title="⚠️ CUDA Not Found",
            border_style="red"
        ))
        return False, None, None

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Display GPU info
    table = Table(title="GPU Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("CUDA Available", "✅ Yes")
    table.add_row("GPU Name", gpu_name)
    table.add_row("CUDA Version", cuda_version)
    table.add_row("GPU Memory", f"{gpu_memory:.1f} GB")
    table.add_row("PyTorch Version", torch.__version__)
    console.print(table)

    # Check minimum memory (LiveAvatar needs 48GB+ VRAM for FP8, 80GB+ for FP16)
    if gpu_memory < 48:
        console.print(
            f"[yellow]⚠️ Warning: GPU has only {gpu_memory:.1f}GB VRAM. "
            "LiveAvatar requires at least 48GB VRAM with FP8 quantization.[/yellow]"
        )
    elif gpu_memory < 80:
        console.print(
            f"[cyan]ℹ️ GPU has {gpu_memory:.1f}GB VRAM. "
            "FP8 quantization will be used for memory efficiency.[/cyan]"
        )

    console.print("[green]✅ CUDA check passed![/green]")
    return True, gpu_name, cuda_version


# =============================================================================
# STEP 2: INSTALL LIVEAVATAR DEPENDENCIES
# =============================================================================

def install_liveavatar_dependencies() -> bool:
    """
    Install LiveAvatar dependencies (diffusers, peft, accelerate, etc.).
    """
    console.print("\n[bold blue]Step 2: Installing LiveAvatar Dependencies...[/bold blue]")

    try:
        dependencies = [
            "diffusers>=0.27.0",
            "transformers>=4.38.0",
            "accelerate>=0.27.0",
            "peft>=0.10.0",
            "librosa",
            "soundfile",
            "einops",
            "safetensors",
        ]

        for dep in dependencies:
            console.print(f"[cyan]Installing {dep}...[/cyan]")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--prefer-binary", dep],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except subprocess.CalledProcessError as e:
                console.print(f"[yellow]⚠️ Warning installing {dep}: {e}[/yellow]")

        console.print("[green]✅ LiveAvatar dependencies installed![/green]")
        return True

    except Exception as e:
        console.print(f"[red]❌ Failed to install LiveAvatar dependencies: {e}[/red]")
        console.print("[yellow]You may need to install manually:[/yellow]")
        console.print("  pip install diffusers transformers accelerate peft librosa soundfile")
        return False


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible."""
    console.print("\n[bold blue]Checking FFmpeg...[/bold blue]")

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if ffmpeg_path and ffprobe_path:
        # Get version
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown"
        console.print(f"[green]✅ FFmpeg found: {version_line}[/green]")
        return True

    linux_install_hint = "  sudo apt install ffmpeg"
    if platform.system().lower() == "linux" and shutil.which("dnf"):
        linux_install_hint = "  sudo dnf install -y ffmpeg"

    console.print(Panel(
        "[bold red]FFmpeg not found in system PATH![/bold red]\n\n"
        "[yellow]Please install FFmpeg:[/yellow]\n\n"
        "[cyan]Windows:[/cyan]\n"
        "  1. Download from https://ffmpeg.org/download.html\n"
        "  2. Extract and add bin/ folder to system PATH\n"
        "  OR use: winget install ffmpeg\n"
        "  OR use: choco install ffmpeg\n\n"
        "[cyan]Linux:[/cyan]\n"
        f"{linux_install_hint}\n\n"
        "[cyan]macOS:[/cyan]\n"
        "  brew install ffmpeg",
        title="⚠️ FFmpeg Required",
        border_style="yellow"
    ))

    # Try to install via pip (imageio-ffmpeg as fallback)
    console.print("[cyan]Installing imageio-ffmpeg as fallback...[/cyan]")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "imageio-ffmpeg"
        ])
        console.print("[yellow]⚠️ Using imageio-ffmpeg fallback. Native FFmpeg is recommended.[/yellow]")
        return True
    except:
        return False


# =============================================================================
# STEP 3: CLONE LIVEAVATAR REPOSITORY
# =============================================================================

def clone_liveavatar_repo() -> bool:
    """Clone the official LiveAvatar repository."""
    console.print("\n[bold blue]Step 3: Setting up LiveAvatar Repository...[/bold blue]")

    liveavatar_path = VENDOR_DIR / "LiveAvatar"

    if liveavatar_path.exists():
        console.print(f"[green]✅ LiveAvatar already exists at {liveavatar_path}[/green]")

        # Check if it's a valid repo
        if (liveavatar_path / ".git").exists():
            console.print("[cyan]Pulling latest changes...[/cyan]")
            try:
                subprocess.run(
                    ["git", "pull"],
                    cwd=liveavatar_path,
                    check=True,
                    capture_output=True
                )
                console.print("[green]✅ Repository updated![/green]")
            except:
                console.print("[yellow]⚠️ Could not update repo (offline?)[/yellow]")
        return True

    # Create vendor directory
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Cloning LiveAvatar repository...[/cyan]")

    try:
        subprocess.run(
            ["git", "clone", "https://github.com/Alibaba-Quark/LiveAvatar.git"],
            cwd=VENDOR_DIR,
            check=True
        )
        console.print("[green]✅ LiveAvatar cloned successfully![/green]")
        return True

    except FileNotFoundError:
        console.print("[red]❌ Git not found. Please install Git first.[/red]")
        console.print("[cyan]Download from: https://git-scm.com/downloads[/cyan]")
        return False

    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to clone LiveAvatar: {e}[/red]")
        return False


def install_liveavatar_repo_deps() -> bool:
    """Install dependencies from the cloned LiveAvatar repository."""
    liveavatar_path = VENDOR_DIR / "LiveAvatar"

    if not liveavatar_path.exists():
        console.print("[yellow]LiveAvatar repo not found, skipping repo-specific dependencies[/yellow]")
        return True

    console.print("\n[cyan]Installing LiveAvatar repository dependencies...[/cyan]")

    # Check for requirements.txt in the repo
    req_files = [
        liveavatar_path / "requirements.txt",
        liveavatar_path / "requirements" / "requirements.txt",
    ]

    for req_file in req_files:
        if req_file.exists():
            console.print(f"[cyan]Installing from {req_file.name}...[/cyan]")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                console.print("[green]✅ LiveAvatar repo dependencies installed![/green]")
                return True
            except subprocess.CalledProcessError as e:
                console.print(f"[yellow]⚠️ Warning: Some dependencies from {req_file.name} may have failed: {e}[/yellow]")

    # Check for setup.py or pyproject.toml
    if (liveavatar_path / "setup.py").exists() or (liveavatar_path / "pyproject.toml").exists():
        console.print("[cyan]Installing LiveAvatar as a package...[/cyan]")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-e", str(liveavatar_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            console.print("[green]✅ LiveAvatar package installed![/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]⚠️ Warning: LiveAvatar package install failed: {e}[/yellow]")
            console.print("[yellow]  Will use diffusers fallback for model loading[/yellow]")

    return True


# =============================================================================
# STEP 4: DOWNLOAD ALL MODEL WEIGHTS
# =============================================================================

def check_disk_space(required_gb: float = 50.0) -> bool:
    """Check if there's enough disk space for model downloads."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(MODELS_DIR.parent)
        free_gb = free / (1024**3)
        console.print(f"[dim]Disk space available: {free_gb:.1f} GB[/dim]")

        if free_gb < required_gb:
            console.print(f"[red]⚠️ Warning: Only {free_gb:.1f}GB free, need ~{required_gb}GB for models[/red]")
            console.print("[yellow]Consider freeing up disk space or using a larger EBS volume.[/yellow]")
            return False
        return True
    except Exception as e:
        console.print(f"[yellow]Could not check disk space: {e}[/yellow]")
        return True  # Continue anyway


def download_all_weights() -> bool:
    """
    Download all required model weights for LiveAvatar using huggingface_hub.

    Required models:
    1. Wan2.2-S2V-14B base model
    2. Live-Avatar LoRA weights
    3. wav2vec2 audio encoder
    """
    console.print("\n[bold blue]Step 4: Downloading Model Weights...[/bold blue]")
    console.print("[yellow]⚠️ This will download approximately 30GB of model weights.[/yellow]")

    # Check disk space first
    if not check_disk_space(50.0):
        console.print("[yellow]Continuing anyway, but download may fail if space runs out.[/yellow]")

    try:
        from huggingface_hub import snapshot_download, HfFolder
    except ImportError:
        console.print("[cyan]Installing huggingface_hub...[/cyan]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download, HfFolder

    # Set HuggingFace cache directory for Linux (avoids permission issues)
    if platform.system().lower() == "linux":
        hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        os.environ["HF_HOME"] = hf_cache
        console.print(f"[dim]HuggingFace cache: {hf_cache}[/dim]")

    # Check for HuggingFace token (some models may require authentication)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        console.print("[green]✓ HuggingFace token found[/green]")
    else:
        console.print("[dim]No HuggingFace token set (HF_TOKEN). Some models may require authentication.[/dim]")
        console.print("[dim]Get a token from: https://huggingface.co/settings/tokens[/dim]")

    # Create directories
    for dir_path in [LIVEAVATAR_BASE_DIR, LIVEAVATAR_LORA_DIR, WAV2VEC2_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    success = True

    # Helper function for download with retry
    def download_with_retry(repo_id: str, local_dir: Path, max_retries: int = 3) -> bool:
        """Download with retry logic for network issues."""
        import time as time_module
        for attempt in range(max_retries):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    token=hf_token,
                    resume_download=True,  # Resume partial downloads
                )
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                    console.print(f"[yellow]  Retry {attempt + 1}/{max_retries} in {wait_time}s: {e}[/yellow]")
                    time_module.sleep(wait_time)
                else:
                    raise e
        return False

    # -------------------------------------------------------------------------
    # 1. Wan2.2-S2V-14B Base Model
    # -------------------------------------------------------------------------
    console.print("\n[cyan]1/3 Downloading Wan2.2-S2V-14B base model...[/cyan]")
    console.print("[dim]This is a large download (~25GB) and may take a while.[/dim]")
    try:
        if _has_base_model_files(LIVEAVATAR_BASE_DIR):
            console.print(f"  [dim]Already exists at {LIVEAVATAR_BASE_DIR}[/dim]")
        else:
            download_with_retry("Wan-AI/Wan2.2-S2V-14B", LIVEAVATAR_BASE_DIR)
            console.print("[green]  ✅ Wan2.2-S2V-14B base model downloaded![/green]")

    except Exception as e:
        console.print(f"[red]  ❌ Failed to download base model: {e}[/red]")
        console.print("[yellow]  You may need to manually download from:[/yellow]")
        console.print("  https://huggingface.co/Wan-AI/Wan2.2-S2V-14B")
        console.print("[yellow]  Or set HF_TOKEN environment variable if authentication is required.[/yellow]")
        success = False

    # -------------------------------------------------------------------------
    # 2. Live-Avatar LoRA Weights
    # -------------------------------------------------------------------------
    console.print("\n[cyan]2/3 Downloading Live-Avatar LoRA weights...[/cyan]")
    try:
        if any(LIVEAVATAR_LORA_DIR.iterdir()) if LIVEAVATAR_LORA_DIR.exists() else False:
            console.print(f"  [dim]Already exists at {LIVEAVATAR_LORA_DIR}[/dim]")
        else:
            download_with_retry("Quark-Vision/Live-Avatar", LIVEAVATAR_LORA_DIR)
            console.print("[green]  ✅ Live-Avatar LoRA weights downloaded![/green]")

    except Exception as e:
        console.print(f"[red]  ❌ Failed to download LoRA weights: {e}[/red]")
        console.print("[yellow]  You may need to manually download from:[/yellow]")
        console.print("  https://huggingface.co/Quark-Vision/Live-Avatar")
        console.print("[yellow]  Or set HF_TOKEN environment variable if authentication is required.[/yellow]")
        success = False

    # -------------------------------------------------------------------------
    # 3. wav2vec2 Audio Encoder
    # -------------------------------------------------------------------------
    console.print("\n[cyan]3/3 Downloading wav2vec2 audio encoder...[/cyan]")
    try:
        if (WAV2VEC2_DIR / "config.json").exists():
            console.print(f"  [dim]Already exists at {WAV2VEC2_DIR}[/dim]")
        else:
            download_with_retry("facebook/wav2vec2-base", WAV2VEC2_DIR)
            console.print("[green]  ✅ wav2vec2 audio encoder downloaded![/green]")

    except Exception as e:
        console.print(f"[yellow]  ⚠️ wav2vec2 download warning: {e}[/yellow]")
        console.print("[yellow]  This may be downloaded automatically at runtime.[/yellow]")

    return success


# =============================================================================
# STEP 5: INSTALL REMAINING REQUIREMENTS
# =============================================================================

def install_requirements() -> bool:
    """Install remaining Python requirements."""
    console.print("\n[bold blue]Step 5: Installing Python Requirements...[/bold blue]")

    requirements_path = PROJECT_ROOT / "requirements.txt"

    if not requirements_path.exists():
        console.print("[yellow]⚠️ requirements.txt not found[/yellow]")
        return True

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
        ])
        console.print("[green]✅ Requirements installed![/green]")
        return True

    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to install requirements: {e}[/red]")
        return False


# =============================================================================
# STEP 6: CREATE ASSETS & DEFAULT FILES
# =============================================================================

def create_default_assets():
    """Create default asset directories and placeholder files."""
    console.print("\n[bold blue]Step 6: Setting up Assets...[/bold blue]")

    # Create assets directory
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Create .env template if not exists
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        env_template = """# Real-Time AI Avatar Configuration
# Copy this file to .env and fill in your API keys

# ElevenLabs (Required for TTS)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Groq (Recommended for fast Llama 3)
GROQ_API_KEY=your_groq_api_key_here

# OpenAI (Optional - for Whisper API or GPT fallback)
OPENAI_API_KEY=your_openai_api_key_here

# LLM Provider (ollama, groq, or openai)
LLM_PROVIDER=groq
LLM_GROQ_MODEL=llama-3.3-70b-versatile

# Avatar settings
AVATAR_NAME=Aria

# Voice settings
VOICE_STABILITY=0.5
VOICE_SIMILARITY_BOOST=0.75
"""
        env_file.write_text(env_template)
        console.print(f"[green]✅ Created .env template at {env_file}[/green]")
        console.print("[yellow]⚠️ Please edit .env and add your API keys![/yellow]")

    # Check for avatar face image
    avatar_face = ASSETS_DIR / "avatar_face.png"
    if not avatar_face.exists():
        console.print(
            f"[yellow]⚠️ No avatar face image found at {avatar_face}[/yellow]\n"
            "   Please add a reference face image for the avatar."
        )

    console.print("[green]✅ Assets setup complete![/green]")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_installation() -> bool:
    """Verify all components are properly installed."""
    console.print("\n[bold blue]Verifying Installation...[/bold blue]")

    table = Table(title="Installation Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Path/Notes")

    all_good = True

    # Check CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_ok else 0
        table.add_row(
            "CUDA",
            "✅" if cuda_ok else "❌",
            f"{torch.cuda.get_device_name(0)} ({gpu_mem:.0f}GB)" if cuda_ok else "Not available"
        )
        if not cuda_ok:
            all_good = False
        elif gpu_mem < 48:
            table.add_row(
                "GPU Memory",
                "⚠️",
                f"{gpu_mem:.0f}GB - LiveAvatar requires 48GB+ VRAM"
            )
    except:
        table.add_row("CUDA", "❌", "PyTorch not installed")
        all_good = False

    # Check LiveAvatar repo
    liveavatar_ok = (VENDOR_DIR / "LiveAvatar").exists()
    table.add_row(
        "LiveAvatar Repo",
        "✅" if liveavatar_ok else "❌",
        str(VENDOR_DIR / "LiveAvatar") if liveavatar_ok else "Not cloned"
    )

    # Check model weights
    if LIVEAVATAR_BASE_DIR.exists():
        base_model_name = "Wan2.2-S2V-14B Base"
        base_model_path = LIVEAVATAR_BASE_DIR
    else:
        base_model_name = "Wan2.2-S2V-14B Base (legacy local Wan2.1 fallback)"
        base_model_path = LIVEAVATAR_BASE_DIR_LEGACY
    weights_checks = [
        (base_model_name, base_model_path, ["config.json", "model_index.json"]),
        ("Live-Avatar LoRA", LIVEAVATAR_LORA_DIR, []),
        ("wav2vec2 Encoder", WAV2VEC2_DIR, ["config.json"]),
    ]

    for name, path, required_files in weights_checks:
        if path.exists():
            if required_files:
                # Base model metadata may vary by repo version.
                if "config.json" in required_files and "model_index.json" in required_files:
                    missing = [] if _has_base_model_files(path) else ["config.json|model_index.json"]
                else:
                    missing = [f for f in required_files if not (path / f).exists()]
                if missing:
                    table.add_row(name, "⚠️", f"Missing: {', '.join(missing)}")
                else:
                    table.add_row(name, "✅", str(path))
            else:
                files = list(path.iterdir()) if path.exists() else []
                table.add_row(name, "✅" if files else "⚠️", str(path))
        else:
            table.add_row(name, "❌", "Not downloaded")
            all_good = False

    # Check FFmpeg
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    table.add_row(
        "FFmpeg",
        "✅" if ffmpeg_ok else "⚠️",
        shutil.which("ffmpeg") or "Using imageio-ffmpeg fallback"
    )

    # Check .env
    env_ok = (PROJECT_ROOT / ".env").exists()
    table.add_row(
        ".env File",
        "✅" if env_ok else "⚠️",
        str(PROJECT_ROOT / ".env") if env_ok else "Not created - using defaults"
    )

    console.print(table)

    return all_good


# =============================================================================
# MAIN WIZARD
# =============================================================================

def main():
    """Run the setup wizard."""
    parser = argparse.ArgumentParser(description="AI Avatar Setup Wizard")
    parser.add_argument("--skip-cuda-check", action="store_true",
                       help="Skip CUDA verification (for CPU-only testing)")
    parser.add_argument("--weights-only", action="store_true",
                       help="Only download model weights")
    parser.add_argument("--verify", action="store_true",
                       help="Only verify installation status")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]Real-Time AI Avatar[/bold cyan]\n"
        "[dim]Setup Wizard v2.0 - LiveAvatar Edition[/dim]\n\n"
        "This wizard will prepare your environment for running the AI Avatar application.\n"
        "It will check hardware, install dependencies, and download LiveAvatar model weights.\n\n"
        "[yellow]Note: LiveAvatar requires 48GB+ GPU VRAM (A100/H100 class)[/yellow]",
        title="🎭 Welcome",
        border_style="cyan"
    ))

    if args.verify:
        verify_installation()
        return

    # Track overall success
    all_success = True

    # Step 1: CUDA Check
    if not args.skip_cuda_check and not args.weights_only:
        cuda_ok, gpu_name, cuda_version = check_cuda_and_gpu()
        if not cuda_ok:
            console.print("\n[red]Setup cannot continue without CUDA. Please install NVIDIA drivers and CUDA Toolkit.[/red]")
            sys.exit(1)

    if not args.weights_only:
        # Step 2: LiveAvatar Dependencies
        if not install_liveavatar_dependencies():
            all_success = False
            console.print("[yellow]⚠️ LiveAvatar dependencies installation had issues. Continuing...[/yellow]")

        # Step 2b: FFmpeg
        if not check_ffmpeg():
            console.print("[yellow]⚠️ FFmpeg not found. Some features may not work.[/yellow]")

        # Step 3: Clone LiveAvatar
        if not clone_liveavatar_repo():
            all_success = False
            console.print("[yellow]⚠️ LiveAvatar clone failed. Weights download may still work.[/yellow]")

        # Step 3b: Install LiveAvatar repo dependencies
        install_liveavatar_repo_deps()

    # Step 4: Download Weights
    if not download_all_weights():
        all_success = False

    if not args.weights_only:
        # Step 5: Install requirements
        if not install_requirements():
            all_success = False

        # Step 6: Setup assets
        create_default_assets()

    # Final verification
    console.print("\n" + "=" * 60)
    verify_installation()

    if all_success:
        console.print(Panel(
            "[bold green]Setup Complete![/bold green]\n\n"
            "Next steps:\n"
            "1. Edit [cyan].env[/cyan] file and add your API keys\n"
            "2. Add an avatar face image to [cyan]assets/avatar_face.png[/cyan]\n"
            "3. Run [cyan]python main.py[/cyan] to start the avatar\n\n"
            "[dim]For troubleshooting, run: python setup_wizard.py --verify[/dim]",
            title="✅ Success",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold yellow]Setup completed with warnings.[/bold yellow]\n\n"
            "Some components may not have installed correctly.\n"
            "Please review the output above and manually install missing components.\n\n"
            "You can re-run this wizard at any time to retry failed steps.",
            title="⚠️ Partial Success",
            border_style="yellow"
        ))


if __name__ == "__main__":
    main()
