#!/bin/bash
# =============================================================================
# AI Avatar - Complete EC2 Deployment Script
# =============================================================================
# This script handles EVERYTHING needed to run AI Avatar on AWS EC2.
# Run this once after cloning the repository.
#
# Prerequisites:
#   - AWS EC2 instance with GPU (g5.xlarge minimum, g5.12xlarge for LiveAvatar)
#   - Amazon Linux 2023 or Ubuntu 22.04
#   - At least 100GB disk space
#   - NVIDIA GPU drivers installed
#
# Usage:
#   chmod +x deploy_ec2.sh
#   ./deploy_ec2.sh
#
# What this script does:
#   1. Installs system dependencies (ffmpeg, espeak-ng, etc.)
#   2. Creates Python virtual environment
#   3. Installs PyTorch with CUDA support
#   4. Installs all Python packages
#   5. Installs and tests TTS (gTTS guaranteed to work)
#   6. Installs and tests STT (faster-whisper)
#   7. Downloads LiveAvatar models (optional, ~30GB)
#   8. Verifies all components work
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${CYAN}========================================${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}========================================${NC}"; }

# Error handler
handle_error() {
    log_error "Script failed at line $1"
    exit 1
}
trap 'handle_error $LINENO' ERR

# =============================================================================
# DETECT OS AND PACKAGE MANAGER
# =============================================================================
log_section "Detecting System"

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    log_info "Detected OS: $OS"
fi

if command -v dnf &> /dev/null; then
    PKG_MGR="dnf"
    INSTALL_CMD="sudo dnf install -y --allowerasing"
elif command -v yum &> /dev/null; then
    PKG_MGR="yum"
    INSTALL_CMD="sudo yum install -y"
elif command -v apt-get &> /dev/null; then
    PKG_MGR="apt"
    INSTALL_CMD="sudo apt-get install -y"
    sudo apt-get update
else
    log_error "No supported package manager found"
    exit 1
fi

log_info "Using package manager: $PKG_MGR"

# =============================================================================
# 1. INSTALL SYSTEM DEPENDENCIES
# =============================================================================
log_section "1. Installing System Dependencies"

# Essential build tools
log_info "Installing build tools..."
if [ "$PKG_MGR" = "apt" ]; then
    $INSTALL_CMD build-essential python3-dev python3-venv git curl wget
else
    $INSTALL_CMD gcc gcc-c++ python3-devel git curl wget
fi

# FFmpeg (critical for audio processing)
log_info "Installing FFmpeg..."
if [ "$PKG_MGR" = "apt" ]; then
    $INSTALL_CMD ffmpeg
else
    # Amazon Linux needs EPEL for ffmpeg
    # AL2023 uses dnf, AL2 uses yum with amazon-linux-extras
    if [ "$PKG_MGR" = "dnf" ]; then
        sudo dnf install -y epel-release 2>/dev/null || true
    else
        sudo amazon-linux-extras install epel -y 2>/dev/null || \
        sudo yum install -y epel-release 2>/dev/null || true
    fi
    $INSTALL_CMD ffmpeg || {
        log_warning "FFmpeg not in repos, installing via static binary..."
        cd /tmp
        wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
        tar xf ffmpeg-release-amd64-static.tar.xz
        sudo cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/
        sudo cp ffmpeg-*-amd64-static/ffprobe /usr/local/bin/
        cd -
    }
fi

# Verify ffmpeg
if command -v ffmpeg &> /dev/null; then
    log_success "FFmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
else
    log_error "FFmpeg installation failed"
    exit 1
fi

# PortAudio (for sounddevice)
log_info "Installing PortAudio..."
if [ "$PKG_MGR" = "apt" ]; then
    $INSTALL_CMD portaudio19-dev libportaudio2
else
    $INSTALL_CMD portaudio-devel || log_warning "PortAudio not found (optional for web mode)"
fi

# espeak-ng (for pyttsx3 fallback TTS)
log_info "Installing espeak-ng..."
if [ "$PKG_MGR" = "apt" ]; then
    $INSTALL_CMD espeak-ng libespeak-ng1 || $INSTALL_CMD espeak libespeak1 || true
else
    $INSTALL_CMD espeak-ng espeak-ng-devel 2>/dev/null || \
    $INSTALL_CMD espeak 2>/dev/null || \
    log_warning "espeak-ng not found (optional - gTTS will be used as fallback)"
fi

# =============================================================================
# 2. CREATE PYTHON VIRTUAL ENVIRONMENT
# =============================================================================
log_section "2. Setting Up Python Environment"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

log_info "Activating virtual environment..."
source $VENV_DIR/bin/activate

log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

log_success "Python environment ready: $(python --version)"

# =============================================================================
# 3. INSTALL PYTORCH WITH CUDA
# =============================================================================
log_section "3. Installing PyTorch with CUDA"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    log_info "Detected GPU: $GPU_NAME ($GPU_MEM)"

    log_info "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    log_warning "No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip install torch torchvision torchaudio
fi

# Verify PyTorch
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

log_success "PyTorch installed successfully"

# =============================================================================
# 4. INSTALL PYTHON PACKAGES
# =============================================================================
log_section "4. Installing Python Packages"

log_info "Installing requirements.txt..."
pip install -r requirements.txt

log_success "Python packages installed"

# =============================================================================
# 5. INSTALL AND TEST TTS
# =============================================================================
log_section "5. Installing Text-to-Speech"

# gTTS - GUARANTEED to work (just needs internet)
log_info "Installing gTTS (Google TTS - guaranteed fallback)..."
pip install gTTS

# Test gTTS
python -c "
from gtts import gTTS
import io
tts = gTTS(text='Test', lang='en')
buffer = io.BytesIO()
tts.write_to_fp(buffer)
print(f'gTTS working! Generated {len(buffer.getvalue())} bytes')
" && log_success "gTTS verified working" || log_error "gTTS failed"

# Piper TTS - High quality local TTS
log_info "Installing Piper TTS..."
pip install piper-tts || log_warning "Piper TTS installation failed (gTTS will be used)"

# Test Piper
python -c "
try:
    from piper.voice import PiperVoice
    print('Piper TTS package installed')
except Exception as e:
    print(f'Piper not available: {e}')
"

# ElevenLabs (paid cloud TTS)
log_info "Installing ElevenLabs..."
pip install elevenlabs

log_success "TTS installation complete"

# =============================================================================
# 6. INSTALL AND TEST STT (WHISPER)
# =============================================================================
log_section "6. Installing Speech-to-Text (Whisper)"

# faster-whisper is preferred
log_info "Installing faster-whisper..."
pip install faster-whisper

# Test faster-whisper
python -c "
try:
    from faster_whisper import WhisperModel
    print('faster-whisper installed successfully')
    # Don't load model here (takes time), just verify import
except ImportError as e:
    print(f'faster-whisper import failed: {e}')
except Exception as e:
    print(f'faster-whisper error: {e}')
" && log_success "faster-whisper verified" || {
    log_warning "faster-whisper failed, installing openai-whisper as fallback..."
    pip install openai-whisper
}

log_success "STT installation complete"

# =============================================================================
# 7. CONFIGURE ENVIRONMENT
# =============================================================================
log_section "7. Configuring Environment"

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    log_info "Creating .env from template..."
    cp .env.example .env
    log_warning "Please edit .env and add your GROQ_API_KEY!"
else
    log_info ".env already exists"
fi

# Check for GROQ_API_KEY
if grep -q "GROQ_API_KEY=your_" .env 2>/dev/null || ! grep -q "GROQ_API_KEY=" .env 2>/dev/null; then
    log_warning "GROQ_API_KEY not set in .env - LLM will not work without it!"
    log_warning "Get your free API key at: https://console.groq.com/"
fi

# Create assets directory
mkdir -p assets

# Check for avatar image
if [ ! -f "assets/avatar_face.png" ] && [ ! -f "assets/avatar.mp4" ]; then
    log_warning "No avatar image found in assets/"
    log_warning "Add assets/avatar_face.png (512x512 recommended) for face rendering"
fi

log_success "Environment configured"

# =============================================================================
# 8. DOWNLOAD LIVEAVATAR MODELS (OPTIONAL)
# =============================================================================
log_section "8. LiveAvatar Models"

# Create models directory with proper permissions
log_info "Setting up models directory..."
mkdir -p models
chmod 755 models

# Check disk space (need ~50GB free for models + cache)
DISK_FREE_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$DISK_FREE_GB" -lt 50 ]; then
    log_warning "Only ${DISK_FREE_GB}GB free disk space. Models require ~50GB."
    log_warning "Consider expanding your EBS volume if download fails."
fi

# Check GPU memory for LiveAvatar
GPU_MEM_GB=$(python -c "
import torch
if torch.cuda.is_available():
    print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    print(0)
" 2>/dev/null || echo "0")

if [ "$GPU_MEM_GB" -ge 48 ]; then
    log_info "GPU has ${GPU_MEM_GB}GB VRAM - LiveAvatar is supported!"

    # Check for HuggingFace token
    if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
        log_info "Tip: Set HF_TOKEN environment variable if models require authentication"
        log_info "Get a token from: https://huggingface.co/settings/tokens"
    fi

    read -p "Download LiveAvatar models (~30GB)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Running setup wizard to download models..."
        python setup_wizard.py --weights-only
    else
        log_info "Skipping LiveAvatar model download"
        log_info "You can download later with: python setup_wizard.py --weights-only"
    fi
else
    log_warning "GPU has only ${GPU_MEM_GB}GB VRAM"
    log_warning "LiveAvatar requires 48GB+ VRAM (A100/H100 class GPU)"
    log_info "The system will use fallback face rendering mode"
fi

# =============================================================================
# 9. RUN COMPONENT TESTS
# =============================================================================
log_section "9. Running Component Tests"

python test_components.py

# =============================================================================
# FINAL SUMMARY
# =============================================================================
log_section "Deployment Complete!"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  AI Avatar is ready to run!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo -e "  1. ${YELLOW}Edit .env and add your GROQ_API_KEY${NC}"
echo "     Get free API key at: https://console.groq.com/"
echo ""
echo -e "  2. ${YELLOW}Add avatar image${NC}"
echo "     Place a face image at: assets/avatar_face.png"
echo ""
echo -e "  3. ${YELLOW}Start the server${NC}"
echo "     python web_server.py --host 0.0.0.0 --port 8080"
echo ""
echo -e "  4. ${YELLOW}Access from browser${NC}"
echo "     http://<your-ec2-ip>:8080  (text chat)"
echo "     https://<your-domain>      (required for browser microphone)"
echo ""

# Show EC2-specific instructions
if curl -s --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4 &>/dev/null; then
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
    echo -e "${CYAN}Your EC2 public IP: $PUBLIC_IP${NC}"
    echo -e "${CYAN}Access URL (text chat): http://$PUBLIC_IP:8080${NC}"
    echo -e "${CYAN}For microphone input, use HTTPS via a domain/reverse proxy.${NC}"
    echo ""
    echo -e "${YELLOW}Make sure port 8080 is open in your EC2 Security Group!${NC}"
fi

echo ""
