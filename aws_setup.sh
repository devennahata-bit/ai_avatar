#!/bin/bash
# =============================================================================
# AI Avatar - AWS Linux Setup Script
# =============================================================================
# Optimized for: Amazon Linux 2/2023, Ubuntu 20.04/22.04/24.04 on EC2
# GPU instances: g4dn, g5, p3, p4d, p5 (NVIDIA GPU required)
#
# Usage:
#   chmod +x aws_setup.sh
#   ./aws_setup.sh              # Full setup
#   ./aws_setup.sh --deps-only  # Only install dependencies
#   ./aws_setup.sh --models-only # Only download models
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
DEPS_ONLY=false
MODELS_ONLY=false
SKIP_CUDA_CHECK=false

for arg in "$@"; do
    case $arg in
        --deps-only)
            DEPS_ONLY=true
            ;;
        --models-only)
            MODELS_ONLY=true
            ;;
        --skip-cuda-check)
            SKIP_CUDA_CHECK=true
            ;;
    esac
done

echo -e "${CYAN}"
echo "============================================================"
echo "       AI Avatar - AWS Linux Setup"
echo "============================================================"
echo -e "${NC}"

# =============================================================================
# DETECT OS
# =============================================================================
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
        OS_VERSION="$VERSION_ID"
    elif [ -f /etc/system-release ]; then
        if grep -q "Amazon Linux" /etc/system-release; then
            OS_ID="amzn"
            OS_VERSION=$(grep -oP '(?<=release )\d+' /etc/system-release || echo "2")
        fi
    else
        OS_ID="unknown"
        OS_VERSION="unknown"
    fi

    echo -e "${CYAN}Detected OS:${NC} $OS_ID $OS_VERSION"
}

# =============================================================================
# CHECK GPU
# =============================================================================
check_gpu() {
    echo -e "\n${CYAN}[1/6] Checking GPU...${NC}"

    if [ "$SKIP_CUDA_CHECK" = true ]; then
        echo -e "${YELLOW}Skipping CUDA check as requested${NC}"
        return 0
    fi

    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}ERROR: nvidia-smi not found!${NC}"
        echo "Please ensure NVIDIA drivers are installed."
        echo ""
        echo "For Amazon Linux 2:"
        echo "  sudo yum install -y gcc kernel-devel-\$(uname -r)"
        echo "  sudo yum install -y nvidia-driver-latest-dkms"
        echo ""
        echo "For Ubuntu:"
        echo "  sudo apt install -y nvidia-driver-535"
        echo ""
        echo "Or use an AWS Deep Learning AMI which has drivers pre-installed."
        exit 1
    fi

    # Check if GPU is accessible
    if ! nvidia-smi &> /dev/null; then
        echo -e "${RED}ERROR: GPU not accessible!${NC}"
        echo "nvidia-smi failed. Check driver installation."
        exit 1
    fi

    # Display GPU info
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

    # Check VRAM
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    VRAM_GB=$((VRAM_MB / 1024))

    if [ "$VRAM_GB" -lt 16 ]; then
        echo -e "${YELLOW}WARNING: Only ${VRAM_GB}GB VRAM detected.${NC}"
        echo "LiveAvatar requires 48GB+ for optimal performance."
        echo "You may need to use CPU-only mode or smaller models."
    elif [ "$VRAM_GB" -lt 48 ]; then
        echo -e "${YELLOW}NOTE: ${VRAM_GB}GB VRAM detected.${NC}"
        echo "FP8 quantization will be used to fit in memory."
    else
        echo -e "${GREEN}Excellent! ${VRAM_GB}GB VRAM available.${NC}"
    fi
}

# =============================================================================
# INSTALL SYSTEM DEPENDENCIES
# =============================================================================
install_system_deps() {
    echo -e "\n${CYAN}[2/6] Installing system dependencies...${NC}"

    detect_os

    case "$OS_ID" in
        ubuntu|debian)
            echo "Using apt package manager..."
            sudo apt-get update -qq

            # Core build tools
            sudo apt-get install -y -qq \
                build-essential \
                git \
                curl \
                wget \
                unzip \
                software-properties-common

            # Python
            sudo apt-get install -y -qq \
                python3 \
                python3-pip \
                python3-venv \
                python3-dev

            # Audio libraries (for sounddevice)
            sudo apt-get install -y -qq \
                portaudio19-dev \
                libportaudio2 \
                libasound2-dev \
                libsndfile1 \
                pulseaudio

            # Video/Image processing
            sudo apt-get install -y -qq \
                ffmpeg \
                libgl1-mesa-glx \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                libgomp1

            # For headless OpenCV
            sudo apt-get install -y -qq \
                libopencv-dev \
                python3-opencv || true
            ;;

        amzn|rhel|centos|fedora)
            echo "Using yum/dnf package manager..."

            # Determine package manager
            if command -v dnf &> /dev/null; then
                PKG_MGR="dnf"
            else
                PKG_MGR="yum"
            fi

            # Enable EPEL for additional packages
            if [ "$OS_ID" = "amzn" ]; then
                # Amazon Linux 2023 uses dnf and doesn't have amazon-linux-extras
                # Amazon Linux 2 uses yum and has amazon-linux-extras
                if [ "$PKG_MGR" = "dnf" ]; then
                    # AL2023: EPEL is available directly
                    sudo dnf install -y epel-release 2>/dev/null || true
                else
                    # AL2: Use amazon-linux-extras
                    sudo amazon-linux-extras install epel -y 2>/dev/null || \
                    sudo $PKG_MGR install -y epel-release || true
                fi
            else
                sudo $PKG_MGR install -y epel-release || true
            fi

            # Core build tools
            sudo $PKG_MGR groupinstall -y "Development Tools" || \
            sudo $PKG_MGR install -y gcc gcc-c++ make

            sudo $PKG_MGR install -y \
                git \
                curl \
                wget \
                unzip

            # Python
            sudo $PKG_MGR install -y \
                python3 \
                python3-pip \
                python3-devel

            # Create venv module if missing
            sudo $PKG_MGR install -y python3-virtualenv || \
            pip3 install --user virtualenv

            # Audio libraries
            sudo $PKG_MGR install -y \
                portaudio-devel \
                alsa-lib-devel \
                libsndfile-devel \
                pulseaudio-libs-devel || true

            # FFmpeg (may need RPM Fusion on some distros)
            sudo $PKG_MGR install -y ffmpeg ffmpeg-devel || {
                echo -e "${YELLOW}FFmpeg not in repos, installing from source...${NC}"
                install_ffmpeg_from_static
            }

            # OpenCV dependencies
            sudo $PKG_MGR install -y \
                mesa-libGL \
                glib2 \
                libSM \
                libXext \
                libXrender || true
            ;;

        *)
            echo -e "${YELLOW}Unknown OS: $OS_ID${NC}"
            echo "Please install manually: git, python3, pip, ffmpeg, portaudio"
            ;;
    esac

    echo -e "${GREEN}System dependencies installed!${NC}"
}

# Install FFmpeg from static build if not available in repos
install_ffmpeg_from_static() {
    if command -v ffmpeg &> /dev/null; then
        return 0
    fi

    echo "Downloading static FFmpeg build..."
    cd /tmp
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar xf ffmpeg-release-amd64-static.tar.xz
    sudo cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/
    sudo cp ffmpeg-*-amd64-static/ffprobe /usr/local/bin/
    rm -rf ffmpeg-*
    cd "$SCRIPT_DIR"
    echo -e "${GREEN}FFmpeg installed to /usr/local/bin${NC}"
}

# =============================================================================
# SETUP PYTHON ENVIRONMENT
# =============================================================================
setup_python_env() {
    echo -e "\n${CYAN}[3/6] Setting up Python environment...${NC}"

    # Create virtual environment
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi

    # Activate
    source .venv/bin/activate

    # Upgrade pip and tools
    echo "Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel --quiet

    # Install PyTorch with CUDA
    echo "Installing PyTorch with CUDA support..."

    # Detect CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    elif [ -f /usr/local/cuda/version.txt ]; then
        CUDA_VERSION=$(cat /usr/local/cuda/version.txt | grep -oP '[0-9]+\.[0-9]+' | head -1)
    else
        CUDA_VERSION="12.1"  # Default
    fi

    echo "Detected CUDA version: $CUDA_VERSION"

    # Select appropriate PyTorch index
    case "$CUDA_VERSION" in
        11.8*)
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            ;;
        12.1*|12.2*|12.3*|12.4*)
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            ;;
        *)
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            ;;
    esac

    pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" --quiet

    echo -e "${GREEN}PyTorch installed!${NC}"

    # Verify CUDA in PyTorch
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
}

# =============================================================================
# INSTALL PYTHON DEPENDENCIES
# =============================================================================
install_python_deps() {
    echo -e "\n${CYAN}[4/6] Installing Python dependencies...${NC}"

    source .venv/bin/activate

    # Install requirements
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt --quiet

    # Install Linux-compatible TTS
    echo "Installing TTS engines..."

    # Try to install RealtimeTTS with Kokoro (may fail on some Linux)
    pip install realtimetts 2>/dev/null || true

    # Install Kokoro engine separately (more reliable)
    pip install kokoro-onnx 2>/dev/null || {
        echo -e "${YELLOW}Kokoro TTS not available, will use ElevenLabs or local TTS${NC}"
    }

    # Install Piper TTS (FREE high-quality neural TTS - RECOMMENDED)
    echo "Installing Piper TTS (free neural TTS)..."
    pip install piper-tts --quiet || {
        echo -e "${YELLOW}Piper TTS installation failed, will fall back to other TTS${NC}"
    }

    # Pre-download default Piper voice model
    echo "Downloading default Piper voice model..."
    python3 -c "
try:
    from piper.download import ensure_voice_exists
    ensure_voice_exists('en_US-lessac-medium', [], None, None)
    print('Piper voice model downloaded successfully')
except Exception as e:
    print(f'Could not pre-download voice model: {e}')
    print('Voice will be downloaded on first use')
" 2>/dev/null || true

    # Install pyttsx3 as fallback (uses espeak on Linux)
    pip install pyttsx3 --quiet || true

    # Install espeak/espeak-ng as fallback for pyttsx3
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y -qq espeak-ng libespeak-ng1 || \
        sudo apt-get install -y -qq espeak libespeak1 || true
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y espeak-ng espeak-ng-devel 2>/dev/null || \
        sudo dnf install -y espeak 2>/dev/null || true
    elif command -v yum &> /dev/null; then
        sudo yum install -y espeak-ng espeak-ng-devel 2>/dev/null || \
        sudo yum install -y espeak 2>/dev/null || true
    fi

    # ElevenLabs (optional paid cloud TTS)
    pip install elevenlabs --quiet

    echo -e "${GREEN}Python dependencies installed!${NC}"
}

# =============================================================================
# SETUP ENVIRONMENT FILE
# =============================================================================
setup_env_file() {
    echo -e "\n${CYAN}[5/6] Setting up environment configuration...${NC}"

    if [ -f .env ]; then
        echo ".env file already exists, keeping existing configuration"

        # Check if it has required keys
        if ! grep -q "GROQ_API_KEY" .env || grep -q "your_groq_api_key_here" .env; then
            echo -e "${YELLOW}WARNING: GROQ_API_KEY may not be configured in .env${NC}"
        fi
    else
        # Create AWS-optimized .env
        cat > .env << 'ENVFILE'
# =============================================================================
# Real-Time AI Avatar - AWS Linux Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# API Keys (Required)
# -----------------------------------------------------------------------------
# Get your keys from:
# - Groq: https://console.groq.com/
# - ElevenLabs: https://elevenlabs.io/
# - OpenAI: https://platform.openai.com/

GROQ_API_KEY=your_groq_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# ElevenLabs Voice ID (Rachel is default)
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# -----------------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------------
LLM_PROVIDER=groq
LLM_GROQ_MODEL=llama-3.3-70b-versatile
LLM_OPENAI_MODEL=gpt-4o
LLM_OLLAMA_MODEL=llama3:8b
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=256

# -----------------------------------------------------------------------------
# Voice/TTS Configuration (AWS Linux optimized)
# -----------------------------------------------------------------------------
# Options: piper (FREE, recommended), elevenlabs (paid), kokoro, local
# "piper" - FREE high-quality neural TTS (default, recommended)
# "elevenlabs" - Paid cloud TTS (requires API key)
# "local" - pyttsx3 + espeak fallback (low quality)
VOICE_TTS_PROVIDER=piper

# Piper TTS settings (FREE neural TTS - recommended)
VOICE_PIPER_MODEL=en_US-lessac-medium

# Kokoro settings (Windows/macOS)
VOICE_KOKORO_VOICE=af_heart
VOICE_KOKORO_SPEED=1.0

# ElevenLabs settings
VOICE_STABILITY=0.5
VOICE_SIMILARITY_BOOST=0.75
VOICE_STYLE=0.0
VOICE_USE_SPEAKER_BOOST=true
VOICE_MODEL_ID=eleven_multilingual_v2

# -----------------------------------------------------------------------------
# Whisper (Speech-to-Text) Configuration
# -----------------------------------------------------------------------------
WHISPER_MODEL_SIZE=base
WHISPER_USE_API=false
WHISPER_DEVICE=cuda

# -----------------------------------------------------------------------------
# LiveAvatar (Lip-Sync) Configuration
# -----------------------------------------------------------------------------
LIVEAVATAR_FPS=24
LIVEAVATAR_DEVICE=cuda
LIVEAVATAR_USE_FP8=true
LIVEAVATAR_USE_FP16=false
LIVEAVATAR_SAMPLING_STEPS=10
LIVEAVATAR_MULTI_GPU=false

# -----------------------------------------------------------------------------
# Application Settings (Headless/Server Mode)
# -----------------------------------------------------------------------------
APP_DEBUG=false
APP_LOG_LEVEL=INFO
APP_SHOW_PREVIEW=false

# Queue sizes
APP_AUDIO_QUEUE_SIZE=50
APP_FRAME_QUEUE_SIZE=30

# -----------------------------------------------------------------------------
# Audio Device Settings (Linux)
# Leave empty for auto-detection, or set to specific device index
# Use 'python -c "import sounddevice; print(sounddevice.query_devices())"' to list
# -----------------------------------------------------------------------------
AUDIO_INPUT_DEVICE=
AUDIO_OUTPUT_DEVICE=

# -----------------------------------------------------------------------------
# Avatar Settings
# -----------------------------------------------------------------------------
AVATAR_NAME=Aria
ENVFILE

        echo -e "${GREEN}Created .env file with AWS Linux defaults${NC}"
        echo -e "${YELLOW}IMPORTANT: Edit .env and add your API keys!${NC}"
    fi
}

# =============================================================================
# DOWNLOAD MODELS
# =============================================================================
download_models() {
    echo -e "\n${CYAN}[6/6] Downloading model weights...${NC}"
    echo -e "${YELLOW}This will download ~30GB of models. This may take a while.${NC}"

    source .venv/bin/activate

    # Create models directory
    mkdir -p models
    chmod 755 models

    # Check disk space
    DISK_FREE_GB=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo "100")
    if [ "$DISK_FREE_GB" -lt 50 ]; then
        echo -e "${YELLOW}Warning: Only ${DISK_FREE_GB}GB free. Models require ~50GB.${NC}"
    fi

    # Check for HuggingFace token
    if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
        echo -e "${YELLOW}Tip: Set HF_TOKEN if model downloads require authentication${NC}"
        echo "  export HF_TOKEN=your_token_here"
        echo "  Get token from: https://huggingface.co/settings/tokens"
    fi

    # Run setup wizard in weights-only mode
    python setup_wizard.py --weights-only

    echo -e "${GREEN}Models downloaded!${NC}"
}

# =============================================================================
# CREATE CONVENIENCE SCRIPTS
# =============================================================================
create_run_scripts() {
    echo -e "\n${CYAN}Creating convenience scripts...${NC}"

    # Start script
    cat > start.sh << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

echo "Starting AI Avatar..."
echo "Web interface will be available at http://0.0.0.0:8080"
echo ""

python web_server.py --host 0.0.0.0 --port 8080
SCRIPT
    chmod +x start.sh

    # Start headless script
    cat > start_headless.sh << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

echo "Starting AI Avatar in headless mode..."
python main.py --no-display
SCRIPT
    chmod +x start_headless.sh

    # Test script
    cat > test_setup.sh << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

echo "Testing AI Avatar setup..."
echo ""

echo "1. Checking Python..."
python --version

echo ""
echo "2. Checking PyTorch and CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "3. Checking core modules..."
python -c "from config import api_config; print('Config: OK')"
python -c "from modules.brain import LLMBrain; print('LLM Brain: OK')"
python -c "from modules.listener import SpeechRecognizer; print('Speech Listener: OK')"

echo ""
echo "4. Checking TTS..."
python -c "
try:
    from modules.voice import RealtimeTTSSynthesizer
    print('Kokoro TTS: OK')
except Exception as e:
    print(f'Kokoro TTS: Not available ({e})')

try:
    from modules.voice import VoiceSynthesizer
    print('ElevenLabs TTS: OK')
except Exception as e:
    print(f'ElevenLabs TTS: Not available ({e})')
"

echo ""
echo "5. Checking FFmpeg..."
ffmpeg -version 2>/dev/null | head -1 || echo "FFmpeg: Not found"

echo ""
echo "6. Running smoke tests..."
python test_smoke.py 2>/dev/null || echo "Smoke tests: Skipped (test_smoke.py not found)"

echo ""
echo "Setup test complete!"
SCRIPT
    chmod +x test_setup.sh

    echo -e "${GREEN}Created: start.sh, start_headless.sh, test_setup.sh${NC}"
}

# =============================================================================
# PRINT SUMMARY
# =============================================================================
print_summary() {
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "YOUR_SERVER_IP")

    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}       Setup Complete!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo ""
    echo "1. Edit .env and add your API keys:"
    echo "   nano .env"
    echo ""
    echo "2. Add an avatar image:"
    echo "   cp your_face.png assets/avatar_face.png"
    echo ""
    echo "3. Run the avatar:"
    echo ""
    echo "   ${CYAN}Web Server (recommended for cloud):${NC}"
    echo "   ./start.sh"
    echo "   Then open: https://<your-domain> (recommended for browser microphone)"
    echo "   If testing without HTTPS, text chat still works at: http://${PUBLIC_IP}:8080"
    echo ""
    echo "   ${CYAN}Headless mode:${NC}"
    echo "   ./start_headless.sh"
    echo ""
    echo "   ${CYAN}Test your setup:${NC}"
    echo "   ./test_setup.sh"
    echo ""
    echo -e "${YELLOW}Make sure port 8080 is open in your security group!${NC}"
    echo -e "${YELLOW}Browser microphone requires HTTPS (or localhost) due to browser security rules.${NC}"
    echo ""
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    # Models only
    if [ "$MODELS_ONLY" = true ]; then
        source .venv/bin/activate 2>/dev/null || {
            echo "Virtual environment not found. Run full setup first."
            exit 1
        }
        download_models
        exit 0
    fi

    # Full setup or deps only
    check_gpu
    install_system_deps
    setup_python_env
    install_python_deps
    setup_env_file

    if [ "$DEPS_ONLY" = true ]; then
        echo -e "\n${GREEN}Dependencies installed!${NC}"
        echo "Run './aws_setup.sh --models-only' to download models."
        exit 0
    fi

    download_models
    create_run_scripts
    print_summary
}

main "$@"
