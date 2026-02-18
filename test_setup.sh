#!/bin/bash
# =============================================================================
# AI Avatar - Test Setup / Installation Verification
# =============================================================================
# Verifies that all components are properly installed
#
# Usage: ./test_setup.sh
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}   AI Avatar - Installation Check${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

ERRORS=0

# Find the virtual environment
if [ -d ".venv" ]; then
    VENV=".venv"
elif [ -d "venv" ]; then
    VENV="venv"
else
    echo -e "${RED}[FAIL]${NC} No virtual environment found"
    echo "       Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source $VENV/bin/activate

# -----------------------------------------------------------------------------
# Check Python
# -----------------------------------------------------------------------------
echo -e "${BLUE}[1/8] Python${NC}"
PYTHON_VERSION=$(python --version 2>&1)
echo -e "      ${GREEN}$PYTHON_VERSION${NC}"

# -----------------------------------------------------------------------------
# Check GPU/CUDA
# -----------------------------------------------------------------------------
echo -e "${BLUE}[2/8] GPU / CUDA${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo -e "      ${GREEN}GPU: $GPU_NAME ($GPU_MEM)${NC}"
else
    echo -e "      ${YELLOW}nvidia-smi not found${NC}"
fi

python -c "
import torch
if torch.cuda.is_available():
    print(f'      \033[0;32mPyTorch CUDA: Yes ({torch.cuda.get_device_name(0)})\033[0m')
else:
    print('      \033[1;33mPyTorch CUDA: No (CPU only)\033[0m')
" 2>/dev/null || echo -e "      ${RED}PyTorch not installed${NC}"

# -----------------------------------------------------------------------------
# Check FFmpeg
# -----------------------------------------------------------------------------
echo -e "${BLUE}[3/8] FFmpeg${NC}"
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3)
    echo -e "      ${GREEN}FFmpeg: $FFMPEG_VERSION${NC}"
else
    echo -e "      ${RED}FFmpeg not found${NC}"
    ((ERRORS++))
fi

# -----------------------------------------------------------------------------
# Check Core Modules
# -----------------------------------------------------------------------------
echo -e "${BLUE}[4/8] Core Modules${NC}"

python -c "from config import api_config; print('      \033[0;32mconfig.py: OK\033[0m')" 2>/dev/null || {
    echo -e "      ${RED}config.py: FAIL${NC}"
    ((ERRORS++))
}

python -c "from modules.brain import LLMBrain; print('      \033[0;32mmodules/brain.py: OK\033[0m')" 2>/dev/null || {
    echo -e "      ${RED}modules/brain.py: FAIL${NC}"
    ((ERRORS++))
}

python -c "from modules.voice import create_tts_synthesizer; print('      \033[0;32mmodules/voice.py: OK\033[0m')" 2>/dev/null || {
    echo -e "      ${RED}modules/voice.py: FAIL${NC}"
    ((ERRORS++))
}

python -c "from modules.listener import SpeechListener; print('      \033[0;32mmodules/listener.py: OK\033[0m')" 2>/dev/null || {
    echo -e "      ${RED}modules/listener.py: FAIL${NC}"
    ((ERRORS++))
}

# -----------------------------------------------------------------------------
# Check TTS Providers
# -----------------------------------------------------------------------------
echo -e "${BLUE}[5/8] TTS Providers${NC}"

python -c "from piper.voice import PiperVoice; print('      \033[0;32mPiper TTS: OK (FREE)\033[0m')" 2>/dev/null || {
    echo -e "      ${YELLOW}Piper TTS: Not installed (pip install piper-tts)${NC}"
}

python -c "import elevenlabs; print('      \033[0;32mElevenLabs: OK (paid)\033[0m')" 2>/dev/null || {
    echo -e "      ${YELLOW}ElevenLabs: Not installed${NC}"
}

python -c "import pyttsx3; print('      \033[0;32mpyttsx3: OK (fallback)\033[0m')" 2>/dev/null || {
    echo -e "      ${YELLOW}pyttsx3: Not installed${NC}"
}

# -----------------------------------------------------------------------------
# Check LLM Providers
# -----------------------------------------------------------------------------
echo -e "${BLUE}[6/8] LLM Providers${NC}"

python -c "import groq; print('      \033[0;32mGroq: OK\033[0m')" 2>/dev/null || {
    echo -e "      ${RED}Groq: Not installed${NC}"
    ((ERRORS++))
}

python -c "import openai; print('      \033[0;32mOpenAI: OK\033[0m')" 2>/dev/null || {
    echo -e "      ${YELLOW}OpenAI: Not installed${NC}"
}

python -c "import ollama; print('      \033[0;32mOllama: OK\033[0m')" 2>/dev/null || {
    echo -e "      ${YELLOW}Ollama: Not installed${NC}"
}

# -----------------------------------------------------------------------------
# Check Environment
# -----------------------------------------------------------------------------
echo -e "${BLUE}[7/8] Environment (.env)${NC}"

if [ -f ".env" ]; then
    echo -e "      ${GREEN}.env file: Found${NC}"

    if grep -q "GROQ_API_KEY=.*[^[:space:]]" .env 2>/dev/null; then
        echo -e "      ${GREEN}GROQ_API_KEY: Set${NC}"
    else
        echo -e "      ${YELLOW}GROQ_API_KEY: Not set (required for LLM)${NC}"
    fi

    if grep -q "ELEVENLABS_API_KEY=.*[^[:space:]]" .env 2>/dev/null; then
        echo -e "      ${GREEN}ELEVENLABS_API_KEY: Set${NC}"
    else
        echo -e "      ${YELLOW}ELEVENLABS_API_KEY: Not set (optional, using Piper instead)${NC}"
    fi
else
    echo -e "      ${YELLOW}.env file: Not found${NC}"
    echo -e "      ${YELLOW}Copy .env.example to .env and add your API keys${NC}"
fi

# -----------------------------------------------------------------------------
# Check Avatar Assets
# -----------------------------------------------------------------------------
echo -e "${BLUE}[8/8] Avatar Assets${NC}"

if [ -f "assets/avatar_face.png" ]; then
    echo -e "      ${GREEN}assets/avatar_face.png: Found${NC}"
elif [ -f "assets/avatar.mp4" ]; then
    echo -e "      ${GREEN}assets/avatar.mp4: Found${NC}"
else
    echo -e "      ${YELLOW}No avatar image found${NC}"
    echo -e "      ${YELLOW}Add assets/avatar_face.png (512x512 recommended)${NC}"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}======================================${NC}"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}   All checks passed!${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    echo "Ready to run:"
    echo -e "  ${GREEN}./start.sh${NC}          - Start web server"
    echo -e "  ${GREEN}./start_headless.sh${NC} - Run without display"
    echo -e "  ${GREEN}make run${NC}            - Start web server"
else
    echo -e "${RED}   $ERRORS check(s) failed${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    echo "Fix the issues above and run this script again."
fi
echo ""
