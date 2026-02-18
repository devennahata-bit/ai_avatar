#!/bin/bash
# =============================================================================
# TTS Installation Script for AWS Linux
# =============================================================================
# Installs Piper TTS and all dependencies on Amazon Linux / RHEL / CentOS
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  TTS Installation for AWS Linux${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Detect package manager
if command -v dnf &> /dev/null; then
    PKG_MGR="dnf"
elif command -v yum &> /dev/null; then
    PKG_MGR="yum"
elif command -v apt-get &> /dev/null; then
    PKG_MGR="apt-get"
else
    echo -e "${RED}No supported package manager found${NC}"
    exit 1
fi

echo -e "${YELLOW}Using package manager: $PKG_MGR${NC}"

# -----------------------------------------------------------------------------
# Install system dependencies
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}[1/5] Installing system dependencies...${NC}"

if [ "$PKG_MGR" = "apt-get" ]; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y \
        espeak-ng \
        libespeak-ng1 \
        libespeak-ng-dev \
        portaudio19-dev \
        libportaudio2 \
        ffmpeg \
        gcc \
        g++ \
        python3-dev
else
    # Amazon Linux / RHEL / CentOS
    sudo $PKG_MGR install -y \
        gcc \
        gcc-c++ \
        python3-devel \
        portaudio-devel \
        ffmpeg \
        || true

    # espeak-ng is not in default Amazon Linux repos
    # We need to install it from EPEL or build from source
    echo -e "${YELLOW}Installing espeak-ng...${NC}"

    # Try EPEL first
    sudo $PKG_MGR install -y epel-release 2>/dev/null || true
    sudo $PKG_MGR install -y espeak-ng espeak-ng-devel 2>/dev/null || {
        echo -e "${YELLOW}espeak-ng not in repos, building from source...${NC}"

        # Install build dependencies
        sudo $PKG_MGR install -y \
            autoconf \
            automake \
            libtool \
            make \
            gcc \
            gcc-c++ \
            pcaudiolib-devel \
            sonic-devel \
            || sudo $PKG_MGR install -y autoconf automake libtool make gcc gcc-c++

        # Clone and build espeak-ng
        TEMP_DIR=$(mktemp -d)
        cd "$TEMP_DIR"

        git clone --depth 1 https://github.com/espeak-ng/espeak-ng.git
        cd espeak-ng

        ./autogen.sh
        ./configure --prefix=/usr/local
        make -j$(nproc)
        sudo make install
        sudo ldconfig

        cd ~
        rm -rf "$TEMP_DIR"

        echo -e "${GREEN}espeak-ng built and installed${NC}"
    }
fi

# Verify espeak-ng
if command -v espeak-ng &> /dev/null; then
    echo -e "${GREEN}espeak-ng installed: $(espeak-ng --version 2>&1 | head -1)${NC}"
elif command -v espeak &> /dev/null; then
    echo -e "${GREEN}espeak installed: $(espeak --version 2>&1 | head -1)${NC}"
else
    echo -e "${YELLOW}Warning: espeak-ng not found in PATH, but libraries may still work${NC}"
fi

# -----------------------------------------------------------------------------
# Install Python TTS packages
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}[2/5] Installing Python TTS packages...${NC}"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install onnxruntime first (required by piper)
echo "Installing onnxruntime..."
pip install onnxruntime || pip install onnxruntime-gpu

# Install piper-phonemize (the tricky dependency)
echo "Installing piper-phonemize..."
pip install piper-phonemize || {
    echo -e "${YELLOW}piper-phonemize failed, trying alternative...${NC}"

    # Try building with espeak-ng path
    export ESPEAK_DATA_PATH=/usr/local/share/espeak-ng-data
    pip install piper-phonemize --no-binary :all: || true
}

# Install piper-tts
echo "Installing piper-tts..."
pip install piper-tts || {
    echo -e "${YELLOW}piper-tts pip install failed, trying from GitHub...${NC}"
    pip install git+https://github.com/rhasspy/piper.git || true
}

# -----------------------------------------------------------------------------
# Install fallback TTS options
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}[3/5] Installing fallback TTS options...${NC}"

# ElevenLabs (cloud TTS)
pip install elevenlabs

# pyttsx3 (local fallback)
pip install pyttsx3

# gTTS (Google TTS - another fallback)
pip install gTTS

# -----------------------------------------------------------------------------
# Download Piper voice model
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}[4/5] Downloading Piper voice model...${NC}"

python3 << 'EOF'
import os
import sys

try:
    from piper.download import ensure_voice_exists, get_voices

    # Download the default voice
    voice_name = "en_US-lessac-medium"
    print(f"Downloading Piper voice: {voice_name}")

    try:
        model_path, config_path = ensure_voice_exists(voice_name, [], None, None)
        print(f"Voice downloaded to: {model_path}")
    except Exception as e:
        print(f"Could not download voice automatically: {e}")
        print("Voice will be downloaded on first use")

except ImportError as e:
    print(f"Piper not available: {e}")
    print("Will use fallback TTS")
except Exception as e:
    print(f"Error: {e}")
EOF

# -----------------------------------------------------------------------------
# Test TTS installation
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}[5/5] Testing TTS installation...${NC}"

python3 << 'EOF'
import sys

print("\n=== TTS Installation Test ===\n")

# Test Piper
print("1. Piper TTS:", end=" ")
try:
    from piper.voice import PiperVoice
    print("\033[92mOK\033[0m")
except Exception as e:
    print(f"\033[91mFAILED - {e}\033[0m")

# Test piper-phonemize
print("2. Piper Phonemize:", end=" ")
try:
    import piper_phonemize
    print("\033[92mOK\033[0m")
except Exception as e:
    print(f"\033[91mFAILED - {e}\033[0m")

# Test ElevenLabs
print("3. ElevenLabs:", end=" ")
try:
    import elevenlabs
    print("\033[92mOK\033[0m")
except Exception as e:
    print(f"\033[91mFAILED - {e}\033[0m")

# Test pyttsx3
print("4. pyttsx3:", end=" ")
try:
    import pyttsx3
    engine = pyttsx3.init()
    print("\033[92mOK\033[0m")
except Exception as e:
    print(f"\033[91mFAILED - {e}\033[0m")

# Test gTTS
print("5. gTTS:", end=" ")
try:
    from gtts import gTTS
    print("\033[92mOK\033[0m")
except Exception as e:
    print(f"\033[91mFAILED - {e}\033[0m")

print("\n=== End of Test ===\n")
EOF

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  TTS Installation Complete${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "If Piper failed, the system will fall back to other TTS options."
echo "You can also use ElevenLabs by setting ELEVENLABS_API_KEY in .env"
echo ""
