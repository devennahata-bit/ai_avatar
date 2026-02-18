# =============================================================================
# AI Avatar - Makefile
# =============================================================================
# Simple commands to setup and run the AI Avatar
#
# Usage:
#   make setup      - Full AWS Linux setup
#   make install    - Install dependencies only
#   make models     - Download models only
#   make run        - Run web server
#   make headless   - Run in headless mode
#   make test       - Run tests
#   make clean      - Clean up
# =============================================================================

.PHONY: setup install models run headless test clean help

# Default Python
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    OS := linux
endif
ifeq ($(UNAME_S),Darwin)
    OS := macos
endif
ifeq ($(OS),Windows_NT)
    OS := windows
    VENV := .venv
    PIP := $(VENV)/Scripts/pip
    PYTHON_VENV := $(VENV)/Scripts/python
endif

# =============================================================================
# MAIN TARGETS
# =============================================================================

help:
	@echo "AI Avatar - Available commands:"
	@echo ""
	@echo "  make setup      - Full setup (deps + models) for AWS Linux"
	@echo "  make install    - Install dependencies only"
	@echo "  make models     - Download model weights (~30GB)"
	@echo "  make run        - Start web server on port 8080"
	@echo "  make headless   - Run avatar without display"
	@echo "  make test       - Run smoke tests"
	@echo "  make clean      - Clean temporary files"
	@echo ""
	@echo "AWS Linux quickstart:"
	@echo "  make setup && make run"
	@echo ""

# Full setup for AWS Linux
setup: check-gpu install-system venv install-deps models
	@echo ""
	@echo "Setup complete! Run 'make run' to start the web server."

# Quick setup (deps only, no models)
quick-setup: install-system venv install-deps
	@echo ""
	@echo "Dependencies installed! Run 'make models' to download models."

# =============================================================================
# INSTALLATION
# =============================================================================

check-gpu:
	@echo "Checking GPU..."
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || \
		echo "WARNING: No NVIDIA GPU detected or nvidia-smi not available"

install-system:
ifeq ($(OS),linux)
	@echo "Installing system dependencies..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update -qq && \
		sudo apt-get install -y -qq python3 python3-pip python3-venv python3-dev \
			portaudio19-dev libportaudio2 ffmpeg git libgl1-mesa-glx espeak; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y python3 python3-pip python3-devel \
			portaudio-devel ffmpeg git mesa-libGL espeak || true; \
	fi
else
	@echo "Skipping system deps (not Linux)"
endif

venv:
	@echo "Creating virtual environment..."
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip setuptools wheel -q

install-deps: venv
	@echo "Installing Python dependencies..."
	@$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
	@$(PIP) install -r requirements.txt -q
	@echo "Dependencies installed!"

install: install-deps

models: venv
	@echo "Downloading model weights (~30GB)..."
	@$(PYTHON_VENV) setup_wizard.py --skip-cuda-check

# =============================================================================
# RUNNING
# =============================================================================

run: venv
	@echo "Starting AI Avatar web server..."
	@echo "Open http://localhost:8080 in your browser"
	@echo ""
	@$(PYTHON_VENV) web_server.py --host 0.0.0.0 --port 8080

run-debug: venv
	@$(PYTHON_VENV) web_server.py --host 0.0.0.0 --port 8080 --debug

headless: venv
	@echo "Starting AI Avatar in headless mode..."
	@$(PYTHON_VENV) main.py --no-display

headless-debug: venv
	@$(PYTHON_VENV) main.py --no-display --debug

# Run with local display (if available)
gui: venv
	@$(PYTHON_VENV) main.py

# =============================================================================
# TESTING & DIAGNOSTICS
# =============================================================================

test: venv
	@echo "Running smoke tests..."
	@$(PYTHON_VENV) test_smoke.py

test-audio: venv
	@echo "Testing audio devices..."
	@$(PYTHON_VENV) test_audio.py

test-all: venv
	@echo "Running all tests..."
	@$(PYTHON_VENV) -m pytest -v

check: venv
	@echo "Checking installation..."
	@echo ""
	@echo "Python version:"
	@$(PYTHON_VENV) --version
	@echo ""
	@echo "PyTorch & CUDA:"
	@$(PYTHON_VENV) -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
	@echo ""
	@echo "Core modules:"
	@$(PYTHON_VENV) -c "from config import api_config; print('Config: OK')"
	@$(PYTHON_VENV) -c "from modules.brain import LLMBrain; print('LLM Brain: OK')"
	@$(PYTHON_VENV) -c "from modules.voice import VoiceSynthesizer; print('Voice: OK')"
	@echo ""
	@echo "FFmpeg:"
	@ffmpeg -version 2>/dev/null | head -1 || echo "FFmpeg not found"

config: venv
	@$(PYTHON_VENV) main.py --config

# =============================================================================
# UTILITIES
# =============================================================================

shell: venv
	@echo "Activating virtual environment..."
	@bash --init-file <(echo "source $(VENV)/bin/activate")

clean:
	@echo "Cleaning temporary files..."
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf .pytest_cache */.pytest_cache
	@rm -rf *.pyc */*.pyc
	@rm -rf logs/*.log
	@rm -rf .mypy_cache

clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)

clean-models:
	@echo "WARNING: This will delete all downloaded models (~30GB)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf models/* || echo "Cancelled"

# List audio devices
list-audio: venv
	@$(PYTHON_VENV) -c "import sounddevice; print(sounddevice.query_devices())"

# Show environment info
info:
	@echo "OS: $(OS)"
	@echo "Python: $(PYTHON)"
	@echo "Venv: $(VENV)"
	@uname -a 2>/dev/null || echo ""
	@cat /etc/os-release 2>/dev/null | head -2 || echo ""

# =============================================================================
# DOCKER (optional)
# =============================================================================

docker-build:
	docker build -t ai-avatar .

docker-run:
	docker run -it --gpus all -p 8080:8080 ai-avatar

# =============================================================================
# AWS SPECIFIC
# =============================================================================

# Full AWS setup using the shell script
aws-setup:
	@chmod +x aws_setup.sh
	@./aws_setup.sh

aws-deps:
	@chmod +x aws_setup.sh
	@./aws_setup.sh --deps-only

aws-models:
	@chmod +x aws_setup.sh
	@./aws_setup.sh --models-only
