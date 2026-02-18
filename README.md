# Real-Time AI Avatar

A production-ready real-time AI avatar application that combines speech recognition, large language models, text-to-speech, and lip-sync technology.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Microphone │────▶│  Whisper    │────▶│  Groq LLM   │────▶│  Piper TTS  │
│   (Input)   │     │   (STT)     │     │  (Llama 3)  │     │   (FREE)    │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                    │
                    ┌─────────────┐     ┌─────────────┐             │
                    │   Display   │◀────│ LiveAvatar  │◀────────────┘
                    │  (Output)   │     │  (Lip-Sync) │
                    └─────────────┘     └─────────────┘
```

## Features

- **Speech-to-Text**: Whisper with Voice Activity Detection
- **LLM Integration**: Groq (recommended), Ollama (local), or OpenAI
- **Text-to-Speech**: Piper TTS (FREE, default) or ElevenLabs (paid)
- **Lip-Sync**: LiveAvatar for realistic mouth movements
- **Producer-Consumer Architecture**: Low-latency threaded pipeline
- **Customizable Personality**: Define your avatar's character via system prompt

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with CUDA 11.8+ (6GB+ VRAM recommended)
- **FFmpeg** installed and in PATH
- **Microphone** for voice input

## Quick Start

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Run Setup Wizard

The setup wizard will:
- Verify CUDA/GPU availability
- Install dependencies
- Clone the LiveAvatar repository
- Download all required model weights

```bash
python setup_wizard.py
```

### 3. Configure API Keys

```bash
# Copy example config
cp .env.example .env

# Edit .env and add your API keys:
# - GROQ_API_KEY (required for LLM)
# - ELEVENLABS_API_KEY (optional - only if using ElevenLabs TTS instead of free Piper)
```

### 4. Add Avatar Face

Place your reference face image at:
```
assets/avatar_face.png
```

Requirements:
- Clear frontal face shot
- 512x512 recommended
- Good lighting, neutral expression

### 5. Run the Avatar

```bash
python main.py
```

### AWS Linux / EC2 Quick Start

```bash
# Recommended project-local venv
python3 -m venv .venv
source .venv/bin/activate

# EC2-friendly dependency install (headless-safe)
python install.py --deps --server

# Download models
python install.py --models

# Run headless on EC2
python main.py --no-display
```

## Usage

### Command Line Options

```bash
# Default run
python main.py

# Debug mode
python main.py --debug

# Use local Ollama
python main.py --provider ollama --model llama3:8b

# Use Groq (fastest)
python main.py --provider groq

# Headless mode (no display window)
python main.py --no-display

# Show configuration
python main.py --config

# Run setup wizard
python main.py --setup
```

### Controls

- **Speak** into your microphone to chat with the avatar
- Press **'q'** or **ESC** in the avatar window to quit
- Press **Ctrl+C** in terminal to stop

## Configuration

### Voice Settings (TTS)

Edit `.env`:

```bash
# TTS Provider - "piper" (FREE, default), "elevenlabs" (paid), "kokoro", "local"
VOICE_TTS_PROVIDER=piper

# Piper settings (FREE neural TTS - recommended)
VOICE_PIPER_MODEL=en_US-lessac-medium

# ElevenLabs settings (paid - only if using elevenlabs provider)
VOICE_STABILITY=0.5
VOICE_SIMILARITY_BOOST=0.75
```

Available Piper voices:
- `en_US-lessac-medium` (default, female)
- `en_US-ryan-medium` (male)
- `en_GB-alan-medium` (British male)
- `en_US-libritts-high` (high quality)

### Avatar Personality

Edit the system prompt in `config.py`:

```python
system_prompt = """You are Aria, a friendly AI assistant...
- Keep responses brief (2-3 sentences)
- Sound natural and conversational
- Express genuine interest
"""
```

### LLM Provider

```bash
# Groq (recommended - fastest)
LLM_PROVIDER=groq
LLM_GROQ_MODEL=llama-3.3-70b-versatile

# Local Ollama
LLM_PROVIDER=ollama
LLM_OLLAMA_MODEL=llama3:8b

# OpenAI
LLM_PROVIDER=openai
LLM_OPENAI_MODEL=gpt-4o
```

## Project Structure

```
ai_avatar/
├── main.py              # Main orchestrator
├── config.py            # Configuration management
├── setup_wizard.py      # Automated setup script
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── .env                 # Your configuration (create this)
├── modules/
│   ├── __init__.py
│   ├── listener.py      # Whisper STT
│   ├── brain.py         # LLM integration
│   ├── voice.py         # Piper/ElevenLabs TTS
│   └── face.py          # LiveAvatar lip-sync
├── models/              # Downloaded model weights
│   ├── liveavatar/
│   ├── dwpose/
│   ├── sd-vae-ft-mse/
│   └── whisper/
├── vendor/              # Cloned repositories
│   └── LiveAvatar/
└── assets/
    └── avatar_face.png  # Reference face image
```

## Troubleshooting

### CUDA Not Available

1. Install NVIDIA drivers (525+)
2. Install CUDA Toolkit 11.8+
3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Model Download Issues

Re-run the setup wizard with weights-only:
```bash
python setup_wizard.py --weights-only
```

### Audio Issues

- Ensure microphone is connected and working
- Check audio device permissions
- Try adjusting VAD aggressiveness in config

### Performance Issues

- Reduce `LIVEAVATAR_FPS` (e.g., 15 instead of 25)
- Use smaller Whisper model (`tiny` or `base`)
- Ensure GPU is being used (check with `nvidia-smi`)
- Enable FP8 quantization: `LIVEAVATAR_USE_FP8=true`

## API Costs

| Service | Approximate Cost |
|---------|-----------------|
| Piper TTS | **FREE** (default, recommended) |
| Groq LLM | Free tier available |
| Local Whisper | Free (GPU required) |
| ElevenLabs TTS | ~$0.30 per 1000 characters (paid option) |
| OpenAI Whisper API | ~$0.006 per minute (optional) |

## License

MIT License - see LICENSE file for details.

## Credits

- [LiveAvatar](https://github.com/your-org/LiveAvatar) - Lip-sync technology
- [Piper TTS](https://github.com/rhasspy/piper) - Free neural voice synthesis
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Groq](https://groq.com/) - Fast LLM inference
- [ElevenLabs](https://elevenlabs.io/) - Voice synthesis (paid option)

##

Script to run in Command Prompt:
cd c:\Users\Deven\ai_avatar
venv\Scripts\activate
python main.py

