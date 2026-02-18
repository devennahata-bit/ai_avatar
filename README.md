# Real-Time AI Avatar

A production-ready real-time AI avatar application that combines speech recognition, large language models, text-to-speech, and lip-sync technology.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Microphone │────▶│  Whisper    │────▶│  Llama 3    │────▶│ ElevenLabs  │
│   (Input)   │     │   (STT)     │     │   (LLM)     │     │   (TTS)     │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                    │
                    ┌─────────────┐     ┌─────────────┐             │
                    │   Display   │◀────│  MuseTalk   │◀────────────┘
                    │  (Output)   │     │  (Lip-Sync) │
                    └─────────────┘     └─────────────┘
```

## Features

- **Speech-to-Text**: Whisper with Voice Activity Detection
- **LLM Integration**: Supports Ollama (local), Groq (fast API), and OpenAI
- **Text-to-Speech**: ElevenLabs with configurable voice settings
- **Lip-Sync**: MuseTalk for realistic mouth movements
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
- Install OpenMMLab dependencies (mmcv, mmdet, mmpose)
- Clone the MuseTalk repository
- Download all required model weights

```bash
python setup_wizard.py
```

### 3. Configure API Keys

```bash
# Copy example config
cp .env.example .env

# Edit .env and add your API keys:
# - ELEVENLABS_API_KEY (required for TTS)
# - GROQ_API_KEY (recommended for fast LLM)
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

### Voice Settings

Edit `.env` or `config.py`:

```python
# Voice characteristics (0.0 - 1.0)
VOICE_STABILITY=0.5          # Lower = more expressive
VOICE_SIMILARITY_BOOST=0.75  # Higher = closer to original voice
VOICE_STYLE=0.0              # Style exaggeration
```

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
│   ├── voice.py         # ElevenLabs TTS
│   └── face.py          # MuseTalk lip-sync
├── models/              # Downloaded model weights
│   ├── musetalk/
│   ├── dwpose/
│   ├── sd-vae-ft-mse/
│   └── whisper/
├── vendor/              # Cloned repositories
│   └── MuseTalk/
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

- Reduce `MUSETALK_FPS` (e.g., 15 instead of 25)
- Use smaller Whisper model (`tiny` or `base`)
- Ensure GPU is being used (check with `nvidia-smi`)

## API Costs

| Service | Approximate Cost |
|---------|-----------------|
| Groq | Free tier available |
| ElevenLabs | ~$0.30 per 1000 characters |
| OpenAI Whisper API | ~$0.006 per minute |
| Local Whisper | Free (GPU required) |

## License

MIT License - see LICENSE file for details.

## Credits

- [MuseTalk](https://github.com/TMElyralab/MuseTalk) - Lip-sync technology
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [ElevenLabs](https://elevenlabs.io/) - Voice synthesis
- [Groq](https://groq.com/) - Fast LLM inference

##

Script to run in Command Prompt:
cd c:\Users\Deven\ai_avatar
venv\Scripts\activate
python main.py

