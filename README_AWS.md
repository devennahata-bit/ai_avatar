# AI Avatar - AWS Linux Deployment Guide

Quick setup guide for deploying AI Avatar on AWS EC2 GPU instances.

## Recommended Instance Types

| Instance | GPU | VRAM | Cost | Recommendation |
|----------|-----|------|------|----------------|
| g4dn.xlarge | T4 | 16GB | $ | Development/testing only |
| g5.xlarge | A10G | 24GB | $$ | Limited functionality |
| g5.2xlarge | A10G | 24GB | $$ | Good for TTS + LLM |
| g5.4xlarge | A10G | 24GB | $$$ | Full avatar (FP8) |
| p4d.24xlarge | A100 | 80GB | $$$$ | Full quality |

**Note:** LiveAvatar lip-sync requires 48GB+ VRAM for optimal performance.

## Quick Start (3 Commands)

```bash
# 1. Clone and enter directory
git clone <your-repo> ai_avatar && cd ai_avatar

# 2. Run setup (installs everything)
chmod +x aws_setup.sh && ./aws_setup.sh

# 3. Start the avatar
./start.sh
```

Then open `http://<your-ec2-ip>:8080` in your browser.

Note: browser microphone capture requires HTTPS (or localhost). On plain HTTP, text chat still works but mic input is blocked by the browser.

## Detailed Setup

### 1. Launch EC2 Instance

- **AMI:** Ubuntu 22.04 LTS or Amazon Linux 2023
- **Instance type:** g5.xlarge or larger (GPU required)
- **Storage:** 100GB+ (models are ~30GB)
- **Security Group:** Open port 8080 (and 22 for SSH)

### 2. Connect and Setup

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone repository
git clone <your-repo> ai_avatar
cd ai_avatar

# Run full setup
chmod +x aws_setup.sh
./aws_setup.sh
```

### 3. Configure API Keys

Edit `.env` with your API keys:

```bash
nano .env
```

Required keys:
- `GROQ_API_KEY` - Get from https://console.groq.com/
- `ELEVENLABS_API_KEY` - Get from https://elevenlabs.io/

### 4. Add Avatar Image

```bash
# Copy your avatar image
cp /path/to/your/face.png assets/avatar_face.png
```

### 5. Run the Avatar

```bash
# Web server (recommended for cloud)
./start.sh

# Or headless mode
./start_headless.sh

# Or using make
make run
```

### 6. Enable HTTPS for Browser Microphone (Recommended)

Browser microphone capture only works on secure origins (`https://...`) or localhost.

Prerequisites:
- A DNS record (for example `avatar.yourdomain.com`) pointing to your EC2 public IP
- Security group inbound rules for TCP `80` and `443`

```bash
# Start the app on localhost:8080 first
./start.sh

# In a second shell, configure Nginx + Let's Encrypt
chmod +x setup_https_nginx.sh
./setup_https_nginx.sh --domain avatar.yourdomain.com --email you@yourdomain.com
```

Then open:
```text
https://avatar.yourdomain.com
```

## Available Commands

### Using Shell Scripts
```bash
./start.sh           # Start web server on port 8080
./start_headless.sh  # Run without display
./test_setup.sh      # Verify installation
```

### Using Make
```bash
make setup           # Full setup
make install         # Install dependencies only
make models          # Download models only
make run             # Start web server
make headless        # Run headless
make test            # Run tests
make check           # Check installation
make clean           # Clean temp files
```

### Using Python Directly
```bash
source .venv/bin/activate
python web_server.py --host 0.0.0.0 --port 8080
python main.py --no-display
python main.py --config
```

## Configuration

### TTS Provider (Text-to-Speech)

On AWS Linux, **Piper** is the default - it's FREE and high-quality neural TTS:

```env
# FREE option (default) - high-quality neural TTS
VOICE_TTS_PROVIDER=piper
VOICE_PIPER_MODEL=en_US-lessac-medium

# OR paid option (cloud) - requires API key
VOICE_TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_key_here
```

Options:
- `piper` - FREE high-quality neural TTS (DEFAULT, recommended)
- `elevenlabs` - High quality cloud TTS (PAID, requires API key)
- `kokoro` - Fast local neural TTS (Windows/macOS)
- `local` - pyttsx3 + espeak fallback (low quality)

Available Piper voices:
- `en_US-lessac-medium` (default, female)
- `en_US-ryan-medium` (male)
- `en_GB-alan-medium` (British male)
- `en_US-libritts-high` (high quality)

### LLM Provider

```env
LLM_PROVIDER=groq           # Recommended (fast, free tier)
GROQ_API_KEY=your_key_here
```

Other options:
- `openai` - OpenAI GPT models
- `ollama` - Local LLMs (requires Ollama installation)

## Troubleshooting

### No GPU detected
```bash
# Check NVIDIA driver
nvidia-smi

# If not working, install drivers
sudo apt install nvidia-driver-535
sudo reboot
```

### Audio issues
```bash
# Install audio libraries
sudo apt install portaudio19-dev ffmpeg espeak

# Verify
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### TTS not working
```bash
# Test TTS
source .venv/bin/activate
python -c "from modules.voice import create_tts_synthesizer; print('OK')"

# If Kokoro fails, use ElevenLabs
# Edit .env: VOICE_TTS_PROVIDER=elevenlabs
```

### Web server not accessible
```bash
# Check if running
curl http://localhost:8080

# Check security group allows port 8080
# AWS Console > EC2 > Security Groups > Edit inbound rules
```

### Browser microphone not working
Browser microphone APIs require a secure context.

- Works: `https://your-domain` (recommended), `http://localhost`
- Blocked by browser: `http://<public-ec2-ip>:8080`

If you need voice input from remote clients, put the app behind HTTPS (for example, Nginx + TLS).
Use the included script:

```bash
./setup_https_nginx.sh --domain your-domain --email you@example.com
```

### Out of memory
```bash
# Use FP8 quantization (in .env)
LIVEAVATAR_USE_FP8=true

# Or reduce model size
WHISPER_MODEL_SIZE=tiny
```

## Performance Tips

1. **Use spot instances** for development (70-90% cheaper)
2. **Pre-download models** using `./aws_setup.sh --models-only`
3. **Use ElevenLabs** for TTS (more reliable than local on Linux)
4. **Enable FP8** for LiveAvatar to reduce VRAM usage
5. **Use Groq** for LLM (faster than local models)

## Architecture

```
Browser ──> Web Server (Flask) ──> AI Pipeline
                                   ├── Whisper (STT)
                                   ├── Groq/LLM (Brain)
                                   ├── ElevenLabs (TTS)
                                   └── LiveAvatar (Lip-sync)
```

## Security Notes

- Keep API keys in `.env` (never commit to git)
- Use security groups to restrict access
- Consider using HTTPS with a reverse proxy (nginx)
- Don't expose port 8080 to 0.0.0.0/0 in production

## Support

- Check logs: `tail -f logs/*.log`
- Run diagnostics: `./test_setup.sh`
- Verify config: `make check`
