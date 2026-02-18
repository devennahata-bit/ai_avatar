#!/usr/bin/env python3
"""
Component Test Script
=====================
Tests all 4 components of the AI Avatar system.
Run this before starting the server to diagnose issues.

Usage:
    python test_components.py
"""

import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Colors for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def test_section(name):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  {name}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def success(msg):
    print(f"  {GREEN}✓ {msg}{RESET}")

def warning(msg):
    print(f"  {YELLOW}⚠ {msg}{RESET}")

def error(msg):
    print(f"  {RED}✗ {msg}{RESET}")

def info(msg):
    print(f"  {msg}")

def main():
    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}  AI Avatar - Component Diagnostics{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")

    errors = []
    warnings = []

    # =========================================================================
    # 1. ENVIRONMENT & CONFIG
    # =========================================================================
    test_section("1. Environment & Configuration")

    # Check .env file
    env_path = Path(".env")
    if env_path.exists():
        success(".env file exists")

        # Check for GROQ_API_KEY
        env_content = env_path.read_text()
        if "GROQ_API_KEY=" in env_content and "your_" not in env_content.split("GROQ_API_KEY=")[1].split("\n")[0]:
            success("GROQ_API_KEY is set")
        else:
            error("GROQ_API_KEY not set or has placeholder value")
            errors.append("Set GROQ_API_KEY in .env")
    else:
        error(".env file not found")
        errors.append("Create .env file from .env.example")

    # Check config loading
    try:
        from config import api_config, voice_settings, llm_config, avatar_personality
        success("Configuration loaded successfully")
        info(f"    LLM Provider: {llm_config.provider}")
        info(f"    TTS Provider: {voice_settings.tts_provider}")
    except Exception as e:
        error(f"Config loading failed: {e}")
        errors.append(f"Fix config.py: {e}")

    # =========================================================================
    # 2. LLM (Brain)
    # =========================================================================
    test_section("2. LLM (Brain) - Groq/OpenAI/Ollama")

    # Check groq package
    try:
        import groq
        success("groq package installed")
    except ImportError:
        error("groq package not installed")
        errors.append("pip install groq")

    # Test Groq connection
    try:
        from config import api_config, llm_config
        if api_config.groq_api_key:
            from modules.brain import LLMBrain
            brain = LLMBrain(
                provider="groq",
                model=llm_config.groq_model,
                groq_api_key=api_config.groq_api_key,
            )
            response = brain.generate("Say 'test successful' in exactly 2 words")
            if response:
                success(f"Groq LLM working! Response: {response[:50]}...")
            else:
                error("Groq LLM returned empty response")
                errors.append("Check GROQ_API_KEY")
        else:
            warning("GROQ_API_KEY not set, skipping LLM test")
            warnings.append("Set GROQ_API_KEY for LLM functionality")
    except Exception as e:
        error(f"LLM test failed: {e}")
        errors.append(f"Fix LLM: {e}")

    # =========================================================================
    # 3. TTS (Voice)
    # =========================================================================
    test_section("3. TTS (Text-to-Speech)")

    tts_working = False

    # Test Piper
    info("Testing Piper TTS...")
    try:
        from piper.voice import PiperVoice
        success("piper-tts package installed")

        from modules.voice import PiperTTSSynthesizer
        import queue
        piper = PiperTTSSynthesizer(model="en_US-lessac-medium", audio_queue=queue.Queue())
        audio = piper.synthesize("Test")
        if audio and len(audio) > 0:
            success(f"Piper TTS working! Generated {len(audio)} bytes")
            tts_working = True
        else:
            warning("Piper TTS returned empty audio")
    except ImportError as e:
        warning(f"Piper not available: {e}")
    except Exception as e:
        warning(f"Piper TTS failed: {e}")

    # Test gTTS
    if not tts_working:
        info("Testing gTTS (Google TTS)...")
        try:
            from gtts import gTTS
            success("gTTS package installed")

            # Quick test
            tts = gTTS(text="test", lang="en")
            success("gTTS working!")
            tts_working = True
        except ImportError:
            warning("gTTS not installed (pip install gTTS)")
        except Exception as e:
            warning(f"gTTS failed: {e}")

    # Test ElevenLabs
    if not tts_working:
        info("Testing ElevenLabs...")
        try:
            import elevenlabs
            success("elevenlabs package installed")
            if api_config.elevenlabs_api_key:
                success("ElevenLabs API key set")
                tts_working = True
            else:
                warning("ElevenLabs API key not set")
        except ImportError:
            warning("elevenlabs not installed")

    # Test Local TTS (pyttsx3)
    if not tts_working:
        info("Testing Local TTS (pyttsx3)...")
        try:
            import pyttsx3
            engine = pyttsx3.init()
            success("pyttsx3 working!")
            tts_working = True
        except Exception as e:
            warning(f"Local TTS failed: {e}")

    if tts_working:
        success("At least one TTS provider is working")
    else:
        error("NO TTS provider is working!")
        errors.append("Install at least one TTS: pip install gTTS (easiest)")

    # =========================================================================
    # 4. STT (Whisper)
    # =========================================================================
    test_section("4. STT (Speech-to-Text) - Whisper")

    try:
        from faster_whisper import WhisperModel
        success("faster-whisper installed")
    except ImportError:
        warning("faster-whisper not installed, trying openai-whisper")
        try:
            import whisper
            success("openai-whisper installed")
        except ImportError:
            error("No whisper package installed")
            errors.append("pip install faster-whisper or pip install openai-whisper")

    # =========================================================================
    # 5. Face Renderer (LiveAvatar)
    # =========================================================================
    test_section("5. Face Renderer (LiveAvatar)")

    # Check avatar image
    from config import avatar_personality, MODELS_DIR
    img_path = avatar_personality.reference_image_path
    vid_path = avatar_personality.reference_media_path

    if vid_path.exists():
        success(f"Reference video found: {vid_path}")
    elif img_path.exists():
        success(f"Reference image found: {img_path}")
    else:
        error(f"No avatar media found!")
        error(f"  Expected: {img_path}")
        error(f"  Or: {vid_path}")
        errors.append("Add assets/avatar_face.png (512x512 face image)")

    # Check models
    base_model = MODELS_DIR / "Wan2.1-S2V-14B"
    lora_model = MODELS_DIR / "Live-Avatar"

    if base_model.exists():
        success(f"LiveAvatar base model found: {base_model}")
    else:
        warning(f"LiveAvatar base model not found (will use fallback mode)")
        warnings.append("Download LiveAvatar models for lip-sync")

    if lora_model.exists():
        success(f"LiveAvatar LoRA weights found: {lora_model}")
    else:
        warning(f"LiveAvatar LoRA weights not found (will use fallback mode)")

    # Check if face module loads
    try:
        from modules.face import FaceRenderer
        success("Face module imports successfully")
    except Exception as e:
        error(f"Face module import failed: {e}")
        errors.append(f"Fix face module: {e}")

    # =========================================================================
    # 6. Web Server Dependencies
    # =========================================================================
    test_section("6. Web Server Dependencies")

    try:
        from flask import Flask
        success("Flask installed")
    except ImportError:
        error("Flask not installed")
        errors.append("pip install flask")

    try:
        from flask_socketio import SocketIO
        success("Flask-SocketIO installed")
    except ImportError:
        error("Flask-SocketIO not installed")
        errors.append("pip install flask-socketio")

    try:
        import cv2
        success(f"OpenCV installed: {cv2.__version__}")
    except ImportError:
        error("OpenCV not installed")
        errors.append("pip install opencv-python")

    # =========================================================================
    # 7. GPU / CUDA
    # =========================================================================
    test_section("7. GPU / CUDA")

    try:
        import torch
        success(f"PyTorch installed: {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            success(f"CUDA available: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            warning("CUDA not available - will run on CPU (slow)")
            warnings.append("Install CUDA-enabled PyTorch for better performance")
    except ImportError:
        error("PyTorch not installed")
        errors.append("pip install torch")

    # Check FFmpeg (needed for gTTS audio conversion)
    import shutil
    if shutil.which("ffmpeg"):
        success("FFmpeg found in PATH")
    else:
        warning("FFmpeg not found - needed for some TTS providers")
        warnings.append("Install ffmpeg: sudo yum install ffmpeg")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    if errors:
        print(f"\n{RED}ERRORS ({len(errors)}):{RESET}")
        for e in errors:
            print(f"  {RED}• {e}{RESET}")

    if warnings:
        print(f"\n{YELLOW}WARNINGS ({len(warnings)}):{RESET}")
        for w in warnings:
            print(f"  {YELLOW}• {w}{RESET}")

    if not errors:
        print(f"\n{GREEN}All critical components are working!{RESET}")
        print(f"\n{GREEN}You can start the server with:{RESET}")
        print(f"  python web_server.py --host 0.0.0.0 --port 8080")
    else:
        print(f"\n{RED}Fix the errors above before running the server.{RESET}")

    print()
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
