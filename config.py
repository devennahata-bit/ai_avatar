"""
Central Configuration for Real-Time AI Avatar Application
=========================================================
All API keys, model paths, and avatar personality settings.

Platform Support:
- Windows: Full GUI support with Realtek audio auto-detection
- Linux (AWS): Headless mode with ElevenLabs TTS recommended
- macOS: Full support
"""

import os
import sys
import platform
from pathlib import Path
from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# PLATFORM DETECTION
# =============================================================================

IS_LINUX = platform.system().lower() == "linux"
IS_WINDOWS = platform.system().lower() == "windows"
IS_MACOS = platform.system().lower() == "darwin"
IS_AWS = os.environ.get("AWS_EXECUTION_ENV") is not None or Path("/sys/hypervisor/uuid").exists()
IS_HEADLESS = os.environ.get("DISPLAY") is None and IS_LINUX


# =============================================================================
# BASE PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
VENDOR_DIR = PROJECT_ROOT / "vendor"
LIVEAVATAR_DIR = VENDOR_DIR / "LiveAvatar"


# =============================================================================
# API CONFIGURATION
# =============================================================================

class APIConfig(BaseSettings):
    """API Keys and Endpoints"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ElevenLabs
    elevenlabs_api_key: str = Field(default="", validation_alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", validation_alias="ELEVENLABS_VOICE_ID")  # Rachel

    # OpenAI (for Whisper API or GPT fallback)
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # Groq (for fast Llama 3 inference)
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")

    # Ollama (local LLM)
    ollama_host: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_HOST")


# =============================================================================
# VOICE SETTINGS (TTS)
# =============================================================================

def get_default_tts_provider() -> str:
    """Get the best default TTS provider based on platform."""
    if IS_LINUX or IS_AWS or IS_HEADLESS:
        # On Linux/AWS: Use Piper (free, high-quality neural TTS)
        return "piper"

    # On Windows/macOS, Kokoro works well locally
    return "kokoro"


class VoiceSettings(BaseSettings):
    """Text-to-Speech Settings"""

    model_config = SettingsConfigDict(env_prefix="VOICE_")

    # TTS Provider options:
    # - "piper" (recommended for Linux/AWS) - FREE, high-quality neural TTS
    # - "kokoro" (recommended for Windows/macOS) - Fast local neural TTS
    # - "elevenlabs" - High-quality cloud TTS (PAID, requires API key)
    # - "tortoise" - High-quality local TTS (slow)
    # - "local" - pyttsx3/espeak fallback (low quality)
    tts_provider: Literal["piper", "kokoro", "tortoise", "elevenlabs", "local"] = Field(default_factory=get_default_tts_provider)

    # Piper TTS settings (FREE neural TTS - recommended for Linux)
    piper_model: str = Field(default="en_US-lessac-medium")  # Voice model name
    piper_model_path: Optional[str] = Field(default=None)  # Custom model path (optional)

    # Voice characteristics (0.0 - 1.0) - ElevenLabs only
    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)
    style: float = Field(default=0.0, ge=0.0, le=1.0)
    use_speaker_boost: bool = Field(default=True)

    # Model selection - ElevenLabs only
    model_id: str = Field(default="eleven_multilingual_v2")

    # Local TTS settings (pyttsx3)
    local_rate: int = Field(default=175)  # Words per minute
    local_voice_index: int = Field(default=0)  # 0 = first voice

    # Tortoise-TTS settings
    tortoise_voice: str = Field(default="random")  # Voice name or "random" or path to voice samples
    tortoise_preset: Literal["ultra_fast", "fast", "standard", "high_quality"] = Field(default="fast")
    tortoise_use_deepspeed: bool = Field(default=False)  # Enable DeepSpeed for faster inference
    tortoise_kv_cache: bool = Field(default=True)  # Enable KV cache for efficiency
    tortoise_half: bool = Field(default=True)  # Use FP16 for reduced memory
    tortoise_samples_dir: Path = Field(default=MODELS_DIR / "tortoise_voices")  # Custom voice samples

    # Kokoro TTS settings (RealtimeTTS) - fast local TTS, 210x real-time on GPU
    kokoro_voice: str = Field(default="af_heart")  # Voice name (af_*, am_*, bf_*, bm_*)
    kokoro_speed: float = Field(default=1.0, ge=0.5, le=2.0)  # Speech speed multiplier

    # Output format
    output_format: str = Field(default="pcm_24000")  # 24kHz PCM for LiveAvatar

    # Streaming settings
    chunk_size: int = Field(default=1024)
    latency_optimization: int = Field(default=3, ge=0, le=4)  # 0=default, 4=max optimization


# =============================================================================
# LLM SETTINGS
# =============================================================================

class LLMConfig(BaseSettings):
    """Large Language Model Configuration"""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    # Provider: "ollama", "groq", "openai"
    provider: Literal["ollama", "groq", "openai"] = Field(default="groq")

    # Model names per provider
    ollama_model: str = Field(default="llama3:8b")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    openai_model: str = Field(default="gpt-4o")

    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    # Streaming
    stream: bool = Field(default=True)


# =============================================================================
# AVATAR PERSONALITY
# =============================================================================

class AvatarPersonality(BaseSettings):
    """Avatar Character Definition"""

    model_config = SettingsConfigDict(env_prefix="AVATAR_")

    name: str = Field(default="Aria")

    system_prompt: str = Field(default="""You are Aria, a friendly and helpful AI assistant with a warm, conversational personality.

PERSONA:
You are knowledgeable, patient, and genuinely interested in helping people. You speak naturally and conversationally, like a helpful friend who happens to know a lot about many topics. You're upbeat but not over-the-top, and you adapt your tone to match the conversation.

COMMUNICATION STYLE:
- Be conversational and natural - avoid sounding robotic or scripted
- Keep responses concise (1-3 sentences for simple questions, longer for complex topics)
- Use a warm, friendly tone while remaining professional
- Ask clarifying questions when needed
- Admit when you don't know something

CAPABILITIES:
- General knowledge and conversation
- Answering questions on a wide range of topics
- Providing explanations and tutorials
- Casual chat and companionship
- Task assistance and brainstorming

GUIDELINES:
- Be helpful, harmless, and honest
- Respect user privacy
- If asked to do something harmful or unethical, politely decline
- Keep responses natural for spoken conversation (this is a voice interface)""")

    # Reference media for face synthesis (can be PNG/JPG image or MP4 video)
    # Video provides more natural head movements and better lip-sync results
    reference_media_path: Path = Field(default=PROJECT_ROOT / "assets" / "avatar.mp4")

    # Fallback to image if video not found
    reference_image_path: Path = Field(default=PROJECT_ROOT / "assets" / "avatar_face.png")

    # Conversation memory
    max_history_turns: int = Field(default=10)


# =============================================================================
# WHISPER (STT) SETTINGS
# =============================================================================

class WhisperConfig(BaseSettings):
    """Speech-to-Text Configuration"""

    model_config = SettingsConfigDict(env_prefix="WHISPER_")

    # Model size: "tiny", "base", "small", "medium", "large", "large-v3"
    model_size: str = Field(default="base")

    # Use local model or OpenAI API
    use_api: bool = Field(default=False)

    # Audio settings
    sample_rate: int = Field(default=16000)
    channels: int = Field(default=1)

    # Voice Activity Detection
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)  # 0-3, higher = more aggressive

    # Silence thresholds (seconds)
    silence_threshold: float = Field(default=0.8)
    min_speech_duration: float = Field(default=0.5)

    # Device
    device: str = Field(default="cuda")
    compute_type: str = Field(default="float16")


# =============================================================================
# LIVEAVATAR SETTINGS
# =============================================================================

class LiveAvatarConfig(BaseSettings):
    """LiveAvatar Lip-Sync Configuration (Alibaba-Quark)"""

    model_config = SettingsConfigDict(env_prefix="LIVEAVATAR_")

    # Model paths (relative to MODELS_DIR)
    base_model_dir: Path = Field(default=MODELS_DIR / "Wan2.1-S2V-14B")
    lora_weights_dir: Path = Field(default=MODELS_DIR / "Live-Avatar")
    audio_encoder_dir: Path = Field(default=MODELS_DIR / "wav2vec2-base")

    # Inference settings
    fps: int = Field(default=24)
    sampling_steps: int = Field(default=10)  # Reduced for speed (default is 40)
    infer_frames: int = Field(default=81)  # Frames per inference batch
    guide_scale: float = Field(default=5.0)  # CFG scale

    # Output resolution
    output_width: int = Field(default=512)
    output_height: int = Field(default=512)

    # GPU settings
    device: str = Field(default="cuda")
    use_fp8: bool = Field(default=True)  # FP8 quantization for 48GB+ GPUs
    use_fp16: bool = Field(default=True)  # Fallback precision
    multi_gpu: bool = Field(default=False)  # Split model across multiple GPUs (for g5.12xlarge etc)

    # Audio buffering
    audio_sample_rate: int = Field(default=24000)  # Expected from TTS


# =============================================================================
# AUDIO DEVICE SETTINGS
# =============================================================================

class AudioConfig(BaseSettings):
    """Audio Device Configuration"""

    model_config = SettingsConfigDict(env_prefix="AUDIO_")

    # Device indices (None = auto-detect)
    input_device: Optional[int] = Field(default=None)
    output_device: Optional[int] = Field(default=None)


# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

def get_default_show_preview() -> bool:
    """Determine if preview should be shown based on platform."""
    if IS_HEADLESS or IS_AWS:
        return False
    return True


class AppConfig(BaseSettings):
    """Main Application Configuration"""

    model_config = SettingsConfigDict(env_prefix="APP_")

    # Debug mode
    debug: bool = Field(default=False)

    # Logging level
    log_level: str = Field(default="INFO")

    # Queue sizes
    audio_queue_size: int = Field(default=50)
    frame_queue_size: int = Field(default=30)

    # Display settings (auto-disabled on headless Linux/AWS)
    show_preview: bool = Field(default_factory=get_default_show_preview)
    window_title: str = Field(default="AI Avatar")

    # Performance
    max_concurrent_requests: int = Field(default=3)

    # Audio buffer settings
    audio_buffer_ms: int = Field(default=100)


# =============================================================================
# GLOBAL CONFIG INSTANCES
# =============================================================================

def load_config():
    """Load all configuration instances"""
    return {
        "api": APIConfig(),
        "voice": VoiceSettings(),
        "llm": LLMConfig(),
        "avatar": AvatarPersonality(),
        "whisper": WhisperConfig(),
        "liveavatar": LiveAvatarConfig(),
        "audio": AudioConfig(),
        "app": AppConfig(),
    }


# Singleton instances for easy import
api_config = APIConfig()
voice_settings = VoiceSettings()
llm_config = LLMConfig()
avatar_personality = AvatarPersonality()
whisper_config = WhisperConfig()
liveavatar_config = LiveAvatarConfig()
audio_config = AudioConfig()
app_config = AppConfig()


# =============================================================================
# VALIDATION & HELPERS
# =============================================================================

def validate_config() -> list[str]:
    """Validate configuration and return list of warnings/errors"""
    issues = []

    # Platform info
    if IS_LINUX:
        issues.append(f"INFO: Running on Linux (headless={IS_HEADLESS}, AWS={IS_AWS})")

    # Check TTS provider configuration
    if voice_settings.tts_provider == "elevenlabs" and not api_config.elevenlabs_api_key:
        issues.append("WARNING: ELEVENLABS_API_KEY not set but provider is 'elevenlabs'")

    if voice_settings.tts_provider == "piper":
        # Piper is recommended - just info message
        pass  # No warnings needed, Piper is the recommended free option

    if voice_settings.tts_provider == "kokoro" and IS_LINUX:
        issues.append("INFO: Kokoro TTS on Linux may require additional setup. Consider using 'piper' instead (free, works great on Linux).")

    if voice_settings.tts_provider == "tortoise":
        # Check if custom voice samples exist (if not using built-in voice)
        if voice_settings.tortoise_voice not in ["random"] and not voice_settings.tortoise_samples_dir.exists():
            issues.append(f"INFO: Tortoise voice samples dir not found at {voice_settings.tortoise_samples_dir}")

    if voice_settings.tts_provider == "local":
        # Check if espeak is available on Linux
        if IS_LINUX:
            import shutil
            if not shutil.which("espeak") and not shutil.which("espeak-ng"):
                issues.append("WARNING: Local TTS requires 'espeak' on Linux. Consider using 'piper' instead (better quality).")

    if llm_config.provider == "groq" and not api_config.groq_api_key:
        issues.append("WARNING: GROQ_API_KEY not set but provider is 'groq'")

    if llm_config.provider == "openai" and not api_config.openai_api_key:
        issues.append("WARNING: OPENAI_API_KEY not set but provider is 'openai'")

    # Check paths - need either video OR image
    has_video = avatar_personality.reference_media_path.exists()
    has_image = avatar_personality.reference_image_path.exists()
    if not has_video and not has_image:
        issues.append(f"WARNING: No avatar media found. Add avatar.mp4 or avatar_face.png to assets/")

    # Check model directories
    if not liveavatar_config.base_model_dir.exists():
        issues.append(f"WARNING: LiveAvatar base model not found - run setup_wizard.py first")
    if not liveavatar_config.lora_weights_dir.exists():
        issues.append(f"WARNING: LiveAvatar LoRA weights not found - run setup_wizard.py first")

    # Audio device warnings for Linux
    if IS_LINUX and IS_HEADLESS:
        issues.append("INFO: Headless mode detected. Audio playback will be disabled unless using web server.")

    return issues


def print_config_summary():
    """Print a summary of current configuration"""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="AI Avatar Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Avatar Name", avatar_personality.name)
    table.add_row("LLM Provider", llm_config.provider)
    table.add_row("LLM Model", getattr(llm_config, f"{llm_config.provider}_model"))
    table.add_row("Whisper Model", whisper_config.model_size)
    table.add_row("TTS Provider", voice_settings.tts_provider)
    if voice_settings.tts_provider == "kokoro":
        table.add_row("Kokoro Voice", voice_settings.kokoro_voice)
        table.add_row("Kokoro Speed", str(voice_settings.kokoro_speed))
    elif voice_settings.tts_provider == "tortoise":
        table.add_row("Tortoise Voice", voice_settings.tortoise_voice)
        table.add_row("Tortoise Preset", voice_settings.tortoise_preset)
    elif voice_settings.tts_provider == "elevenlabs":
        table.add_row("Voice Model", voice_settings.model_id)
    table.add_row("LiveAvatar FPS", str(liveavatar_config.fps))
    table.add_row("LiveAvatar Steps", str(liveavatar_config.sampling_steps))
    table.add_row("Debug Mode", str(app_config.debug))

    console.print(table)


if __name__ == "__main__":
    # Test configuration loading
    print_config_summary()

    issues = validate_config()
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  - {issue}")
