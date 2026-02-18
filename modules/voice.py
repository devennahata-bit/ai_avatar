"""
Voice Module - Text-to-Speech
=============================
Handles text-to-speech synthesis using multiple providers.

Providers:
- Kokoro (RealtimeTTS): Fast local TTS, 210x real-time on GPU, streaming support
- ElevenLabs: High-quality cloud TTS with voice cloning
- Tortoise-TTS: High-quality local TTS with voice cloning (slow)
- pyttsx3: Free local TTS fallback (low quality)

Features:
- Streaming audio generation
- Configurable voice settings
- Real-time audio chunk output for lip-sync
- Thread-safe audio queue for producer-consumer pattern
- LLM generator support for real-time streaming

Platform Support:
- Windows: All providers work
- Linux/AWS: ElevenLabs recommended, Kokoro may require additional setup
- macOS: All providers work
"""

import threading
import queue
import time
import io
import platform
import os
from typing import Optional, Callable, Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Platform detection
IS_LINUX = platform.system().lower() == "linux"
IS_HEADLESS = IS_LINUX and os.environ.get("DISPLAY") is None


@dataclass
class VoiceConfig:
    """Voice synthesis configuration."""
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    model_id: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    output_format: str = "pcm_24000"  # 24kHz PCM for MuseTalk
    latency_optimization: int = 3


class VoiceSynthesizer:
    """
    ElevenLabs voice synthesizer for real-time TTS.

    Produces audio chunks that can be consumed by the face renderer
    for synchronized lip movement.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
        output_format: str = "pcm_24000",
        latency_optimization: int = 3,
        audio_queue: Optional[queue.Queue] = None,
        output_device: Optional[int] = None,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
        on_synthesis_start: Optional[Callable[[], None]] = None,
        on_synthesis_complete: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the voice synthesizer.

        Args:
            api_key: ElevenLabs API key
            voice_id: ElevenLabs voice ID
            model_id: ElevenLabs model ID
            stability: Voice stability (0-1)
            similarity_boost: Similarity boost (0-1)
            style: Style exaggeration (0-1)
            use_speaker_boost: Enable speaker boost
            output_format: Audio output format
            latency_optimization: Latency optimization level (0-4)
            audio_queue: Queue for audio chunks (for face renderer)
            output_device: Audio output device index (None for default)
            on_audio_chunk: Callback for each audio chunk
            on_synthesis_start: Called when synthesis starts
            on_synthesis_complete: Called when synthesis completes
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.use_speaker_boost = use_speaker_boost
        self.output_format = output_format
        self.latency_optimization = latency_optimization
        self.output_device = output_device

        # Queue for audio chunks (producer-consumer pattern)
        self.audio_queue = audio_queue or queue.Queue(maxsize=100)

        # Callbacks
        self.on_audio_chunk = on_audio_chunk
        self.on_synthesis_start = on_synthesis_start
        self.on_synthesis_complete = on_synthesis_complete

        # State
        self._synthesizing = False
        self._client = None

        # Initialize client
        self._init_client()

        # Output audio parameters based on format
        self._parse_format()

        # Auto-detect Realtek output device if not specified
        if self.output_device is None:
            self._detect_output_device()

    def _init_client(self):
        """Initialize ElevenLabs client."""
        try:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self.api_key)
            logger.info(f"ElevenLabs client initialized (voice: {self.voice_id})")
        except ImportError:
            raise ImportError("elevenlabs package required")

    def _parse_format(self):
        """Parse output format to get audio parameters."""
        # Format: <encoding>_<sample_rate>
        # Examples: pcm_24000, mp3_44100, pcm_16000
        parts = self.output_format.split("_")
        self.encoding = parts[0] if parts else "pcm"
        self.sample_rate = int(parts[1]) if len(parts) > 1 else 24000

        # Calculate bytes per sample
        if self.encoding == "pcm":
            self.bytes_per_sample = 2  # 16-bit PCM
            self.channels = 1
        elif self.encoding == "mp3":
            self.bytes_per_sample = None  # Variable
            self.channels = 1
        else:
            self.bytes_per_sample = 2
            self.channels = 1

    def _detect_output_device(self):
        """Auto-detect the best output device (prefer Realtek)."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()

            # Look for Realtek output device
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    name = device['name'].lower()
                    if 'realtek' in name or 'speakers' in name:
                        self.output_device = i
                        logger.info(f"Auto-detected output device: [{i}] {device['name']}")
                        return

            # Fall back to default
            self.output_device = sd.default.device[1]
            logger.info(f"Using default output device: [{self.output_device}]")

        except Exception as e:
            logger.warning(f"Could not detect output device: {e}")
            self.output_device = None

    def update_settings(
        self,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
        style: Optional[float] = None,
        use_speaker_boost: Optional[bool] = None,
    ):
        """Update voice settings."""
        if stability is not None:
            self.stability = max(0.0, min(1.0, stability))
        if similarity_boost is not None:
            self.similarity_boost = max(0.0, min(1.0, similarity_boost))
        if style is not None:
            self.style = max(0.0, min(1.0, style))
        if use_speaker_boost is not None:
            self.use_speaker_boost = use_speaker_boost

        logger.info(
            f"Voice settings updated: stability={self.stability}, "
            f"similarity={self.similarity_boost}, style={self.style}"
        )

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech (non-streaming).

        Args:
            text: Text to synthesize

        Returns:
            Complete audio data as bytes
        """
        if not text.strip():
            return b""

        try:
            from elevenlabs import VoiceSettings

            audio = self._client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id=self.model_id,
                output_format=self.output_format,
                voice_settings=VoiceSettings(
                    stability=self.stability,
                    similarity_boost=self.similarity_boost,
                    style=self.style,
                    use_speaker_boost=self.use_speaker_boost,
                ),
            )

            # Collect all chunks
            audio_data = b"".join(audio)
            return audio_data

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return b""

    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        """
        Synthesize text to speech with streaming.

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks as bytes
        """
        if not text.strip():
            return

        self._synthesizing = True

        if self.on_synthesis_start:
            self.on_synthesis_start()

        try:
            from elevenlabs import VoiceSettings

            audio_stream = self._client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id=self.model_id,
                output_format=self.output_format,
                voice_settings=VoiceSettings(
                    stability=self.stability,
                    similarity_boost=self.similarity_boost,
                    style=self.style,
                    use_speaker_boost=self.use_speaker_boost,
                ),
            )

            for chunk in audio_stream:
                if chunk:
                    # Push to queue for face renderer
                    try:
                        self.audio_queue.put_nowait(chunk)
                    except queue.Full:
                        logger.warning("Audio queue full, dropping chunk")

                    # Callback
                    if self.on_audio_chunk:
                        self.on_audio_chunk(chunk)

                    yield chunk

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")

        finally:
            self._synthesizing = False
            # Signal end of audio
            self.audio_queue.put(None)

            if self.on_synthesis_complete:
                self.on_synthesis_complete()

    def synthesize_to_queue(self, text: str, play_audio: bool = True):
        """
        Synthesize text and push audio chunks to queue.

        This method runs synchronously and pushes all audio chunks
        to self.audio_queue for the face renderer to consume.
        Optionally also plays the audio through speakers.

        Args:
            text: Text to synthesize
            play_audio: Whether to play audio through speakers
        """
        if play_audio:
            # Collect all audio and play while also sending to queue
            audio_chunks = []
            for chunk in self.synthesize_stream(text):
                audio_chunks.append(chunk)

            # Play the collected audio
            if audio_chunks:
                full_audio = b"".join(audio_chunks)
                self._play_audio(full_audio)
        else:
            for _ in self.synthesize_stream(text):
                pass  # Chunks are pushed to queue in synthesize_stream

    def _play_audio(self, audio_data: bytes):
        """Play audio data through speakers."""
        try:
            import sounddevice as sd
            import numpy as np

            # Convert bytes to numpy array (16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32767.0

            # Play on specified device or default
            logger.info(f"Playing audio on device: {self.output_device}")
            sd.play(audio_float, self.sample_rate, device=self.output_device)
            sd.wait()

        except ImportError:
            logger.warning("sounddevice not installed - cannot play audio. Install with: pip install sounddevice")
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    def synthesize_async(self, text: str, callback: Optional[Callable[[bytes], None]] = None):
        """
        Synthesize text asynchronously in a separate thread.

        Args:
            text: Text to synthesize
            callback: Called with complete audio data when done
        """
        def _run():
            audio_data = self.synthesize(text)
            if callback:
                callback(audio_data)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def synthesize_stream_async(self, text: str):
        """
        Start streaming synthesis asynchronously.

        Audio chunks are pushed to self.audio_queue.

        Args:
            text: Text to synthesize
        """
        def _run():
            self.synthesize_to_queue(text)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    @property
    def is_synthesizing(self) -> bool:
        """Check if currently synthesizing."""
        return self._synthesizing

    def get_voices(self) -> list[dict]:
        """Get list of available voices."""
        try:
            response = self._client.voices.get_all()
            return [
                {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category,
                }
                for voice in response.voices
            ]
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return []


# =============================================================================
# LOCAL TTS (pyttsx3 - free, no API needed)
# =============================================================================

class LocalVoiceSynthesizer:
    """
    Local voice synthesizer using pyttsx3.

    Free alternative to ElevenLabs - uses system TTS engine.
    Quality is lower but works without API credits.
    """

    def __init__(
        self,
        rate: int = 175,  # Words per minute
        volume: float = 1.0,
        voice_index: int = 0,  # 0 = first voice, 1 = second, etc.
        output_device: Optional[int] = None,
        audio_queue: Optional[queue.Queue] = None,
        on_synthesis_start: Optional[Callable[[], None]] = None,
        on_synthesis_complete: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize local voice synthesizer.

        Args:
            rate: Speech rate in words per minute (default 175)
            volume: Volume from 0.0 to 1.0 (default 1.0)
            voice_index: Index of voice to use (0 for first available)
            output_device: Audio output device index
            audio_queue: Queue for audio chunks (for face renderer)
            on_synthesis_start: Called when synthesis starts
            on_synthesis_complete: Called when synthesis completes
        """
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self.output_device = output_device
        self.audio_queue = audio_queue or queue.Queue(maxsize=100)
        self.on_synthesis_start = on_synthesis_start
        self.on_synthesis_complete = on_synthesis_complete

        self._engine = None
        self._synthesizing = False
        self.sample_rate = 22050  # pyttsx3 default

        # Auto-detect output device
        if self.output_device is None:
            self._detect_output_device()

        self._init_engine()

    def _init_engine(self):
        """Initialize pyttsx3 engine."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()

            # Set properties
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)

            # Set voice
            voices = self._engine.getProperty('voices')
            if voices and self.voice_index < len(voices):
                self._engine.setProperty('voice', voices[self.voice_index].id)
                logger.info(f"Local TTS initialized with voice: {voices[self.voice_index].name}")
            else:
                logger.info("Local TTS initialized with default voice")

        except ImportError:
            raise ImportError("pyttsx3 package required. Install with: pip install pyttsx3")
        except Exception as e:
            error_msg = str(e).lower()
            if IS_LINUX and ("espeak" in error_msg or "no module" in error_msg or "not found" in error_msg):
                raise RuntimeError(
                    "Local TTS failed - espeak not installed. "
                    "Install with: sudo apt install espeak espeak-ng\n"
                    "Or use ElevenLabs TTS by setting VOICE_TTS_PROVIDER=elevenlabs in .env"
                ) from e
            logger.error(f"Failed to initialize pyttsx3: {e}")
            raise

    def _detect_output_device(self):
        """Auto-detect the best output device (prefer Realtek)."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()

            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    name = device['name'].lower()
                    if 'realtek' in name and 'speaker' in name:
                        self.output_device = i
                        logger.info(f"Local TTS auto-detected output: [{i}] {device['name']}")
                        return

            self.output_device = sd.default.device[1]
            logger.info(f"Local TTS using default output: [{self.output_device}]")

        except Exception as e:
            logger.warning(f"Could not detect output device: {e}")
            self.output_device = None

    def get_voices(self) -> list[dict]:
        """Get list of available local voices."""
        if not self._engine:
            return []

        voices = self._engine.getProperty('voices')
        return [
            {
                "voice_id": v.id,
                "name": v.name,
                "languages": getattr(v, 'languages', []),
                "gender": getattr(v, 'gender', 'unknown'),
            }
            for v in voices
        ]

    def set_voice(self, voice_index: int):
        """Set voice by index."""
        voices = self._engine.getProperty('voices')
        if voice_index < len(voices):
            self._engine.setProperty('voice', voices[voice_index].id)
            self.voice_index = voice_index
            logger.info(f"Voice set to: {voices[voice_index].name}")

    def synthesize_to_queue(self, text: str, play_audio: bool = True):
        """
        Synthesize text and optionally play through speakers.

        Args:
            text: Text to synthesize
            play_audio: Whether to play audio through speakers
        """
        if not text.strip():
            return

        self._synthesizing = True

        if self.on_synthesis_start:
            self.on_synthesis_start()

        try:
            if play_audio:
                # pyttsx3 plays directly through default audio
                # We need to save to file and play with sounddevice for device control
                import tempfile
                import os

                # Save to temp WAV file
                temp_path = os.path.join(tempfile.gettempdir(), 'tts_temp.wav')
                self._engine.save_to_file(text, temp_path)
                self._engine.runAndWait()

                # Play through specified device
                self._play_wav_file(temp_path)

                # Clean up
                try:
                    os.remove(temp_path)
                except:
                    pass
            else:
                # Just speak through default
                self._engine.say(text)
                self._engine.runAndWait()

        except Exception as e:
            logger.error(f"Local TTS error: {e}")

        finally:
            self._synthesizing = False
            # Signal end of audio to queue
            try:
                self.audio_queue.put(None)
            except:
                pass

            if self.on_synthesis_complete:
                self.on_synthesis_complete()

    def _play_wav_file(self, wav_path: str):
        """Play WAV file through specified output device and send to queue for lip-sync."""
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np

            data, samplerate = sf.read(wav_path)

            # Convert to 16-bit PCM bytes for the audio queue (lip-sync)
            if data.dtype == np.float64 or data.dtype == np.float32:
                audio_int16 = (data * 32767).astype(np.int16)
            else:
                audio_int16 = data.astype(np.int16)

            audio_bytes = audio_int16.tobytes()

            # Send audio chunks to queue for face renderer lip-sync
            chunk_size = 4800  # ~0.1 seconds at 24kHz (or proportional)
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass  # Skip if queue is full

            logger.info(f"Playing local TTS on device: {self.output_device}")
            sd.play(data, samplerate, device=self.output_device)
            sd.wait()

        except ImportError as e:
            logger.warning(f"soundfile/sounddevice not available: {e}")
            # Fallback: just use pyttsx3 directly
            self._engine.say("")  # Already played via save_to_file
        except Exception as e:
            logger.error(f"Error playing WAV: {e}")

    def synthesize_stream_async(self, text: str):
        """Start synthesis asynchronously."""
        def _run():
            self.synthesize_to_queue(text, play_audio=True)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    @property
    def is_synthesizing(self) -> bool:
        """Check if currently synthesizing."""
        return self._synthesizing


# =============================================================================
# TORTOISE TTS (high-quality local TTS with voice cloning)
# =============================================================================

class TortoiseTTSSynthesizer:
    """
    Tortoise-TTS voice synthesizer for high-quality local TTS.

    Features:
    - High-quality neural TTS without API costs
    - Voice cloning from audio samples
    - Multiple quality presets (ultra_fast, fast, standard, high_quality)
    - GPU acceleration with optional DeepSpeed
    """

    def __init__(
        self,
        voice: str = "random",
        preset: str = "fast",
        samples_dir: Optional[str] = None,
        use_deepspeed: bool = False,
        kv_cache: bool = True,
        half: bool = True,
        device: str = "cuda",
        output_device: Optional[int] = None,
        audio_queue: Optional[queue.Queue] = None,
        on_synthesis_start: Optional[Callable[[], None]] = None,
        on_synthesis_complete: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize Tortoise-TTS synthesizer.

        Args:
            voice: Voice name (built-in), "random", or name of folder in samples_dir
            preset: Quality preset - "ultra_fast", "fast", "standard", "high_quality"
            samples_dir: Directory containing voice sample folders
            use_deepspeed: Enable DeepSpeed for faster inference
            kv_cache: Enable KV cache for efficiency
            half: Use FP16 precision for reduced memory
            device: Device to run on ("cuda" or "cpu")
            output_device: Audio output device index
            audio_queue: Queue for audio chunks (for face renderer)
            on_synthesis_start: Called when synthesis starts
            on_synthesis_complete: Called when synthesis completes
        """
        self.voice = voice
        self.preset = preset
        self.samples_dir = samples_dir
        self.use_deepspeed = use_deepspeed
        self.kv_cache = kv_cache
        self.half = half
        self.device = device
        self.output_device = output_device
        self.audio_queue = audio_queue or queue.Queue(maxsize=100)
        self.on_synthesis_start = on_synthesis_start
        self.on_synthesis_complete = on_synthesis_complete

        self._synthesizing = False
        self._tts = None
        self.sample_rate = 24000  # Tortoise outputs 24kHz

        # Voice samples cache
        self._voice_samples = None
        self._conditioning_latents = None

        # Auto-detect output device
        if self.output_device is None:
            self._detect_output_device()

        # Initialize Tortoise-TTS
        self._init_tts()

    def _init_tts(self):
        """Initialize Tortoise-TTS model."""
        try:
            from tortoise.api import TextToSpeech
            from tortoise.utils.audio import load_audio, load_voices

            logger.info(f"Initializing Tortoise-TTS (preset={self.preset}, device={self.device})...")

            self._tts = TextToSpeech(
                use_deepspeed=self.use_deepspeed,
                kv_cache=self.kv_cache,
                half=self.half,
                device=self.device,
            )

            # Load voice samples if using custom voice
            if self.voice != "random" and self.samples_dir:
                self._load_voice_samples()

            logger.info(f"Tortoise-TTS initialized (voice={self.voice})")

        except ImportError as e:
            raise ImportError(
                "tortoise-tts package required. Install with: pip install tortoise-tts"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Tortoise-TTS: {e}")
            raise

    def _load_voice_samples(self):
        """Load voice samples for voice cloning."""
        try:
            from tortoise.utils.audio import load_voices
            import os

            voice_dir = os.path.join(self.samples_dir, self.voice)
            if os.path.exists(voice_dir):
                self._voice_samples, self._conditioning_latents = load_voices(
                    [self.voice], [self.samples_dir]
                )
                logger.info(f"Loaded voice samples from {voice_dir}")
            else:
                logger.warning(f"Voice directory not found: {voice_dir}, using random voice")
                self.voice = "random"

        except Exception as e:
            logger.warning(f"Failed to load voice samples: {e}, using random voice")
            self.voice = "random"

    def _detect_output_device(self):
        """Auto-detect the best output device."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()

            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    name = device['name'].lower()
                    if 'realtek' in name or 'speakers' in name:
                        self.output_device = i
                        logger.info(f"Tortoise TTS auto-detected output: [{i}] {device['name']}")
                        return

            self.output_device = sd.default.device[1]
            logger.info(f"Tortoise TTS using default output: [{self.output_device}]")

        except Exception as e:
            logger.warning(f"Could not detect output device: {e}")
            self.output_device = None

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech (non-streaming).

        Args:
            text: Text to synthesize

        Returns:
            Audio data as bytes (16-bit PCM at 24kHz)
        """
        if not text.strip():
            return b""

        try:
            import torch
            import numpy as np

            # Generate speech using tts_with_preset for proper preset handling
            if self.voice == "random":
                gen = self._tts.tts_with_preset(text, preset=self.preset)
            else:
                gen = self._tts.tts_with_preset(
                    text,
                    voice_samples=self._voice_samples,
                    conditioning_latents=self._conditioning_latents,
                    preset=self.preset,
                )

            # Convert to numpy and then to bytes
            audio_tensor = gen.squeeze(0).cpu()
            audio_np = audio_tensor.numpy()

            # Normalize and convert to 16-bit PCM
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)

            return audio_int16.tobytes()

        except Exception as e:
            logger.error(f"Tortoise synthesis error: {e}")
            return b""

    def synthesize_to_queue(self, text: str, play_audio: bool = True):
        """
        Synthesize text and push audio to queue for lip-sync.

        Args:
            text: Text to synthesize
            play_audio: Whether to play audio through speakers
        """
        if not text.strip():
            return

        self._synthesizing = True

        if self.on_synthesis_start:
            self.on_synthesis_start()

        try:
            import numpy as np

            logger.info(f"Tortoise synthesizing: {text[:50]}...")

            # Synthesize audio
            audio_bytes = self.synthesize(text)

            if audio_bytes:
                # Send audio chunks to queue for face renderer lip-sync
                chunk_size = 4800  # ~0.1 seconds at 24kHz (in bytes, 16-bit = 2 bytes/sample)
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    try:
                        self.audio_queue.put_nowait(chunk)
                    except queue.Full:
                        logger.warning("Audio queue full, dropping chunk")

                # Play audio if requested
                if play_audio:
                    self._play_audio(audio_bytes)

        except Exception as e:
            logger.error(f"Tortoise TTS error: {e}")

        finally:
            self._synthesizing = False
            # Signal end of audio
            try:
                self.audio_queue.put(None)
            except:
                pass

            if self.on_synthesis_complete:
                self.on_synthesis_complete()

    def _play_audio(self, audio_bytes: bytes):
        """Play audio data through speakers."""
        try:
            import sounddevice as sd
            import numpy as np

            # Convert bytes to numpy array (16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32767.0

            logger.info(f"Playing Tortoise audio on device: {self.output_device}")
            sd.play(audio_float, self.sample_rate, device=self.output_device)
            sd.wait()

        except ImportError:
            logger.warning("sounddevice not installed - cannot play audio")
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    def synthesize_stream_async(self, text: str):
        """Start synthesis asynchronously in a separate thread."""
        def _run():
            self.synthesize_to_queue(text, play_audio=True)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    @property
    def is_synthesizing(self) -> bool:
        """Check if currently synthesizing."""
        return self._synthesizing

    def get_available_voices(self) -> list[str]:
        """Get list of available built-in voices."""
        try:
            import os
            from tortoise.utils.audio import get_voices

            voices = get_voices()

            # Add custom voices from samples_dir
            if self.samples_dir and os.path.exists(self.samples_dir):
                custom_voices = [
                    d for d in os.listdir(self.samples_dir)
                    if os.path.isdir(os.path.join(self.samples_dir, d))
                ]
                voices.extend(custom_voices)

            return voices

        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return ["random"]


# =============================================================================
# PIPER TTS (Fast, high-quality local neural TTS - FREE)
# =============================================================================

class PiperTTSSynthesizer:
    """
    Piper TTS voice synthesizer - fast, high-quality, FREE local neural TTS.

    Features:
    - High-quality neural voices (much better than espeak)
    - Fast inference using ONNX
    - Runs completely offline
    - No API keys required
    - Multiple voice options
    - Works great on Linux/AWS

    Install: pip install piper-tts
    Voices: https://huggingface.co/rhasspy/piper-voices
    """

    def __init__(
        self,
        model: str = "en_US-lessac-medium",
        model_path: Optional[str] = None,
        output_device: Optional[int] = None,
        audio_queue: Optional[queue.Queue] = None,
        on_synthesis_start: Optional[Callable[[], None]] = None,
        on_synthesis_complete: Optional[Callable[[], None]] = None,
        use_cuda: bool = False,
    ):
        """
        Initialize Piper TTS synthesizer.

        Args:
            model: Voice model name (e.g., "en_US-lessac-medium", "en_GB-alan-medium")
            model_path: Path to local .onnx model file (overrides model name)
            output_device: Audio output device index
            audio_queue: Queue for audio chunks (for face renderer)
            on_synthesis_start: Called when synthesis starts
            on_synthesis_complete: Called when synthesis completes
            use_cuda: Use GPU acceleration if available
        """
        self.model_name = model
        self.model_path = model_path
        self.output_device = output_device
        self.audio_queue = audio_queue or queue.Queue(maxsize=100)
        self.on_synthesis_start = on_synthesis_start
        self.on_synthesis_complete = on_synthesis_complete
        self.use_cuda = use_cuda

        self._synthesizing = False
        self._voice = None
        self.sample_rate = 22050  # Piper default, may vary by model

        # Auto-detect output device
        if self.output_device is None:
            self._detect_output_device()

        # Initialize Piper
        self._init_piper()

    def _init_piper(self):
        """Initialize Piper TTS engine."""
        try:
            from piper.voice import PiperVoice
            from piper.download import find_voice, get_voices

            logger.info(f"Initializing Piper TTS (model={self.model_name})...")

            if self.model_path:
                # Load from specific path
                self._voice = PiperVoice.load(self.model_path, use_cuda=self.use_cuda)
            else:
                # Download/find voice model
                model_path, config_path = find_voice(self.model_name, [])
                self._voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=self.use_cuda)

            # Get sample rate from config
            if hasattr(self._voice, 'config') and self._voice.config:
                self.sample_rate = self._voice.config.sample_rate

            logger.info(f"Piper TTS initialized (model={self.model_name}, sample_rate={self.sample_rate})")

        except ImportError as e:
            raise ImportError(
                "piper-tts package required. Install with: pip install piper-tts"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            raise

    def _detect_output_device(self):
        """Auto-detect the best output device."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()

            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    name = device['name'].lower()
                    if 'realtek' in name or 'speakers' in name or 'default' in name:
                        self.output_device = i
                        logger.info(f"Piper TTS auto-detected output: [{i}] {device['name']}")
                        return

            self.output_device = sd.default.device[1]
            logger.info(f"Piper TTS using default output: [{self.output_device}]")

        except Exception as e:
            logger.warning(f"Could not detect output device: {e}")
            self.output_device = None

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as bytes (16-bit PCM)
        """
        if not text.strip():
            return b""

        try:
            import numpy as np

            # Synthesize to audio
            audio_chunks = []
            for audio_bytes in self._voice.synthesize_stream_raw(text):
                audio_chunks.append(audio_bytes)

            if audio_chunks:
                return b"".join(audio_chunks)
            return b""

        except Exception as e:
            logger.error(f"Piper synthesis error: {e}")
            return b""

    def synthesize_to_queue(self, text: str, play_audio: bool = True):
        """
        Synthesize text and push audio to queue for lip-sync.

        Args:
            text: Text to synthesize
            play_audio: Whether to play audio through speakers
        """
        if not text.strip():
            return

        self._synthesizing = True

        if self.on_synthesis_start:
            self.on_synthesis_start()

        try:
            import numpy as np

            logger.info(f"Piper synthesizing: {text[:50]}...")

            # Collect audio for playback and queue
            all_audio = []

            for audio_bytes in self._voice.synthesize_stream_raw(text):
                # Send to queue for lip-sync
                try:
                    self.audio_queue.put_nowait(audio_bytes)
                except queue.Full:
                    logger.warning("Audio queue full, dropping chunk")

                all_audio.append(audio_bytes)

            # Play audio if requested
            if play_audio and all_audio:
                full_audio = b"".join(all_audio)
                self._play_audio(full_audio)

        except Exception as e:
            logger.error(f"Piper TTS error: {e}")

        finally:
            self._synthesizing = False
            # Signal end of audio
            try:
                self.audio_queue.put(None)
            except:
                pass

            if self.on_synthesis_complete:
                self.on_synthesis_complete()

    def _play_audio(self, audio_bytes: bytes):
        """Play audio data through speakers."""
        try:
            import sounddevice as sd
            import numpy as np

            # Convert bytes to numpy array (16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32767.0

            logger.info(f"Playing Piper audio on device: {self.output_device}")
            sd.play(audio_float, self.sample_rate, device=self.output_device)
            sd.wait()

        except ImportError:
            logger.warning("sounddevice not installed - cannot play audio")
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    def synthesize_stream_async(self, text: str):
        """Start synthesis asynchronously in a separate thread."""
        def _run():
            self.synthesize_to_queue(text, play_audio=True)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    @property
    def is_synthesizing(self) -> bool:
        """Check if currently synthesizing."""
        return self._synthesizing

    @staticmethod
    def get_available_voices() -> list[str]:
        """Get list of recommended Piper voices."""
        # Common high-quality voices
        return [
            "en_US-lessac-medium",      # US English - Lessac (recommended)
            "en_US-amy-medium",         # US English - Amy
            "en_US-ryan-medium",        # US English - Ryan (male)
            "en_GB-alan-medium",        # British English - Alan (male)
            "en_GB-alba-medium",        # British English - Alba
            "en_US-libritts-high",      # US English - LibriTTS (high quality)
            "en_US-lessac-high",        # US English - Lessac (high quality)
        ]


# =============================================================================
# REALTIME TTS (Kokoro - fast local TTS with streaming)
# =============================================================================

class RealtimeTTSSynthesizer:
    """
    RealtimeTTS voice synthesizer using Kokoro engine.

    Features:
    - Extremely fast: 210x real-time on GPU, 3-11x on CPU
    - True streaming: audio starts playing before text is fully processed
    - Local processing: no API costs
    - LLM-friendly: designed for real-time LLM output streaming
    - 82M parameter model with high quality output
    """

    def __init__(
        self,
        voice: str = "af_heart",  # Default Kokoro voice
        speed: float = 1.0,
        output_device: Optional[int] = None,
        audio_queue: Optional[queue.Queue] = None,
        on_synthesis_start: Optional[Callable[[], None]] = None,
        on_synthesis_complete: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize RealtimeTTS with Kokoro engine.

        Args:
            voice: Kokoro voice name (e.g., "af_heart", "af_bella", "am_adam")
            speed: Speech speed multiplier (default 1.0)
            output_device: Audio output device index
            audio_queue: Queue for audio chunks (for face renderer)
            on_synthesis_start: Called when synthesis starts
            on_synthesis_complete: Called when synthesis completes
        """
        self.voice = voice
        self.speed = speed
        self.output_device = output_device
        self.audio_queue = audio_queue or queue.Queue(maxsize=100)
        self.on_synthesis_start = on_synthesis_start
        self.on_synthesis_complete = on_synthesis_complete

        self._synthesizing = False
        self._stream = None
        self._engine = None
        self.sample_rate = 24000  # Kokoro outputs 24kHz

        # Auto-detect output device
        if self.output_device is None:
            self._detect_output_device()

        # Initialize RealtimeTTS
        self._init_engine()

    def _init_engine(self):
        """Initialize RealtimeTTS with Kokoro engine."""
        try:
            from RealtimeTTS import TextToAudioStream, KokoroEngine

            logger.info(f"Initializing Kokoro TTS (voice={self.voice})...")

            # Initialize Kokoro engine
            self._engine = KokoroEngine(
                voice=self.voice,
                speed=self.speed,
            )

            # Create streaming audio handler
            self._stream = TextToAudioStream(
                self._engine,
                on_audio_stream_start=self._on_stream_start,
                on_audio_stream_stop=self._on_stream_stop,
            )

            logger.info(f"Kokoro TTS initialized (voice={self.voice})")

        except ImportError as e:
            raise ImportError(
                "RealtimeTTS with Kokoro required. Install with: pip install realtimetts[kokoro]"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            raise

    def _detect_output_device(self):
        """Auto-detect the best output device."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()

            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    name = device['name'].lower()
                    if 'realtek' in name or 'speakers' in name:
                        self.output_device = i
                        logger.info(f"Kokoro TTS auto-detected output: [{i}] {device['name']}")
                        return

            self.output_device = sd.default.device[1]
            logger.info(f"Kokoro TTS using default output: [{self.output_device}]")

        except Exception as e:
            logger.warning(f"Could not detect output device: {e}")
            self.output_device = None

    def _on_stream_start(self):
        """Called when audio stream starts."""
        self._synthesizing = True
        if self.on_synthesis_start:
            self.on_synthesis_start()

    def _on_stream_stop(self):
        """Called when audio stream stops."""
        self._synthesizing = False
        # Signal end of audio to queue
        try:
            self.audio_queue.put(None)
        except:
            pass
        if self.on_synthesis_complete:
            self.on_synthesis_complete()

    def synthesize_to_queue(self, text: str, play_audio: bool = True):
        """
        Synthesize text with streaming and push audio to queue for lip-sync.

        Args:
            text: Text to synthesize
            play_audio: Whether to play audio through speakers
        """
        if not text.strip():
            return

        try:
            import numpy as np

            logger.info(f"Kokoro synthesizing: {text[:50]}...")

            # Feed text to stream
            self._stream.feed(text)

            if play_audio:
                # Play with custom callback to also feed queue
                def on_audio_chunk(chunk):
                    # Convert to bytes for queue
                    if isinstance(chunk, np.ndarray):
                        audio_int16 = (chunk * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                    else:
                        audio_bytes = chunk

                    try:
                        self.audio_queue.put_nowait(audio_bytes)
                    except queue.Full:
                        logger.warning("Audio queue full, dropping chunk")

                # Play synchronously with callback
                self._stream.play(
                    on_audio_chunk=on_audio_chunk,
                    output_device_index=self.output_device,
                )
            else:
                # Just generate without playing
                audio_data = self._stream.retrieve_audio()
                if audio_data is not None:
                    import numpy as np
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()

                    # Send to queue in chunks
                    chunk_size = 4800
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i:i + chunk_size]
                        try:
                            self.audio_queue.put_nowait(chunk)
                        except queue.Full:
                            pass

        except Exception as e:
            logger.error(f"Kokoro TTS error: {e}")
            self._synthesizing = False
            try:
                self.audio_queue.put(None)
            except:
                pass

    def synthesize_stream_async(self, text: str):
        """Start synthesis asynchronously in a separate thread."""
        def _run():
            self.synthesize_to_queue(text, play_audio=True)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def feed_generator(self, text_generator):
        """
        Feed a generator (e.g., LLM output) for real-time streaming TTS.

        This is ideal for streaming LLM responses - audio starts
        playing as soon as the first sentence is complete.

        Args:
            text_generator: Generator yielding text chunks (e.g., from LLM)
        """
        try:
            def _run():
                self._stream.feed(text_generator)
                self._stream.play(output_device_index=self.output_device)

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

        except Exception as e:
            logger.error(f"Kokoro generator feed error: {e}")

    def set_voice(self, voice: str):
        """Change the voice."""
        self.voice = voice
        if self._engine:
            self._engine.set_voice(voice)
            logger.info(f"Kokoro voice changed to: {voice}")

    def set_speed(self, speed: float):
        """Change speech speed."""
        self.speed = speed
        if self._engine:
            self._engine.speed = speed
            logger.info(f"Kokoro speed changed to: {speed}")

    @property
    def is_synthesizing(self) -> bool:
        """Check if currently synthesizing."""
        return self._synthesizing

    @staticmethod
    def get_available_voices() -> list[str]:
        """Get list of available Kokoro voices."""
        # Kokoro built-in voices
        return [
            "af_heart",    # American Female - Heart
            "af_bella",    # American Female - Bella
            "af_nicole",   # American Female - Nicole
            "af_sarah",    # American Female - Sarah
            "af_sky",      # American Female - Sky
            "am_adam",     # American Male - Adam
            "am_michael",  # American Male - Michael
            "bf_emma",     # British Female - Emma
            "bf_isabella", # British Female - Isabella
            "bm_george",   # British Male - George
            "bm_lewis",    # British Male - Lewis
        ]


# =============================================================================
# AUDIO PLAYER (for testing)
# =============================================================================

class AudioPlayer:
    """Simple audio player for testing voice synthesis."""

    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = None

    def play(self, audio_data: bytes):
        """Play audio data."""
        try:
            import sounddevice as sd
            import numpy as np

            # Convert bytes to numpy array (16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32767.0

            sd.play(audio_float, self.sample_rate)
            sd.wait()

        except ImportError:
            logger.error("sounddevice not installed")
        except Exception as e:
            logger.error(f"Playback error: {e}")

    def play_stream(self, audio_queue: queue.Queue):
        """Play audio from queue."""
        try:
            import sounddevice as sd
            import numpy as np

            def callback(outdata, frames, time_info, status):
                try:
                    chunk = audio_queue.get_nowait()
                    if chunk is None:
                        raise sd.CallbackStop

                    audio = np.frombuffer(chunk, dtype=np.int16)
                    audio_float = audio.astype(np.float32) / 32767.0

                    # Pad or truncate to match frames
                    if len(audio_float) < frames:
                        audio_float = np.pad(audio_float, (0, frames - len(audio_float)))
                    else:
                        audio_float = audio_float[:frames]

                    outdata[:, 0] = audio_float

                except queue.Empty:
                    outdata.fill(0)

            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback,
                blocksize=2048,
            ):
                while True:
                    try:
                        # Check for end signal
                        if audio_queue.queue and audio_queue.queue[-1] is None:
                            time.sleep(0.5)  # Let remaining audio play
                            break
                        time.sleep(0.1)
                    except:
                        break

        except Exception as e:
            logger.error(f"Stream playback error: {e}")


# =============================================================================
# TTS FACTORY (Platform-aware provider selection)
# =============================================================================

def create_tts_synthesizer(
    provider: str = "auto",
    api_key: Optional[str] = None,
    voice_id: Optional[str] = None,
    audio_queue: Optional[queue.Queue] = None,
    output_device: Optional[int] = None,
    on_synthesis_start: Optional[Callable[[], None]] = None,
    on_synthesis_complete: Optional[Callable[[], None]] = None,
    **kwargs
):
    """
    Factory function to create the appropriate TTS synthesizer.

    Handles platform-specific issues and graceful fallback.

    Args:
        provider: "piper", "kokoro", "elevenlabs", "tortoise", "local", or "auto"
        api_key: API key for cloud providers (ElevenLabs)
        voice_id: Voice ID for ElevenLabs
        audio_queue: Queue for audio chunks
        output_device: Audio output device index
        on_synthesis_start: Callback when synthesis starts
        on_synthesis_complete: Callback when synthesis completes
        **kwargs: Additional provider-specific arguments

    Returns:
        Appropriate TTS synthesizer instance
    """
    providers_tried = []

    # Auto-select based on platform
    if provider == "auto":
        if IS_LINUX or IS_HEADLESS:
            # On Linux, prefer Piper (free, high quality neural TTS)
            provider = "piper"
        else:
            provider = "kokoro"

    # Try requested provider first, then fall back
    # Piper is the best free option, so it's high in fallback order
    fallback_order = {
        "piper": ["piper", "kokoro", "elevenlabs", "local"],
        "kokoro": ["kokoro", "piper", "elevenlabs", "local"],
        "elevenlabs": ["elevenlabs", "piper", "kokoro", "local"],
        "tortoise": ["tortoise", "piper", "elevenlabs", "local"],
        "local": ["local", "piper"],
    }

    providers_to_try = fallback_order.get(provider, ["piper", "local"])

    for p in providers_to_try:
        providers_tried.append(p)

        try:
            if p == "piper":
                return PiperTTSSynthesizer(
                    model=kwargs.get("piper_model", "en_US-lessac-medium"),
                    model_path=kwargs.get("piper_model_path"),
                    output_device=output_device,
                    audio_queue=audio_queue,
                    on_synthesis_start=on_synthesis_start,
                    on_synthesis_complete=on_synthesis_complete,
                    use_cuda=kwargs.get("use_cuda", False),
                )

            elif p == "kokoro":
                return RealtimeTTSSynthesizer(
                    voice=kwargs.get("kokoro_voice", "af_heart"),
                    speed=kwargs.get("kokoro_speed", 1.0),
                    output_device=output_device,
                    audio_queue=audio_queue,
                    on_synthesis_start=on_synthesis_start,
                    on_synthesis_complete=on_synthesis_complete,
                )

            elif p == "elevenlabs":
                if not api_key:
                    logger.warning("ElevenLabs requires API key, skipping...")
                    continue
                return VoiceSynthesizer(
                    api_key=api_key,
                    voice_id=voice_id or "21m00Tcm4TlvDq8ikWAM",
                    model_id=kwargs.get("model_id", "eleven_multilingual_v2"),
                    stability=kwargs.get("stability", 0.5),
                    similarity_boost=kwargs.get("similarity_boost", 0.75),
                    style=kwargs.get("style", 0.0),
                    use_speaker_boost=kwargs.get("use_speaker_boost", True),
                    output_format=kwargs.get("output_format", "pcm_24000"),
                    audio_queue=audio_queue,
                    output_device=output_device,
                    on_synthesis_start=on_synthesis_start,
                    on_synthesis_complete=on_synthesis_complete,
                )

            elif p == "tortoise":
                return TortoiseTTSSynthesizer(
                    voice=kwargs.get("tortoise_voice", "random"),
                    preset=kwargs.get("tortoise_preset", "fast"),
                    device=kwargs.get("device", "cuda"),
                    output_device=output_device,
                    audio_queue=audio_queue,
                    on_synthesis_start=on_synthesis_start,
                    on_synthesis_complete=on_synthesis_complete,
                )

            elif p == "local":
                return LocalVoiceSynthesizer(
                    rate=kwargs.get("local_rate", 175),
                    voice_index=kwargs.get("local_voice_index", 0),
                    output_device=output_device,
                    audio_queue=audio_queue,
                    on_synthesis_start=on_synthesis_start,
                    on_synthesis_complete=on_synthesis_complete,
                )

        except Exception as e:
            logger.warning(f"Failed to initialize {p} TTS: {e}")
            continue

    # If all providers failed, raise error
    raise RuntimeError(
        f"Failed to initialize any TTS provider. Tried: {providers_tried}. "
        f"On Linux, install piper-tts: pip install piper-tts"
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    api_key = os.getenv("ELEVENLABS_API_KEY")

    print("Initializing voice synthesizer...")
    print(f"Platform: {platform.system()}, Headless: {IS_HEADLESS}")

    # Use factory for automatic provider selection
    try:
        synthesizer = create_tts_synthesizer(
            provider="auto",
            api_key=api_key,
        )
        print(f"Using: {type(synthesizer).__name__}")
    except Exception as e:
        print(f"Failed to create synthesizer: {e}")
        exit(1)

    player = AudioPlayer(sample_rate=24000)

    print("\nType text to synthesize (or 'quit' to exit):\n")

    while True:
        text = input("Text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break

        print("Synthesizing...")

        # Use synthesize_to_queue for consistent interface
        synthesizer.synthesize_to_queue(text, play_audio=True)
