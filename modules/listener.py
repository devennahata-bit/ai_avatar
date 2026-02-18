"""
Speech Listener Module - Whisper STT
=====================================
Handles microphone input and speech-to-text conversion using Whisper.

Features:
- Voice Activity Detection (VAD) for efficient processing
- Configurable silence thresholds
- Support for both local Whisper and OpenAI API
- Thread-safe audio buffering
"""

import threading
import queue
import time
import numpy as np
from typing import Optional, Callable, Generator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SpeechListener:
    """
    Real-time speech-to-text listener using Whisper.

    Uses Voice Activity Detection (VAD) to detect speech segments,
    then transcribes them using either local Whisper or OpenAI API.
    """

    def __init__(
        self,
        model_size: str = "base",
        sample_rate: int = 16000,
        channels: int = 1,
        use_api: bool = False,
        api_key: Optional[str] = None,
        device: str = "cuda",
        input_device: Optional[int] = None,
        vad_aggressiveness: int = 2,
        silence_threshold: float = 0.8,
        min_speech_duration: float = 0.5,
        on_transcription: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the speech listener.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            sample_rate: Audio sample rate (default 16kHz for Whisper)
            channels: Number of audio channels (1 for mono)
            use_api: Use OpenAI Whisper API instead of local model
            api_key: OpenAI API key (required if use_api=True)
            device: Device for local inference (cuda/cpu)
            input_device: Audio input device index (None for auto-detect)
            vad_aggressiveness: VAD aggressiveness level (0-3)
            silence_threshold: Seconds of silence to end speech segment
            min_speech_duration: Minimum speech duration to process
            on_transcription: Callback function for transcription results
        """
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.use_api = use_api
        self.api_key = api_key
        self.device = device
        self.input_device = input_device
        self.vad_aggressiveness = vad_aggressiveness
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.on_transcription = on_transcription

        # State
        self._running = False
        self._paused = False
        self._model = None
        self._vad = None
        self._stream = None
        self._use_openai_whisper = False

        # Audio buffer
        self._audio_queue: queue.Queue = queue.Queue()
        self._speech_buffer: list = []

        # Threads
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

        # Initialize components
        self._init_vad()

        # Auto-detect input device if not specified
        if self.input_device is None:
            self._detect_input_device()

    def _detect_input_device(self):
        """Auto-detect the best input device (prefer Realtek)."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()

            # Look for Realtek input device
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    name = device['name'].lower()
                    if 'realtek' in name or 'microphone' in name:
                        self.input_device = i
                        logger.info(f"Auto-detected input device: [{i}] {device['name']}")
                        return

            # Fall back to default
            self.input_device = sd.default.device[0]
            logger.info(f"Using default input device: [{self.input_device}]")

        except Exception as e:
            logger.warning(f"Could not detect input device: {e}")
            self.input_device = None

    def _init_vad(self):
        """Initialize Voice Activity Detection."""
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.vad_aggressiveness)
            logger.info(f"VAD initialized with aggressiveness={self.vad_aggressiveness}")
        except ImportError:
            logger.warning("webrtcvad not available, using energy-based VAD")
            self._vad = None

    def _init_model(self):
        """Initialize Whisper model (lazy loading)."""
        if self._model is not None:
            return

        if self.use_api:
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=self.api_key)
                logger.info("Using OpenAI Whisper API")
            except ImportError:
                raise ImportError("openai package required for API mode")
        else:
            try:
                from faster_whisper import WhisperModel

                compute_type = "float16" if self.device == "cuda" else "float32"
                logger.info(f"Loading faster-whisper model: {self.model_size}")
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=compute_type,
                )
                self._use_openai_whisper = False
                logger.info("faster-whisper model loaded successfully")
            except ImportError:
                logger.warning("faster-whisper not available, trying openai-whisper")
                try:
                    import whisper

                    logger.info(f"Loading openai-whisper model: {self.model_size}")
                    self._model = whisper.load_model(self.model_size, device=self.device)
                    self._use_openai_whisper = True
                    logger.info("openai-whisper model loaded successfully")
                except ImportError:
                    raise ImportError(
                        "Neither faster-whisper nor openai-whisper is installed "
                        "for local transcription mode"
                    )

    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech.

        Args:
            audio_chunk: Audio data as numpy array

        Returns:
            True if speech detected
        """
        if self._vad is not None:
            # Use WebRTC VAD
            try:
                # Convert to 16-bit PCM
                audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()

                # VAD expects 10, 20, or 30ms frames at 8, 16, or 32kHz
                frame_duration_ms = 30
                frame_size = int(self.sample_rate * frame_duration_ms / 1000)

                # Check multiple frames
                speech_frames = 0
                total_frames = 0

                for i in range(0, len(audio_chunk) - frame_size, frame_size):
                    frame = audio_bytes[i * 2:(i + frame_size) * 2]
                    if len(frame) == frame_size * 2:
                        total_frames += 1
                        if self._vad.is_speech(frame, self.sample_rate):
                            speech_frames += 1

                # Consider speech if more than 30% of frames are speech
                return total_frames > 0 and (speech_frames / total_frames) > 0.3

            except Exception as e:
                logger.debug(f"VAD error: {e}, falling back to energy-based")

        # Fallback: energy-based VAD
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        threshold = 0.01  # Adjust based on your microphone
        return energy > threshold

    def _capture_audio(self):
        """Audio capture thread - reads from microphone."""
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed")
            return

        # Calculate chunk size for ~100ms of audio
        chunk_duration = 0.1  # seconds
        chunk_size = int(self.sample_rate * chunk_duration)

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            if not self._paused and self._running:
                self._audio_queue.put(indata.copy().flatten())

        try:
            logger.info(f"Opening audio input on device: {self.input_device}")
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=chunk_size,
                callback=audio_callback,
                device=self.input_device,
            ):
                logger.info("Audio capture started")
                while self._running:
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            self._running = False

    def _process_audio(self):
        """Audio processing thread - handles VAD and transcription."""
        self._init_model()

        speech_buffer = []
        silence_duration = 0.0
        is_speaking = False
        chunk_duration = 0.1  # Match capture chunk duration

        logger.info("Audio processing started")

        while self._running:
            try:
                # Get audio chunk with timeout
                try:
                    audio_chunk = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if self._paused:
                    continue

                # Check for speech
                has_speech = self._is_speech(audio_chunk)

                if has_speech:
                    if not is_speaking:
                        logger.debug("Speech started")
                        is_speaking = True
                        silence_duration = 0.0

                    speech_buffer.append(audio_chunk)
                    silence_duration = 0.0

                elif is_speaking:
                    # Accumulate silence
                    speech_buffer.append(audio_chunk)
                    silence_duration += chunk_duration

                    # Check if silence threshold reached
                    if silence_duration >= self.silence_threshold:
                        logger.debug("Speech ended")
                        is_speaking = False

                        # Process accumulated speech
                        if speech_buffer:
                            speech_audio = np.concatenate(speech_buffer)
                            speech_duration = len(speech_audio) / self.sample_rate

                            if speech_duration >= self.min_speech_duration:
                                # Transcribe
                                text = self._transcribe(speech_audio)
                                if text and text.strip():
                                    logger.info(f"Transcription: {text}")
                                    if self.on_transcription:
                                        self.on_transcription(text.strip())

                        speech_buffer = []
                        silence_duration = 0.0

            except Exception as e:
                logger.error(f"Processing error: {e}")

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array

        Returns:
            Transcribed text
        """
        if self.use_api:
            return self._transcribe_api(audio)
        else:
            return self._transcribe_local(audio)

    def _transcribe_local(self, audio: np.ndarray) -> str:
        """Transcribe using local Whisper model."""
        try:
            audio = audio.astype(np.float32)

            if self._use_openai_whisper:
                result = self._model.transcribe(
                    audio,
                    language="en",
                    fp16=(self.device == "cuda"),
                    task="transcribe",
                )
                return result.get("text", "")

            segments, _ = self._model.transcribe(
                audio,
                language="en",
                beam_size=5,
            )
            text = " ".join(segment.text for segment in segments)
            return text.strip()

        except Exception as e:
            logger.error(f"Local transcription error: {e}")
            return ""

    def _transcribe_api(self, audio: np.ndarray) -> str:
        """Transcribe using OpenAI Whisper API."""
        try:
            import tempfile
            import soundfile as sf

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, self.sample_rate)
                temp_path = f.name

            try:
                with open(temp_path, "rb") as audio_file:
                    response = self._openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en",
                    )
                return response.text
            finally:
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"API transcription error: {e}")
            return ""

    def start(self):
        """Start listening for speech."""
        if self._running:
            logger.warning("Listener already running")
            return

        self._running = True
        self._paused = False

        # Start capture thread
        self._capture_thread = threading.Thread(
            target=self._capture_audio,
            name="AudioCapture",
            daemon=True,
        )
        self._capture_thread.start()

        # Start processing thread
        self._process_thread = threading.Thread(
            target=self._process_audio,
            name="AudioProcess",
            daemon=True,
        )
        self._process_thread.start()

        logger.info("Speech listener started")

    def stop(self):
        """Stop listening."""
        self._running = False

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

        logger.info("Speech listener stopped")

    def pause(self):
        """Pause listening (while avatar is speaking)."""
        self._paused = True
        logger.debug("Listener paused")

    def resume(self):
        """Resume listening."""
        self._paused = False
        # Clear any buffered audio
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("Listener resumed")

    @property
    def is_running(self) -> bool:
        """Check if listener is running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if listener is paused."""
        return self._paused


# =============================================================================
# FILE TRANSCRIBER (for browser audio)
# =============================================================================

class SpeechRecognizer:
    """
    Simple speech recognizer for transcribing audio files.
    Used for browser audio transcription in web_server.py.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
    ):
        self.model_size = model_size
        self.device = device
        self._model = None
        self._init_model()

    def _init_model(self):
        """Initialize faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            compute_type = "float16" if self.device == "cuda" else "float32"
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )
            logger.info(f"Loaded faster-whisper model: {self.model_size}")
        except ImportError:
            logger.warning("faster-whisper not available, trying openai-whisper")
            try:
                import whisper
                self._model = whisper.load_model(self.model_size, device=self.device)
                self._use_openai_whisper = True
                logger.info(f"Loaded openai-whisper model: {self.model_size}")
            except ImportError:
                raise ImportError("Neither faster-whisper nor openai-whisper installed")
        else:
            self._use_openai_whisper = False

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file (wav, mp3, webm, etc.)

        Returns:
            Transcribed text
        """
        try:
            if self._use_openai_whisper:
                # OpenAI Whisper
                result = self._model.transcribe(
                    audio_path,
                    language="en",
                    fp16=(self.device == "cuda"),
                )
                return result.get("text", "")
            else:
                # Faster Whisper
                segments, _ = self._model.transcribe(
                    audio_path,
                    language="en",
                    beam_size=5,
                )
                text = " ".join(segment.text for segment in segments)
                return text.strip()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""


# =============================================================================
# STREAMING VARIANT
# =============================================================================

class StreamingSpeechListener(SpeechListener):
    """
    Streaming variant that yields transcriptions as a generator.

    Useful for async processing patterns.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._transcription_queue: queue.Queue = queue.Queue()

        # Override callback to put in queue
        self.on_transcription = self._queue_transcription

    def _queue_transcription(self, text: str):
        """Put transcription in queue."""
        self._transcription_queue.put(text)

    def transcriptions(self) -> Generator[str, None, None]:
        """
        Generator that yields transcriptions.

        Yields:
            Transcribed text strings
        """
        while self._running:
            try:
                text = self._transcription_queue.get(timeout=0.5)
                yield text
            except queue.Empty:
                continue


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    def on_text(text: str):
        print(f"\n>>> You said: {text}\n")

    print("Starting speech listener... (Ctrl+C to stop)")
    print("Speak into your microphone!\n")

    listener = SpeechListener(
        model_size="base",
        on_transcription=on_text,
        device="cuda",
    )

    try:
        listener.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()
