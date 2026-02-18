#!/usr/bin/env python3
"""
Real-Time AI Avatar - Main Application
======================================
Orchestrates the avatar pipeline:
  Mic -> Whisper STT -> Llama 3 -> ElevenLabs TTS -> LiveAvatar Face

Producer-Consumer Architecture:
- Listener produces transcriptions
- Brain consumes transcriptions, produces responses
- Voice consumes responses, produces audio chunks
- Face consumes audio chunks, produces video frames
- Display consumes video frames

Usage:
    python main.py
    python main.py --debug
    python main.py --provider ollama --model llama3:8b
"""

import os
import sys
import warnings

# Suppress torch FutureWarnings from tortoise-tts
warnings.filterwarnings("ignore", category=FutureWarning, module="tortoise")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

import threading
import queue
import time
import signal
import argparse
from pathlib import Path
from typing import Optional
import logging

# Rich console for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich import print as rprint
except ImportError:
    print("Installing rich...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich import print as rprint

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import configuration
from config import (
    api_config,
    voice_settings,
    llm_config,
    avatar_personality,
    whisper_config,
    liveavatar_config,
    audio_config,
    app_config,
    validate_config,
    print_config_summary,
    PROJECT_ROOT,
    MODELS_DIR,
)

# Import modules
from modules.listener import SpeechListener
from modules.brain import LLMBrain
from modules.voice import VoiceSynthesizer, LocalVoiceSynthesizer, TortoiseTTSSynthesizer, RealtimeTTSSynthesizer, PiperTTSSynthesizer
from modules.face import FaceRenderer, DisplayWindow

console = Console()
logger = logging.getLogger(__name__)


class AvatarPipeline:
    """
    Main avatar pipeline orchestrator.

    Manages the flow:
    Speech Input -> STT -> LLM -> TTS -> Lip-Sync -> Display
    """

    def __init__(
        self,
        # Override configs via args
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        whisper_model: Optional[str] = None,
        voice_id: Optional[str] = None,
        debug: bool = False,
    ):
        """Initialize the avatar pipeline."""
        self.debug = debug or app_config.debug

        # Setup logging
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Override configs
        provider = llm_provider or llm_config.provider
        model = llm_model or getattr(llm_config, f"{provider}_model")
        whisper_size = whisper_model or whisper_config.model_size
        voice = voice_id or api_config.elevenlabs_voice_id

        # State
        self._running = False
        self._paused = False
        self._speaking = False

        # Queues for inter-module communication
        self.audio_queue = queue.Queue(maxsize=app_config.audio_queue_size)
        self.frame_queue = queue.Queue(maxsize=app_config.frame_queue_size)

        # Current conversation state
        self._last_transcription = ""
        self._last_response = ""
        self._status = "Initializing..."

        # Initialize components
        console.print("[cyan]Initializing avatar components...[/cyan]")

        # 1. Speech Listener (STT)
        self.listener = SpeechListener(
            model_size=whisper_size,
            sample_rate=whisper_config.sample_rate,
            device=whisper_config.device,
            input_device=audio_config.input_device,
            vad_aggressiveness=whisper_config.vad_aggressiveness,
            silence_threshold=whisper_config.silence_threshold,
            min_speech_duration=whisper_config.min_speech_duration,
            on_transcription=self._on_transcription,
        )
        console.print("  [green]✓[/green] Speech listener initialized")

        # 2. LLM Brain
        self.brain = LLMBrain(
            provider=provider,
            model=model,
            system_prompt=avatar_personality.system_prompt,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            max_history_turns=avatar_personality.max_history_turns,
            stream=llm_config.stream,
            groq_api_key=api_config.groq_api_key,
            openai_api_key=api_config.openai_api_key,
            ollama_host=api_config.ollama_host,
            on_response_start=self._on_response_start,
            on_response_chunk=self._on_response_chunk,
            on_response_complete=self._on_response_complete,
        )
        console.print("  [green]✓[/green] LLM brain initialized")

        # 3. Voice Synthesizer (TTS)
        tts_provider = voice_settings.tts_provider

        def _init_local_tts():
            self.voice = LocalVoiceSynthesizer(
                rate=voice_settings.local_rate,
                voice_index=voice_settings.local_voice_index,
                output_device=audio_config.output_device,
                audio_queue=self.audio_queue,
                on_synthesis_start=self._on_synthesis_start,
                on_synthesis_complete=self._on_synthesis_complete,
            )
            console.print("  [green]✓[/green] Local TTS initialized (pyttsx3)")

        def _init_elevenlabs_tts():
            if not api_config.elevenlabs_api_key:
                raise RuntimeError("ELEVENLABS_API_KEY is not configured")
            self.voice = VoiceSynthesizer(
                api_key=api_config.elevenlabs_api_key,
                voice_id=voice,
                model_id=voice_settings.model_id,
                stability=voice_settings.stability,
                similarity_boost=voice_settings.similarity_boost,
                style=voice_settings.style,
                use_speaker_boost=voice_settings.use_speaker_boost,
                output_format=voice_settings.output_format,
                audio_queue=self.audio_queue,
                output_device=audio_config.output_device,
                on_synthesis_start=self._on_synthesis_start,
                on_synthesis_complete=self._on_synthesis_complete,
            )
            console.print("  [green]✓[/green] ElevenLabs TTS initialized")

        try:
            if tts_provider == "piper":
                self.voice = PiperTTSSynthesizer(
                    model=voice_settings.piper_model,
                    model_path=voice_settings.piper_model_path,
                    output_device=audio_config.output_device,
                    audio_queue=self.audio_queue,
                    on_synthesis_start=self._on_synthesis_start,
                    on_synthesis_complete=self._on_synthesis_complete,
                )
                console.print(
                    f"  [green]✓[/green] Piper TTS initialized "
                    f"(model={voice_settings.piper_model})"
                )
            elif tts_provider == "kokoro":
                self.voice = RealtimeTTSSynthesizer(
                    voice=voice_settings.kokoro_voice,
                    speed=voice_settings.kokoro_speed,
                    output_device=audio_config.output_device,
                    audio_queue=self.audio_queue,
                    on_synthesis_start=self._on_synthesis_start,
                    on_synthesis_complete=self._on_synthesis_complete,
                )
                console.print(
                    f"  [green]✓[/green] Kokoro TTS initialized "
                    f"(voice={voice_settings.kokoro_voice}, speed={voice_settings.kokoro_speed}x)"
                )
            elif tts_provider == "tortoise":
                self.voice = TortoiseTTSSynthesizer(
                    voice=voice_settings.tortoise_voice,
                    preset=voice_settings.tortoise_preset,
                    samples_dir=str(voice_settings.tortoise_samples_dir),
                    use_deepspeed=voice_settings.tortoise_use_deepspeed,
                    kv_cache=voice_settings.tortoise_kv_cache,
                    half=voice_settings.tortoise_half,
                    device=liveavatar_config.device,
                    output_device=audio_config.output_device,
                    audio_queue=self.audio_queue,
                    on_synthesis_start=self._on_synthesis_start,
                    on_synthesis_complete=self._on_synthesis_complete,
                )
                console.print(
                    f"  [green]✓[/green] Tortoise-TTS initialized "
                    f"(voice={voice_settings.tortoise_voice}, preset={voice_settings.tortoise_preset})"
                )
            elif tts_provider == "elevenlabs":
                _init_elevenlabs_tts()
            elif tts_provider == "local":
                _init_local_tts()
            else:
                raise RuntimeError(f"Unsupported TTS provider: {tts_provider}")
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] TTS init failed for provider '{tts_provider}': {e}")
            # Try fallback chain: piper -> elevenlabs -> local
            fallback_success = False

            if tts_provider != "piper":
                try:
                    self.voice = PiperTTSSynthesizer(
                        model="en_US-lessac-medium",
                        output_device=audio_config.output_device,
                        audio_queue=self.audio_queue,
                        on_synthesis_start=self._on_synthesis_start,
                        on_synthesis_complete=self._on_synthesis_complete,
                    )
                    console.print("  [green]✓[/green] Piper TTS fallback initialized")
                    fallback_success = True
                except Exception as piper_err:
                    console.print(f"  [yellow]⚠[/yellow] Piper fallback unavailable: {piper_err}")

            if not fallback_success and tts_provider != "elevenlabs":
                try:
                    _init_elevenlabs_tts()
                    fallback_success = True
                except Exception as cloud_err:
                    console.print(f"  [yellow]⚠[/yellow] ElevenLabs fallback unavailable: {cloud_err}")

            if not fallback_success:
                _init_local_tts()

        # 4. Face Renderer (Lip-Sync)
        # Check for video file first, fallback to image
        video_path = avatar_personality.reference_media_path
        if not video_path.exists():
            video_path = None  # Will use image fallback

        self.face = FaceRenderer(
            reference_image_path=str(avatar_personality.reference_image_path),
            reference_video_path=str(video_path) if video_path else None,
            models_dir=str(MODELS_DIR),
            audio_queue=self.audio_queue,
            frame_queue=self.frame_queue,
            fps=liveavatar_config.fps,
            output_width=liveavatar_config.output_width,
            output_height=liveavatar_config.output_height,
            device=liveavatar_config.device,
            use_fp16=liveavatar_config.use_fp16,
            use_fp8=liveavatar_config.use_fp8,
            sampling_steps=liveavatar_config.sampling_steps,
            infer_frames=liveavatar_config.infer_frames,
            guide_scale=liveavatar_config.guide_scale,
            multi_gpu=liveavatar_config.multi_gpu,
        )
        console.print("  [green]✓[/green] Face renderer initialized")

        # 5. Display Window
        if app_config.show_preview:
            self.display = DisplayWindow(
                window_name=app_config.window_title,
                width=liveavatar_config.output_width,
                height=liveavatar_config.output_height,
            )
            console.print("  [green]✓[/green] Display window ready")
        else:
            self.display = None

        self._status = "Ready"

    def _on_transcription(self, text: str):
        """Handle transcription from speech listener."""
        self._last_transcription = text
        self._status = "Processing..."

        logger.info(f"User: {text}")
        console.print(f"\n[bold blue]You:[/bold blue] {text}")

        # Pause listening while processing
        self.listener.pause()

        # Generate response in a separate thread
        threading.Thread(
            target=self._process_input,
            args=(text,),
            daemon=True,
        ).start()

    def _process_input(self, text: str):
        """Process user input through the pipeline."""
        try:
            # Generate LLM response
            response = self.brain.generate(text)

            # Synthesize and play audio
            if self.voice and response:
                self.voice.synthesize_to_queue(response)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._status = f"Error: {e}"

        finally:
            # Resume listening after a short delay
            time.sleep(0.5)
            self.listener.resume()
            self._status = "Listening..."

    def _on_response_start(self):
        """Called when LLM response starts."""
        self._speaking = True
        self._status = "Thinking..."
        console.print(f"\n[bold green]{avatar_personality.name}:[/bold green] ", end="")

    def _on_response_chunk(self, chunk: str):
        """Called for each LLM response chunk."""
        console.print(chunk, end="")

    def _on_response_complete(self, response: str):
        """Called when LLM response is complete."""
        self._last_response = response
        console.print()  # New line
        self._status = "Speaking..."

    def _on_synthesis_start(self):
        """Called when TTS synthesis starts."""
        self._speaking = True

    def _on_synthesis_complete(self):
        """Called when TTS synthesis is complete."""
        self._speaking = False
        self._status = "Listening..."

    def start(self):
        """Start the avatar pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return

        self._running = True
        self._status = "Starting..."

        # Start components
        self.face.start()
        self.listener.start()

        # Start display in main thread (OpenCV requirement)
        if self.display:
            self._status = "Listening..."
            console.print(Panel(
                f"[bold green]{avatar_personality.name} is ready![/bold green]\n\n"
                "Speak into your microphone to chat.\n"
                "Press [bold]'q'[/bold] or [bold]ESC[/bold] in the avatar window to quit.",
                title="🎭 AI Avatar",
                border_style="green",
            ))

            # Display loop (blocks until window closed)
            self.display.start(self.frame_queue)

            # Window closed - stop everything
            self.stop()
        else:
            # No display - run in headless mode
            self._status = "Listening..."
            console.print(Panel(
                f"[bold green]{avatar_personality.name} is ready![/bold green]\n\n"
                "Speak into your microphone to chat.\n"
                "Press [bold]Ctrl+C[/bold] to quit.",
                title="🎭 AI Avatar (Headless Mode)",
                border_style="green",
            ))

            # Keep running until interrupted
            try:
                while self._running:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

            self.stop()

    def stop(self):
        """Stop the avatar pipeline."""
        if not self._running:
            return

        self._running = False
        self._status = "Stopping..."

        console.print("\n[yellow]Shutting down...[/yellow]")

        # Stop components in reverse order
        self.listener.stop()
        self.face.stop()

        if self.display:
            self.display.stop()

        console.print("[green]Goodbye![/green]")

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running


def check_prerequisites():
    """Check if prerequisites are met before starting."""
    issues = validate_config()

    if issues:
        console.print(Panel(
            "\n".join(f"• {issue}" for issue in issues),
            title="⚠️ Configuration Issues",
            border_style="yellow",
        ))

    # Check if models are downloaded
    if not MODELS_DIR.exists() or not any(MODELS_DIR.iterdir()):
        console.print(Panel(
            "[bold red]Model weights not found![/bold red]\n\n"
            "Please run the setup wizard first:\n"
            "[cyan]python setup_wizard.py[/cyan]",
            title="⚠️ Setup Required",
            border_style="red",
        ))
        return False

    # Check for reference media (video or image)
    has_video = avatar_personality.reference_media_path.exists()
    has_image = avatar_personality.reference_image_path.exists()
    if has_video:
        console.print(f"[green]✓[/green] Using video avatar: {avatar_personality.reference_media_path.name}")
    elif has_image:
        console.print(f"[green]✓[/green] Using image avatar: {avatar_personality.reference_image_path.name}")
    else:
        console.print(
            f"[yellow]⚠️ No avatar media found[/yellow]\n"
            "   Add avatar.mp4 or avatar_face.png to the assets/ folder."
        )

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time AI Avatar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          # Run with default settings
    python main.py --debug                  # Run with debug logging
    python main.py --provider ollama        # Use local Ollama
    python main.py --provider groq          # Use Groq API (fast)
    python main.py --no-display             # Headless mode (no window)
        """,
    )

    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["ollama", "groq", "openai"],
        default=None,
        help="LLM provider",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model name",
    )
    parser.add_argument(
        "--whisper", "-w",
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,
        help="Whisper model size",
    )
    parser.add_argument(
        "--voice", "-v",
        default=None,
        help="ElevenLabs voice ID",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display window",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show current configuration and exit",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run setup wizard",
    )

    args = parser.parse_args()

    # Show config
    if args.config:
        print_config_summary()
        return

    # Run setup wizard
    if args.setup:
        import setup_wizard
        setup_wizard.main()
        return

    # Welcome banner
    console.print(Panel(
        "[bold cyan]Real-Time AI Avatar[/bold cyan]\n"
        "[dim]Whisper → Llama 3 → ElevenLabs → LiveAvatar[/dim]",
        border_style="cyan",
    ))

    # Check prerequisites
    if not check_prerequisites():
        console.print("\n[red]Please resolve the issues above before running.[/red]")
        sys.exit(1)

    # Override display setting
    if args.no_display:
        app_config.show_preview = False

    # Create and run pipeline
    try:
        pipeline = AvatarPipeline(
            llm_provider=args.provider,
            llm_model=args.model,
            whisper_model=args.whisper,
            voice_id=args.voice,
            debug=args.debug,
        )

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            console.print("\n[yellow]Interrupt received...[/yellow]")
            pipeline.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Start the pipeline
        pipeline.start()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
