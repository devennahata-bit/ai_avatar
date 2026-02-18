"""
Face Module - LiveAvatar Lip-Sync
=================================
Handles face rendering with lip-sync using LiveAvatar (Alibaba-Quark).

Features:
- Load and run LiveAvatar inference (WanS2V 14B diffusion model)
- Audio buffering for utterance-based processing
- Frame queue for display
- Reference image management
"""

import threading
import queue
import time
import sys
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Add vendor path for LiveAvatar
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
VENDOR_DIR = PROJECT_ROOT / "vendor"
LIVEAVATAR_DIR = VENDOR_DIR / "LiveAvatar"

if LIVEAVATAR_DIR.exists():
    sys.path.insert(0, str(LIVEAVATAR_DIR))


class FaceRenderer:
    """
    Real-time face renderer using LiveAvatar for lip-sync.

    Consumes audio chunks from a queue and produces video frames
    with synchronized lip movements.

    Key difference from MuseTalk: LiveAvatar processes complete utterances
    rather than streaming chunks. Audio is buffered until the TTS signals
    end-of-utterance (via None marker), then batch processed.
    """

    def __init__(
        self,
        reference_image_path: str,
        models_dir: str,
        audio_queue: Optional[queue.Queue] = None,
        frame_queue: Optional[queue.Queue] = None,
        fps: int = 24,
        output_width: int = 512,
        output_height: int = 512,
        device: str = "cuda",
        use_fp16: bool = True,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
        reference_video_path: Optional[str] = None,
        # LiveAvatar inference settings
        sampling_steps: int = 10,
        infer_frames: int = 81,
        guide_scale: float = 5.0,
        use_fp8: bool = True,
        multi_gpu: bool = False,  # Enable multi-GPU model splitting
    ):
        """
        Initialize the face renderer.

        Args:
            reference_image_path: Path to reference face image
            models_dir: Path to models directory
            audio_queue: Queue to receive audio chunks from voice module
            frame_queue: Queue to output rendered frames
            fps: Output frame rate
            output_width: Output frame width
            output_height: Output frame height
            device: Device for inference (cuda/cpu)
            use_fp16: Use half precision (mapped to FP8 for LiveAvatar)
            on_frame: Callback for each rendered frame
            reference_video_path: Path to reference video (for idle animation)
            sampling_steps: Diffusion sampling steps (lower = faster)
            infer_frames: Frames per inference batch
            guide_scale: Classifier-free guidance scale
            use_fp8: Use FP8 quantization (requires 48GB+ VRAM)
        """
        self.reference_image_path = Path(reference_image_path)
        self.reference_video_path = Path(reference_video_path) if reference_video_path else None
        self.models_dir = Path(models_dir)
        self.fps = fps
        self.output_width = output_width
        self.output_height = output_height
        self.device = device
        self.use_fp16 = use_fp16
        self.use_fp8 = use_fp8
        self.on_frame = on_frame
        self.sampling_steps = sampling_steps
        self.infer_frames = infer_frames
        self.guide_scale = guide_scale
        self.multi_gpu = multi_gpu

        # Queues
        self.audio_queue = audio_queue or queue.Queue(maxsize=100)
        self.frame_queue = frame_queue or queue.Queue(maxsize=50)

        # State
        self._running = False
        self._model_loaded = False
        self._render_thread: Optional[threading.Thread] = None
        self._liveavatar_ready = False

        # LiveAvatar components (lazy loaded)
        self._pipeline = None
        self._device = None

        # Reference face data
        self._reference_face = None
        self._idle_frames = []
        self._idle_frame_idx = 0

        # Video frames (if using video input for idle animation)
        self._video_frames = []
        self._video_frame_idx = 0
        self._video_fps = fps

        # Audio buffer for accumulating chunks
        self._audio_buffer = bytearray()
        self._audio_sample_rate = 24000  # Expected from TTS
        self._temp_dir = tempfile.mkdtemp(prefix="liveavatar_")

    def load_models(self):
        """Load all required models for LiveAvatar inference."""
        if self._model_loaded:
            logger.info("Models already loaded")
            return

        logger.info("Loading LiveAvatar models...")

        try:
            import torch
            import cv2

            # Verify CUDA
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"

            self._device = torch.device(self.device)

            # Load reference media (video or image)
            self._load_reference_media()

            # Try to load LiveAvatar
            if self._try_load_liveavatar():
                logger.info("LiveAvatar models loaded successfully")
            else:
                logger.warning("LiveAvatar not available, using fallback mode")
                self._setup_fallback_mode()

            self._model_loaded = True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self._setup_fallback_mode()
            self._model_loaded = True

    def _load_reference_media(self):
        """Load reference video or image."""
        import cv2

        # Try video first (for idle animation)
        if self.reference_video_path and self.reference_video_path.exists():
            if self._load_reference_video():
                return

        # Fall back to image
        self._load_reference_image()

    def _load_reference_video(self) -> bool:
        """Load and cache video frames for idle animation."""
        import cv2

        video_path = str(self.reference_video_path)
        logger.info(f"Loading reference video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return False

        # Get video properties
        self._video_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {total_frames} frames at {self._video_fps} FPS")

        # Load frames for idle animation
        self._video_frames = []
        max_frames = min(total_frames, 300)  # Limit to ~10-12 seconds

        while len(self._video_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to output dimensions
            frame = cv2.resize(
                frame,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_LINEAR
            )
            self._video_frames.append(frame)

        cap.release()

        if len(self._video_frames) > 0:
            self._reference_face = self._video_frames[0]
            logger.info(f"Loaded {len(self._video_frames)} video frames for idle animation")
            return True

        return False

    def _load_reference_image(self):
        """Load and preprocess reference face image."""
        import cv2

        if not self.reference_image_path.exists():
            logger.warning(f"Reference image not found: {self.reference_image_path}")
            # Create placeholder
            self._reference_face = np.zeros(
                (self.output_height, self.output_width, 3),
                dtype=np.uint8
            )
            return

        # Load image
        image = cv2.imread(str(self.reference_image_path))
        if image is None:
            logger.error(f"Failed to load reference image: {self.reference_image_path}")
            self._reference_face = np.zeros(
                (self.output_height, self.output_width, 3),
                dtype=np.uint8
            )
            return

        # Resize to output dimensions
        self._reference_face = cv2.resize(
            image,
            (self.output_width, self.output_height),
            interpolation=cv2.INTER_LANCZOS4
        )

        logger.info(f"Reference image loaded: {self.reference_image_path}")

    def _get_idle_frame(self) -> np.ndarray:
        """Get the next frame for idle animation."""
        if self._video_frames:
            frame = self._video_frames[self._video_frame_idx].copy()
            self._video_frame_idx = (self._video_frame_idx + 1) % len(self._video_frames)
            return frame
        return self._reference_face.copy()

    def _try_load_liveavatar(self) -> bool:
        """Try to load LiveAvatar models."""
        try:
            import torch

            # Check for model directories
            base_model_dir = self.models_dir / "Wan2.1-S2V-14B"
            lora_dir = self.models_dir / "Live-Avatar"

            if not base_model_dir.exists():
                logger.warning(f"LiveAvatar base model not found at {base_model_dir}")
                return False

            if not lora_dir.exists():
                logger.warning(f"LiveAvatar LoRA weights not found at {lora_dir}")
                return False

            # Try to import LiveAvatar modules
            try:
                from liveavatar.models import WanS2VPipeline
                from peft import PeftModel

                logger.info("Loading WanS2V pipeline...")

                # Determine dtype based on settings
                if self.use_fp8 and hasattr(torch, 'float8_e4m3fn'):
                    dtype = torch.float8_e4m3fn
                    logger.info("Using FP8 quantization")
                elif self.use_fp16:
                    dtype = torch.float16
                    logger.info("Using FP16 precision")
                else:
                    dtype = torch.float32
                    logger.info("Using FP32 precision")

                # Load the pipeline
                if self.multi_gpu:
                    # Multi-GPU: automatically split model across GPUs
                    logger.info("Using multi-GPU mode (device_map=auto)")
                    self._pipeline = WanS2VPipeline.from_pretrained(
                        str(base_model_dir),
                        torch_dtype=dtype,
                        device_map="auto",  # Automatically distribute across GPUs
                    )
                else:
                    # Single GPU
                    self._pipeline = WanS2VPipeline.from_pretrained(
                        str(base_model_dir),
                        torch_dtype=dtype,
                    )

                # Load LoRA weights
                logger.info("Loading LoRA weights...")
                self._pipeline.load_lora_weights(str(lora_dir))

                # Move to device (skip if using device_map)
                if not self.multi_gpu:
                    self._pipeline = self._pipeline.to(self._device)

                self._liveavatar_ready = True
                return True

            except ImportError as e:
                logger.warning(f"LiveAvatar import failed: {e}")
                logger.info("Trying alternative loading method...")

                # Alternative: Try loading via diffusers
                try:
                    from diffusers import DiffusionPipeline

                    if self.multi_gpu:
                        self._pipeline = DiffusionPipeline.from_pretrained(
                            str(base_model_dir),
                            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                            device_map="auto",
                        )
                    else:
                        self._pipeline = DiffusionPipeline.from_pretrained(
                            str(base_model_dir),
                            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                        )
                        self._pipeline = self._pipeline.to(self._device)
                    self._liveavatar_ready = True
                    return True

                except Exception as e2:
                    logger.error(f"Alternative loading also failed: {e2}")
                    return False

        except Exception as e:
            logger.error(f"LiveAvatar loading failed: {e}")
            return False

    def _setup_fallback_mode(self):
        """Setup fallback mode when LiveAvatar is not available."""
        logger.info("Setting up fallback mode (static face / video loop)")
        self._liveavatar_ready = False

    def _flush_audio_to_wav(self) -> Optional[str]:
        """Write buffered audio to WAV file and return path."""
        if not self._audio_buffer:
            return None

        try:
            # Generate unique filename
            wav_path = os.path.join(
                self._temp_dir,
                f"audio_{time.time_ns()}.wav"
            )

            # Write WAV file
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self._audio_sample_rate)
                wf.writeframes(bytes(self._audio_buffer))

            # Clear buffer
            self._audio_buffer.clear()

            logger.debug(f"Wrote audio to {wav_path}")
            return wav_path

        except Exception as e:
            logger.error(f"Failed to write audio WAV: {e}")
            self._audio_buffer.clear()
            return None

    def _generate_video(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Generate video frames from audio using LiveAvatar.

        Args:
            audio_path: Path to WAV file

        Returns:
            numpy array of shape (N_frames, H, W, 3) with uint8 BGR values
        """
        if not self._liveavatar_ready or self._pipeline is None:
            return None

        try:
            import torch

            logger.info(f"Generating video from {audio_path}")

            # Generate video tensor
            video_tensor = self._pipeline.generate(
                input_prompt="",
                ref_image_path=str(self.reference_image_path),
                audio_path=audio_path,
                num_repeat=1,
                infer_frames=self.infer_frames,
                sampling_steps=self.sampling_steps,
                guide_scale=self.guide_scale,
            )

            # Convert tensor to frames
            frames = self._tensor_to_frames(video_tensor)
            logger.info(f"Generated {len(frames)} frames")

            return frames

        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None

    def _tensor_to_frames(self, tensor) -> np.ndarray:
        """
        Convert LiveAvatar output tensor to numpy frames.

        Input tensor shape: (3, N_frames, H, W) or (batch, 3, N_frames, H, W)
        Input values: [0, 1] float
        Output shape: (N_frames, H, W, 3)
        Output values: [0, 255] uint8 BGR
        """
        import torch

        # Handle batch dimension
        if tensor.dim() == 5:
            tensor = tensor[0]  # Take first batch

        # tensor shape: (3, N, H, W) -> (N, H, W, 3)
        video = tensor.permute(1, 2, 3, 0)

        # Convert to numpy uint8
        video = (video * 255).clamp(0, 255).byte()
        video = video.cpu().numpy()

        # Convert RGB to BGR for OpenCV
        video = video[..., ::-1].copy()

        return video

    def _render_loop(self):
        """Main rendering loop - consumes audio and produces frames."""
        logger.info("Render loop started")

        frame_duration = 1.0 / self.fps

        while self._running:
            try:
                # Try to get audio chunk with timeout
                try:
                    audio_chunk = self.audio_queue.get(timeout=frame_duration)
                except queue.Empty:
                    # No audio - output idle frame
                    self._output_frame(self._get_idle_frame())
                    continue

                if audio_chunk is None:
                    # End of utterance signal - process buffered audio
                    if self._audio_buffer:
                        wav_path = self._flush_audio_to_wav()
                        if wav_path:
                            # Generate and output video frames
                            frames = self._generate_video(wav_path)

                            if frames is not None and len(frames) > 0:
                                self._output_frames_at_rate(frames)
                            else:
                                # Generation failed - output idle frames
                                for _ in range(10):
                                    self._output_frame(self._get_idle_frame())
                                    time.sleep(frame_duration)

                            # Clean up temp file
                            try:
                                os.unlink(wav_path)
                            except Exception:
                                pass
                    continue

                # Accumulate audio chunk
                self._audio_buffer.extend(audio_chunk)

                # Output idle frame while buffering
                self._output_frame(self._get_idle_frame())

            except Exception as e:
                logger.error(f"Render loop error: {e}")
                time.sleep(0.1)

        logger.info("Render loop stopped")

    def _output_frames_at_rate(self, frames: np.ndarray):
        """Output frames at the correct FPS."""
        import cv2

        frame_duration = 1.0 / self.fps

        for frame in frames:
            if not self._running:
                break

            # Resize if needed
            if frame.shape[:2] != (self.output_height, self.output_width):
                frame = cv2.resize(
                    frame,
                    (self.output_width, self.output_height),
                    interpolation=cv2.INTER_LINEAR
                )

            self._output_frame(frame)
            time.sleep(frame_duration)

    def _output_frame(self, frame: np.ndarray):
        """Output a rendered frame."""
        # Push to frame queue
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest frame
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

        # Callback
        if self.on_frame:
            self.on_frame(frame)

    def start(self):
        """Start the face renderer."""
        if self._running:
            logger.warning("Renderer already running")
            return

        # Load models if not loaded
        if not self._model_loaded:
            self.load_models()

        self._running = True

        # Start render thread
        self._render_thread = threading.Thread(
            target=self._render_loop,
            name="FaceRenderer",
            daemon=True,
        )
        self._render_thread.start()

        logger.info("Face renderer started")

    def stop(self):
        """Stop the face renderer."""
        self._running = False

        if self._render_thread:
            self._render_thread.join(timeout=2.0)
            self._render_thread = None

        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass

        logger.info("Face renderer stopped")

    def push_audio(self, audio_bytes: bytes):
        """Push audio chunk for rendering."""
        if audio_bytes:
            try:
                self.audio_queue.put_nowait(audio_bytes)
            except queue.Full:
                logger.warning("Audio queue full")

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get the next rendered frame."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        """Check if renderer is running."""
        return self._running


# =============================================================================
# DISPLAY WINDOW (for testing/preview)
# =============================================================================

class DisplayWindow:
    """OpenCV window for displaying rendered frames."""

    def __init__(
        self,
        window_name: str = "AI Avatar",
        width: int = 512,
        height: int = 512,
    ):
        self.window_name = window_name
        self.width = width
        self.height = height
        self._running = False

    def start(self, frame_queue: queue.Queue):
        """Start display loop."""
        import cv2

        self._running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

        while self._running:
            try:
                frame = frame_queue.get(timeout=0.1)
                if frame is not None:
                    cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Display error: {e}")
                break

        cv2.destroyAllWindows()
        self._running = False

    def stop(self):
        """Stop display."""
        self._running = False


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    models_dir = PROJECT_ROOT / "models"
    reference_image = PROJECT_ROOT / "assets" / "avatar_face.png"

    # Create test reference image if not exists
    if not reference_image.exists():
        import cv2
        reference_image.parent.mkdir(parents=True, exist_ok=True)
        # Create a simple test image
        test_img = np.zeros((512, 512, 3), dtype=np.uint8)
        test_img[:] = (100, 100, 100)  # Gray background
        # Draw a simple face placeholder
        cv2.circle(test_img, (256, 200), 80, (200, 180, 160), -1)  # Face
        cv2.circle(test_img, (220, 180), 15, (50, 50, 50), -1)  # Left eye
        cv2.circle(test_img, (292, 180), 15, (50, 50, 50), -1)  # Right eye
        cv2.ellipse(test_img, (256, 240), (30, 15), 0, 0, 180, (50, 50, 50), 2)  # Mouth
        cv2.imwrite(str(reference_image), test_img)
        print(f"Created test reference image: {reference_image}")

    print("Initializing face renderer...")

    audio_queue = queue.Queue()
    frame_queue = queue.Queue()

    renderer = FaceRenderer(
        reference_image_path=str(reference_image),
        models_dir=str(models_dir),
        audio_queue=audio_queue,
        frame_queue=frame_queue,
        fps=24,
    )

    print("Starting renderer...")
    renderer.start()

    print("Starting display window (press 'q' to quit)...")
    display = DisplayWindow()

    try:
        display.start(frame_queue)
    except KeyboardInterrupt:
        pass
    finally:
        renderer.stop()
        print("Done")
