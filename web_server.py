#!/usr/bin/env python3
"""
Web Streaming Server for AI Avatar
===================================
Allows accessing the avatar via web browser when running on a cloud VM.

Features:
- MJPEG video stream (works in any browser)
- WebSocket for audio streaming
- Text input fallback (no microphone needed)
- Real-time chat interface

Usage:
    python web_server.py              # Start on port 8080
    python web_server.py --port 5000  # Custom port

Then open http://<server-ip>:8080 in your browser
"""

import os
import sys
import json
import time
import queue
import threading
import argparse
import logging
import base64
import tempfile
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

# Load environment
from dotenv import load_dotenv
load_dotenv()

try:
    from flask import Flask, Response, render_template_string, request, jsonify
    from flask_socketio import SocketIO, emit
    _WEB_DEPS_ERROR = None
except ImportError as exc:
    Flask = None
    Response = None
    render_template_string = None
    request = None
    jsonify = None
    SocketIO = None
    emit = None
    _WEB_DEPS_ERROR = exc

# Import config
from config import (
    api_config, voice_settings, llm_config, avatar_personality,
    liveavatar_config, app_config, whisper_config, MODELS_DIR
)

# Import modules
from modules.brain import LLMBrain
from modules.voice import RealtimeTTSSynthesizer, LocalVoiceSynthesizer, PiperTTSSynthesizer, GTTSSynthesizer
from modules.face import FaceRenderer
from modules.listener import SpeechRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _NoopFlaskApp:
    """Fallback app object so module imports succeed when Flask is missing."""
    config = {}

    @staticmethod
    def route(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator


class _NoopSocketIO:
    """Fallback socket object so decorators do not fail at import time."""

    @staticmethod
    def on(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator

    @staticmethod
    def emit(*_args, **_kwargs):
        raise RuntimeError("Web server dependencies are not installed")

    @staticmethod
    def run(*_args, **_kwargs):
        raise RuntimeError("Web server dependencies are not installed")


if _WEB_DEPS_ERROR is None:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "ai-avatar-secret"
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
else:
    app = _NoopFlaskApp()
    socketio = _NoopSocketIO()

# Global state
avatar_pipeline = None
frame_queue = queue.Queue(maxsize=30)
is_processing = False


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Avatar</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 20px;
        }
        .container {
            display: flex;
            gap: 20px;
            max-width: 1200px;
            width: 100%;
        }
        .video-container {
            flex: 1;
            background: #0f0f23;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        #video-feed {
            width: 100%;
            height: auto;
            display: block;
        }
        .chat-container {
            width: 400px;
            background: #16213e;
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .chat-header {
            padding: 15px;
            background: #1a1a2e;
            border-radius: 12px 12px 0 0;
            font-weight: bold;
            color: #00d4ff;
        }
        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            max-height: 400px;
            min-height: 400px;
        }
        .message {
            margin-bottom: 12px;
            padding: 10px 14px;
            border-radius: 8px;
            max-width: 85%;
        }
        .message.user {
            background: #00d4ff;
            color: #000;
            margin-left: auto;
        }
        .message.assistant {
            background: #2d3748;
            color: #fff;
        }
        .message .name {
            font-size: 11px;
            opacity: 0.7;
            margin-bottom: 4px;
        }
        .chat-input {
            padding: 15px;
            border-top: 1px solid #2d3748;
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: #1a1a2e;
            color: #fff;
            font-size: 14px;
        }
        input[type="text"]:focus {
            outline: 2px solid #00d4ff;
        }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            background: #00d4ff;
            color: #000;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover {
            background: #00b8e6;
        }
        button:disabled {
            background: #4a5568;
            cursor: not-allowed;
        }
        #mic-btn {
            background: #4a5568;
            padding: 12px 16px;
        }
        #mic-btn.recording {
            background: #fc8181;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .status {
            margin-top: 15px;
            padding: 10px;
            background: #2d3748;
            border-radius: 8px;
            font-size: 12px;
            text-align: center;
        }
        .status.connected { color: #48bb78; }
        .status.disconnected { color: #fc8181; }
        .status.processing { color: #f6e05e; }
    </style>
</head>
<body>
    <h1>AI Avatar</h1>

    <div class="container">
        <div class="video-container">
            <img id="video-feed" src="/video_feed" alt="Avatar Video Feed">
        </div>

        <div class="chat-container">
            <div class="chat-header">Chat with {{ avatar_name }}</div>
            <div class="chat-messages" id="chat-messages"></div>
            <div class="chat-input">
                <button onclick="toggleMic()" id="mic-btn" title="Hold to speak">Mic</button>
                <input type="text" id="message-input" placeholder="Type or use mic..."
                       onkeypress="if(event.key==='Enter')sendMessage()">
                <button onclick="sendMessage()" id="send-btn">Send</button>
            </div>
            <div class="status" id="status">Connecting...</div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        const socket = io();
        const chatMessages = document.getElementById('chat-messages');
        const messageInput = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');
        const micBtn = document.getElementById('mic-btn');
        const status = document.getElementById('status');

        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        socket.on('connect', () => {
            status.textContent = 'Connected';
            status.className = 'status connected';
        });

        socket.on('disconnect', () => {
            status.textContent = 'Disconnected';
            status.className = 'status disconnected';
        });

        socket.on('processing', (data) => {
            status.textContent = data.status;
            status.className = 'status processing';
            sendBtn.disabled = data.status !== 'Ready';
            micBtn.disabled = data.status !== 'Ready';
        });

        socket.on('transcription', (data) => {
            // Received transcription from server, show it in chat
            if (data.text && data.text.trim()) {
                addMessage(data.text, 'user', 'You (voice)');
            }
        });

        socket.on('response', (data) => {
            addMessage(data.text, 'assistant', '{{ avatar_name }}');
            status.textContent = 'Ready';
            status.className = 'status connected';
            sendBtn.disabled = false;
            micBtn.disabled = false;
        });

        function addMessage(text, type, name) {
            const msg = document.createElement('div');
            msg.className = 'message ' + type;
            msg.innerHTML = '<div class="name">' + name + '</div>' + text;
            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function sendMessage() {
            const text = messageInput.value.trim();
            if (!text) return;

            addMessage(text, 'user', 'You');
            socket.emit('message', {text: text});
            messageInput.value = '';
            sendBtn.disabled = true;
            micBtn.disabled = true;
            status.textContent = 'Processing...';
            status.className = 'status processing';
        }

        async function toggleMic() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    sendAudio(audioBlob);
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                isRecording = true;
                micBtn.classList.add('recording');
                micBtn.textContent = 'Stop';
                status.textContent = 'Recording...';
                status.className = 'status processing';
            } catch (err) {
                console.error('Microphone access denied:', err);
                status.textContent = 'Mic access denied';
                status.className = 'status disconnected';
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                micBtn.classList.remove('recording');
                micBtn.textContent = 'Mic';
                status.textContent = 'Transcribing...';
            }
        }

        function sendAudio(blob) {
            const reader = new FileReader();
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                socket.emit('audio', { data: base64 });
                sendBtn.disabled = true;
                micBtn.disabled = true;
                status.textContent = 'Processing audio...';
                status.className = 'status processing';
            };
            reader.readAsDataURL(blob);
        }
    </script>
</body>
</html>
"""


class WebAvatarPipeline:
    """Avatar pipeline adapted for web streaming."""

    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=100)
        self.frame_queue = frame_queue
        self._running = False
        self._current_frame = None

        logger.info("Initializing web avatar pipeline...")

        # Initialize LLM
        provider = llm_config.provider
        model = getattr(llm_config, f"{provider}_model")

        self.brain = LLMBrain(
            provider=provider,
            model=model,
            system_prompt=avatar_personality.system_prompt,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            groq_api_key=api_config.groq_api_key,
            openai_api_key=api_config.openai_api_key,
            ollama_host=api_config.ollama_host,
        )
        logger.info("LLM initialized")

        # Initialize TTS (Piper -> Kokoro -> Local fallback chain)
        tts_initialized = False

        # Try Piper first (best free option for Linux)
        try:
            self.voice = PiperTTSSynthesizer(
                model=voice_settings.piper_model,
                audio_queue=self.audio_queue,
            )
            logger.info("Piper TTS initialized")
            tts_initialized = True
        except Exception as e:
            logger.warning(f"Piper TTS failed: {e}")

        # Try Kokoro (works well on Windows/macOS)
        if not tts_initialized:
            try:
                self.voice = RealtimeTTSSynthesizer(
                    voice=voice_settings.kokoro_voice,
                    speed=voice_settings.kokoro_speed,
                    audio_queue=self.audio_queue,
                )
                logger.info("Kokoro TTS initialized")
                tts_initialized = True
            except Exception as e:
                logger.warning(f"Kokoro TTS failed: {e}")

        # Fall back to local TTS
        if not tts_initialized:
            logger.info("Using local TTS fallback")
            self.voice = LocalVoiceSynthesizer(
                audio_queue=self.audio_queue,
            )

        # Initialize Face Renderer
        video_path = avatar_personality.reference_media_path
        if not video_path.exists():
            video_path = None

        self.face = FaceRenderer(
            reference_image_path=str(avatar_personality.reference_image_path),
            reference_video_path=str(video_path) if video_path else None,
            models_dir=str(MODELS_DIR),
            audio_queue=self.audio_queue,
            frame_queue=self.frame_queue,
            fps=liveavatar_config.fps,
            device=liveavatar_config.device,
            use_fp8=liveavatar_config.use_fp8,
        )
        logger.info("Face renderer initialized")

        # Initialize Whisper for browser audio transcription
        try:
            self.ears = SpeechRecognizer(
                model_size=whisper_config.model_size,
                device=whisper_config.device,
            )
            logger.info("Whisper initialized for browser audio")
        except Exception as e:
            logger.warning(f"Whisper init failed: {e}")
            self.ears = None

    def start(self):
        """Start the pipeline."""
        self._running = True
        self.face.start()

        # Start frame capture thread
        threading.Thread(target=self._frame_loop, daemon=True).start()
        logger.info("Pipeline started")

    def stop(self):
        """Stop the pipeline."""
        self._running = False
        self.face.stop()

    def _frame_loop(self):
        """Capture frames for web streaming."""
        while self._running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is not None:
                    self._current_frame = frame
            except queue.Empty:
                pass

    def get_current_frame(self):
        """Get the current frame for streaming."""
        if self._current_frame is not None:
            return self._current_frame
        # Return a placeholder frame
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Initializing...", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder

    def process_message(self, text: str) -> str:
        """Process a text message and return response."""
        global is_processing
        is_processing = True

        try:
            # Generate response
            response = self.brain.generate(text)

            # Synthesize speech (this will feed audio to face renderer)
            if response:
                self.voice.synthesize_to_queue(response, play_audio=False)

            return response

        finally:
            is_processing = False

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file using Whisper."""
        if self.ears is None:
            raise RuntimeError("Whisper not initialized")
        return self.ears.transcribe_file(audio_path)


class TextOnlyPipeline:
    """Minimal pipeline for text chat with optional voice (no face)."""

    def __init__(self):
        logger.info("Initializing text-only pipeline (with optional TTS)...")

        self.audio_queue = queue.Queue(maxsize=100)
        self.voice = None

        # Initialize LLM
        provider = llm_config.provider
        model = getattr(llm_config, f"{provider}_model")

        self.brain = LLMBrain(
            provider=provider,
            model=model,
            system_prompt=avatar_personality.system_prompt,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            groq_api_key=api_config.groq_api_key,
            openai_api_key=api_config.openai_api_key,
            ollama_host=api_config.ollama_host,
        )
        logger.info("LLM initialized")

        # Try to initialize TTS (Piper preferred)
        self._init_tts()

        # Placeholder frame
        self._placeholder_frame = self._create_placeholder()

    def _init_tts(self):
        """Try to initialize TTS with graceful fallbacks."""
        # Try Piper first (best free option)
        try:
            self.voice = PiperTTSSynthesizer(
                model=voice_settings.piper_model,
                audio_queue=self.audio_queue,
            )
            logger.info("Piper TTS initialized successfully")
            return
        except Exception as e:
            logger.warning(f"Piper TTS failed: {e}")

        # Try ElevenLabs if API key is set
        if api_config.elevenlabs_api_key:
            try:
                from modules.voice import VoiceSynthesizer
                self.voice = VoiceSynthesizer(
                    api_key=api_config.elevenlabs_api_key,
                    voice_id=voice_settings.voice_id if hasattr(voice_settings, 'voice_id') else "21m00Tcm4TlvDq8ikWAM",
                    audio_queue=self.audio_queue,
                )
                logger.info("ElevenLabs TTS initialized")
                return
            except Exception as e:
                logger.warning(f"ElevenLabs TTS failed: {e}")

        # Try gTTS (Google TTS - reliable cloud fallback, free)
        try:
            self.voice = GTTSSynthesizer(
                lang="en",
                audio_queue=self.audio_queue,
            )
            logger.info("gTTS (Google TTS) initialized successfully")
            return
        except Exception as e:
            logger.warning(f"gTTS failed: {e}")

        # Try Kokoro
        try:
            self.voice = RealtimeTTSSynthesizer(
                voice=voice_settings.kokoro_voice,
                speed=voice_settings.kokoro_speed,
                audio_queue=self.audio_queue,
            )
            logger.info("Kokoro TTS initialized")
            return
        except Exception as e:
            logger.warning(f"Kokoro TTS failed: {e}")

        # Try local TTS
        try:
            self.voice = LocalVoiceSynthesizer(
                audio_queue=self.audio_queue,
            )
            logger.info("Local TTS initialized")
            return
        except Exception as e:
            logger.warning(f"Local TTS failed: {e}")

        logger.warning("No TTS available - text responses only")

    def _create_placeholder(self):
        """Create a placeholder frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 40)  # Dark background

        # Show mode based on what's available
        if self.voice:
            mode_text = "Voice Mode (No Face)"
            sub_text = "TTS enabled - audio will play"
        else:
            mode_text = "Text-Only Mode"
            sub_text = "Voice & Face not available"

        cv2.putText(frame, mode_text, (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 212, 255), 2)
        cv2.putText(frame, sub_text, (150, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        cv2.putText(frame, "Type a message to chat", (170, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        return frame

    def start(self):
        """No-op for text-only mode."""
        pass

    def stop(self):
        """No-op for text-only mode."""
        pass

    def get_current_frame(self):
        """Return placeholder frame."""
        return self._placeholder_frame

    def process_message(self, text: str) -> str:
        """Process a text message and return response."""
        global is_processing
        is_processing = True
        try:
            response = self.brain.generate(text)

            # Synthesize speech if TTS is available
            if self.voice and response:
                try:
                    self.voice.synthesize_to_queue(response, play_audio=False)
                except Exception as e:
                    logger.warning(f"TTS synthesis failed: {e}")

            return response
        finally:
            is_processing = False

    def transcribe_audio(self, audio_path: str) -> str:
        """Not available in text-only mode."""
        raise RuntimeError("Audio transcription not available in text-only mode")


def generate_frames():
    """Generator for MJPEG video stream."""
    while True:
        if avatar_pipeline:
            frame = avatar_pipeline.get_current_frame()

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(1/30)  # 30 FPS


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE, avatar_name=avatar_personality.name)


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    logger.info("Client connected")
    emit('processing', {'status': 'Ready'})


@socketio.on('message')
def handle_message(data):
    """Handle incoming chat message."""
    text = data.get('text', '').strip()
    if not text:
        return

    logger.info(f"Received message: {text}")
    emit('processing', {'status': 'Processing...'})

    # Process in background thread
    def process():
        response = avatar_pipeline.process_message(text)
        socketio.emit('response', {'text': response})
        socketio.emit('processing', {'status': 'Ready'})

    threading.Thread(target=process, daemon=True).start()


@socketio.on('audio')
def handle_audio(data):
    """Handle incoming audio from browser microphone."""
    audio_data = data.get('data', '')
    if not audio_data:
        return

    logger.info("Received audio from browser")
    emit('processing', {'status': 'Transcribing...'})

    def process_audio():
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            # Transcribe with Whisper
            text = avatar_pipeline.transcribe_audio(temp_path)

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass

            if text and text.strip():
                logger.info(f"Transcribed: {text}")
                socketio.emit('transcription', {'text': text})
                socketio.emit('processing', {'status': 'Processing...'})

                # Process the transcribed message
                response = avatar_pipeline.process_message(text)
                socketio.emit('response', {'text': response})
            else:
                socketio.emit('processing', {'status': 'No speech detected'})

            socketio.emit('processing', {'status': 'Ready'})

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            socketio.emit('processing', {'status': f'Error: {str(e)}'})
            socketio.emit('processing', {'status': 'Ready'})

    threading.Thread(target=process_audio, daemon=True).start()


def main():
    global avatar_pipeline
    if _WEB_DEPS_ERROR is not None:
        raise RuntimeError(
            "Missing web server dependencies (flask/flask-socketio). "
            "Install them with `pip install -r requirements.txt`. "
            f"Original error: {_WEB_DEPS_ERROR}"
        )

    parser = argparse.ArgumentParser(description="AI Avatar Web Server")
    parser.add_argument('--port', type=int, default=8080, help='Port to run on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           🎭 AI Avatar Web Server                        ║
╠══════════════════════════════════════════════════════════╣
║  Starting server on http://{args.host}:{args.port}              ║
║                                                          ║
║  Open this URL in your browser to chat with the avatar   ║
╚══════════════════════════════════════════════════════════╝
""")

    # Initialize pipeline
    try:
        avatar_pipeline = WebAvatarPipeline()
        avatar_pipeline.start()
    except Exception as e:
        logger.error(f"Failed to initialize full pipeline: {e}")
        logger.info("Starting in text-only mode (LLM chat without voice/face)...")

        # Create a minimal text-only pipeline
        try:
            avatar_pipeline = TextOnlyPipeline()
            logger.info("Text-only pipeline initialized successfully")
        except Exception as e2:
            logger.error(f"Text-only pipeline also failed: {e2}")
            avatar_pipeline = None

    # Run server
    socketio.run(app, host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
