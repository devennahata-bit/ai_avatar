"""
Brain Module - LLM Integration
==============================
Handles Llama 3 generation for avatar responses.

Supports:
- Ollama (local)
- Groq API (fast cloud inference)
- OpenAI API (fallback)

Features:
- System prompt for personality
- Conversation history management
- Streaming responses
- Token-aware context windowing
"""

import threading
import queue
import time
from typing import Optional, Callable, Generator, Literal
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Chat message."""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ConversationHistory:
    """Manages conversation history with context windowing."""

    messages: list[Message] = field(default_factory=list)
    max_turns: int = 10

    def add_user_message(self, content: str):
        """Add a user message."""
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        self.messages.append(Message(role="assistant", content=content))
        self._trim()

    def _trim(self):
        """Trim history to max_turns (pairs of user/assistant messages)."""
        # Keep system message if present
        system_msg = None
        if self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]
            history = self.messages[1:]
        else:
            history = self.messages

        # Calculate max messages (2 per turn)
        max_messages = self.max_turns * 2

        if len(history) > max_messages:
            # Keep most recent messages
            history = history[-max_messages:]

        # Reconstruct
        self.messages = [system_msg] + history if system_msg else history

    def to_list(self) -> list[dict]:
        """Convert to list of dicts for API calls."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self):
        """Clear history (keeps system prompt)."""
        if self.messages and self.messages[0].role == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []


class LLMBrain:
    """
    LLM-powered brain for the avatar.

    Generates conversational responses using Llama 3 (or compatible models)
    through various providers.
    """

    def __init__(
        self,
        provider: Literal["ollama", "groq", "openai"] = "groq",
        model: Optional[str] = None,
        system_prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_history_turns: int = 10,
        stream: bool = True,
        # API keys
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        ollama_host: str = "http://localhost:11434",
        # Callbacks
        on_response_start: Optional[Callable[[], None]] = None,
        on_response_chunk: Optional[Callable[[str], None]] = None,
        on_response_complete: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the LLM brain.

        Args:
            provider: LLM provider (ollama, groq, openai)
            model: Model name (auto-selected if None)
            system_prompt: System prompt defining avatar personality
            max_tokens: Maximum response tokens
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            max_history_turns: Maximum conversation turns to keep
            stream: Enable streaming responses
            groq_api_key: Groq API key
            openai_api_key: OpenAI API key
            ollama_host: Ollama server URL
            on_response_start: Called when response starts
            on_response_chunk: Called for each response chunk (streaming)
            on_response_complete: Called when response is complete
        """
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream

        # API configuration
        self.groq_api_key = groq_api_key
        self.openai_api_key = openai_api_key
        self.ollama_host = ollama_host

        # Callbacks
        self.on_response_start = on_response_start
        self.on_response_chunk = on_response_chunk
        self.on_response_complete = on_response_complete

        # Default models per provider
        self._default_models = {
            "ollama": "llama3:8b",
            "groq": "llama-3.3-70b-versatile",
            "openai": "gpt-4o",
        }
        self.model = model or self._default_models.get(provider, "llama3")

        # Conversation history
        self.history = ConversationHistory(max_turns=max_history_turns)

        # Set system prompt
        if system_prompt:
            self.history.messages.append(Message(role="system", content=system_prompt))

        # Initialize client
        self._client = None
        self._init_client()

        # Threading
        self._response_queue: queue.Queue = queue.Queue()
        self._processing = False

    def _init_client(self):
        """Initialize the appropriate API client."""
        if self.provider == "groq":
            try:
                from groq import Groq
                self._client = Groq(api_key=self.groq_api_key)
                logger.info(f"Groq client initialized (model: {self.model})")
            except ImportError:
                raise ImportError("groq package required for Groq provider")

        elif self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.openai_api_key)
                logger.info(f"OpenAI client initialized (model: {self.model})")
            except ImportError:
                raise ImportError("openai package required for OpenAI provider")

        elif self.provider == "ollama":
            try:
                import ollama
                self._client = ollama.Client(host=self.ollama_host)
                logger.info(f"Ollama client initialized (model: {self.model})")
            except ImportError:
                raise ImportError("ollama package required for Ollama provider")

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        # Remove existing system message
        if self.history.messages and self.history.messages[0].role == "system":
            self.history.messages[0] = Message(role="system", content=prompt)
        else:
            self.history.messages.insert(0, Message(role="system", content=prompt))

    def generate(self, user_input: str) -> str:
        """
        Generate a response to user input.

        Args:
            user_input: User's message

        Returns:
            Complete response text
        """
        # Add user message to history
        self.history.add_user_message(user_input)

        if self.on_response_start:
            self.on_response_start()

        if self.stream:
            # Streaming mode
            response_text = ""
            for chunk in self._generate_stream():
                response_text += chunk
                if self.on_response_chunk:
                    self.on_response_chunk(chunk)
        else:
            # Non-streaming mode
            response_text = self._generate_complete()

        # Add to history
        self.history.add_assistant_message(response_text)

        if self.on_response_complete:
            self.on_response_complete(response_text)

        return response_text

    def _generate_stream(self) -> Generator[str, None, None]:
        """Generate streaming response."""
        messages = self.history.to_list()

        try:
            if self.provider == "groq":
                yield from self._stream_groq(messages)
            elif self.provider == "openai":
                yield from self._stream_openai(messages)
            elif self.provider == "ollama":
                yield from self._stream_ollama(messages)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"I apologize, but I encountered an error: {str(e)}"

    def _generate_complete(self) -> str:
        """Generate complete (non-streaming) response."""
        messages = self.history.to_list()

        try:
            if self.provider == "groq":
                return self._complete_groq(messages)
            elif self.provider == "openai":
                return self._complete_openai(messages)
            elif self.provider == "ollama":
                return self._complete_ollama(messages)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

    # -------------------------------------------------------------------------
    # GROQ
    # -------------------------------------------------------------------------

    def _stream_groq(self, messages: list[dict]) -> Generator[str, None, None]:
        """Stream from Groq API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _complete_groq(self, messages: list[dict]) -> str:
        """Complete response from Groq API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=False,
        )
        return response.choices[0].message.content

    # -------------------------------------------------------------------------
    # OPENAI
    # -------------------------------------------------------------------------

    def _stream_openai(self, messages: list[dict]) -> Generator[str, None, None]:
        """Stream from OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _complete_openai(self, messages: list[dict]) -> str:
        """Complete response from OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=False,
        )
        return response.choices[0].message.content

    # -------------------------------------------------------------------------
    # OLLAMA
    # -------------------------------------------------------------------------

    def _stream_ollama(self, messages: list[dict]) -> Generator[str, None, None]:
        """Stream from Ollama."""
        response = self._client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        )

        for chunk in response:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]

    def _complete_ollama(self, messages: list[dict]) -> str:
        """Complete response from Ollama."""
        response = self._client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options={
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        )
        return response["message"]["content"]

    # -------------------------------------------------------------------------
    # ASYNC GENERATION (for threaded use)
    # -------------------------------------------------------------------------

    def generate_async(self, user_input: str, callback: Callable[[str], None]):
        """
        Generate response asynchronously in a separate thread.

        Args:
            user_input: User's message
            callback: Called with complete response
        """
        def _run():
            response = self.generate(user_input)
            callback(response)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def clear_history(self):
        """Clear conversation history (keeps system prompt)."""
        self.history.clear()
        logger.info("Conversation history cleared")


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

    system_prompt = """You are Aria, a friendly AI assistant having a voice conversation.
Keep your responses brief and conversational (2-3 sentences).
Be warm, helpful, and natural in your speech."""

    brain = LLMBrain(
        provider="groq",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        system_prompt=system_prompt,
        max_tokens=150,
        stream=True,
        on_response_chunk=lambda chunk: print(chunk, end="", flush=True),
    )

    print("Chat with the AI (type 'quit' to exit):\n")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        print("\nAria: ", end="")
        brain.generate(user_input)
        print()
