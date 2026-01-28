"""TTS Backend implementations.

This module provides abstract base classes and concrete implementations
for different TTS engines. The backend abstraction allows TTS Toolkit
to work with multiple TTS providers.

Available backends:
    - QwenBackend: Qwen3-TTS (requires qwen-tts package)
    - MockBackend: For testing (no dependencies)

Example:
    from tts_toolkit.backends import QwenBackend

    backend = QwenBackend(model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    audio, sr = backend.generate("Hello, world!", voice_prompt=voice_prompt)
"""

from .base import TTSBackend, VoicePrompt
from .mock import MockBackend

# Lazy import for QwenBackend to avoid requiring qwen-tts
def get_qwen_backend():
    """Get QwenBackend (imports qwen-tts on demand)."""
    from .qwen import QwenBackend
    return QwenBackend

__all__ = [
    "TTSBackend",
    "VoicePrompt",
    "MockBackend",
    "get_qwen_backend",
]

# Try to import QwenBackend if qwen-tts is available
try:
    from .qwen import QwenBackend
    __all__.append("QwenBackend")
except ImportError:
    pass
