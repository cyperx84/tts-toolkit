"""TTS Backend implementations.

This module provides abstract base classes and concrete implementations
for different TTS engines. The backend abstraction allows TTS Toolkit
to work with multiple TTS providers.

Available backends:
    - QwenBackend: Qwen3-TTS (requires qwen-tts package)
    - ChatterboxBackend: Resemble AI Chatterbox (requires chatterbox-tts)
    - KokoroBackend: Lightweight 82M model (requires kokoro)
    - FishSpeechBackend: Fish Audio API (requires fish-audio-sdk)
    - BarkBackend: Suno Bark (requires transformers)
    - CosyVoice2Backend: Alibaba CosyVoice2 (requires cosyvoice)
    - CoquiXTTSBackend: Coqui XTTS v2 (requires coqui-tts)
    - MockBackend: For testing (no dependencies)

Example:
    from tts_toolkit.backends import QwenBackend, ChatterboxBackend

    # Use Qwen for voice cloning
    backend = QwenBackend(model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base")

    # Or use Chatterbox for emotion control
    backend = ChatterboxBackend(model_type="turbo", device="cuda")

    audio, sr = backend.generate("Hello, world!", voice_prompt=voice_prompt)
"""

from .base import TTSBackend, VoicePrompt
from .mock import MockBackend

__all__ = [
    "TTSBackend",
    "VoicePrompt",
    "MockBackend",
]


# Lazy import functions for backends with heavy dependencies
def get_qwen_backend():
    """Get QwenBackend (imports qwen-tts on demand)."""
    from .qwen import QwenBackend
    return QwenBackend


def get_chatterbox_backend():
    """Get ChatterboxBackend (imports chatterbox-tts on demand)."""
    from .chatterbox import ChatterboxBackend
    return ChatterboxBackend


def get_kokoro_backend():
    """Get KokoroBackend (imports kokoro on demand)."""
    from .kokoro import KokoroBackend
    return KokoroBackend


def get_fish_speech_backend():
    """Get FishSpeechBackend (imports fish-audio-sdk on demand)."""
    from .fish_speech import FishSpeechBackend
    return FishSpeechBackend


def get_bark_backend():
    """Get BarkBackend (imports transformers on demand)."""
    from .bark import BarkBackend
    return BarkBackend


def get_cosyvoice_backend():
    """Get CosyVoice2Backend (imports cosyvoice on demand)."""
    from .cosyvoice import CosyVoice2Backend
    return CosyVoice2Backend


def get_coqui_xtts_backend():
    """Get CoquiXTTSBackend (imports coqui-tts on demand)."""
    from .coqui_xtts import CoquiXTTSBackend
    return CoquiXTTSBackend


__all__.extend([
    "get_qwen_backend",
    "get_chatterbox_backend",
    "get_kokoro_backend",
    "get_fish_speech_backend",
    "get_bark_backend",
    "get_cosyvoice_backend",
    "get_coqui_xtts_backend",
])


# Try to import backends if their dependencies are available
# This allows direct imports like: from tts_toolkit.backends import QwenBackend

try:
    from .qwen import QwenBackend
    __all__.append("QwenBackend")
except ImportError:
    pass

try:
    from .chatterbox import ChatterboxBackend
    __all__.append("ChatterboxBackend")
except ImportError:
    pass

try:
    from .kokoro import KokoroBackend
    __all__.append("KokoroBackend")
except ImportError:
    pass

try:
    from .fish_speech import FishSpeechBackend
    __all__.append("FishSpeechBackend")
except ImportError:
    pass

try:
    from .bark import BarkBackend
    __all__.append("BarkBackend")
except ImportError:
    pass

try:
    from .cosyvoice import CosyVoice2Backend
    __all__.append("CosyVoice2Backend")
except ImportError:
    pass

try:
    from .coqui_xtts import CoquiXTTSBackend
    __all__.append("CoquiXTTSBackend")
except ImportError:
    pass


# Backend registry for discovery
AVAILABLE_BACKENDS = {
    "qwen": ("QwenBackend", "qwen-tts torch", "Voice cloning, 11 languages, streaming"),
    "chatterbox": ("ChatterboxBackend", "chatterbox-tts torch torchaudio", "Emotion control, 23 languages, paralinguistic tags"),
    "kokoro": ("KokoroBackend", "kokoro soundfile", "Lightweight 82M params, fast, high quality"),
    "fish_speech": ("FishSpeechBackend", "fish-audio-sdk", "API-based, multilingual, voice cloning"),
    "bark": ("BarkBackend", "transformers torch scipy", "Expressive, non-verbal sounds, multilingual"),
    "cosyvoice": ("CosyVoice2Backend", "cosyvoice (manual install)", "Ultra-low latency streaming, Chinese dialects"),
    "coqui_xtts": ("CoquiXTTSBackend", "coqui-tts torch", "17 languages, 6s voice cloning, streaming"),
    "mock": ("MockBackend", "(none)", "Testing backend, no dependencies"),
}


def list_backends():
    """List all available backends with their requirements."""
    print("Available TTS Backends:")
    print("-" * 80)
    for key, (name, deps, desc) in AVAILABLE_BACKENDS.items():
        print(f"\n{name}")
        print(f"  Install: pip install {deps}" if deps != "(none)" else "  Install: (no dependencies)")
        print(f"  Features: {desc}")
    print()
