"""TTS Toolkit - Extensible Text-to-Speech toolkit for podcasts, audiobooks, and voiceovers.

TTS Toolkit is a modular toolkit that supports multiple TTS backends and provides
high-level abstractions for creating podcasts, audiobooks, voiceovers, and dialogues.

Example:
    from tts_toolkit import Pipeline
    from tts_toolkit.backends import QwenBackend

    pipeline = Pipeline(backend=QwenBackend())
    pipeline.generate_voiceover(
        text="Hello, world!",
        voice_ref="sample.wav",
        output_path="output.wav",
    )
"""

from .core.pipeline import Pipeline
from .core.chunker import TextChunker
from .core.stitcher import AudioStitcher
from .voices.profile import VoiceProfile
from .voices.registry import VoiceRegistry

__version__ = "0.1.0"

__all__ = [
    "Pipeline",
    "TextChunker",
    "AudioStitcher",
    "VoiceProfile",
    "VoiceRegistry",
]
