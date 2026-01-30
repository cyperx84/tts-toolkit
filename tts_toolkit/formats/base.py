"""Abstract base classes for format handlers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..backends.base import TTSBackend, VoicePrompt

logger = logging.getLogger("tts-toolkit")


@dataclass
class Segment:
    """A segment of content to be synthesized."""

    text: str
    speaker_id: Optional[str] = None  # For multi-speaker content
    voice_id: Optional[str] = None  # Voice profile to use
    emotion: Optional[str] = None  # Emotion preset
    pause_before_ms: int = 0  # Pause before this segment
    pause_after_ms: int = 0  # Pause after this segment
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioOutput:
    """Output from format generation."""

    audio: np.ndarray
    sample_rate: int
    segments: List[Segment]
    duration_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FormatHandler(ABC):
    """Abstract base class for content format handlers.

    Format handlers are responsible for:
    1. Parsing input content into segments
    2. Generating audio for each segment using a TTSBackend
    3. Combining segments into final output
    """

    def __init__(
        self,
        backend: Optional["TTSBackend"] = None,
        stitcher=None,
    ):
        """
        Initialize format handler.

        Args:
            backend: TTSBackend instance (created lazily if None)
            stitcher: AudioStitcher instance (created if None)
        """
        self._backend = backend
        self._stitcher = stitcher

    @property
    def backend(self) -> "TTSBackend":
        """Lazy-load backend."""
        if self._backend is None:
            # Try to create QwenBackend, fall back to MockBackend
            try:
                from ..backends import QwenBackend
                self._backend = QwenBackend()
                logger.info("Using QwenBackend (default)")
            except ImportError as e:
                from ..backends import MockBackend
                logger.warning(
                    f"qwen-tts not available ({e}), falling back to MockBackend. "
                    "Install with: pip install tts-toolkit[qwen]"
                )
                self._backend = MockBackend()
        return self._backend

    def set_backend(self, backend: "TTSBackend") -> None:
        """Set the TTS backend."""
        self._backend = backend

    @property
    def stitcher(self):
        """Lazy-load stitcher."""
        if self._stitcher is None:
            from ..core.stitcher import AudioStitcher
            self._stitcher = AudioStitcher()
        return self._stitcher

    @abstractmethod
    def parse(self, input_text: str) -> List[Segment]:
        """
        Parse input content into segments.

        Args:
            input_text: Raw input content

        Returns:
            List of Segment objects
        """
        pass

    @abstractmethod
    def generate(
        self,
        segments: List[Segment],
        output_path: Optional[str] = None,
        **kwargs,
    ) -> AudioOutput:
        """
        Generate audio from segments.

        Args:
            segments: List of Segment objects
            output_path: Optional path to save output
            **kwargs: Additional generation parameters

        Returns:
            AudioOutput with combined audio
        """
        pass

    def process(
        self,
        input_text: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> AudioOutput:
        """
        Parse input and generate audio in one step.

        Args:
            input_text: Raw input content
            output_path: Optional path to save output
            **kwargs: Additional generation parameters

        Returns:
            AudioOutput with combined audio
        """
        segments = self.parse(input_text)
        return self.generate(segments, output_path, **kwargs)

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> AudioOutput:
        """
        Process a file and generate audio.

        Args:
            input_path: Path to input file
            output_path: Optional path to save output
            encoding: File encoding
            **kwargs: Additional generation parameters

        Returns:
            AudioOutput with combined audio
        """
        with open(input_path, "r", encoding=encoding) as f:
            input_text = f.read()
        return self.process(input_text, output_path, **kwargs)

    def _add_silence(self, audio: np.ndarray, duration_ms: int, sample_rate: int) -> np.ndarray:
        """Add silence to audio."""
        if duration_ms <= 0:
            return audio
        silence_samples = int(duration_ms * sample_rate / 1000)
        silence = np.zeros(silence_samples, dtype=np.float32)
        return np.concatenate([audio, silence])
