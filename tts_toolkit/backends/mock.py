"""Mock TTS backend for testing.

This backend generates silent audio or sine waves for testing purposes.
It requires no external dependencies and is useful for:
- Unit testing without loading heavy models
- CI/CD pipelines
- Development without GPU

Example:
    from tts_toolkit.backends import MockBackend

    backend = MockBackend()
    voice_prompt = backend.create_voice_prompt("dummy.wav", "text")
    audio, sr = backend.generate("Hello", voice_prompt)
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from .base import TTSBackend, VoicePrompt


class MockBackend(TTSBackend):
    """Mock backend for testing.

    Generates silent audio or sine waves based on text length.
    Useful for testing the pipeline without loading actual TTS models.

    Args:
        sample_rate: Output sample rate (default: 24000)
        mode: "silent" for silent audio, "sine" for sine wave
        words_per_second: Estimated speaking rate for duration calculation
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        mode: str = "silent",
        words_per_second: float = 2.5,
    ):
        """Initialize mock backend."""
        self._sample_rate = sample_rate
        self.mode = mode
        self.words_per_second = words_per_second
        self._loaded = False

    def load_model(self) -> None:
        """Mock model loading (instant)."""
        self._loaded = True

    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str,
        **kwargs,
    ) -> VoicePrompt:
        """Create a mock voice prompt.

        Args:
            reference_audio: Path (not actually loaded)
            reference_text: Reference text (stored but not used)

        Returns:
            VoicePrompt with mock data
        """
        return VoicePrompt(
            reference_audio=reference_audio,
            reference_text=reference_text,
            backend_data={"mock": True},
        )

    def generate(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        language: str = "Auto",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate mock audio.

        Args:
            text: Text to "synthesize"
            voice_prompt: Voice prompt (ignored)
            language: Language (ignored)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Estimate duration based on word count
        word_count = len(text.split())
        duration_sec = max(0.5, word_count / self.words_per_second)
        num_samples = int(duration_sec * self._sample_rate)

        if self.mode == "sine":
            # Generate a simple sine wave (440 Hz)
            t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        else:
            # Generate silence
            audio = np.zeros(num_samples, dtype=np.float32)

        return audio, self._sample_rate

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate

    def supports_voice_cloning(self) -> bool:
        """Mock backend pretends to support voice cloning."""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Get backend info."""
        info = super().get_info()
        info.update({
            "mode": self.mode,
            "words_per_second": self.words_per_second,
            "is_mock": True,
        })
        return info
