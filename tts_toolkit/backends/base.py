"""Abstract base class for TTS backends.

This module defines the interface that all TTS backends must implement.
By abstracting the TTS engine, TTS Toolkit can work with different
providers (Qwen, Coqui, Bark, etc.) through a unified API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VoicePrompt:
    """Voice prompt data for voice cloning.

    This is a backend-agnostic representation of a voice prompt.
    Each backend converts this to its native format.
    """

    reference_audio: str  # Path to reference audio file
    reference_text: str  # Transcript of reference audio
    speaker_embedding: Optional[np.ndarray] = None  # Cached speaker embedding
    backend_data: Any = None  # Backend-specific cached data
    metadata: Dict[str, Any] = field(default_factory=dict)


class TTSBackend(ABC):
    """Abstract base class for TTS backends.

    All TTS backends must implement this interface to work with TTS Toolkit.
    This allows swapping between different TTS engines (Qwen, Coqui, Bark, etc.)
    without changing the rest of the codebase.

    Example implementation:
        class MyBackend(TTSBackend):
            def load_model(self):
                self.model = load_my_model()

            def create_voice_prompt(self, ref_audio, ref_text, **kwargs):
                embedding = self.model.encode_speaker(ref_audio)
                return VoicePrompt(
                    reference_audio=ref_audio,
                    reference_text=ref_text,
                    speaker_embedding=embedding,
                )

            def generate(self, text, voice_prompt, **kwargs):
                audio = self.model.synthesize(text, voice_prompt.speaker_embedding)
                return audio, self.sample_rate

            @property
            def sample_rate(self):
                return 24000

            def supports_voice_cloning(self):
                return True
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load the TTS model.

        This should be idempotent - calling it multiple times should not
        reload the model if it's already loaded.
        """
        pass

    @abstractmethod
    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str,
        **kwargs,
    ) -> VoicePrompt:
        """Create a voice prompt from reference audio.

        Args:
            reference_audio: Path to reference audio file (3-10 seconds)
            reference_text: Exact transcript of the reference audio
            **kwargs: Backend-specific options

        Returns:
            VoicePrompt object containing the voice data
        """
        pass

    @abstractmethod
    def generate(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        language: str = "Auto",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            voice_prompt: VoicePrompt for voice cloning
            language: Language code ("Auto", "English", "Chinese", etc.)
            **kwargs: Generation parameters (temperature, top_k, etc.)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get the output sample rate in Hz."""
        pass

    @abstractmethod
    def supports_voice_cloning(self) -> bool:
        """Check if this backend supports voice cloning."""
        pass

    def supports_streaming(self) -> bool:
        """Check if this backend supports streaming generation."""
        return False

    def supports_emotions(self) -> bool:
        """Check if this backend supports emotion control."""
        return False

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ["Auto"]

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        pass

    def cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory after operations.

        Should be called after batch processing or large operations
        to free GPU memory. Default implementation does nothing.
        Backends using GPU should override this.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get information about this backend."""
        return {
            "name": self.__class__.__name__,
            "sample_rate": self.sample_rate,
            "supports_voice_cloning": self.supports_voice_cloning(),
            "supports_streaming": self.supports_streaming(),
            "supports_emotions": self.supports_emotions(),
            "languages": self.get_supported_languages(),
        }
