"""Fish Speech TTS backend implementation.

This backend uses the fish-audio-sdk for high-quality multilingual
text-to-speech with voice cloning capabilities.

Requirements:
    pip install fish-audio-sdk

For audio playback utilities:
    pip install fish-audio-sdk[utils]

Supported Languages:
    English, Japanese, Korean, Chinese, French, German, Arabic, Spanish

Example:
    from tts_toolkit.backends import FishSpeechBackend

    backend = FishSpeechBackend(api_key="your-api-key")
    backend.load_model()

    voice_prompt = backend.create_voice_prompt(
        reference_audio="sample.wav",
        reference_text="Hello, this is my voice.",
    )

    audio, sr = backend.generate("Generate this text.", voice_prompt)
"""

from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np

from .base import TTSBackend, VoicePrompt


class FishSpeechBackend(TTSBackend):
    """Fish Speech TTS backend via Fish Audio API.

    Features:
        - High-quality voice cloning from 10-30s reference
        - Multilingual support (8 languages)
        - DualAR transformer architecture
        - Captures timbre, style, and emotional tendencies

    Args:
        api_key: Fish Audio API key (or set FISH_AUDIO_API_KEY env var)
        model_id: Model to use (default: "speech-01-turbo")
        reference_id: Pre-uploaded voice model ID (optional, faster)

    Note:
        This backend requires an API key from https://fish.audio
        For local usage, consider using the fish-speech local installation.
    """

    SUPPORTED_LANGUAGES = [
        "Auto", "English", "Japanese", "Korean", "Chinese",
        "French", "German", "Arabic", "Spanish",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "speech-01-turbo",
        reference_id: Optional[str] = None,
    ):
        """Initialize Fish Speech backend."""
        self.api_key = api_key or os.environ.get("FISH_AUDIO_API_KEY")
        self.model_id = model_id
        self.reference_id = reference_id
        self._client = None
        self._sample_rate = 44100

    def load_model(self) -> None:
        """Initialize the Fish Audio client."""
        if self._client is not None:
            return

        if not self.api_key:
            raise ValueError(
                "Fish Audio API key required. "
                "Set api_key parameter or FISH_AUDIO_API_KEY environment variable."
            )

        try:
            from fish_audio_sdk import Session
        except ImportError as e:
            raise ImportError(
                "fish-audio-sdk is required for FishSpeechBackend. "
                "Install with: pip install fish-audio-sdk"
            ) from e

        print("Initializing Fish Audio client")
        self._client = Session(api_key=self.api_key)
        print("Client initialized.")

    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str,
        reference_id: Optional[str] = None,
        **kwargs,
    ) -> VoicePrompt:
        """Create voice prompt from reference audio.

        Args:
            reference_audio: Path to reference audio (10-30 seconds ideal)
            reference_text: Transcript of the reference audio
            reference_id: Pre-uploaded voice model ID (faster if reusing)

        Returns:
            VoicePrompt with reference data
        """
        return VoicePrompt(
            reference_audio=reference_audio,
            reference_text=reference_text,
            backend_data={
                "reference_id": reference_id or self.reference_id,
                "reference_audio": reference_audio,
                "reference_text": reference_text,
            },
            metadata={"model_id": self.model_id},
        )

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
            voice_prompt: VoicePrompt from create_voice_prompt()
            language: Language hint (optional)
            **kwargs: Additional parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self._client is None:
            self.load_model()

        try:
            from fish_audio_sdk import TTSRequest, ReferenceAudio
        except ImportError as e:
            raise ImportError(
                "fish-audio-sdk is required. Install with: pip install fish-audio-sdk"
            ) from e

        backend_data = voice_prompt.backend_data or {}
        ref_id = backend_data.get("reference_id")

        # Build request
        if ref_id:
            # Use pre-uploaded reference
            request = TTSRequest(
                text=text,
                reference_id=ref_id,
            )
        else:
            # Use on-the-fly voice cloning
            ref_audio_path = backend_data.get("reference_audio") or voice_prompt.reference_audio
            ref_text = backend_data.get("reference_text") or voice_prompt.reference_text

            if not ref_audio_path or not os.path.exists(ref_audio_path):
                raise ValueError(
                    "Reference audio required for voice cloning. "
                    "Provide reference_audio path or reference_id."
                )

            with open(ref_audio_path, "rb") as f:
                audio_data = f.read()

            request = TTSRequest(
                text=text,
                references=[
                    ReferenceAudio(
                        audio=audio_data,
                        text=ref_text,
                    )
                ],
            )

        # Generate audio
        audio_bytes = b""
        for chunk in self._client.tts(request):
            audio_bytes += chunk

        # Convert to numpy array
        audio = self._bytes_to_array(audio_bytes)

        return audio, self._sample_rate

    def _bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        import io
        try:
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            self._sample_rate = sr
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return audio
        except Exception:
            # Fallback: assume raw PCM
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0
            return audio

    @property
    def sample_rate(self) -> int:
        """Get output sample rate."""
        return self._sample_rate

    def supports_voice_cloning(self) -> bool:
        """Fish Speech supports voice cloning."""
        return True

    def supports_streaming(self) -> bool:
        """Fish Speech supports streaming."""
        return True

    def supports_emotions(self) -> bool:
        """Emotions transferred from reference audio."""
        return False

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return self.SUPPORTED_LANGUAGES

    def unload_model(self) -> None:
        """Close the client session."""
        if self._client is not None:
            self._client = None

    def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        info = super().get_info()
        info.update({
            "model_id": self.model_id,
            "reference_id": self.reference_id,
            "loaded": self._client is not None,
            "api_based": True,
        })
        return info
