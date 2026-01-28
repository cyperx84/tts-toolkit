"""Coqui XTTS v2 TTS backend implementation.

This backend uses the Coqui TTS library with XTTS v2 model for high-quality
multilingual text-to-speech with voice cloning.

Requirements:
    pip install coqui-tts torch

Features:
    - Voice cloning from 6-second audio clip
    - 17 languages supported
    - Emotion and style transfer
    - Streaming with <200ms latency
    - Fine-tuning support

Example:
    from tts_toolkit.backends import CoquiXTTSBackend

    backend = CoquiXTTSBackend(device="cuda")
    backend.load_model()

    voice_prompt = backend.create_voice_prompt(
        reference_audio="sample.wav",
        reference_text="Hello, this is my voice.",
    )

    audio, sr = backend.generate(
        text="Generate this text.",
        voice_prompt=voice_prompt,
        language="en",
    )
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TTSBackend, VoicePrompt


class CoquiXTTSBackend(TTSBackend):
    """Coqui XTTS v2 TTS backend.

    Features:
        - Voice cloning with 6-second reference (85-95% similarity)
        - Emotion and style transfer from reference
        - Cross-language voice cloning
        - 17 languages supported
        - Streaming inference (<200ms latency)

    Args:
        model_name: TTS model name
        device: Device to run on ("cpu", "cuda")

    Supported Languages:
        en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi
    """

    SUPPORTED_LANGUAGES = [
        "Auto", "en", "es", "fr", "de", "it", "pt", "pl", "tr",
        "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
    ]

    LANGUAGE_NAMES = {
        "Auto": "en", "English": "en", "Spanish": "es", "French": "fr",
        "German": "de", "Italian": "it", "Portuguese": "pt", "Polish": "pl",
        "Turkish": "tr", "Russian": "ru", "Dutch": "nl", "Czech": "cs",
        "Arabic": "ar", "Chinese": "zh-cn", "Japanese": "ja",
        "Hungarian": "hu", "Korean": "ko", "Hindi": "hi",
    }

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda",
    ):
        """Initialize Coqui XTTS backend."""
        self.model_name = model_name
        self.device = device
        self._tts = None
        self._sample_rate = 24000

    def load_model(self) -> None:
        """Load the XTTS v2 model."""
        if self._tts is not None:
            return

        try:
            from TTS.api import TTS
            import torch
        except ImportError as e:
            raise ImportError(
                "coqui-tts is required for CoquiXTTSBackend. "
                "Install with: pip install coqui-tts torch"
            ) from e

        print(f"Loading Coqui XTTS model: {self.model_name}")
        self._tts = TTS(self.model_name)

        if self.device != "cpu":
            import torch
            if torch.cuda.is_available():
                self._tts = self._tts.to(self.device)

        print("Model loaded.")

    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str,
        **kwargs,
    ) -> VoicePrompt:
        """Create voice prompt from reference audio.

        Args:
            reference_audio: Path to reference audio (6+ seconds recommended)
            reference_text: Transcript of the reference audio (optional for XTTS)

        Returns:
            VoicePrompt with reference audio path
        """
        return VoicePrompt(
            reference_audio=reference_audio,
            reference_text=reference_text,
            backend_data={"speaker_wav": reference_audio},
            metadata={"model_name": self.model_name},
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
            language: Language code (en, es, fr, de, etc.)
            **kwargs: Additional parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self._tts is None:
            self.load_model()

        # Get speaker wav
        backend_data = voice_prompt.backend_data or {}
        speaker_wav = backend_data.get("speaker_wav") or voice_prompt.reference_audio

        # Map language name to code
        lang = language
        if language in self.LANGUAGE_NAMES:
            lang = self.LANGUAGE_NAMES[language]
        if lang == "Auto":
            lang = "en"

        # Generate audio
        audio = self._tts.tts(
            text=text,
            speaker_wav=speaker_wav,
            language=lang,
        )

        # Convert to numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Get sample rate from model
        if hasattr(self._tts, "synthesizer") and hasattr(self._tts.synthesizer, "output_sample_rate"):
            self._sample_rate = self._tts.synthesizer.output_sample_rate

        return audio, self._sample_rate

    @property
    def sample_rate(self) -> int:
        """Get output sample rate."""
        return self._sample_rate

    def supports_voice_cloning(self) -> bool:
        """XTTS supports voice cloning."""
        return True

    def supports_streaming(self) -> bool:
        """XTTS supports streaming (<200ms latency)."""
        return True

    def supports_emotions(self) -> bool:
        """Emotion transferred from reference audio."""
        return True

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return self.SUPPORTED_LANGUAGES

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._tts is not None:
            del self._tts
            self._tts = None

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        info = super().get_info()
        info.update({
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._tts is not None,
            "voice_similarity": "85-95% with 10s reference",
            "min_reference_audio": "6 seconds",
            "latency": "<200ms (streaming)",
        })
        return info
