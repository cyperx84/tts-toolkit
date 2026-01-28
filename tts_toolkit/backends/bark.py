"""Bark TTS backend implementation.

This backend uses the Bark model from Suno AI for expressive text-to-audio
generation with support for non-speech sounds.

Requirements:
    pip install transformers torch scipy

For full version (requires ~12GB VRAM):
    Set SUNO_USE_SMALL_MODELS=False

For small version (~8GB VRAM):
    Set SUNO_USE_SMALL_MODELS=True (default)

Features:
    - Multilingual speech synthesis
    - Non-verbal sounds: laughing, sighing, crying
    - Music and background noise generation
    - Highly expressive and natural output

Example:
    from tts_toolkit.backends import BarkBackend

    backend = BarkBackend(device="cuda", use_small_models=True)
    backend.load_model()

    voice_prompt = backend.create_voice_prompt(
        reference_audio="",
        reference_text="",
        speaker="v2/en_speaker_6",
    )

    audio, sr = backend.generate(
        text="Hello! [laughs] How are you today?",
        voice_prompt=voice_prompt,
    )
"""

from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np

from .base import TTSBackend, VoicePrompt


class BarkBackend(TTSBackend):
    """Bark TTS backend from Suno AI.

    Features:
        - Highly expressive and natural speech
        - Non-verbal sounds: [laughter], [sighs], [music], etc.
        - Multilingual support (13+ languages)
        - Multiple speaker presets per language

    Args:
        device: Device to run on ("cpu", "cuda")
        use_small_models: Use smaller models for lower VRAM (default: True)
        speaker: Default speaker preset (e.g., "v2/en_speaker_6")

    Speaker Presets:
        Format: "v2/{lang}_speaker_{n}" where:
        - lang: en, zh, fr, de, hi, it, ja, ko, pl, pt, ru, es, tr
        - n: 0-9 (speaker variation)

    Special Tokens:
        Use in text for non-verbal sounds:
        - [laughter], [laughs], [sighs], [music]
        - [gasps], [clears throat], [singing]
        - ... (hesitation), ♪ (music note)
    """

    SUPPORTED_LANGUAGES = [
        "Auto", "English", "Chinese", "French", "German", "Hindi",
        "Italian", "Japanese", "Korean", "Polish", "Portuguese",
        "Russian", "Spanish", "Turkish",
    ]

    LANGUAGE_CODES = {
        "Auto": "en", "English": "en", "Chinese": "zh", "French": "fr",
        "German": "de", "Hindi": "hi", "Italian": "it", "Japanese": "ja",
        "Korean": "ko", "Polish": "pl", "Portuguese": "pt", "Russian": "ru",
        "Spanish": "es", "Turkish": "tr",
    }

    def __init__(
        self,
        device: str = "cuda",
        use_small_models: bool = True,
        speaker: str = "v2/en_speaker_6",
    ):
        """Initialize Bark backend."""
        self.device = device
        self.use_small_models = use_small_models
        self.speaker = speaker
        self._pipeline = None
        self._sample_rate = 24000

        # Set environment variable for model size
        if use_small_models:
            os.environ["SUNO_USE_SMALL_MODELS"] = "True"

    def load_model(self) -> None:
        """Load the Bark model via transformers pipeline."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for BarkBackend. "
                "Install with: pip install transformers torch scipy"
            ) from e

        print(f"Loading Bark model on {self.device}")
        print(f"Using small models: {self.use_small_models}")

        device_idx = 0 if self.device.startswith("cuda") else -1
        if ":" in self.device:
            device_idx = int(self.device.split(":")[1])

        self._pipeline = pipeline(
            "text-to-speech",
            model="suno/bark",
            device=device_idx if self.device != "cpu" else -1,
        )
        print("Model loaded.")

    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str,
        speaker: Optional[str] = None,
        **kwargs,
    ) -> VoicePrompt:
        """Create voice prompt (uses preset speakers).

        Args:
            reference_audio: Ignored (Bark uses preset speakers)
            reference_text: Ignored
            speaker: Speaker preset (e.g., "v2/en_speaker_6")

        Returns:
            VoicePrompt with speaker preset in backend_data

        Note:
            Bark's official API restricts custom voice cloning to prevent misuse.
            Use the preset speakers instead.
        """
        selected_speaker = speaker or self.speaker

        return VoicePrompt(
            reference_audio=reference_audio,
            reference_text=reference_text,
            backend_data={"speaker": selected_speaker},
            metadata={"preset_voice": True, "model": "suno/bark"},
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
            text: Text to synthesize. Include special tokens for effects:
                  [laughter], [laughs], [sighs], [music], [gasps], etc.
            voice_prompt: VoicePrompt from create_voice_prompt()
            language: Language for auto-selecting speaker
            **kwargs: Additional parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self._pipeline is None:
            self.load_model()

        # Get speaker from voice prompt or select by language
        speaker = self.speaker
        if voice_prompt.backend_data and "speaker" in voice_prompt.backend_data:
            speaker = voice_prompt.backend_data["speaker"]
        elif language != "Auto":
            lang_code = self.LANGUAGE_CODES.get(language, "en")
            speaker = f"v2/{lang_code}_speaker_6"

        # Generate audio
        result = self._pipeline(
            text,
            forward_params={"do_sample": True},
        )

        audio = result["audio"]
        sr = result["sampling_rate"]
        self._sample_rate = sr

        # Ensure proper shape
        if audio.ndim > 1:
            audio = audio.squeeze()

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        return audio, sr

    @property
    def sample_rate(self) -> int:
        """Get output sample rate (24000 Hz)."""
        return self._sample_rate

    def supports_voice_cloning(self) -> bool:
        """Bark uses preset speakers (restricted cloning)."""
        return False

    def supports_streaming(self) -> bool:
        """Bark does not support streaming."""
        return False

    def supports_emotions(self) -> bool:
        """Bark supports non-verbal sounds via special tokens."""
        return True

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return self.SUPPORTED_LANGUAGES

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

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
            "speaker": self.speaker,
            "device": self.device,
            "use_small_models": self.use_small_models,
            "loaded": self._pipeline is not None,
            "special_tokens": [
                "[laughter]", "[laughs]", "[sighs]", "[music]",
                "[gasps]", "[clears throat]", "...", "♪",
            ],
            "vram_requirement": "~8GB (small) or ~12GB (full)",
        })
        return info
