"""Kokoro TTS backend implementation.

This backend uses the kokoro package for lightweight, fast, high-quality
text-to-speech synthesis with only 82M parameters.

Requirements:
    pip install kokoro soundfile
    # Linux: apt-get install espeak-ng
    # macOS: brew install espeak

Supported Languages:
    - American English ('a')
    - British English ('b')
    - Japanese ('j') - requires: pip install misaki[ja]
    - Mandarin Chinese ('z') - requires: pip install misaki[zh]
    - Spanish ('e'), French ('f'), Hindi ('h'), Italian ('i')
    - Brazilian Portuguese ('p')

Example:
    from tts_toolkit.backends import KokoroBackend

    backend = KokoroBackend(lang_code='a', voice='af_heart')
    backend.load_model()

    voice_prompt = backend.create_voice_prompt(
        reference_audio="",  # Not needed for Kokoro
        reference_text="",
    )

    audio, sr = backend.generate("Hello world!", voice_prompt)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TTSBackend, VoicePrompt


class KokoroBackend(TTSBackend):
    """Kokoro TTS backend - lightweight 82M parameter model.

    Features:
        - Extremely fast inference
        - High quality despite small size (82M params)
        - Multiple voices and languages
        - Apache 2.0 license
        - Voice blending support

    Args:
        lang_code: Language code:
            - 'a': American English
            - 'b': British English
            - 'j': Japanese
            - 'z': Mandarin Chinese
            - 'e': Spanish
            - 'f': French
            - 'h': Hindi
            - 'i': Italian
            - 'p': Brazilian Portuguese
        voice: Voice preset name (e.g., 'af_heart', 'af_sarah', 'am_adam')
        speed: Speech speed multiplier (default: 1.0)

    Note:
        Kokoro uses preset voices, not voice cloning from reference audio.
        The create_voice_prompt() method accepts a voice name override.
    """

    LANGUAGE_MAP = {
        "Auto": "a",
        "English": "a",
        "American English": "a",
        "British English": "b",
        "Japanese": "j",
        "Chinese": "z",
        "Mandarin": "z",
        "Spanish": "e",
        "French": "f",
        "Hindi": "h",
        "Italian": "i",
        "Portuguese": "p",
    }

    SUPPORTED_LANGUAGES = list(LANGUAGE_MAP.keys())

    # Common voice presets
    VOICES = [
        "af_heart", "af_sarah", "af_nicole", "af_bella", "af_sky",
        "am_adam", "am_michael", "am_fenrir",
        "bf_emma", "bf_isabella",
        "bm_george", "bm_lewis",
    ]

    def __init__(
        self,
        lang_code: str = "a",
        voice: str = "af_heart",
        speed: float = 1.0,
    ):
        """Initialize Kokoro backend."""
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self._pipeline = None
        self._sample_rate = 24000

    def load_model(self) -> None:
        """Load the Kokoro pipeline."""
        if self._pipeline is not None:
            return

        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise ImportError(
                "kokoro is required for KokoroBackend. "
                "Install with: pip install kokoro soundfile\n"
                "Also install espeak-ng (Linux) or espeak (macOS)"
            ) from e

        print(f"Loading Kokoro pipeline (lang={self.lang_code}, voice={self.voice})")
        self._pipeline = KPipeline(lang_code=self.lang_code)
        print("Pipeline loaded.")

    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str,
        voice: Optional[str] = None,
        **kwargs,
    ) -> VoicePrompt:
        """Create voice prompt (uses preset voices, not cloning).

        Args:
            reference_audio: Ignored (Kokoro uses preset voices)
            reference_text: Ignored
            voice: Override voice preset name

        Returns:
            VoicePrompt with voice preset in backend_data
        """
        selected_voice = voice or self.voice

        return VoicePrompt(
            reference_audio=reference_audio,
            reference_text=reference_text,
            backend_data={"voice": selected_voice},
            metadata={"lang_code": self.lang_code, "preset_voice": True},
        )

    def generate(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        language: str = "Auto",
        speed: Optional[float] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            voice_prompt: VoicePrompt from create_voice_prompt()
            language: Language (mapped to lang_code)
            speed: Speech speed multiplier (default: instance speed)
            **kwargs: Additional parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self._pipeline is None:
            self.load_model()

        voice = self.voice
        if voice_prompt.backend_data and "voice" in voice_prompt.backend_data:
            voice = voice_prompt.backend_data["voice"]

        speed_val = speed if speed is not None else self.speed

        # Generate audio chunks
        audio_chunks = []
        generator = self._pipeline(
            text,
            voice=voice,
            speed=speed_val,
            split_pattern=r'\n+',
        )

        for _, _, audio_chunk in generator:
            audio_chunks.append(audio_chunk)

        if not audio_chunks:
            return np.zeros(0, dtype=np.float32), self._sample_rate

        # Concatenate all chunks
        audio = np.concatenate(audio_chunks, axis=0)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        return audio, self._sample_rate

    @property
    def sample_rate(self) -> int:
        """Get output sample rate (24000 Hz)."""
        return self._sample_rate

    def supports_voice_cloning(self) -> bool:
        """Kokoro uses preset voices, not cloning."""
        return False

    def supports_streaming(self) -> bool:
        """Kokoro generates in chunks (streaming-like)."""
        return True

    def supports_emotions(self) -> bool:
        """Different voices have different styles."""
        return False

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return self.SUPPORTED_LANGUAGES

    def unload_model(self) -> None:
        """Unload pipeline."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

    def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        info = super().get_info()
        info.update({
            "lang_code": self.lang_code,
            "voice": self.voice,
            "speed": self.speed,
            "loaded": self._pipeline is not None,
            "model_size": "82M parameters",
            "license": "Apache 2.0",
            "available_voices": self.VOICES,
        })
        return info
