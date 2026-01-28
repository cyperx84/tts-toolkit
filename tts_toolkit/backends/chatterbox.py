"""Chatterbox TTS backend implementation.

This backend uses the chatterbox-tts package from Resemble AI for high-quality
text-to-speech with voice cloning, emotion control, and multilingual support.

Requirements:
    pip install chatterbox-tts torch torchaudio

Supported Models:
    - ChatterboxTTS: Standard 500M model (English)
    - ChatterboxTurboTTS: Fast 350M model with paralinguistic tags
    - ChatterboxMultilingualTTS: 23+ languages

Example:
    from tts_toolkit.backends import ChatterboxBackend

    backend = ChatterboxBackend(device="cuda", model_type="turbo")
    backend.load_model()

    voice_prompt = backend.create_voice_prompt(
        reference_audio="sample.wav",
        reference_text="Hello, this is a sample.",
    )

    audio, sr = backend.generate(
        text="Generate this text with [laugh] sound effects.",
        voice_prompt=voice_prompt,
    )
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TTSBackend, VoicePrompt


class ChatterboxBackend(TTSBackend):
    """Chatterbox TTS backend from Resemble AI.

    Features:
        - Voice cloning with 10s reference audio
        - Emotion exaggeration control
        - Paralinguistic tags: [laugh], [chuckle], [cough] (Turbo model)
        - 23+ languages (Multilingual model)
        - Perth neural watermarking (imperceptible, survives compression)

    Args:
        model_type: Model variant to use:
            - "standard": ChatterboxTTS (500M, English, emotion control)
            - "turbo": ChatterboxTurboTTS (350M, English, paralinguistic tags)
            - "multilingual": ChatterboxMultilingualTTS (500M, 23+ languages)
        device: Device to run on ("cpu", "cuda", "cuda:0")

    Paralinguistic Tags (Turbo model only):
        Use tags like [laugh], [chuckle], [cough] directly in text:
        "Hi there [chuckle], how are you?"
    """

    SUPPORTED_LANGUAGES = [
        "Auto", "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
    ]

    def __init__(
        self,
        model_type: str = "turbo",
        device: str = "cuda",
    ):
        """Initialize Chatterbox backend."""
        if model_type not in ("standard", "turbo", "multilingual"):
            raise ValueError(
                f"model_type must be 'standard', 'turbo', or 'multilingual', got: {model_type}"
            )
        self.model_type = model_type
        self.device = device
        self._model = None
        self._sample_rate: Optional[int] = None

    def load_model(self) -> None:
        """Load the Chatterbox model."""
        if self._model is not None:
            return

        try:
            if self.model_type == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                print(f"Loading Chatterbox Turbo model on {self.device}")
                self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            elif self.model_type == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                print(f"Loading Chatterbox Multilingual model on {self.device}")
                self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            else:
                from chatterbox.tts import ChatterboxTTS
                print(f"Loading Chatterbox Standard model on {self.device}")
                self._model = ChatterboxTTS.from_pretrained(device=self.device)

            self._sample_rate = self._model.sr
            print("Model loaded.")

        except ImportError as e:
            raise ImportError(
                "chatterbox-tts is required for ChatterboxBackend. "
                "Install with: pip install chatterbox-tts torch torchaudio"
            ) from e

    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str,
        **kwargs,
    ) -> VoicePrompt:
        """Create voice prompt from reference audio.

        Args:
            reference_audio: Path to reference audio (ideally 10 seconds)
            reference_text: Transcript of the audio (used for metadata only)
            **kwargs: Additional options

        Returns:
            VoicePrompt with audio path stored in backend_data
        """
        return VoicePrompt(
            reference_audio=reference_audio,
            reference_text=reference_text,
            backend_data={"audio_path": reference_audio},
            metadata={"model_type": self.model_type},
        )

    def generate(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        language: str = "Auto",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text.

        Args:
            text: Text to synthesize. Turbo model supports tags like [laugh], [chuckle]
            voice_prompt: VoicePrompt from create_voice_prompt()
            language: Language code (for multilingual model: en, fr, zh, etc.)
            exaggeration: Emotion intensity (0.0-1.0, higher = more dramatic)
            cfg_weight: Style adherence (0.0-1.0, lower = more expressive)
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self._model is None:
            self.load_model()

        audio_path = voice_prompt.backend_data.get("audio_path") if voice_prompt.backend_data else None
        if audio_path is None:
            audio_path = voice_prompt.reference_audio

        # Generate based on model type
        if self.model_type == "turbo":
            wav = self._model.generate(
                text,
                audio_prompt_path=audio_path,
            )
        elif self.model_type == "multilingual":
            lang_id = language if language != "Auto" else "en"
            wav = self._model.generate(
                text,
                audio_prompt_path=audio_path,
                language_id=lang_id,
            )
        else:
            wav = self._model.generate(
                text,
                audio_prompt_path=audio_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        # Convert tensor to numpy
        if hasattr(wav, "numpy"):
            audio = wav.squeeze().cpu().numpy()
        else:
            audio = np.array(wav).squeeze()

        # Normalize to float32 [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        return audio, self.sample_rate

    @property
    def sample_rate(self) -> int:
        """Get output sample rate."""
        return self._sample_rate or 24000

    def supports_voice_cloning(self) -> bool:
        """Chatterbox supports voice cloning."""
        return True

    def supports_streaming(self) -> bool:
        """Turbo model has low latency but no streaming API."""
        return False

    def supports_emotions(self) -> bool:
        """Standard model supports emotion control via exaggeration."""
        return self.model_type == "standard"

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        if self.model_type == "multilingual":
            return self.SUPPORTED_LANGUAGES
        return ["Auto", "en"]

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

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
            "model_type": self.model_type,
            "device": self.device,
            "loaded": self._model is not None,
            "paralinguistic_tags": self.model_type == "turbo",
            "watermarking": "Perth neural watermark (imperceptible)",
        })
        return info
