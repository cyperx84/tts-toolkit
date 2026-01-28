"""Qwen3-TTS backend implementation.

This backend uses the qwen-tts package to provide high-quality
text-to-speech with voice cloning capabilities.

Requirements:
    pip install qwen-tts torch

Example:
    from tts_toolkit.backends import QwenBackend

    backend = QwenBackend(device="cuda:0")
    backend.load_model()

    voice_prompt = backend.create_voice_prompt(
        reference_audio="sample.wav",
        reference_text="Hello, this is a sample.",
    )

    audio, sr = backend.generate(
        text="Generate this text.",
        voice_prompt=voice_prompt,
    )
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TTSBackend, VoicePrompt


class QwenBackend(TTSBackend):
    """Qwen3-TTS backend for voice cloning and speech synthesis.

    This backend wraps the qwen-tts package and provides voice cloning
    capabilities using Qwen3-TTS models.

    Args:
        model_name: HuggingFace model name or local path
            Default: "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        device: Device to run on ("cpu", "cuda:0", "mps")
        torch_dtype: Torch data type (auto-detected if None)

    Supported models:
        - Qwen/Qwen3-TTS-12Hz-0.6B-Base (voice cloning)
        - Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice (9 premium voices)
        - Qwen/Qwen3-TTS-12Hz-1.7B-Base (better quality)
        - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (9 premium voices)
        - Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign (voice description)
    """

    SUPPORTED_LANGUAGES = [
        "Auto", "Chinese", "English", "Japanese", "Korean",
        "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: str = "cpu",
        torch_dtype=None,
    ):
        """Initialize Qwen backend."""
        self.model_name = model_name
        self.device = device
        self._torch_dtype = torch_dtype
        self._model = None
        self._sample_rate: Optional[int] = None

    def load_model(self) -> None:
        """Load the Qwen3-TTS model."""
        if self._model is not None:
            return

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise ImportError(
                "qwen-tts is required for QwenBackend. "
                "Install with: pip install qwen-tts torch"
            ) from e

        # Auto-detect dtype
        if self._torch_dtype is None:
            if self.device == "cpu":
                self._torch_dtype = torch.float32
            else:
                self._torch_dtype = torch.bfloat16

        print(f"Loading Qwen model: {self.model_name}")
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=self._torch_dtype,
        )
        print("Model loaded.")

    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str,
        x_vector_only: bool = False,
        **kwargs,
    ) -> VoicePrompt:
        """Create voice prompt from reference audio.

        Args:
            reference_audio: Path to reference audio (3-10 seconds)
            reference_text: Exact transcript of the audio
            x_vector_only: Use only speaker embedding (faster, lower quality)

        Returns:
            VoicePrompt with cached Qwen voice clone data
        """
        if self._model is None:
            self.load_model()

        print(f"Creating voice prompt from: {reference_audio}")
        qwen_prompt = self._model.create_voice_clone_prompt(
            ref_audio=reference_audio,
            ref_text=reference_text,
            x_vector_only_mode=x_vector_only,
        )

        return VoicePrompt(
            reference_audio=reference_audio,
            reference_text=reference_text,
            backend_data=qwen_prompt,
            metadata={"x_vector_only": x_vector_only},
        )

    def generate(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        language: str = "Auto",
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            voice_prompt: VoicePrompt from create_voice_prompt()
            language: Language code
            temperature: Sampling temperature (0.0-1.0)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self._model is None:
            self.load_model()

        if voice_prompt.backend_data is None:
            raise ValueError(
                "VoicePrompt missing backend_data. "
                "Create it using QwenBackend.create_voice_prompt()"
            )

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 2048),
            "do_sample": kwargs.get("do_sample", True),
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
            "subtalker_dosample": kwargs.get("subtalker_dosample", True),
            "subtalker_top_k": kwargs.get("subtalker_top_k", top_k),
            "subtalker_top_p": kwargs.get("subtalker_top_p", top_p),
            "subtalker_temperature": kwargs.get("subtalker_temperature", temperature),
        }

        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_prompt.backend_data,
            **gen_kwargs,
        )

        self._sample_rate = sr
        return wavs[0], sr

    @property
    def sample_rate(self) -> int:
        """Get output sample rate (24000 Hz for Qwen3-TTS)."""
        return self._sample_rate or 24000

    def supports_voice_cloning(self) -> bool:
        """Qwen3-TTS supports voice cloning."""
        return True

    def supports_streaming(self) -> bool:
        """Qwen3-TTS supports streaming generation."""
        return True

    def supports_emotions(self) -> bool:
        """Emotion control via generation parameters."""
        return True

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return self.SUPPORTED_LANGUAGES

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

            # Try to free GPU memory
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
            "loaded": self._model is not None,
        })
        return info
