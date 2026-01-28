"""CosyVoice2 TTS backend implementation.

This backend uses the CosyVoice2 model from Alibaba FunAudioLLM for
ultra-low latency streaming text-to-speech with voice cloning.

Requirements:
    # Clone and install CosyVoice
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice
    pip install -r requirements.txt

    # Download models
    from huggingface_hub import snapshot_download
    snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

Supported Languages:
    Chinese, English, Japanese, Korean, and Chinese dialects
    (Cantonese, Sichuanese, Shanghainese, Tianjinese, Wuhanese)

Features:
    - Ultra-low latency streaming (150ms first-packet)
    - Zero-shot voice cloning
    - Cross-lingual synthesis
    - Instruction-following generation

Example:
    from tts_toolkit.backends import CosyVoice2Backend

    backend = CosyVoice2Backend(model_dir="pretrained_models/CosyVoice2-0.5B")
    backend.load_model()

    voice_prompt = backend.create_voice_prompt(
        reference_audio="sample.wav",
        reference_text="Hello, this is my voice.",
    )

    audio, sr = backend.generate("Generate this text.", voice_prompt)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TTSBackend, VoicePrompt


class CosyVoice2Backend(TTSBackend):
    """CosyVoice2 TTS backend from Alibaba FunAudioLLM.

    Features:
        - Ultra-low latency streaming (150ms first-packet with vLLM)
        - Zero-shot voice cloning from short reference
        - Cross-lingual voice cloning
        - Instruction-based synthesis
        - Chinese dialect support

    Args:
        model_dir: Path to pretrained model directory
        load_jit: Load TorchScript model for faster inference
        load_trt: Load TensorRT model (4x faster)
        fp16: Use FP16 precision
        stream: Enable streaming mode

    Inference Modes:
        - zero_shot: Clone voice from reference audio
        - cross_lingual: Clone voice, generate in different language
        - instruct: Follow natural language instructions
    """

    SUPPORTED_LANGUAGES = [
        "Auto", "Chinese", "English", "Japanese", "Korean",
        "Cantonese", "Sichuanese", "Shanghainese",
    ]

    def __init__(
        self,
        model_dir: str = "pretrained_models/CosyVoice2-0.5B",
        load_jit: bool = False,
        load_trt: bool = False,
        fp16: bool = False,
        stream: bool = False,
    ):
        """Initialize CosyVoice2 backend."""
        self.model_dir = model_dir
        self.load_jit = load_jit
        self.load_trt = load_trt
        self.fp16 = fp16
        self.stream = stream
        self._model = None
        self._sample_rate = 22050

    def load_model(self) -> None:
        """Load the CosyVoice2 model."""
        if self._model is not None:
            return

        try:
            from cosyvoice import CosyVoice2
        except ImportError as e:
            raise ImportError(
                "CosyVoice is required for CosyVoice2Backend. "
                "Install from: https://github.com/FunAudioLLM/CosyVoice\n"
                "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\n"
                "cd CosyVoice && pip install -r requirements.txt"
            ) from e

        print(f"Loading CosyVoice2 model from: {self.model_dir}")
        self._model = CosyVoice2(
            self.model_dir,
            load_jit=self.load_jit,
            load_trt=self.load_trt,
            fp16=self.fp16,
        )
        self._sample_rate = self._model.sample_rate
        print("Model loaded.")

    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str,
        **kwargs,
    ) -> VoicePrompt:
        """Create voice prompt from reference audio.

        Args:
            reference_audio: Path to reference audio (16kHz recommended)
            reference_text: Transcript of the reference audio

        Returns:
            VoicePrompt with reference data
        """
        return VoicePrompt(
            reference_audio=reference_audio,
            reference_text=reference_text,
            backend_data={
                "reference_audio": reference_audio,
                "reference_text": reference_text,
            },
            metadata={"model_dir": self.model_dir},
        )

    def generate(
        self,
        text: str,
        voice_prompt: VoicePrompt,
        language: str = "Auto",
        mode: str = "zero_shot",
        instruct_text: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            voice_prompt: VoicePrompt from create_voice_prompt()
            language: Target language (for cross_lingual mode)
            mode: Inference mode:
                - "zero_shot": Standard voice cloning
                - "cross_lingual": Clone voice, different language
                - "instruct": Follow instruction text
            instruct_text: Instructions for instruct mode
                          (e.g., "Speak slowly and clearly")
            **kwargs: Additional parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self._model is None:
            self.load_model()

        try:
            import torchaudio
        except ImportError as e:
            raise ImportError("torchaudio required: pip install torchaudio") from e

        # Load reference audio
        backend_data = voice_prompt.backend_data or {}
        ref_audio_path = backend_data.get("reference_audio") or voice_prompt.reference_audio
        ref_text = backend_data.get("reference_text") or voice_prompt.reference_text

        prompt_speech, _ = torchaudio.load(ref_audio_path)

        # Generate based on mode
        audio_chunks = []

        if mode == "cross_lingual":
            generator = self._model.inference_cross_lingual(
                text,
                prompt_speech,
                stream=self.stream,
            )
        elif mode == "instruct" and instruct_text:
            generator = self._model.inference_instruct2(
                text,
                instruct_text,
                prompt_speech,
                stream=self.stream,
            )
        else:
            # Default: zero_shot
            generator = self._model.inference_zero_shot(
                text,
                ref_text,
                prompt_speech,
                stream=self.stream,
            )

        for result in generator:
            audio_chunk = result.get("tts_speech")
            if audio_chunk is not None:
                if hasattr(audio_chunk, "numpy"):
                    audio_chunks.append(audio_chunk.squeeze().numpy())
                else:
                    audio_chunks.append(np.array(audio_chunk).squeeze())

        if not audio_chunks:
            return np.zeros(0, dtype=np.float32), self._sample_rate

        audio = np.concatenate(audio_chunks, axis=0)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        return audio, self._sample_rate

    @property
    def sample_rate(self) -> int:
        """Get output sample rate."""
        return self._sample_rate

    def supports_voice_cloning(self) -> bool:
        """CosyVoice2 supports zero-shot voice cloning."""
        return True

    def supports_streaming(self) -> bool:
        """CosyVoice2 supports streaming (150ms latency)."""
        return True

    def supports_emotions(self) -> bool:
        """Emotion control via instruct mode."""
        return True

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return self.SUPPORTED_LANGUAGES

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
            "model_dir": self.model_dir,
            "load_jit": self.load_jit,
            "load_trt": self.load_trt,
            "fp16": self.fp16,
            "stream": self.stream,
            "loaded": self._model is not None,
            "latency": "150ms first-packet (streaming)",
            "modes": ["zero_shot", "cross_lingual", "instruct"],
        })
        return info
