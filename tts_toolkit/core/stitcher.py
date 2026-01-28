"""Audio stitching with crossfade for seamless transitions."""

import os
from typing import List, Optional

import numpy as np
import soundfile as sf


class AudioStitcher:
    """Concatenate audio chunks with crossfade transitions.

    Args:
        crossfade_ms: Crossfade duration in milliseconds (0 to disable)
        sample_rate: Sample rate (auto-detected from first file if None)

    Example:
        stitcher = AudioStitcher(crossfade_ms=75)
        combined = stitcher.stitch([audio1, audio2, audio3])
        stitcher.stitch_files(["a.wav", "b.wav"], "output.wav")
    """

    def __init__(
        self,
        crossfade_ms: int = 75,
        sample_rate: Optional[int] = None,
    ):
        """Initialize audio stitcher."""
        self.crossfade_ms = crossfade_ms
        self.sample_rate = sample_rate

    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio file and verify sample rate."""
        audio, sr = sf.read(path, dtype="float32")

        if self.sample_rate is None:
            self.sample_rate = sr
        elif sr != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: expected {self.sample_rate}, got {sr} in {path}"
            )

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        return audio

    def _crossfade(self, audio1: np.ndarray, audio2: np.ndarray) -> np.ndarray:
        """Apply crossfade between two audio arrays."""
        if self.crossfade_ms <= 0 or self.sample_rate is None:
            return np.concatenate([audio1, audio2])

        crossfade_samples = int(self.crossfade_ms * self.sample_rate / 1000)
        crossfade_samples = min(crossfade_samples, len(audio1), len(audio2))

        if crossfade_samples <= 0:
            return np.concatenate([audio1, audio2])

        # Create crossfade curves (equal power crossfade)
        t = np.linspace(0, np.pi / 2, crossfade_samples)
        fade_out = np.cos(t) ** 2
        fade_in = np.sin(t) ** 2

        result = np.zeros(len(audio1) + len(audio2) - crossfade_samples, dtype=np.float32)

        # Copy first part (before crossfade)
        result[: len(audio1) - crossfade_samples] = audio1[:-crossfade_samples]

        # Crossfade region
        crossfade_start = len(audio1) - crossfade_samples
        result[crossfade_start : crossfade_start + crossfade_samples] = (
            audio1[-crossfade_samples:] * fade_out + audio2[:crossfade_samples] * fade_in
        )

        # Copy second part (after crossfade)
        result[crossfade_start + crossfade_samples :] = audio2[crossfade_samples:]

        return result

    def stitch(self, audio_arrays: List[np.ndarray]) -> np.ndarray:
        """Stitch multiple audio arrays together with crossfade.

        Args:
            audio_arrays: List of audio arrays to concatenate

        Returns:
            Single concatenated audio array
        """
        if not audio_arrays:
            return np.array([], dtype=np.float32)

        if len(audio_arrays) == 1:
            return audio_arrays[0]

        result = audio_arrays[0]
        for audio in audio_arrays[1:]:
            result = self._crossfade(result, audio)

        return result

    def stitch_files(
        self,
        input_paths: List[str],
        output_path: str,
        normalize: bool = True,
    ) -> str:
        """Load audio files and stitch them together.

        Args:
            input_paths: List of input WAV file paths
            output_path: Path for output WAV file
            normalize: Whether to normalize to prevent clipping

        Returns:
            Output file path
        """
        if not input_paths:
            raise ValueError("No input files provided")

        audio_arrays = []
        for path in input_paths:
            audio = self._load_audio(path)
            audio_arrays.append(audio)

        stitched = self.stitch(audio_arrays)

        if normalize:
            max_val = np.abs(stitched).max()
            if max_val > 1.0:
                stitched = stitched / max_val * 0.99

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, stitched, self.sample_rate)

        return output_path

    def get_duration(self, audio: np.ndarray) -> float:
        """Get duration in seconds."""
        if self.sample_rate is None:
            raise ValueError("Sample rate not set")
        return len(audio) / self.sample_rate

    def add_silence(
        self,
        audio: np.ndarray,
        duration_ms: int,
        position: str = "end",
    ) -> np.ndarray:
        """Add silence to audio.

        Args:
            audio: Audio array
            duration_ms: Duration of silence in milliseconds
            position: Where to add ("start", "end", or "both")

        Returns:
            Audio with silence added
        """
        if self.sample_rate is None:
            raise ValueError("Sample rate not set")

        silence_samples = int(duration_ms * self.sample_rate / 1000)
        silence = np.zeros(silence_samples, dtype=np.float32)

        if position == "start":
            return np.concatenate([silence, audio])
        elif position == "end":
            return np.concatenate([audio, silence])
        elif position == "both":
            return np.concatenate([silence, audio, silence])
        else:
            raise ValueError(f"Invalid position: {position}")
