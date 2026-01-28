"""Audio mixer for background music, sound effects, and transitions."""

import os
from typing import Optional

import numpy as np
import soundfile as sf


class AudioMixer:
    """Mix audio tracks with background music and effects.

    Args:
        sample_rate: Target sample rate for all audio

    Example:
        mixer = AudioMixer(sample_rate=24000)
        bg = mixer.load_audio("music.mp3")
        mixed = mixer.mix_background(speech, bg, background_db=-12)
    """

    def __init__(self, sample_rate: int = 24000):
        """Initialize audio mixer."""
        self.sample_rate = sample_rate

    def load_audio(self, path: str) -> np.ndarray:
        """Load and resample audio file."""
        audio, sr = sf.read(path, dtype="float32")

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            duration = len(audio) / sr
            target_samples = int(duration * self.sample_rate)
            indices = np.linspace(0, len(audio) - 1, target_samples)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        return audio.astype(np.float32)

    def adjust_volume(self, audio: np.ndarray, db: float) -> np.ndarray:
        """Adjust audio volume in decibels."""
        factor = 10 ** (db / 20)
        return audio * factor

    def normalize(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalize audio to target peak level."""
        max_val = np.abs(audio).max()
        if max_val == 0:
            return audio
        target_amp = 10 ** (target_db / 20)
        return audio * (target_amp / max_val)

    def fade_in(self, audio: np.ndarray, duration_ms: int) -> np.ndarray:
        """Apply fade in to audio."""
        fade_samples = int(duration_ms * self.sample_rate / 1000)
        fade_samples = min(fade_samples, len(audio))

        if fade_samples <= 0:
            return audio

        result = audio.copy()
        fade_curve = np.linspace(0, 1, fade_samples) ** 2
        result[:fade_samples] *= fade_curve

        return result

    def fade_out(self, audio: np.ndarray, duration_ms: int) -> np.ndarray:
        """Apply fade out to audio."""
        fade_samples = int(duration_ms * self.sample_rate / 1000)
        fade_samples = min(fade_samples, len(audio))

        if fade_samples <= 0:
            return audio

        result = audio.copy()
        fade_curve = np.linspace(1, 0, fade_samples) ** 2
        result[-fade_samples:] *= fade_curve

        return result

    def mix_background(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        background_db: float = -12.0,
        fade_in_ms: int = 1000,
        fade_out_ms: int = 2000,
    ) -> np.ndarray:
        """Mix foreground audio with background music.

        Args:
            foreground: Main audio (speech)
            background: Background audio (music)
            background_db: Volume of background relative to foreground in dB
            fade_in_ms: Fade in duration for background
            fade_out_ms: Fade out duration for background

        Returns:
            Mixed audio
        """
        fg_len = len(foreground)
        bg = background.copy()

        # Loop background if needed
        if len(bg) < fg_len:
            repeats = (fg_len // len(bg)) + 1
            bg = np.tile(bg, repeats)

        # Trim to match foreground
        bg = bg[:fg_len]

        # Apply fades
        bg = self.fade_in(bg, fade_in_ms)
        bg = self.fade_out(bg, fade_out_ms)

        # Adjust volume
        bg = self.adjust_volume(bg, background_db)

        # Mix
        mixed = foreground + bg

        # Normalize to prevent clipping
        max_val = np.abs(mixed).max()
        if max_val > 0.99:
            mixed = mixed / max_val * 0.99

        return mixed

    def add_intro_outro(
        self,
        main_audio: np.ndarray,
        intro: Optional[np.ndarray] = None,
        outro: Optional[np.ndarray] = None,
        crossfade_ms: int = 500,
    ) -> np.ndarray:
        """Add intro and/or outro with crossfade."""
        result = main_audio

        if intro is not None:
            crossfade_samples = int(crossfade_ms * self.sample_rate / 1000)
            crossfade_samples = min(crossfade_samples, len(intro), len(result))

            t = np.linspace(0, np.pi / 2, crossfade_samples)
            fade_out = np.cos(t) ** 2
            fade_in = np.sin(t) ** 2

            new_result = np.zeros(
                len(intro) + len(result) - crossfade_samples, dtype=np.float32
            )
            new_result[: len(intro) - crossfade_samples] = intro[:-crossfade_samples]

            xf_start = len(intro) - crossfade_samples
            new_result[xf_start : xf_start + crossfade_samples] = (
                intro[-crossfade_samples:] * fade_out
                + result[:crossfade_samples] * fade_in
            )
            new_result[xf_start + crossfade_samples :] = result[crossfade_samples:]

            result = new_result

        if outro is not None:
            crossfade_samples = int(crossfade_ms * self.sample_rate / 1000)
            crossfade_samples = min(crossfade_samples, len(result), len(outro))

            t = np.linspace(0, np.pi / 2, crossfade_samples)
            fade_out = np.cos(t) ** 2
            fade_in = np.sin(t) ** 2

            new_result = np.zeros(
                len(result) + len(outro) - crossfade_samples, dtype=np.float32
            )
            new_result[: len(result) - crossfade_samples] = result[:-crossfade_samples]

            xf_start = len(result) - crossfade_samples
            new_result[xf_start : xf_start + crossfade_samples] = (
                result[-crossfade_samples:] * fade_out
                + outro[:crossfade_samples] * fade_in
            )
            new_result[xf_start + crossfade_samples :] = outro[crossfade_samples:]

            result = new_result

        return result

    def save(self, audio: np.ndarray, output_path: str) -> str:
        """Save audio to file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, audio, self.sample_rate)
        return output_path
