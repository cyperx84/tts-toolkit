"""WAV export utilities."""

import os
from typing import Optional

import numpy as np
import soundfile as sf


def export_wav(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 24000,
    normalize: bool = True,
    subtype: str = "PCM_16",
) -> str:
    """
    Export audio as WAV file.

    Args:
        audio: Audio data as numpy array
        output_path: Output file path
        sample_rate: Sample rate in Hz
        normalize: Whether to normalize audio to prevent clipping
        subtype: WAV subtype (PCM_16, PCM_24, FLOAT, etc.)

    Returns:
        Output file path
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize if requested
    if normalize:
        max_val = np.abs(audio).max()
        if max_val > 0.99:
            audio = audio / max_val * 0.99

    sf.write(output_path, audio, sample_rate, subtype=subtype)
    return output_path


def read_wav(path: str) -> tuple:
    """
    Read WAV file.

    Args:
        path: Path to WAV file

    Returns:
        Tuple of (audio array, sample rate)
    """
    audio, sr = sf.read(path, dtype="float32")

    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sr


def get_wav_info(path: str) -> dict:
    """
    Get information about a WAV file.

    Args:
        path: Path to WAV file

    Returns:
        Dictionary with file info
    """
    info = sf.info(path)
    return {
        "duration_sec": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "frames": info.frames,
    }


def concatenate_wav_files(
    input_paths: list,
    output_path: str,
    crossfade_ms: int = 0,
) -> str:
    """
    Concatenate multiple WAV files.

    Args:
        input_paths: List of input WAV file paths
        output_path: Output file path
        crossfade_ms: Crossfade duration in milliseconds

    Returns:
        Output file path
    """
    if not input_paths:
        raise ValueError("No input files provided")

    # Read all files
    audios = []
    sample_rate = None

    for path in input_paths:
        audio, sr = read_wav(path)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {sr} vs {sample_rate}")
        audios.append(audio)

    # Concatenate with optional crossfade
    if crossfade_ms > 0 and len(audios) > 1:
        crossfade_samples = int(crossfade_ms * sample_rate / 1000)
        result = audios[0]

        for audio in audios[1:]:
            xf = min(crossfade_samples, len(result), len(audio))
            if xf > 0:
                # Equal power crossfade
                t = np.linspace(0, np.pi / 2, xf)
                fade_out = np.cos(t) ** 2
                fade_in = np.sin(t) ** 2

                new_result = np.zeros(len(result) + len(audio) - xf, dtype=np.float32)
                new_result[: len(result) - xf] = result[:-xf]
                new_result[len(result) - xf : len(result)] = (
                    result[-xf:] * fade_out + audio[:xf] * fade_in
                )
                new_result[len(result):] = audio[xf:]
                result = new_result
            else:
                result = np.concatenate([result, audio])
    else:
        result = np.concatenate(audios)

    return export_wav(result, output_path, sample_rate)
