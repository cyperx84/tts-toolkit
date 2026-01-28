"""MP3 export utilities.

Requires pydub and ffmpeg to be installed:
    pip install pydub
    brew install ffmpeg  # or apt-get install ffmpeg
"""

import os
import tempfile
from typing import Optional

import numpy as np


def export_mp3(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 24000,
    bitrate: str = "192k",
    normalize: bool = True,
) -> str:
    """
    Export audio as MP3 file.

    Args:
        audio: Audio data as numpy array
        output_path: Output file path
        sample_rate: Sample rate in Hz
        bitrate: MP3 bitrate (e.g., "128k", "192k", "320k")
        normalize: Whether to normalize audio

    Returns:
        Output file path

    Raises:
        ImportError: If pydub is not installed
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for MP3 export. Install with: pip install pydub"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize if requested
    if normalize:
        max_val = np.abs(audio).max()
        if max_val > 0.99:
            audio = audio / max_val * 0.99

    # Convert to int16 for pydub
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=1,
    )

    # Export as MP3
    audio_segment.export(output_path, format="mp3", bitrate=bitrate)

    return output_path


def convert_wav_to_mp3(
    wav_path: str,
    output_path: Optional[str] = None,
    bitrate: str = "192k",
) -> str:
    """
    Convert WAV file to MP3.

    Args:
        wav_path: Path to input WAV file
        output_path: Output MP3 path (auto-generated if None)
        bitrate: MP3 bitrate

    Returns:
        Output file path
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for MP3 conversion. Install with: pip install pydub"
        )

    if output_path is None:
        output_path = os.path.splitext(wav_path)[0] + ".mp3"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    audio = AudioSegment.from_wav(wav_path)
    audio.export(output_path, format="mp3", bitrate=bitrate)

    return output_path


def check_mp3_support() -> bool:
    """
    Check if MP3 export is supported.

    Returns:
        True if pydub and ffmpeg are available
    """
    try:
        from pydub import AudioSegment
        from pydub.utils import which

        if which("ffmpeg") is None:
            return False

        return True
    except ImportError:
        return False
