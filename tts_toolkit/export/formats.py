"""Audio format definitions and quality presets."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AudioFormat:
    """Audio format specification."""

    name: str
    extension: str
    sample_rate: int
    bit_depth: Optional[int] = None  # For WAV
    bitrate: Optional[str] = None  # For MP3/AAC

    def __str__(self) -> str:
        return f"{self.name} ({self.extension})"


# Standard format definitions
FORMATS: Dict[str, AudioFormat] = {
    "wav_16bit": AudioFormat(
        name="WAV 16-bit",
        extension=".wav",
        sample_rate=24000,
        bit_depth=16,
    ),
    "wav_24bit": AudioFormat(
        name="WAV 24-bit",
        extension=".wav",
        sample_rate=24000,
        bit_depth=24,
    ),
    "wav_hq": AudioFormat(
        name="WAV High Quality",
        extension=".wav",
        sample_rate=44100,
        bit_depth=24,
    ),
    "mp3_128": AudioFormat(
        name="MP3 128kbps",
        extension=".mp3",
        sample_rate=24000,
        bitrate="128k",
    ),
    "mp3_192": AudioFormat(
        name="MP3 192kbps",
        extension=".mp3",
        sample_rate=24000,
        bitrate="192k",
    ),
    "mp3_320": AudioFormat(
        name="MP3 320kbps",
        extension=".mp3",
        sample_rate=24000,
        bitrate="320k",
    ),
}


@dataclass
class QualityPreset:
    """Quality preset for audio export."""

    name: str
    description: str
    format: AudioFormat
    normalize: bool = True


# Quality presets for different use cases
QUALITY_PRESETS: Dict[str, QualityPreset] = {
    "draft": QualityPreset(
        name="Draft",
        description="Fast preview quality",
        format=FORMATS["mp3_128"],
        normalize=True,
    ),
    "standard": QualityPreset(
        name="Standard",
        description="Good balance of quality and file size",
        format=FORMATS["mp3_192"],
        normalize=True,
    ),
    "high": QualityPreset(
        name="High Quality",
        description="High quality for distribution",
        format=FORMATS["mp3_320"],
        normalize=True,
    ),
    "lossless": QualityPreset(
        name="Lossless",
        description="Uncompressed WAV for editing",
        format=FORMATS["wav_16bit"],
        normalize=True,
    ),
    "podcast": QualityPreset(
        name="Podcast",
        description="Optimized for podcast distribution",
        format=AudioFormat(
            name="MP3 Podcast",
            extension=".mp3",
            sample_rate=44100,
            bitrate="96k",  # Mono voice is fine at 96k
        ),
        normalize=True,
    ),
    "audiobook": QualityPreset(
        name="Audiobook",
        description="Optimized for audiobook distribution",
        format=AudioFormat(
            name="MP3 Audiobook",
            extension=".mp3",
            sample_rate=22050,
            bitrate="64k",  # Mono voice, smaller files
        ),
        normalize=True,
    ),
    "video": QualityPreset(
        name="Video",
        description="High quality for video integration",
        format=FORMATS["wav_24bit"],
        normalize=True,
    ),
}


def get_format(name: str) -> Optional[AudioFormat]:
    """
    Get a format by name.

    Args:
        name: Format name

    Returns:
        AudioFormat or None
    """
    return FORMATS.get(name)


def get_preset(name: str) -> Optional[QualityPreset]:
    """
    Get a quality preset by name.

    Args:
        name: Preset name

    Returns:
        QualityPreset or None
    """
    return QUALITY_PRESETS.get(name)


def list_formats() -> Dict[str, str]:
    """
    List available formats with descriptions.

    Returns:
        Dictionary of format names to descriptions
    """
    return {name: str(fmt) for name, fmt in FORMATS.items()}


def list_presets() -> Dict[str, str]:
    """
    List available quality presets.

    Returns:
        Dictionary of preset names to descriptions
    """
    return {name: preset.description for name, preset in QUALITY_PRESETS.items()}
