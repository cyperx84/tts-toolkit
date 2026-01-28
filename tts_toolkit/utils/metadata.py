"""Audio metadata embedding utilities.

Supports embedding metadata in audio files:
- WAV: INFO chunks
- MP3: ID3 tags

Common metadata fields:
- title: Track title
- artist: Artist/creator name
- album: Album/collection name
- genre: Content genre
- comment: Additional comments
- date: Creation date
- language: Content language
- duration: Audio duration
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("tts-toolkit")


# Standard metadata fields mapping
METADATA_FIELDS = {
    "title": "title",
    "artist": "artist",
    "album": "album",
    "genre": "genre",
    "comment": "comment",
    "date": "date",
    "year": "year",
    "language": "language",
    "track": "track",
    "duration": "duration",
    "encoder": "encoder",
}


def embed_metadata_wav(
    audio_path: Union[str, Path],
    metadata: Dict[str, str],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Embed metadata in WAV file using INFO chunks.

    Args:
        audio_path: Path to WAV file
        metadata: Dictionary of metadata fields
        output_path: Output path (default: modify in place)

    Returns:
        Path to output file
    """
    import wave
    import struct

    audio_path = Path(audio_path)
    output_path = Path(output_path) if output_path else audio_path

    # Read existing WAV
    with wave.open(str(audio_path), 'rb') as wav:
        params = wav.getparams()
        frames = wav.readframes(params.nframes)

    # INFO chunk field IDs
    info_fields = {
        "title": b"INAM",
        "artist": b"IART",
        "album": b"IPRD",
        "genre": b"IGNR",
        "comment": b"ICMT",
        "date": b"ICRD",
        "encoder": b"ISFT",
        "language": b"ILNG",
    }

    # Build INFO chunk
    info_data = b""
    for key, value in metadata.items():
        field_id = info_fields.get(key.lower())
        if field_id and value:
            # Encode value with null terminator
            value_bytes = value.encode("utf-8") + b"\x00"
            # Pad to even length
            if len(value_bytes) % 2:
                value_bytes += b"\x00"
            # Add field: ID + size + data
            info_data += field_id + struct.pack("<I", len(value_bytes)) + value_bytes

    if info_data:
        # Wrap in LIST chunk
        list_data = b"INFO" + info_data
        list_chunk = b"LIST" + struct.pack("<I", len(list_data)) + list_data
    else:
        list_chunk = b""

    # Write new WAV with metadata
    with wave.open(str(output_path), 'wb') as wav:
        wav.setparams(params)
        wav.writeframes(frames)

    # Append LIST chunk after RIFF header
    if list_chunk:
        with open(output_path, 'r+b') as f:
            # Read RIFF header
            f.seek(0)
            riff = f.read(4)
            if riff != b"RIFF":
                logger.warning("Not a valid RIFF/WAV file")
                return str(output_path)

            # Read size
            size_bytes = f.read(4)
            size = struct.unpack("<I", size_bytes)[0]

            # Read rest of file
            f.seek(12)  # After RIFF size WAVE
            content = f.read()

            # Write updated file
            f.seek(0)
            f.write(b"RIFF")
            f.write(struct.pack("<I", size + len(list_chunk)))
            f.write(b"WAVE")
            f.write(content)
            f.write(list_chunk)
            f.truncate()

    logger.debug(f"Embedded metadata in WAV: {output_path}")
    return str(output_path)


def embed_metadata_mp3(
    audio_path: Union[str, Path],
    metadata: Dict[str, str],
    output_path: Optional[Union[str, Path]] = None,
    cover_image: Optional[Union[str, Path]] = None,
) -> str:
    """Embed metadata in MP3 file using ID3 tags.

    Args:
        audio_path: Path to MP3 file
        metadata: Dictionary of metadata fields
        output_path: Output path (default: modify in place)
        cover_image: Path to cover image (optional)

    Returns:
        Path to output file
    """
    try:
        from mutagen.mp3 import MP3
        from mutagen.id3 import ID3, TIT2, TPE1, TALB, TCON, COMM, TDRC, TLAN, TRCK, TENC
    except ImportError:
        logger.warning("mutagen not installed. Cannot embed MP3 metadata.")
        logger.warning("Install with: pip install mutagen")
        return str(audio_path)

    audio_path = Path(audio_path)
    output_path = Path(output_path) if output_path else audio_path

    # Copy file if different output
    if output_path != audio_path:
        import shutil
        shutil.copy2(audio_path, output_path)

    # Load or create ID3 tags
    try:
        audio = MP3(str(output_path))
        if audio.tags is None:
            audio.add_tags()
    except Exception:
        audio = MP3(str(output_path))
        audio.add_tags()

    # ID3 frame mapping
    frame_map = {
        "title": lambda v: TIT2(encoding=3, text=v),
        "artist": lambda v: TPE1(encoding=3, text=v),
        "album": lambda v: TALB(encoding=3, text=v),
        "genre": lambda v: TCON(encoding=3, text=v),
        "comment": lambda v: COMM(encoding=3, lang="eng", desc="", text=v),
        "date": lambda v: TDRC(encoding=3, text=v),
        "year": lambda v: TDRC(encoding=3, text=v),
        "language": lambda v: TLAN(encoding=3, text=v),
        "track": lambda v: TRCK(encoding=3, text=v),
        "encoder": lambda v: TENC(encoding=3, text=v),
    }

    for key, value in metadata.items():
        frame_func = frame_map.get(key.lower())
        if frame_func and value:
            audio.tags.add(frame_func(str(value)))

    # Add cover image if provided
    if cover_image:
        try:
            from mutagen.id3 import APIC
            cover_path = Path(cover_image)
            if cover_path.exists():
                mime_types = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                }
                mime = mime_types.get(cover_path.suffix.lower(), "image/jpeg")

                with open(cover_path, "rb") as f:
                    audio.tags.add(APIC(
                        encoding=3,
                        mime=mime,
                        type=3,  # Cover (front)
                        desc="Cover",
                        data=f.read(),
                    ))
        except Exception as e:
            logger.warning(f"Could not embed cover image: {e}")

    audio.save()
    logger.debug(f"Embedded metadata in MP3: {output_path}")
    return str(output_path)


def embed_metadata(
    audio_path: Union[str, Path],
    metadata: Dict[str, str],
    output_path: Optional[Union[str, Path]] = None,
    cover_image: Optional[Union[str, Path]] = None,
) -> str:
    """Embed metadata in audio file (auto-detect format).

    Args:
        audio_path: Path to audio file
        metadata: Dictionary of metadata fields
        output_path: Output path (default: modify in place)
        cover_image: Path to cover image (MP3 only)

    Returns:
        Path to output file

    Example:
        embed_metadata(
            "output.mp3",
            {
                "title": "Chapter 1",
                "artist": "Narrator",
                "album": "My Audiobook",
                "genre": "Audiobook",
                "date": "2024",
            },
            cover_image="cover.jpg",
        )
    """
    audio_path = Path(audio_path)
    suffix = audio_path.suffix.lower()

    if suffix == ".mp3":
        return embed_metadata_mp3(audio_path, metadata, output_path, cover_image)
    elif suffix == ".wav":
        return embed_metadata_wav(audio_path, metadata, output_path)
    else:
        logger.warning(f"Metadata embedding not supported for {suffix} files")
        return str(audio_path)


def read_metadata(audio_path: Union[str, Path]) -> Dict[str, Any]:
    """Read metadata from audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary of metadata fields
    """
    audio_path = Path(audio_path)
    suffix = audio_path.suffix.lower()
    metadata = {}

    if suffix == ".mp3":
        try:
            from mutagen.mp3 import MP3
            audio = MP3(str(audio_path))
            if audio.tags:
                tag_map = {
                    "TIT2": "title",
                    "TPE1": "artist",
                    "TALB": "album",
                    "TCON": "genre",
                    "TDRC": "date",
                    "TLAN": "language",
                    "TRCK": "track",
                }
                for tag_id, field in tag_map.items():
                    if tag_id in audio.tags:
                        metadata[field] = str(audio.tags[tag_id])

            metadata["duration"] = audio.info.length
            metadata["bitrate"] = audio.info.bitrate
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not read MP3 metadata: {e}")

    elif suffix == ".wav":
        try:
            import wave
            with wave.open(str(audio_path), 'rb') as wav:
                params = wav.getparams()
                metadata["duration"] = params.nframes / params.framerate
                metadata["sample_rate"] = params.framerate
                metadata["channels"] = params.nchannels
        except Exception as e:
            logger.debug(f"Could not read WAV metadata: {e}")

    return metadata


def create_tts_metadata(
    text: str,
    backend_name: str,
    voice_name: Optional[str] = None,
    language: str = "Auto",
    duration_sec: float = 0.0,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Create standard TTS metadata dictionary.

    Args:
        text: Source text (first 100 chars used for comment)
        backend_name: TTS backend used
        voice_name: Voice profile name
        language: Language
        duration_sec: Audio duration
        extra: Additional metadata fields

    Returns:
        Metadata dictionary ready for embedding
    """
    metadata = {
        "encoder": f"TTS Toolkit ({backend_name})",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "language": language if language != "Auto" else "",
        "comment": text[:100] + "..." if len(text) > 100 else text,
    }

    if voice_name:
        metadata["artist"] = voice_name

    if duration_sec > 0:
        # Format as MM:SS
        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)
        metadata["duration"] = f"{mins}:{secs:02d}"

    if extra:
        metadata.update(extra)

    # Remove empty values
    return {k: v for k, v in metadata.items() if v}
