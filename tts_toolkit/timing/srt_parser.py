"""SRT and VTT subtitle file parsing."""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class SRTEntry:
    """A single subtitle entry."""

    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        """Duration of this subtitle in milliseconds."""
        return self.end_ms - self.start_ms

    @property
    def start_sec(self) -> float:
        """Start time in seconds."""
        return self.start_ms / 1000

    @property
    def end_sec(self) -> float:
        """End time in seconds."""
        return self.end_ms / 1000


def parse_timestamp(timestamp: str) -> int:
    """
    Parse SRT/VTT timestamp to milliseconds.

    Args:
        timestamp: Timestamp string (HH:MM:SS,mmm or HH:MM:SS.mmm)

    Returns:
        Time in milliseconds
    """
    # Handle both comma (SRT) and period (VTT) separators
    timestamp = timestamp.replace(',', '.')

    # Parse HH:MM:SS.mmm
    match = re.match(r'(\d+):(\d+):(\d+)[.,](\d+)', timestamp)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        millis = int(match.group(4).ljust(3, '0')[:3])

        return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis

    # Try MM:SS.mmm format
    match = re.match(r'(\d+):(\d+)[.,](\d+)', timestamp)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        millis = int(match.group(3).ljust(3, '0')[:3])

        return (minutes * 60 + seconds) * 1000 + millis

    raise ValueError(f"Invalid timestamp format: {timestamp}")


def format_timestamp(ms: int, use_comma: bool = True) -> str:
    """
    Format milliseconds as SRT/VTT timestamp.

    Args:
        ms: Time in milliseconds
        use_comma: Use comma (SRT) vs period (VTT)

    Returns:
        Formatted timestamp string
    """
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    millis = ms % 1000

    sep = ',' if use_comma else '.'
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{sep}{millis:03d}"


def parse_srt(path: str, encoding: str = "utf-8") -> List[SRTEntry]:
    """
    Parse an SRT subtitle file.

    Args:
        path: Path to SRT file
        encoding: File encoding

    Returns:
        List of SRTEntry objects
    """
    with open(path, "r", encoding=encoding) as f:
        content = f.read()

    return parse_srt_content(content)


def parse_srt_content(content: str) -> List[SRTEntry]:
    """
    Parse SRT content from string.

    Args:
        content: SRT file content

    Returns:
        List of SRTEntry objects
    """
    entries = []

    # Split into blocks (separated by blank lines)
    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # First line is index
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Second line is timestamp
        timestamp_match = re.match(
            r'(\d+:\d+:\d+[.,]\d+)\s*-->\s*(\d+:\d+:\d+[.,]\d+)',
            lines[1].strip()
        )
        if not timestamp_match:
            continue

        start_ms = parse_timestamp(timestamp_match.group(1))
        end_ms = parse_timestamp(timestamp_match.group(2))

        # Remaining lines are text
        text = '\n'.join(lines[2:]).strip()

        entries.append(
            SRTEntry(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
        )

    return entries


def parse_vtt(path: str, encoding: str = "utf-8") -> List[SRTEntry]:
    """
    Parse a VTT (WebVTT) subtitle file.

    Args:
        path: Path to VTT file
        encoding: File encoding

    Returns:
        List of SRTEntry objects
    """
    with open(path, "r", encoding=encoding) as f:
        content = f.read()

    return parse_vtt_content(content)


def parse_vtt_content(content: str) -> List[SRTEntry]:
    """
    Parse VTT content from string.

    Args:
        content: VTT file content

    Returns:
        List of SRTEntry objects
    """
    entries = []

    # Remove WEBVTT header
    content = re.sub(r'^WEBVTT\s*\n', '', content)
    content = re.sub(r'^NOTE.*?\n\n', '', content, flags=re.DOTALL)

    # Split into blocks
    blocks = re.split(r'\n\n+', content.strip())

    index = 0
    for block in blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue

        # Find timestamp line
        timestamp_line = None
        text_start = 0

        for i, line in enumerate(lines):
            if '-->' in line:
                timestamp_line = line
                text_start = i + 1
                break

        if not timestamp_line:
            continue

        # Parse timestamp
        timestamp_match = re.match(
            r'(\d+:\d+[:\.]?\d*[.,]?\d*)\s*-->\s*(\d+:\d+[:\.]?\d*[.,]?\d*)',
            timestamp_line.strip()
        )
        if not timestamp_match:
            continue

        start_ms = parse_timestamp(timestamp_match.group(1))
        end_ms = parse_timestamp(timestamp_match.group(2))

        # Get text
        text = '\n'.join(lines[text_start:]).strip()

        # Remove VTT styling tags
        text = re.sub(r'<[^>]+>', '', text)

        index += 1
        entries.append(
            SRTEntry(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
        )

    return entries


def write_srt(entries: List[SRTEntry], path: str, encoding: str = "utf-8") -> str:
    """
    Write entries to SRT file.

    Args:
        entries: List of SRTEntry objects
        path: Output file path
        encoding: File encoding

    Returns:
        Output path
    """
    with open(path, "w", encoding=encoding) as f:
        for i, entry in enumerate(entries, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(entry.start_ms)} --> {format_timestamp(entry.end_ms)}\n")
            f.write(f"{entry.text}\n\n")

    return path


def write_vtt(entries: List[SRTEntry], path: str, encoding: str = "utf-8") -> str:
    """
    Write entries to VTT file.

    Args:
        entries: List of SRTEntry objects
        path: Output file path
        encoding: File encoding

    Returns:
        Output path
    """
    with open(path, "w", encoding=encoding) as f:
        f.write("WEBVTT\n\n")

        for entry in entries:
            f.write(f"{format_timestamp(entry.start_ms, use_comma=False)} --> "
                    f"{format_timestamp(entry.end_ms, use_comma=False)}\n")
            f.write(f"{entry.text}\n\n")

    return path
