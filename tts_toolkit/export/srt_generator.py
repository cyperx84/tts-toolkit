"""SRT/VTT subtitle generation from audio segments."""

import os
from typing import List, Optional

from ..timing.srt_parser import SRTEntry, write_srt, write_vtt
from ..timing.duration_estimator import DurationEstimator


class SRTGenerator:
    """Generate SRT/VTT subtitles from TTS output."""

    def __init__(
        self,
        words_per_minute: float = 150,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
    ):
        """
        Initialize SRT generator.

        Args:
            words_per_minute: Estimated speaking rate
            max_chars_per_line: Maximum characters per subtitle line
            max_lines: Maximum lines per subtitle entry
        """
        self.estimator = DurationEstimator(words_per_minute=words_per_minute)
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines

    def generate_from_segments(
        self,
        segments: list,
        audio_durations: Optional[List[float]] = None,
    ) -> List[SRTEntry]:
        """
        Generate subtitle entries from segments.

        Args:
            segments: List of Segment objects
            audio_durations: Optional list of actual audio durations per segment

        Returns:
            List of SRTEntry objects
        """
        entries = []
        current_time_ms = 0
        index = 0

        for i, segment in enumerate(segments):
            # Handle pause before
            if hasattr(segment, 'pause_before_ms'):
                current_time_ms += segment.pause_before_ms

            # Get text and split if needed
            text = segment.text.strip()
            if not text:
                continue

            # Determine duration
            if audio_durations and i < len(audio_durations):
                duration_ms = int(audio_durations[i] * 1000)
            else:
                duration_ms = self.estimator.estimate_duration_ms(text)

            # Split text into subtitle-friendly chunks
            text_chunks = self._split_text(text)

            # Calculate duration per chunk
            chunk_duration_ms = duration_ms // len(text_chunks) if text_chunks else duration_ms

            for chunk in text_chunks:
                index += 1
                start_ms = current_time_ms
                end_ms = current_time_ms + chunk_duration_ms

                entries.append(
                    SRTEntry(
                        index=index,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=chunk,
                    )
                )

                current_time_ms = end_ms

            # Handle pause after
            if hasattr(segment, 'pause_after_ms'):
                current_time_ms += segment.pause_after_ms

        return entries

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into subtitle-friendly chunks.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        max_chars = self.max_chars_per_line * self.max_lines

        if len(text) <= max_chars:
            return [self._format_lines(text)]

        # Split at sentence boundaries first
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(self._format_lines(current_chunk))
                current_chunk = sentence

        if current_chunk:
            chunks.append(self._format_lines(current_chunk))

        return chunks

    def _format_lines(self, text: str) -> str:
        """
        Format text into subtitle lines.

        Args:
            text: Input text

        Returns:
            Formatted text with line breaks
        """
        if len(text) <= self.max_chars_per_line:
            return text

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            if len(test_line) <= self.max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

                if len(lines) >= self.max_lines:
                    break

        if current_line and len(lines) < self.max_lines:
            lines.append(current_line)

        return "\n".join(lines)

    def export_srt(
        self,
        entries: List[SRTEntry],
        output_path: str,
    ) -> str:
        """
        Export entries as SRT file.

        Args:
            entries: List of SRTEntry objects
            output_path: Output file path

        Returns:
            Output path
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        return write_srt(entries, output_path)

    def export_vtt(
        self,
        entries: List[SRTEntry],
        output_path: str,
    ) -> str:
        """
        Export entries as VTT file.

        Args:
            entries: List of SRTEntry objects
            output_path: Output file path

        Returns:
            Output path
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        return write_vtt(entries, output_path)

    def generate_and_export(
        self,
        segments: list,
        output_path: str,
        format: str = "srt",
        audio_durations: Optional[List[float]] = None,
    ) -> str:
        """
        Generate and export subtitles in one step.

        Args:
            segments: List of Segment objects
            output_path: Output file path
            format: Output format ("srt" or "vtt")
            audio_durations: Optional list of actual audio durations

        Returns:
            Output path
        """
        entries = self.generate_from_segments(segments, audio_durations)

        if format.lower() == "vtt":
            return self.export_vtt(entries, output_path)
        else:
            return self.export_srt(entries, output_path)
