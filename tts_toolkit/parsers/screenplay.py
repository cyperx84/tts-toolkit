"""Screenplay format parser."""

import re
from typing import List, Optional, Tuple

from ..formats.base import Segment


class ScreenplayParser:
    """Parser for screenplay-style dialogue format.

    Supports formats like:
        JOHN: Hello, how are you?
        MARY: I'm doing well, thanks!
        JOHN: That's great to hear.
    """

    def __init__(
        self,
        speaker_pause_ms: int = 400,
        same_speaker_pause_ms: int = 150,
    ):
        """
        Initialize parser.

        Args:
            speaker_pause_ms: Pause when speaker changes
            same_speaker_pause_ms: Pause between lines from same speaker
        """
        self.speaker_pause_ms = speaker_pause_ms
        self.same_speaker_pause_ms = same_speaker_pause_ms

    def parse(self, text: str) -> List[Segment]:
        """
        Parse screenplay text into segments.

        Args:
            text: Raw screenplay content

        Returns:
            List of Segment objects
        """
        segments = []
        prev_speaker = None

        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # Parse speaker line
            parsed = self._parse_line(line)
            if parsed:
                speaker, dialogue = parsed

                # Determine pause
                if prev_speaker is None:
                    pause_before = 0
                elif speaker != prev_speaker:
                    pause_before = self.speaker_pause_ms
                else:
                    pause_before = self.same_speaker_pause_ms

                segments.append(
                    Segment(
                        text=dialogue,
                        speaker_id=speaker,
                        pause_before_ms=pause_before,
                    )
                )
                prev_speaker = speaker
            elif segments:
                # Continuation line (no speaker prefix)
                segments[-1].text += " " + line

        return segments

    def _parse_line(self, line: str) -> Optional[Tuple[str, str]]:
        """
        Parse a single screenplay line.

        Args:
            line: Line to parse

        Returns:
            Tuple of (speaker, dialogue) or None
        """
        # SPEAKER: dialogue
        match = re.match(r'^([A-Z][A-Za-z0-9_]*)\s*:\s*(.+)$', line)
        if match:
            return match.group(1), match.group(2)

        # ALL CAPS SPEAKER: dialogue
        match = re.match(r'^([A-Z][A-Z0-9_]+)\s*:\s*(.+)$', line)
        if match:
            return match.group(1), match.group(2)

        return None

    def parse_file(self, path: str, encoding: str = "utf-8") -> List[Segment]:
        """
        Parse a screenplay file.

        Args:
            path: Path to screenplay file
            encoding: File encoding

        Returns:
            List of Segment objects
        """
        with open(path, "r", encoding=encoding) as f:
            text = f.read()
        return self.parse(text)

    def detect_speakers(self, segments: List[Segment]) -> List[str]:
        """
        Get list of unique speakers in order of appearance.

        Args:
            segments: List of parsed segments

        Returns:
            List of speaker IDs
        """
        seen = set()
        speakers = []
        for segment in segments:
            if segment.speaker_id and segment.speaker_id not in seen:
                seen.add(segment.speaker_id)
                speakers.append(segment.speaker_id)
        return speakers
