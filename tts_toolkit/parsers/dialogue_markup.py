"""Dialogue markup parser supporting multiple formats."""

import re
from typing import List, Optional, Tuple

from ..formats.base import Segment


class DialogueMarkupParser:
    """Parser for various dialogue markup formats.

    Supports:
    - [S1]: text / [S2]: text (Dia-style)
    - [SPEAKER]: text (bracket style)
    - **Speaker**: text (Markdown bold)
    - SPEAKER: text (screenplay uppercase)
    """

    # Pattern definitions
    PATTERNS = {
        "dia": re.compile(r'\[S(\d)\]:\s*(.+?)(?=\[S\d\]:|$)', re.DOTALL),
        "bracket": re.compile(r'\[([A-Z][A-Za-z0-9_]+)\]:\s*(.+?)(?=\[[A-Z]|$)', re.DOTALL),
        "markdown": re.compile(r'\*\*([^*]+)\*\*:\s*(.+?)(?=\*\*[^*]+\*\*:|$)', re.DOTALL),
        "screenplay": re.compile(r'^([A-Z][A-Z0-9_]+):\s*(.+)$', re.MULTILINE),
    }

    def __init__(
        self,
        speaker_pause_ms: int = 300,
        same_speaker_pause_ms: int = 150,
        auto_detect: bool = True,
        format: Optional[str] = None,
    ):
        """
        Initialize parser.

        Args:
            speaker_pause_ms: Pause when speaker changes
            same_speaker_pause_ms: Pause between lines from same speaker
            auto_detect: Auto-detect format from content
            format: Force specific format ("dia", "bracket", "markdown", "screenplay")
        """
        self.speaker_pause_ms = speaker_pause_ms
        self.same_speaker_pause_ms = same_speaker_pause_ms
        self.auto_detect = auto_detect
        self.format = format

    def detect_format(self, text: str) -> Optional[str]:
        """
        Detect the dialogue format used in text.

        Args:
            text: Raw dialogue text

        Returns:
            Format name or None if not detected
        """
        # Check for [S1]/[S2] style first
        if re.search(r'\[S\d\]:', text):
            return "dia"

        # Check for [SPEAKER]: style
        if re.search(r'\[[A-Z][A-Za-z0-9_]+\]:', text):
            return "bracket"

        # Check for **Speaker**: style
        if re.search(r'\*\*[^*]+\*\*:', text):
            return "markdown"

        # Check for SPEAKER: style (uppercase only)
        if re.search(r'^[A-Z][A-Z0-9_]+:', text, re.MULTILINE):
            return "screenplay"

        return None

    def parse(self, text: str) -> List[Segment]:
        """
        Parse dialogue text into segments.

        Args:
            text: Raw dialogue content

        Returns:
            List of Segment objects
        """
        # Determine format
        format_to_use = self.format
        if self.auto_detect and not format_to_use:
            format_to_use = self.detect_format(text)

        if not format_to_use:
            # No dialogue format detected, treat as plain text
            return [Segment(text=text.strip())]

        pattern = self.PATTERNS.get(format_to_use)
        if not pattern:
            return [Segment(text=text.strip())]

        # Parse matches
        segments = []
        prev_speaker = None

        if format_to_use == "screenplay":
            # Line-by-line parsing for screenplay
            for line in text.strip().split('\n'):
                match = pattern.match(line.strip())
                if match:
                    speaker, dialogue = match.group(1), match.group(2).strip()
                    segments.append(
                        self._create_segment(speaker, dialogue, prev_speaker)
                    )
                    prev_speaker = speaker
        else:
            # Multi-line parsing for other formats
            matches = pattern.findall(text)
            for match in matches:
                speaker, dialogue = match[0].strip(), match[1].strip()
                if dialogue:
                    segments.append(
                        self._create_segment(speaker, dialogue, prev_speaker)
                    )
                    prev_speaker = speaker

        return segments

    def _create_segment(
        self,
        speaker: str,
        dialogue: str,
        prev_speaker: Optional[str],
    ) -> Segment:
        """Create a segment with appropriate pause."""
        if prev_speaker is None:
            pause_before = 0
        elif speaker != prev_speaker:
            pause_before = self.speaker_pause_ms
        else:
            pause_before = self.same_speaker_pause_ms

        return Segment(
            text=dialogue,
            speaker_id=speaker,
            pause_before_ms=pause_before,
        )

    def parse_with_emotions(self, text: str) -> List[Segment]:
        """
        Parse dialogue with inline emotion annotations.

        Supports:
        - [S1 (happy)]: text
        - [SPEAKER (excited)]: text

        Args:
            text: Raw dialogue with emotions

        Returns:
            List of Segment objects with emotions
        """
        segments = []
        prev_speaker = None

        # Extended pattern for emotions
        pattern = re.compile(
            r'\[([A-Za-z0-9_]+)(?:\s*\(([^)]+)\))?\]:\s*(.+?)(?=\[[A-Za-z]|$)',
            re.DOTALL
        )

        matches = pattern.findall(text)
        for match in matches:
            speaker = match[0].strip()
            emotion = match[1].strip() if match[1] else None
            dialogue = match[2].strip()

            if not dialogue:
                continue

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
                    emotion=emotion,
                    pause_before_ms=pause_before,
                )
            )
            prev_speaker = speaker

        return segments

    def parse_file(self, path: str, encoding: str = "utf-8") -> List[Segment]:
        """
        Parse a dialogue file.

        Args:
            path: Path to dialogue file
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
