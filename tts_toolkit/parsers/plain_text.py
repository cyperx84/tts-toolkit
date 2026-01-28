"""Plain text parser."""

import re
from typing import List

from ..formats.base import Segment


class PlainTextParser:
    """Parser for plain text input."""

    def __init__(
        self,
        paragraph_pause_ms: int = 500,
        sentence_pause_ms: int = 100,
    ):
        """
        Initialize parser.

        Args:
            paragraph_pause_ms: Pause between paragraphs
            sentence_pause_ms: Pause between sentences
        """
        self.paragraph_pause_ms = paragraph_pause_ms
        self.sentence_pause_ms = sentence_pause_ms

    def parse(self, text: str) -> List[Segment]:
        """
        Parse plain text into segments.

        Args:
            text: Raw text content

        Returns:
            List of Segment objects
        """
        segments = []

        # Split into paragraphs
        paragraphs = re.split(r'\n\n+', text.strip())

        for para_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # Determine pause after paragraph
            is_last = para_idx == len(paragraphs) - 1
            pause_after = 0 if is_last else self.paragraph_pause_ms

            segments.append(
                Segment(
                    text=para,
                    pause_after_ms=pause_after,
                )
            )

        return segments

    def parse_file(self, path: str, encoding: str = "utf-8") -> List[Segment]:
        """
        Parse a text file.

        Args:
            path: Path to text file
            encoding: File encoding

        Returns:
            List of Segment objects
        """
        with open(path, "r", encoding=encoding) as f:
            text = f.read()
        return self.parse(text)
