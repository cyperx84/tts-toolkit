"""Text chunking with sentence boundary detection."""

import re
from typing import List


class TextChunker:
    """Split text into chunks at sentence boundaries.

    This chunker is designed to split long texts into optimal segments
    for TTS generation. It respects sentence boundaries and common
    abbreviations to avoid awkward splits.

    Args:
        min_chars: Minimum characters per chunk (soft limit)
        max_chars: Maximum characters per chunk (hard limit)
        target_chars: Target chunk size for optimal TTS

    Example:
        chunker = TextChunker(min_chars=100, max_chars=300)
        chunks = chunker.chunk("Your long text here...")
    """

    # Common abbreviations to protect from sentence splitting
    ABBREVIATIONS = {
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
        "vs.", "etc.", "i.e.", "e.g.", "cf.", "viz.",
        "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
        "St.", "Ave.", "Blvd.", "Rd.", "Mt.",
        "Inc.", "Ltd.", "Corp.", "Co.",
        "No.", "Vol.", "Ch.", "Sec.", "Fig.", "pp.",
    }

    def __init__(
        self,
        min_chars: int = 100,
        max_chars: int = 300,
        target_chars: int = 200,
    ):
        """Initialize chunker."""
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.target_chars = target_chars

        # Build abbreviation pattern
        abbrev_pattern = "|".join(re.escape(a) for a in self.ABBREVIATIONS)
        self._abbrev_re = re.compile(f"({abbrev_pattern})", re.IGNORECASE)

    def _protect_abbreviations(self, text: str) -> str:
        """Replace periods in abbreviations with placeholder."""
        return self._abbrev_re.sub(lambda m: m.group(0).replace(".", "\x00"), text)

    def _restore_abbreviations(self, text: str) -> str:
        """Restore periods in abbreviations."""
        return text.replace("\x00", ".")

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, respecting abbreviations."""
        protected = self._protect_abbreviations(text)

        # Split on sentence-ending punctuation followed by whitespace and capital letter
        pattern = r'(?<=[.!?])[\s\"\'\)\]]+(?=[A-Z\"\'\(\[])'
        raw_sentences = re.split(pattern, protected)

        sentences = []
        for s in raw_sentences:
            restored = self._restore_abbreviations(s).strip()
            if restored:
                sentences.append(restored)

        return sentences

    def chunk(self, text: str) -> List[str]:
        """Split text into chunks suitable for TTS.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text.strip())

        if not text:
            return []

        sentences = self._split_sentences(text)

        if not sentences:
            return [text] if text else []

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            potential = (current_chunk + " " + sentence).strip() if current_chunk else sentence

            if len(potential) > self.max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            elif len(potential) > self.max_chars and not current_chunk:
                # Single sentence exceeds max - force split at word boundaries
                words = sentence.split()
                temp = ""
                for word in words:
                    test = (temp + " " + word).strip() if temp else word
                    if len(test) > self.max_chars and temp:
                        chunks.append(temp)
                        temp = word
                    else:
                        temp = test
                if temp:
                    current_chunk = temp
            else:
                current_chunk = potential

            if len(current_chunk) >= self.target_chars:
                chunks.append(current_chunk)
                current_chunk = ""

        if current_chunk:
            if len(current_chunk) < self.min_chars and chunks:
                last = chunks.pop()
                merged = last + " " + current_chunk
                if len(merged) <= self.max_chars:
                    chunks.append(merged)
                else:
                    chunks.append(last)
                    chunks.append(current_chunk)
            else:
                chunks.append(current_chunk)

        return chunks

    def chunk_file(self, file_path: str, encoding: str = "utf-8") -> List[str]:
        """Read and chunk a text file."""
        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()
        return self.chunk(text)
