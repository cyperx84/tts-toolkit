"""Duration estimation for speech synthesis."""

import re
from typing import Dict, Optional


class DurationEstimator:
    """Estimate speech duration for text.

    Uses word count and average speaking rates to estimate
    how long synthesized speech will be.
    """

    # Average words per minute by speaking style
    SPEAKING_RATES: Dict[str, float] = {
        "slow": 120,
        "normal": 150,
        "fast": 180,
        "audiobook": 140,
        "podcast": 160,
        "news": 165,
        "conversational": 145,
    }

    def __init__(
        self,
        words_per_minute: float = 150,
        style: Optional[str] = None,
    ):
        """
        Initialize estimator.

        Args:
            words_per_minute: Speaking rate in words per minute
            style: Speaking style (overrides words_per_minute)
        """
        if style and style in self.SPEAKING_RATES:
            self.wpm = self.SPEAKING_RATES[style]
        else:
            self.wpm = words_per_minute

    def count_words(self, text: str) -> int:
        """
        Count words in text.

        Args:
            text: Input text

        Returns:
            Word count
        """
        # Split on whitespace and filter empty
        words = [w for w in re.split(r'\s+', text) if w]
        return len(words)

    def estimate_duration_sec(self, text: str) -> float:
        """
        Estimate speech duration in seconds.

        Args:
            text: Input text

        Returns:
            Estimated duration in seconds
        """
        word_count = self.count_words(text)
        minutes = word_count / self.wpm
        return minutes * 60

    def estimate_duration_ms(self, text: str) -> int:
        """
        Estimate speech duration in milliseconds.

        Args:
            text: Input text

        Returns:
            Estimated duration in milliseconds
        """
        return int(self.estimate_duration_sec(text) * 1000)

    def format_duration(self, seconds: float) -> str:
        """
        Format duration as human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string like "5m 30s"
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

    def estimate_formatted(self, text: str) -> str:
        """
        Estimate and format duration.

        Args:
            text: Input text

        Returns:
            Formatted duration string
        """
        seconds = self.estimate_duration_sec(text)
        return self.format_duration(seconds)

    def adjust_for_pauses(
        self,
        duration_sec: float,
        num_sentences: int,
        pause_between_sentences_ms: int = 100,
    ) -> float:
        """
        Adjust duration for inter-sentence pauses.

        Args:
            duration_sec: Base duration in seconds
            num_sentences: Number of sentences
            pause_between_sentences_ms: Pause duration in ms

        Returns:
            Adjusted duration in seconds
        """
        pause_seconds = (num_sentences - 1) * pause_between_sentences_ms / 1000
        return duration_sec + pause_seconds

    def estimate_for_chunks(
        self,
        chunks: list,
        pause_between_ms: int = 75,
    ) -> float:
        """
        Estimate total duration for multiple chunks with pauses.

        Args:
            chunks: List of text chunks
            pause_between_ms: Pause between chunks in ms

        Returns:
            Total estimated duration in seconds
        """
        total = 0
        for chunk in chunks:
            total += self.estimate_duration_sec(chunk)

        # Add pauses
        if len(chunks) > 1:
            pause_seconds = (len(chunks) - 1) * pause_between_ms / 1000
            total += pause_seconds

        return total
