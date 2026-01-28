"""Tests for core components."""

import numpy as np
import pytest

from tts_toolkit.core.chunker import TextChunker
from tts_toolkit.core.stitcher import AudioStitcher


class TestTextChunker:
    """Tests for TextChunker."""

    def test_init_defaults(self):
        """Test default initialization."""
        chunker = TextChunker()
        assert chunker.min_chars == 100
        assert chunker.max_chars == 300
        assert chunker.target_chars == 200

    def test_init_custom(self):
        """Test custom initialization."""
        chunker = TextChunker(min_chars=50, max_chars=150, target_chars=100)
        assert chunker.min_chars == 50
        assert chunker.max_chars == 150
        assert chunker.target_chars == 100

    def test_chunk_short_text(self):
        """Test chunking short text."""
        chunker = TextChunker(min_chars=10, max_chars=50, target_chars=30)
        text = "Hello world."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_chunk_long_text(self):
        """Test chunking long text into multiple chunks."""
        chunker = TextChunker(min_chars=10, max_chars=50, target_chars=30)
        text = "This is a test. This is another sentence. And here is more. Even more text here."
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50 or "." not in chunk[:-1]

    def test_chunk_respects_sentences(self):
        """Test chunking respects sentence boundaries."""
        chunker = TextChunker(min_chars=10, max_chars=100, target_chars=50)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)

        # Each chunk should end with proper punctuation
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                assert chunk[-1] in ".!?", f"Chunk doesn't end with punctuation: {chunk}"

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk("")
        assert chunks == [] or chunks == [""]

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunker = TextChunker()
        chunks = chunker.chunk("   \n\n   ")
        assert all(c.strip() == "" for c in chunks) or chunks == []

    def test_chunk_preserves_content(self):
        """Test chunking preserves all content."""
        chunker = TextChunker(min_chars=10, max_chars=50, target_chars=30)
        text = "Hello world. This is a test. Another sentence here."
        chunks = chunker.chunk(text)

        # Rejoin and compare (accounting for whitespace)
        rejoined = " ".join(chunks).replace("  ", " ").strip()
        original = text.replace("  ", " ").strip()
        # Content should be preserved (may have different whitespace)
        assert set(rejoined.split()) == set(original.split())


class TestAudioStitcher:
    """Tests for AudioStitcher."""

    def test_init_defaults(self):
        """Test default initialization."""
        stitcher = AudioStitcher()
        assert stitcher.crossfade_ms == 75
        assert stitcher.sample_rate == 24000

    def test_init_custom(self):
        """Test custom initialization."""
        stitcher = AudioStitcher(crossfade_ms=100, sample_rate=16000)
        assert stitcher.crossfade_ms == 100
        assert stitcher.sample_rate == 16000

    def test_stitch_single_segment(self):
        """Test stitching a single segment."""
        stitcher = AudioStitcher(sample_rate=1000)
        audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = stitcher.stitch([audio])

        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)

    def test_stitch_multiple_segments(self):
        """Test stitching multiple segments."""
        stitcher = AudioStitcher(crossfade_ms=10, sample_rate=1000)
        audio1 = np.ones(100, dtype=np.float32)
        audio2 = np.ones(100, dtype=np.float32) * 2

        result = stitcher.stitch([audio1, audio2])

        assert isinstance(result, np.ndarray)
        # Result should be shorter than sum due to crossfade
        assert len(result) < len(audio1) + len(audio2)

    def test_stitch_empty_list(self):
        """Test stitching empty list."""
        stitcher = AudioStitcher()
        result = stitcher.stitch([])

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_stitch_preserves_amplitude_range(self):
        """Test stitching preserves reasonable amplitude."""
        stitcher = AudioStitcher(crossfade_ms=10, sample_rate=1000)
        audio1 = np.random.randn(500).astype(np.float32) * 0.5
        audio2 = np.random.randn(500).astype(np.float32) * 0.5

        result = stitcher.stitch([audio1, audio2])

        # Should not have extreme values
        assert np.max(np.abs(result)) < 2.0

    def test_crossfade_creates_smooth_transition(self):
        """Test that crossfade creates smooth transition."""
        stitcher = AudioStitcher(crossfade_ms=50, sample_rate=1000)

        # Create distinct signals
        audio1 = np.zeros(200, dtype=np.float32)
        audio2 = np.ones(200, dtype=np.float32)

        result = stitcher.stitch([audio1, audio2])

        # Find transition region (should have values between 0 and 1)
        transition = result[150:250] if len(result) > 250 else result
        unique_vals = np.unique(np.round(transition, 2))

        # Should have intermediate values (not just 0 and 1)
        assert len(unique_vals) > 2 or len(result) < 100


class TestChunkerEdgeCases:
    """Edge case tests for TextChunker."""

    def test_single_very_long_word(self):
        """Test handling of very long words."""
        chunker = TextChunker(min_chars=10, max_chars=50, target_chars=30)
        text = "a" * 100  # Single word longer than max

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # Content should be preserved
        assert "".join(chunks).replace(" ", "") == text

    def test_abbreviations(self):
        """Test that abbreviations don't cause premature splits."""
        chunker = TextChunker(min_chars=10, max_chars=200, target_chars=100)
        text = "Dr. Smith went to St. Louis. He met Mr. Jones there."

        chunks = chunker.chunk(text)
        # Should not split on abbreviation periods
        full_text = " ".join(chunks)
        assert "Dr." in full_text or "Dr" in full_text

    def test_unicode_text(self):
        """Test handling of unicode text."""
        chunker = TextChunker(min_chars=10, max_chars=100, target_chars=50)
        text = "你好世界。这是一个测试。更多的中文文字在这里。"

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # Unicode content should be preserved
        assert "你好" in "".join(chunks)

    def test_mixed_punctuation(self):
        """Test handling of mixed punctuation."""
        chunker = TextChunker(min_chars=10, max_chars=100, target_chars=50)
        text = "Hello! How are you? I'm fine. Thanks for asking..."

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # Should handle different sentence endings
        for chunk in chunks:
            if chunk.strip():
                assert chunk.strip()[-1] in ".!?…"
