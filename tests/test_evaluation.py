"""Tests for evaluation module."""

import numpy as np
import pytest

from tts_toolkit.evaluation.metrics import (
    _compute_wer,
    _estimate_mos_fallback,
    _get_simple_embedding,
)
from tts_toolkit.evaluation.evaluator import TTSEvaluator, EvaluationResult


class TestWERComputation:
    """Tests for Word Error Rate computation."""

    def test_identical_text(self):
        """Test WER is 0 for identical text."""
        wer = _compute_wer("hello world", "hello world")
        assert wer == 0.0

    def test_completely_different(self):
        """Test WER for completely different text."""
        wer = _compute_wer("hello world", "foo bar")
        assert wer == 1.0  # All words wrong

    def test_partial_match(self):
        """Test WER for partial match."""
        wer = _compute_wer("hello world test", "hello world")
        assert 0 < wer < 1

    def test_empty_reference(self):
        """Test WER with empty reference."""
        wer = _compute_wer("", "hello")
        assert wer == 1.0

    def test_empty_hypothesis(self):
        """Test WER with empty hypothesis."""
        wer = _compute_wer("hello world", "")
        assert wer == 1.0

    def test_both_empty(self):
        """Test WER with both empty."""
        wer = _compute_wer("", "")
        assert wer == 0.0

    def test_case_insensitive(self):
        """Test WER is case insensitive."""
        wer = _compute_wer("Hello World", "hello world")
        assert wer == 0.0

    def test_insertion_error(self):
        """Test WER with insertion."""
        wer = _compute_wer("hello world", "hello big world")
        assert wer > 0

    def test_deletion_error(self):
        """Test WER with deletion."""
        wer = _compute_wer("hello big world", "hello world")
        assert wer > 0

    def test_substitution_error(self):
        """Test WER with substitution."""
        wer = _compute_wer("hello world", "hello earth")
        assert wer == 0.5  # 1 wrong out of 2


class TestMOSFallback:
    """Tests for fallback MOS estimation."""

    def test_silent_audio(self):
        """Test MOS for silent audio."""
        audio = np.zeros(1000, dtype=np.float32)
        mos = _estimate_mos_fallback(audio, 24000)
        assert mos < 2.0  # Silent should have low MOS

    def test_normal_audio(self):
        """Test MOS for normal audio."""
        audio = np.random.randn(24000).astype(np.float32) * 0.3
        mos = _estimate_mos_fallback(audio, 24000)
        assert 2.0 <= mos <= 4.0

    def test_clipped_audio(self):
        """Test MOS for clipped audio."""
        audio = np.ones(1000, dtype=np.float32)  # All at max
        mos = _estimate_mos_fallback(audio, 24000)
        assert mos <= 2.0  # Clipping should lower MOS


class TestSimpleEmbedding:
    """Tests for simple speaker embedding fallback."""

    def test_embedding_shape(self):
        """Test embedding has correct shape."""
        audio = np.random.randn(24000).astype(np.float32)
        embedding = _get_simple_embedding(audio, 24000)
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 20  # 20 bins

    def test_different_audio_different_embedding(self):
        """Test different audio produces different embeddings."""
        audio1 = np.sin(np.linspace(0, 100, 24000)).astype(np.float32)
        audio2 = np.sin(np.linspace(0, 200, 24000)).astype(np.float32)

        emb1 = _get_simple_embedding(audio1, 24000)
        emb2 = _get_simple_embedding(audio2, 24000)

        # Should not be identical
        assert not np.allclose(emb1, emb2)


class TestTTSEvaluator:
    """Tests for TTSEvaluator class."""

    def test_init_defaults(self):
        """Test default initialization."""
        evaluator = TTSEvaluator()
        assert evaluator.compute_mos is True
        assert evaluator.compute_speaker_sim is True
        assert evaluator.compute_wer is True

    def test_init_custom(self):
        """Test custom initialization."""
        evaluator = TTSEvaluator(
            compute_mos=False,
            compute_wer=False,
        )
        assert evaluator.compute_mos is False
        assert evaluator.compute_wer is False


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = EvaluationResult(audio_path="test.wav")
        assert result.audio_path == "test.wav"
        assert result.utmos is None
        assert result.wer is None

    def test_with_metrics(self):
        """Test result with metrics."""
        result = EvaluationResult(
            audio_path="test.wav",
            utmos=4.2,
            speaker_similarity=0.85,
            wer=0.05,
        )
        assert result.utmos == 4.2
        assert result.speaker_similarity == 0.85
        assert result.wer == 0.05

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EvaluationResult(
            audio_path="test.wav",
            utmos=4.0,
        )
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["audio_path"] == "test.wav"
        assert d["utmos"] == 4.0
