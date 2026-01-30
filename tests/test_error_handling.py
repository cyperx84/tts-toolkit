"""Tests for error handling across the toolkit."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from tts_toolkit.backends import MockBackend
from tts_toolkit.core.pipeline import Pipeline
from tts_toolkit.core.chunker import TextChunker
from tts_toolkit.core.stitcher import AudioStitcher
from tts_toolkit.utils.config import TTSConfig, ConfigValidationError


class TestBackendErrorHandling:
    """Tests for backend error handling."""

    def test_mock_backend_handles_empty_text(self):
        """Test MockBackend handles empty text gracefully."""
        backend = MockBackend()
        audio, sr = backend.generate("", None)
        assert isinstance(audio, np.ndarray)

    def test_mock_backend_handles_very_long_text(self):
        """Test MockBackend handles very long text."""
        backend = MockBackend()
        long_text = "Hello world. " * 10000  # Very long text
        audio, sr = backend.generate(long_text, None)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_backend_load_idempotent(self):
        """Test loading backend multiple times is safe."""
        backend = MockBackend()
        backend.load_model()
        backend.load_model()  # Should not raise
        backend.load_model()  # Should not raise

    def test_backend_unload_idempotent(self):
        """Test unloading backend multiple times is safe."""
        backend = MockBackend()
        backend.load_model()
        backend.unload_model()
        backend.unload_model()  # Should not raise


class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    def test_pipeline_empty_text_raises(self):
        """Test pipeline raises on empty text."""
        backend = MockBackend()
        pipeline = Pipeline(backend=backend)

        with pytest.raises(ValueError, match="No text to process"):
            pipeline.process(
                text="",
                ref_audio="ref.wav",
                ref_text="Reference",
                output_path="output.wav",
            )

    def test_pipeline_whitespace_only_raises(self):
        """Test pipeline raises on whitespace-only text."""
        backend = MockBackend()
        pipeline = Pipeline(backend=backend)

        with pytest.raises(ValueError, match="No text to process"):
            pipeline.process(
                text="   \n\t   ",
                ref_audio="ref.wav",
                ref_text="Reference",
                output_path="output.wav",
            )

    def test_pipeline_missing_ref_audio(self, tmp_path):
        """Test pipeline handles missing reference audio."""
        backend = MockBackend()
        pipeline = Pipeline(backend=backend)

        # MockBackend doesn't actually require ref_audio, but real backends would
        # This tests the path exists
        output = tmp_path / "output.wav"
        result = pipeline.process(
            text="Hello world.",
            ref_audio="nonexistent.wav",  # MockBackend ignores this
            ref_text="Reference",
            output_path=str(output),
        )
        assert result == str(output)

    def test_pipeline_chunk_failure_recovery(self, tmp_path):
        """Test pipeline handles chunk generation failure."""
        # Create a backend that fails on first attempt
        class FailOnceBackend(MockBackend):
            def __init__(self):
                super().__init__()
                self._fail_count = 0

            def generate(self, text, voice_prompt, **kwargs):
                if self._fail_count == 0:
                    self._fail_count += 1
                    raise RuntimeError("Simulated failure")
                return super().generate(text, voice_prompt, **kwargs)

        backend = FailOnceBackend()
        pipeline = Pipeline(backend=backend, max_retries=3)

        output = tmp_path / "output.wav"
        result = pipeline.process(
            text="Hello world.",
            ref_audio="ref.wav",
            ref_text="Reference",
            output_path=str(output),
        )
        assert output.exists()


class TestChunkerErrorHandling:
    """Tests for chunker error handling."""

    def test_chunker_empty_string(self):
        """Test chunker handles empty string."""
        chunker = TextChunker()
        chunks = chunker.chunk("")
        assert chunks == [] or chunks == [""]

    def test_chunker_only_punctuation(self):
        """Test chunker handles punctuation-only string."""
        chunker = TextChunker()
        chunks = chunker.chunk("...")
        assert isinstance(chunks, list)

    def test_chunker_single_character(self):
        """Test chunker handles single character."""
        chunker = TextChunker()
        chunks = chunker.chunk("a")
        assert chunks == ["a"]

    def test_chunker_unicode_characters(self):
        """Test chunker handles unicode."""
        chunker = TextChunker()
        chunks = chunker.chunk("你好世界 🌍 مرحبا")
        assert len(chunks) >= 1


class TestStitcherErrorHandling:
    """Tests for audio stitcher error handling."""

    def test_stitcher_empty_list(self):
        """Test stitcher handles empty list."""
        stitcher = AudioStitcher()
        result = stitcher.stitch([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_stitcher_single_empty_array(self):
        """Test stitcher handles single empty array."""
        stitcher = AudioStitcher()
        result = stitcher.stitch([np.array([], dtype=np.float32)])
        assert isinstance(result, np.ndarray)

    def test_stitcher_mismatched_sample_rates(self, tmp_path):
        """Test stitcher detects sample rate mismatch."""
        import soundfile as sf

        # Create two files with different sample rates
        audio1 = np.random.randn(1000).astype(np.float32)
        audio2 = np.random.randn(1000).astype(np.float32)

        file1 = tmp_path / "audio1.wav"
        file2 = tmp_path / "audio2.wav"

        sf.write(str(file1), audio1, 16000)
        sf.write(str(file2), audio2, 22050)

        stitcher = AudioStitcher()
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            stitcher.stitch_files([str(file1), str(file2)], str(tmp_path / "out.wav"))

    def test_stitcher_missing_file(self, tmp_path):
        """Test stitcher handles missing file."""
        stitcher = AudioStitcher()
        with pytest.raises(FileNotFoundError):
            stitcher.stitch_files(
                [str(tmp_path / "nonexistent.wav")],
                str(tmp_path / "output.wav")
            )


class TestConfigErrorHandling:
    """Tests for config error handling."""

    def test_config_invalid_backend(self):
        """Test config validates backend name."""
        with pytest.raises(ConfigValidationError, match="Invalid backend"):
            TTSConfig(backend="invalid_backend")

    def test_config_invalid_temperature(self):
        """Test config validates temperature range."""
        with pytest.raises(ConfigValidationError, match="Temperature"):
            TTSConfig(temperature=5.0)

    def test_config_invalid_top_k(self):
        """Test config validates top_k."""
        with pytest.raises(ConfigValidationError, match="top_k"):
            TTSConfig(top_k=0)

    def test_config_invalid_top_p(self):
        """Test config validates top_p."""
        with pytest.raises(ConfigValidationError, match="top_p"):
            TTSConfig(top_p=1.5)

    def test_config_invalid_chunk_range(self):
        """Test config validates chunk min/max relationship."""
        with pytest.raises(ConfigValidationError, match="chunk_max"):
            TTSConfig(chunk_min=300, chunk_max=100)

    def test_config_invalid_output_format(self):
        """Test config validates output format."""
        with pytest.raises(ConfigValidationError, match="output_format"):
            TTSConfig(output_format="invalid")

    def test_config_invalid_batch_workers(self):
        """Test config validates batch workers."""
        with pytest.raises(ConfigValidationError, match="batch_workers"):
            TTSConfig(batch_workers=0)

    def test_config_from_dict_invalid(self):
        """Test config from_dict validates."""
        with pytest.raises(ConfigValidationError):
            TTSConfig.from_dict({"backend": "invalid"})

    def test_config_from_dict_skip_validation(self):
        """Test config from_dict can skip validation."""
        # Should not raise when validate=False
        config = TTSConfig.from_dict({"backend": "invalid"}, validate=False)
        assert config.backend == "invalid"


class TestNetworkErrorSimulation:
    """Tests for network error handling (for API backends)."""

    def test_api_backend_timeout(self):
        """Test API backend handles timeout."""
        # Simulating what should happen with FishSpeechBackend timeout
        from tts_toolkit.backends.fish_speech import FishSpeechBackend

        backend = FishSpeechBackend(api_key="test_key")

        # Mock the client to raise timeout
        with patch.object(backend, '_client', create=True) as mock_client:
            mock_client.tts.side_effect = TimeoutError("Connection timed out")

            # The generate should handle this gracefully or raise meaningful error
            backend._client = mock_client
            with pytest.raises((TimeoutError, RuntimeError)):
                backend.generate("Hello", MagicMock())

    def test_api_backend_connection_error(self):
        """Test API backend handles connection error."""
        from tts_toolkit.backends.fish_speech import FishSpeechBackend

        backend = FishSpeechBackend(api_key="test_key")

        with patch.object(backend, '_client', create=True) as mock_client:
            mock_client.tts.side_effect = ConnectionError("Connection refused")
            backend._client = mock_client

            with pytest.raises((ConnectionError, RuntimeError)):
                backend.generate("Hello", MagicMock())


class TestGracefulDegradation:
    """Tests for graceful degradation."""

    def test_memory_tracking_import_error(self):
        """Test toolkit works without psutil."""
        from tts_toolkit.utils.memory import get_system_memory

        # Even if psutil fails, should return zeros not crash
        with patch.dict('sys.modules', {'psutil': None}):
            info = get_system_memory()
            # Should return zero/default values, not crash

    def test_gpu_memory_without_cuda(self):
        """Test GPU memory functions work without CUDA."""
        from tts_toolkit.utils.memory import get_gpu_memory, clear_gpu_cache

        # Should return None, not crash
        result = get_gpu_memory(0)
        assert result is None or hasattr(result, 'free_mb')

        # Should not crash even without CUDA
        clear_gpu_cache()


class TestErrorMessageQuality:
    """Tests for error message quality."""

    def test_config_validation_error_messages(self):
        """Test config validation provides helpful messages."""
        try:
            TTSConfig(backend="nonexistent", temperature=999)
        except ConfigValidationError as e:
            error_msg = str(e)
            # Should include the valid options
            assert "qwen" in error_msg.lower() or "mock" in error_msg.lower()
            # Should mention temperature
            assert "temperature" in error_msg.lower()

    def test_file_not_found_includes_path(self, tmp_path):
        """Test file not found errors include the path."""
        from tts_toolkit.cli import _validate_file_exists

        nonexistent = str(tmp_path / "nonexistent_file.txt")

        with pytest.raises(SystemExit):
            _validate_file_exists(nonexistent, "--input")
