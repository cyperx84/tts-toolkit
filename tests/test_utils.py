"""Tests for utility modules."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestConfig:
    """Tests for configuration module."""

    def test_tts_config_defaults(self):
        """Test TTSConfig default values."""
        from tts_toolkit.utils.config import TTSConfig

        config = TTSConfig()
        assert config.backend == "qwen"
        assert config.device == "cpu"
        assert config.temperature == 0.9
        assert config.chunk_min == 100
        assert config.chunk_max == 300

    def test_tts_config_custom(self):
        """Test TTSConfig with custom values."""
        from tts_toolkit.utils.config import TTSConfig

        config = TTSConfig(
            backend="kokoro",
            device="cuda:0",
            temperature=0.7,
        )
        assert config.backend == "kokoro"
        assert config.device == "cuda:0"
        assert config.temperature == 0.7

    def test_tts_config_to_dict(self):
        """Test config to_dict conversion."""
        from tts_toolkit.utils.config import TTSConfig

        config = TTSConfig(backend="mock")
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["backend"] == "mock"
        assert "device" in d
        assert "temperature" in d

    def test_tts_config_from_dict(self):
        """Test config from_dict creation."""
        from tts_toolkit.utils.config import TTSConfig

        data = {"backend": "bark", "device": "cuda:1"}
        config = TTSConfig.from_dict(data)

        assert config.backend == "bark"
        assert config.device == "cuda:1"
        # Defaults should still be set
        assert config.temperature == 0.9

    def test_tts_config_merge(self):
        """Test config merging."""
        from tts_toolkit.utils.config import TTSConfig

        base = TTSConfig(backend="qwen", device="cpu")
        override = TTSConfig(device="cuda:0", temperature=0.5)

        merged = base.merge(override)

        assert merged.backend == "qwen"  # From base
        assert merged.device == "cuda:0"  # Overridden
        assert merged.temperature == 0.5  # Overridden

    def test_save_and_load_yaml(self):
        """Test YAML save and load."""
        pytest.importorskip("yaml")
        from tts_toolkit.utils.config import save_yaml, load_yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yml"
            data = {"backend": "kokoro", "device": "cpu"}

            save_yaml(path, data)
            loaded = load_yaml(path)

            assert loaded == data

    def test_save_and_load_config(self):
        """Test full config save and load."""
        pytest.importorskip("yaml")
        from tts_toolkit.utils.config import TTSConfig, save_config, load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".tts_toolkit.yml"
            config = TTSConfig(backend="chatterbox", temperature=0.8)

            save_config(config, path)
            # Change directory to test loading
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                loaded = load_config()
                assert loaded.backend == "chatterbox"
                assert loaded.temperature == 0.8
            finally:
                os.chdir(old_cwd)


class TestMemory:
    """Tests for memory tracking module."""

    def test_get_system_memory(self):
        """Test getting system memory info."""
        from tts_toolkit.utils.memory import get_system_memory, MemoryInfo

        info = get_system_memory()
        assert isinstance(info, MemoryInfo)
        # Values may be 0 if psutil not available
        assert info.total_mb >= 0
        assert info.percent_used >= 0

    def test_memory_info_str(self):
        """Test MemoryInfo string representation."""
        from tts_toolkit.utils.memory import MemoryInfo

        info = MemoryInfo(
            total_mb=16000,
            available_mb=8000,
            used_mb=8000,
            percent_used=50.0,
        )
        s = str(info)
        assert "Memory" in s
        assert "50.0%" in s

    def test_estimate_model_memory(self):
        """Test model memory estimation."""
        from tts_toolkit.utils.memory import estimate_model_memory

        # Known backends should have estimates
        assert estimate_model_memory("qwen") > 0
        assert estimate_model_memory("kokoro") > 0
        assert estimate_model_memory("mock") == 0

        # Unknown defaults to something reasonable
        assert estimate_model_memory("unknown") > 0

    def test_estimate_audio_memory(self):
        """Test audio memory estimation."""
        from tts_toolkit.utils.memory import estimate_audio_memory

        # 1 second at 24000 Hz = 24000 * 4 bytes = ~0.09 MB
        mem = estimate_audio_memory(1.0, 24000)
        assert 0.08 < mem < 0.1

        # 10 seconds should be ~10x
        mem10 = estimate_audio_memory(10.0, 24000)
        assert 0.8 < mem10 < 1.0

    def test_estimate_text_memory(self):
        """Test text memory estimation."""
        from tts_toolkit.utils.memory import estimate_text_memory

        text_mb, audio_mb = estimate_text_memory(1000)  # 1000 chars

        assert text_mb > 0
        assert audio_mb > 0
        # Audio should be larger than text
        assert audio_mb > text_mb

    def test_check_memory_available(self):
        """Test memory availability check."""
        from tts_toolkit.utils.memory import check_memory_available

        # Very small amount should be available
        available, msg = check_memory_available(1.0, use_gpu=False)
        # May fail on very low memory systems, but should return tuple
        assert isinstance(available, bool)
        assert isinstance(msg, str)


class TestBatch:
    """Tests for batch processing module."""

    def test_batch_job_creation(self):
        """Test BatchJob creation."""
        from tts_toolkit.utils.batch import BatchJob

        job = BatchJob(
            input_path="input.txt",
            output_path="output.wav",
            ref_audio="voice.wav",
            ref_text="Reference",
        )

        assert job.input_path == "input.txt"
        assert job.output_path == "output.wav"
        assert job.language == "Auto"  # Default

    def test_batch_result(self):
        """Test BatchResult creation."""
        from tts_toolkit.utils.batch import BatchJob, BatchResult

        job = BatchJob(input_path="in.txt", output_path="out.wav")
        result = BatchResult(
            job=job,
            success=True,
            duration_sec=5.0,
            output_path="out.wav",
        )

        assert result.success is True
        assert result.duration_sec == 5.0

    def test_batch_summary(self):
        """Test BatchSummary creation."""
        from tts_toolkit.utils.batch import BatchJob, BatchResult, BatchSummary

        job = BatchJob(input_path="in.txt", output_path="out.wav")
        results = [
            BatchResult(job=job, success=True, duration_sec=5.0),
            BatchResult(job=job, success=True, duration_sec=3.0),
            BatchResult(job=job, success=False, error="Test error"),
        ]

        summary = BatchSummary(
            total_jobs=3,
            successful=2,
            failed=1,
            total_duration_sec=8.0,
            total_processing_time_sec=10.0,
            results=results,
        )

        assert summary.success_rate == pytest.approx(66.67, rel=0.1)
        assert len(summary.to_dict()["jobs"]) == 3

    def test_create_jobs_from_directory(self):
        """Test creating jobs from directory."""
        from tts_toolkit.utils.batch import create_jobs_from_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "script1.txt").write_text("Hello")
            (Path(tmpdir) / "script2.txt").write_text("World")
            (Path(tmpdir) / "other.md").write_text("Ignored")

            jobs = create_jobs_from_directory(
                input_dir=tmpdir,
                output_dir=tmpdir,
                ref_audio="voice.wav",
                ref_text="Ref",
                pattern="*.txt",
            )

            assert len(jobs) == 2
            assert all(j.ref_audio == "voice.wav" for j in jobs)

    def test_create_jobs_from_manifest(self):
        """Test creating jobs from manifest."""
        from tts_toolkit.utils.batch import create_jobs_from_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "defaults": {
                    "ref_audio": "voice.wav",
                    "ref_text": "Reference transcript",
                },
                "jobs": [
                    {"input": "script1.txt", "output": "out1.wav"},
                    {"input": "script2.txt", "output": "out2.wav", "language": "English"},
                ],
            }

            manifest_path = Path(tmpdir) / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            jobs = create_jobs_from_manifest(manifest_path)

            assert len(jobs) == 2
            assert jobs[0].language == "Auto"
            assert jobs[1].language == "English"


class TestMetadata:
    """Tests for metadata embedding module."""

    def test_create_tts_metadata(self):
        """Test TTS metadata creation."""
        from tts_toolkit.utils.metadata import create_tts_metadata

        metadata = create_tts_metadata(
            text="Hello world, this is a test.",
            backend_name="qwen",
            voice_name="narrator",
            language="English",
            duration_sec=65.5,
        )

        assert "encoder" in metadata
        assert "TTS Toolkit" in metadata["encoder"]
        assert metadata["artist"] == "narrator"
        assert metadata["language"] == "English"
        assert "1:05" in metadata["duration"]  # 65.5 seconds

    def test_create_tts_metadata_long_text(self):
        """Test metadata with long text truncation."""
        from tts_toolkit.utils.metadata import create_tts_metadata

        long_text = "A" * 200
        metadata = create_tts_metadata(
            text=long_text,
            backend_name="mock",
        )

        # Comment should be truncated
        assert len(metadata["comment"]) <= 104  # 100 chars + "..."

    def test_embed_metadata_wav_basic(self):
        """Test WAV metadata embedding."""
        from tts_toolkit.utils.metadata import embed_metadata_wav
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test WAV
            audio = np.random.randn(24000).astype(np.float32) * 0.5
            wav_path = Path(tmpdir) / "test.wav"
            sf.write(wav_path, audio, 24000)

            # Embed metadata
            output = embed_metadata_wav(
                wav_path,
                {"title": "Test Title", "artist": "Test Artist"},
            )

            assert Path(output).exists()
            # File should still be readable
            data, sr = sf.read(output)
            assert len(data) == len(audio)

    def test_read_metadata_wav(self):
        """Test reading WAV metadata."""
        from tts_toolkit.utils.metadata import read_metadata
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = np.random.randn(48000).astype(np.float32) * 0.5
            wav_path = Path(tmpdir) / "test.wav"
            sf.write(wav_path, audio, 24000)

            metadata = read_metadata(wav_path)

            assert "duration" in metadata
            assert metadata["duration"] == pytest.approx(2.0, rel=0.01)
            assert metadata["sample_rate"] == 24000


class TestPipelineParallel:
    """Tests for parallel chunk processing."""

    def test_pipeline_has_parallel_method(self):
        """Test Pipeline has process_parallel method."""
        from tts_toolkit.core.pipeline import Pipeline

        pipeline = Pipeline()
        assert hasattr(pipeline, "process_parallel")
        assert callable(pipeline.process_parallel)
