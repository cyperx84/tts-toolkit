"""Tests for export modules."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


class TestWavExport:
    """Tests for WAV export utilities."""

    def test_export_wav_basic(self):
        """Test basic WAV export."""
        from tts_toolkit.export.wav import export_wav

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = np.random.randn(24000).astype(np.float32) * 0.5
            output_path = Path(tmpdir) / "test.wav"

            result = export_wav(audio, str(output_path), sample_rate=24000)

            assert Path(result).exists()
            data, sr = sf.read(result)
            assert sr == 24000
            assert len(data) == len(audio)

    def test_export_wav_normalize(self):
        """Test WAV export with normalization."""
        from tts_toolkit.export.wav import export_wav

        with tempfile.TemporaryDirectory() as tmpdir:
            # Audio that would clip without normalization
            audio = np.ones(1000, dtype=np.float32) * 1.5
            output_path = Path(tmpdir) / "test.wav"

            export_wav(audio, str(output_path), normalize=True)

            data, _ = sf.read(str(output_path))
            assert np.max(np.abs(data)) <= 0.99

    def test_export_wav_creates_directory(self):
        """Test that export_wav creates missing directories."""
        from tts_toolkit.export.wav import export_wav

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = np.random.randn(1000).astype(np.float32)
            output_path = Path(tmpdir) / "subdir" / "deep" / "test.wav"

            result = export_wav(audio, str(output_path))
            assert Path(result).exists()

    def test_read_wav(self):
        """Test reading WAV file."""
        from tts_toolkit.export.wav import read_wav, export_wav

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = np.random.randn(24000).astype(np.float32) * 0.5
            wav_path = Path(tmpdir) / "test.wav"
            export_wav(audio, str(wav_path), sample_rate=24000)

            loaded_audio, sr = read_wav(str(wav_path))

            assert sr == 24000
            assert len(loaded_audio) == len(audio)
            assert loaded_audio.dtype == np.float32

    def test_read_wav_file_not_found(self):
        """Test read_wav raises error for missing file."""
        from tts_toolkit.export.wav import read_wav

        with pytest.raises(FileNotFoundError):
            read_wav("/nonexistent/path.wav")

    def test_read_wav_stereo_to_mono(self):
        """Test that stereo WAV is converted to mono."""
        from tts_toolkit.export.wav import read_wav

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create stereo audio
            stereo = np.random.randn(1000, 2).astype(np.float32) * 0.5
            wav_path = Path(tmpdir) / "stereo.wav"
            sf.write(wav_path, stereo, 24000)

            audio, sr = read_wav(str(wav_path))

            assert audio.ndim == 1  # Should be mono
            assert len(audio) == 1000

    def test_get_wav_info(self):
        """Test getting WAV file information."""
        from tts_toolkit.export.wav import get_wav_info, export_wav

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = np.random.randn(48000).astype(np.float32) * 0.5
            wav_path = Path(tmpdir) / "test.wav"
            export_wav(audio, str(wav_path), sample_rate=24000)

            info = get_wav_info(str(wav_path))

            assert info["sample_rate"] == 24000
            assert info["duration_sec"] == pytest.approx(2.0, rel=0.01)
            assert info["channels"] == 1
            assert "format" in info

    def test_get_wav_info_file_not_found(self):
        """Test get_wav_info raises error for missing file."""
        from tts_toolkit.export.wav import get_wav_info

        with pytest.raises(FileNotFoundError):
            get_wav_info("/nonexistent/path.wav")

    def test_concatenate_wav_files(self):
        """Test concatenating multiple WAV files."""
        from tts_toolkit.export.wav import concatenate_wav_files, export_wav

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            audio1 = np.random.randn(24000).astype(np.float32) * 0.3
            audio2 = np.random.randn(24000).astype(np.float32) * 0.3

            path1 = Path(tmpdir) / "test1.wav"
            path2 = Path(tmpdir) / "test2.wav"
            output = Path(tmpdir) / "concat.wav"

            export_wav(audio1, str(path1), sample_rate=24000)
            export_wav(audio2, str(path2), sample_rate=24000)

            result = concatenate_wav_files([str(path1), str(path2)], str(output))

            assert Path(result).exists()
            data, sr = sf.read(result)
            assert sr == 24000
            # Combined should be ~2 seconds
            assert len(data) == len(audio1) + len(audio2)

    def test_concatenate_wav_files_with_crossfade(self):
        """Test concatenating WAV files with crossfade."""
        from tts_toolkit.export.wav import concatenate_wav_files, export_wav

        with tempfile.TemporaryDirectory() as tmpdir:
            audio1 = np.ones(24000, dtype=np.float32) * 0.3
            audio2 = np.ones(24000, dtype=np.float32) * 0.3

            path1 = Path(tmpdir) / "test1.wav"
            path2 = Path(tmpdir) / "test2.wav"
            output = Path(tmpdir) / "concat.wav"

            export_wav(audio1, str(path1), sample_rate=24000)
            export_wav(audio2, str(path2), sample_rate=24000)

            result = concatenate_wav_files(
                [str(path1), str(path2)],
                str(output),
                crossfade_ms=100,
            )

            data, _ = sf.read(result)
            # With crossfade, result should be shorter
            crossfade_samples = int(100 * 24000 / 1000)
            expected_len = len(audio1) + len(audio2) - crossfade_samples
            assert len(data) == expected_len

    def test_concatenate_wav_files_empty_list(self):
        """Test concatenate raises error for empty list."""
        from tts_toolkit.export.wav import concatenate_wav_files

        with pytest.raises(ValueError, match="No input files"):
            concatenate_wav_files([], "output.wav")

    def test_concatenate_wav_files_missing_file(self):
        """Test concatenate raises error for missing file."""
        from tts_toolkit.export.wav import concatenate_wav_files

        with pytest.raises(FileNotFoundError):
            concatenate_wav_files(["/nonexistent.wav"], "output.wav")


class TestMp3Export:
    """Tests for MP3 export utilities."""

    def test_check_mp3_support(self):
        """Test MP3 support check."""
        from tts_toolkit.export.mp3 import check_mp3_support

        # Should return bool without error
        result = check_mp3_support()
        assert isinstance(result, bool)


class TestSrtGenerator:
    """Tests for SRT subtitle generation."""

    def test_generate_srt_from_segments(self):
        """Test SRT generation from segments."""
        from tts_toolkit.export.srt_generator import SRTGenerator
        from tts_toolkit.formats.base import Segment

        segments = [
            Segment(text="Hello world", speaker_id="S1"),
            Segment(text="How are you?", speaker_id="S2"),
        ]

        # Create audio durations (in seconds)
        audio_durations = [1.5, 1.5]

        generator = SRTGenerator()
        entries = generator.generate_from_segments(segments, audio_durations)

        assert len(entries) >= 2
        # Check first entry
        assert entries[0].text == "Hello world" or "Hello world" in entries[0].text
        # Check entries have timing info
        assert entries[0].start_ms == 0

    def test_srt_entry_format(self):
        """Test SRT entry has correct structure."""
        from tts_toolkit.export.srt_generator import SRTGenerator
        from tts_toolkit.formats.base import Segment
        from tts_toolkit.timing.srt_parser import SRTEntry

        segments = [Segment(text="Test text")]
        generator = SRTGenerator()
        entries = generator.generate_from_segments(segments)

        assert len(entries) >= 1
        assert isinstance(entries[0], SRTEntry)
        assert entries[0].index >= 1
        assert entries[0].start_ms >= 0
        assert entries[0].end_ms > entries[0].start_ms
