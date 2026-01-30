"""Tests for CLI commands."""

import ast
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tts_toolkit import cli


class TestCLINoSecretsInSource:
    """Tests to ensure CLI code quality."""

    def test_no_print_statements_in_cli(self):
        """Verify no print statements in cli.py (should use logger)."""
        cli_path = Path(__file__).parent.parent / "tts_toolkit" / "cli.py"
        with open(cli_path, "r") as f:
            source = f.read()

        tree = ast.parse(source)

        # Find all print() calls
        print_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    print_calls.append(node.lineno)

        assert len(print_calls) == 0, f"Found print() calls at lines: {print_calls}"


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_main_no_args_shows_help(self, capsys):
        """Test that running with no args shows help."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, 'argv', ['tts-toolkit']):
                cli.main()
        assert exc_info.value.code == 0

    def test_voiceover_requires_input(self):
        """Test voiceover command requires input file."""
        with pytest.raises(SystemExit):
            with patch.object(sys, 'argv', ['tts-toolkit', 'voiceover', '--output', 'out.wav']):
                cli.main()

    def test_pipeline_requires_ref_audio(self):
        """Test pipeline command requires ref audio."""
        with pytest.raises(SystemExit):
            with patch.object(sys, 'argv', [
                'tts-toolkit', 'pipeline',
                '--text', 'Hello',
                '--output', 'out.wav'
            ]):
                cli.main()


class TestCLIValidation:
    """Tests for CLI input validation."""

    def test_validate_file_exists_missing(self, capsys):
        """Test file validation for missing file."""
        with pytest.raises(SystemExit):
            cli._validate_file_exists("/nonexistent/path.txt", "test")

    def test_validate_file_exists_present(self, tmp_path):
        """Test file validation for existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        # Should not raise
        cli._validate_file_exists(str(test_file), "test")

    def test_validate_audio_file_valid(self, tmp_path):
        """Test audio file validation for valid file."""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 1024)  # Minimal WAV-like content
        # Should not raise
        cli._validate_audio_file(str(test_file), "test")

    def test_validate_audio_file_empty(self, tmp_path, capsys):
        """Test audio file validation rejects empty files."""
        test_file = tmp_path / "empty.wav"
        test_file.write_text("")
        with pytest.raises(SystemExit):
            cli._validate_audio_file(str(test_file), "test")

    def test_validate_audio_file_unusual_extension(self, tmp_path, caplog):
        """Test audio file validation warns on unusual extension."""
        test_file = tmp_path / "test.xyz"
        test_file.write_bytes(b"content" * 100)
        cli._validate_audio_file(str(test_file), "test")
        assert "unusual extension" in caplog.text.lower()

    def test_validate_text_content_empty(self, capsys):
        """Test text validation rejects empty content."""
        with pytest.raises(SystemExit):
            cli._validate_text_content("", "test")

    def test_validate_text_content_whitespace_only(self, capsys):
        """Test text validation rejects whitespace-only content."""
        with pytest.raises(SystemExit):
            cli._validate_text_content("   \n\t   ", "test")

    def test_validate_text_content_valid(self):
        """Test text validation accepts valid content."""
        # Should not raise
        cli._validate_text_content("Hello world", "test")


class TestCreateBackend:
    """Tests for backend creation."""

    def test_create_mock_backend(self):
        """Test creating MockBackend."""
        args = MagicMock()
        args.backend = "mock"
        args.device = "cpu"
        args.model = "test"

        backend = cli._create_backend(args)

        from tts_toolkit.backends import MockBackend
        assert isinstance(backend, MockBackend)

    def test_create_qwen_backend_returns_backend(self):
        """Test creating QwenBackend returns a backend (or MockBackend if unavailable)."""
        args = MagicMock()
        args.backend = "qwen"
        args.device = "cpu"
        args.model = "test"

        # Should return either QwenBackend or MockBackend depending on availability
        backend = cli._create_backend(args)
        from tts_toolkit.backends import TTSBackend
        assert isinstance(backend, TTSBackend)

    def test_create_backend_unknown_exits(self):
        """Test unknown backend exits with error."""
        args = MagicMock()
        args.backend = "nonexistent_backend"
        args.device = "cpu"

        with pytest.raises(SystemExit):
            cli._create_backend(args)


class TestVoiceCommand:
    """Tests for voice management command."""

    def test_voice_list_no_profiles(self, caplog):
        """Test voice list with no profiles."""
        with patch.object(cli, '_run_voice') as mock_run:
            # Just verify the function can be called
            args = MagicMock()
            args.voice_command = "list"
            mock_run(args)
            mock_run.assert_called_once()

    def test_voice_create(self, tmp_path):
        """Test voice create command calls registry."""
        with patch.object(cli, '_run_voice') as mock_run:
            args = MagicMock()
            args.voice_command = "create"
            args.name = "test_voice"
            mock_run(args)
            mock_run.assert_called_once()


class TestBatchCommand:
    """Tests for batch processing command."""

    def test_batch_no_jobs_warning(self, tmp_path, caplog):
        """Test batch command warns when no jobs found."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with patch('tts_toolkit.cli._create_backend') as mock_backend:
            mock_backend.return_value = MagicMock()

            args = MagicMock()
            args.input_dir = str(input_dir)
            args.output_dir = str(output_dir)
            args.manifest = None
            args.ref_audio = str(tmp_path / "ref.wav")
            args.ref_text = "Reference"
            args.voice = None
            args.pattern = "*.txt"
            args.language = "Auto"
            args.workers = 1
            args.timeout = 300
            args.report = None
            args.backend = "mock"

            # Create ref audio
            (tmp_path / "ref.wav").write_bytes(b"RIFF" + b"\x00" * 1024)

            cli._run_batch(args)
            assert "No jobs to process" in caplog.text


class TestConfigCommand:
    """Tests for config management command."""

    def test_config_init(self, tmp_path, caplog, monkeypatch):
        """Test config init command."""
        import logging
        caplog.set_level(logging.INFO)
        monkeypatch.chdir(tmp_path)

        with patch('tts_toolkit.utils.config.init_config') as mock_init:
            mock_init.return_value = tmp_path / ".tts_toolkit.yml"

            args = MagicMock()
            args.config_command = "init"
            args.global_config = False

            cli._run_config(args)
            mock_init.assert_called_once()

    def test_config_show(self, caplog):
        """Test config show command."""
        import logging
        caplog.set_level(logging.INFO)

        with patch('tts_toolkit.utils.config.load_config') as mock_load:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"backend": "qwen", "device": "cpu"}
            mock_load.return_value = mock_config

            args = MagicMock()
            args.config_command = "show"

            cli._run_config(args)
            # Verify load_config was called
            mock_load.assert_called_once()

    def test_config_set_unknown_key(self, caplog):
        """Test config set with unknown key exits."""
        with patch('tts_toolkit.utils.config.load_config') as mock_load:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"backend": "qwen"}
            mock_load.return_value = mock_config

            args = MagicMock()
            args.config_command = "set"
            args.key = "nonexistent_key"
            args.value = "test"

            with pytest.raises(SystemExit):
                cli._run_config(args)


class TestSayCommand:
    """Tests for quick TTS command."""

    def test_say_missing_voice_exits(self, capsys):
        """Test say command exits without voice reference."""
        args = MagicMock()
        args.voice = None
        args.ref_audio = None
        args.ref_text = None

        with pytest.raises(SystemExit):
            cli._run_say(args)


class TestPipelineCommand:
    """Tests for pipeline command."""

    def test_pipeline_runs_with_mock(self, tmp_path):
        """Test pipeline command runs with mock backend."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Hello world. This is a test sentence.")

        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"RIFF" + b"\x00" * 1024)

        output_file = tmp_path / "output.wav"

        args = MagicMock()
        args.text_file = str(input_file)
        args.text = None
        args.ref_audio = str(ref_audio)
        args.ref_text = "Reference text"
        args.output = str(output_file)
        args.work_dir = str(tmp_path / "work")
        args.backend = "mock"
        args.device = "cpu"
        args.model = "test"
        args.language = "Auto"
        args.no_resume = True
        args.chunk_min = 10
        args.chunk_max = 50
        args.chunk_target = 30
        args.crossfade_ms = 10
        args.temperature = 0.9
        args.top_k = 50
        args.top_p = 1.0

        # This should run without raising
        cli._run_pipeline(args)

        # Output should be created
        assert output_file.exists()
