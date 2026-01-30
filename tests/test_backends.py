"""Tests for TTS backends."""

import numpy as np
import pytest

from tts_toolkit.backends import MockBackend, TTSBackend, VoicePrompt


class TestMockBackend:
    """Tests for MockBackend."""

    def test_init_defaults(self):
        """Test default initialization."""
        backend = MockBackend()
        assert backend.sample_rate == 24000
        assert backend.mode == "silent"

    def test_init_custom(self):
        """Test custom initialization."""
        backend = MockBackend(sample_rate=16000, mode="sine")
        assert backend.sample_rate == 16000
        assert backend.mode == "sine"

    def test_load_model(self):
        """Test model loading is idempotent."""
        backend = MockBackend()
        backend.load_model()
        backend.load_model()  # Should not raise

    def test_create_voice_prompt(self):
        """Test voice prompt creation."""
        backend = MockBackend()
        prompt = backend.create_voice_prompt("ref.wav", "Reference text")

        assert isinstance(prompt, VoicePrompt)
        assert prompt.reference_audio == "ref.wav"
        assert prompt.reference_text == "Reference text"

    def test_generate_silent(self):
        """Test silent audio generation."""
        backend = MockBackend(mode="silent")
        audio, sr = backend.generate("Hello world", None)

        assert isinstance(audio, np.ndarray)
        assert sr == 24000
        assert len(audio) > 0
        assert np.max(np.abs(audio)) < 0.01  # Should be silent

    def test_generate_sine(self):
        """Test sine wave generation."""
        backend = MockBackend(mode="sine")
        audio, sr = backend.generate("Hello world", None)

        assert isinstance(audio, np.ndarray)
        assert sr == 24000
        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0.1  # Should have signal

    def test_generate_with_voice_prompt(self):
        """Test generation with voice prompt."""
        backend = MockBackend()
        prompt = backend.create_voice_prompt("ref.wav", "text")
        audio, sr = backend.generate("Hello", prompt)

        assert isinstance(audio, np.ndarray)
        assert sr == 24000

    def test_supports_voice_cloning(self):
        """Test voice cloning support flag."""
        backend = MockBackend()
        assert backend.supports_voice_cloning() is True

    def test_supports_streaming(self):
        """Test streaming support flag."""
        backend = MockBackend()
        assert backend.supports_streaming() is False

    def test_get_supported_languages(self):
        """Test supported languages."""
        backend = MockBackend()
        languages = backend.get_supported_languages()
        assert isinstance(languages, list)
        assert "Auto" in languages

    def test_get_info(self):
        """Test backend info."""
        backend = MockBackend(sample_rate=22050, mode="sine")
        info = backend.get_info()

        assert info["sample_rate"] == 22050
        assert info["mode"] == "sine"
        assert info["supports_voice_cloning"] is True


class TestVoicePrompt:
    """Tests for VoicePrompt dataclass."""

    def test_basic_creation(self):
        """Test basic voice prompt creation."""
        prompt = VoicePrompt(
            reference_audio="test.wav",
            reference_text="Hello",
        )
        assert prompt.reference_audio == "test.wav"
        assert prompt.reference_text == "Hello"
        assert prompt.speaker_embedding is None
        assert prompt.backend_data is None

    def test_with_embedding(self):
        """Test voice prompt with embedding."""
        embedding = np.array([0.1, 0.2, 0.3])
        prompt = VoicePrompt(
            reference_audio="test.wav",
            reference_text="Hello",
            speaker_embedding=embedding,
        )
        assert np.array_equal(prompt.speaker_embedding, embedding)

    def test_with_backend_data(self):
        """Test voice prompt with backend-specific data."""
        prompt = VoicePrompt(
            reference_audio="test.wav",
            reference_text="Hello",
            backend_data={"cached_prompt": "xyz"},
        )
        assert prompt.backend_data["cached_prompt"] == "xyz"


class TestTTSBackendInterface:
    """Tests for TTSBackend abstract interface."""

    def test_mock_implements_interface(self):
        """Test MockBackend implements TTSBackend."""
        backend = MockBackend()
        assert isinstance(backend, TTSBackend)

    def test_abstract_methods_exist(self):
        """Test abstract methods are defined."""
        backend = MockBackend()

        # Required methods
        assert hasattr(backend, "load_model")
        assert hasattr(backend, "create_voice_prompt")
        assert hasattr(backend, "generate")
        assert hasattr(backend, "sample_rate")
        assert hasattr(backend, "supports_voice_cloning")

        # Optional methods with defaults
        assert hasattr(backend, "supports_streaming")
        assert hasattr(backend, "supports_emotions")
        assert hasattr(backend, "get_supported_languages")
        assert hasattr(backend, "unload_model")
        assert hasattr(backend, "get_info")


class TestBackendRegistry:
    """Tests for backend registry."""

    def test_available_backends_dict(self):
        """Test AVAILABLE_BACKENDS is defined."""
        from tts_toolkit.backends import AVAILABLE_BACKENDS

        assert isinstance(AVAILABLE_BACKENDS, dict)
        assert "mock" in AVAILABLE_BACKENDS
        assert "qwen" in AVAILABLE_BACKENDS

    def test_list_backends(self, capsys):
        """Test list_backends function."""
        from tts_toolkit.backends import list_backends

        list_backends()
        captured = capsys.readouterr()
        assert "MockBackend" in captured.out
        assert "Available TTS Backends" in captured.out
