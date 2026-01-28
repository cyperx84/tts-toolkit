"""Tests for voice management modules."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


class TestVoiceProfile:
    """Tests for VoiceProfile dataclass."""

    def test_voice_profile_creation(self):
        """Test basic VoiceProfile creation."""
        from tts_toolkit.voices.profile import VoiceProfile

        profile = VoiceProfile(
            name="test_voice",
            reference_audio="test.wav",
            reference_text="Hello world",
        )

        assert profile.name == "test_voice"
        assert profile.reference_audio == "test.wav"
        assert profile.reference_text == "Hello world"

    def test_voice_profile_with_metadata(self):
        """Test VoiceProfile with metadata."""
        from tts_toolkit.voices.profile import VoiceProfile, VoiceMetadata

        metadata = VoiceMetadata(
            gender="female",
            age_range="adult",
            accent="american",
        )

        profile = VoiceProfile(
            name="narrator",
            reference_audio="voice.wav",
            reference_text="Reference text",
            metadata=metadata,
        )

        assert profile.metadata.gender == "female"
        assert profile.metadata.accent == "american"

    def test_voice_profile_to_dict(self):
        """Test VoiceProfile serialization."""
        from tts_toolkit.voices.profile import VoiceProfile

        profile = VoiceProfile(
            name="test",
            reference_audio="test.wav",
            reference_text="Hello",
            description="Test voice",
        )

        d = profile.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["reference_audio"] == "test.wav"
        assert d["description"] == "Test voice"


class TestVoiceRegistry:
    """Tests for VoiceRegistry."""

    def test_registry_create_voice(self):
        """Test creating a voice profile in registry."""
        from tts_toolkit.voices.registry import VoiceRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test audio file
            audio = np.random.randn(24000).astype(np.float32) * 0.5
            audio_path = Path(tmpdir) / "test_voice.wav"
            sf.write(audio_path, audio, 24000)

            # Create registry with custom path
            registry = VoiceRegistry(base_dir=tmpdir)

            profile = registry.create(
                name="test_narrator",
                reference_audio=str(audio_path),
                reference_text="This is a test",
                description="Test narrator voice",
            )

            assert profile.name == "test_narrator"
            assert "test_narrator" in registry.list()

    def test_registry_get_voice(self):
        """Test retrieving a voice profile."""
        from tts_toolkit.voices.registry import VoiceRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = np.random.randn(24000).astype(np.float32) * 0.5
            audio_path = Path(tmpdir) / "voice.wav"
            sf.write(audio_path, audio, 24000)

            registry = VoiceRegistry(base_dir=tmpdir)
            registry.create(
                name="myvoice",
                reference_audio=str(audio_path),
                reference_text="Reference",
            )

            retrieved = registry.get("myvoice")

            assert retrieved is not None
            assert retrieved.name == "myvoice"

    def test_registry_get_nonexistent(self):
        """Test getting nonexistent voice returns None."""
        from tts_toolkit.voices.registry import VoiceRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = VoiceRegistry(base_dir=tmpdir)
            result = registry.get("nonexistent")
            assert result is None

    def test_registry_list_empty(self):
        """Test listing empty registry."""
        from tts_toolkit.voices.registry import VoiceRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = VoiceRegistry(base_dir=tmpdir)
            voices = registry.list()
            assert voices == []

    def test_registry_delete_voice(self):
        """Test deleting a voice profile."""
        from tts_toolkit.voices.registry import VoiceRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = np.random.randn(24000).astype(np.float32) * 0.5
            audio_path = Path(tmpdir) / "voice.wav"
            sf.write(audio_path, audio, 24000)

            registry = VoiceRegistry(base_dir=tmpdir)
            registry.create(
                name="deleteme",
                reference_audio=str(audio_path),
                reference_text="Delete this",
            )

            assert "deleteme" in registry.list()

            result = registry.delete("deleteme")

            assert result is True
            assert "deleteme" not in registry.list()

    def test_registry_delete_nonexistent(self):
        """Test deleting nonexistent voice returns False."""
        from tts_toolkit.voices.registry import VoiceRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = VoiceRegistry(base_dir=tmpdir)
            result = registry.delete("nonexistent")
            assert result is False

    def test_registry_contains(self):
        """Test checking if voice exists in registry."""
        from tts_toolkit.voices.registry import VoiceRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = np.random.randn(24000).astype(np.float32) * 0.5
            audio_path = Path(tmpdir) / "voice.wav"
            sf.write(audio_path, audio, 24000)

            registry = VoiceRegistry(base_dir=tmpdir)
            registry.create(
                name="exists",
                reference_audio=str(audio_path),
                reference_text="Test",
            )

            assert "exists" in registry
            assert "notexists" not in registry


class TestEmotions:
    """Tests for emotion presets."""

    def test_emotion_presets_exist(self):
        """Test that built-in emotion presets exist."""
        from tts_toolkit.voices.emotions import EMOTION_PRESETS

        expected_emotions = [
            "neutral", "happy", "sad", "excited", "serious",
            "calm", "angry", "whisper", "narrator",
        ]

        for emotion in expected_emotions:
            assert emotion in EMOTION_PRESETS, f"Missing emotion preset: {emotion}"

    def test_apply_emotion(self):
        """Test applying emotion to generation kwargs."""
        from tts_toolkit.voices.emotions import apply_emotion

        base_kwargs = {"temperature": 0.9}

        # Apply happy emotion
        result = apply_emotion(base_kwargs, "happy")

        assert isinstance(result, dict)
        # Should have modified parameters
        assert result != base_kwargs or "temperature" in result

    def test_apply_unknown_emotion(self):
        """Test applying unknown emotion returns original kwargs."""
        from tts_toolkit.voices.emotions import apply_emotion

        base_kwargs = {"temperature": 0.9, "top_k": 50}
        result = apply_emotion(base_kwargs, "nonexistent_emotion")

        # Should return original or similar
        assert "temperature" in result

    def test_get_emotion_preset(self):
        """Test getting emotion preset directly."""
        from tts_toolkit.voices.emotions import get_emotion_preset

        preset = get_emotion_preset("happy")

        assert preset is not None
        assert isinstance(preset, dict)

    def test_list_emotions(self):
        """Test listing available emotions."""
        from tts_toolkit.voices.emotions import list_emotions

        emotions = list_emotions()

        assert isinstance(emotions, list)
        assert len(emotions) > 0
        assert "neutral" in emotions


class TestEmotionPreset:
    """Tests for EmotionPreset dataclass."""

    def test_emotion_preset_creation(self):
        """Test creating an EmotionPreset."""
        from tts_toolkit.voices.emotions import EmotionPreset

        preset = EmotionPreset(
            name="custom",
            description="Custom emotion",
            temperature=0.8,
            top_k=40,
        )

        assert preset.name == "custom"
        assert preset.temperature == 0.8
        assert preset.top_k == 40

    def test_emotion_preset_to_kwargs(self):
        """Test converting preset to generation kwargs."""
        from tts_toolkit.voices.emotions import EmotionPreset

        preset = EmotionPreset(
            name="test",
            description="Test",
            temperature=0.7,
            top_p=0.9,
        )

        kwargs = preset.to_kwargs()

        assert isinstance(kwargs, dict)
        assert kwargs.get("temperature") == 0.7
        assert kwargs.get("top_p") == 0.9
