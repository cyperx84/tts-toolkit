"""Emotion and tone control for TTS generation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EmotionPreset:
    """Preset for controlling emotional expression in TTS."""

    name: str
    description: str
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to generation kwargs."""
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


# Built-in emotion presets
EMOTION_PRESETS: Dict[str, EmotionPreset] = {
    "neutral": EmotionPreset(
        name="neutral",
        description="Neutral, natural speaking tone",
        temperature=0.9,
        top_k=50,
        top_p=1.0,
    ),
    "happy": EmotionPreset(
        name="happy",
        description="Cheerful, upbeat tone",
        temperature=0.95,
        top_k=60,
        top_p=1.0,
    ),
    "sad": EmotionPreset(
        name="sad",
        description="Melancholic, subdued tone",
        temperature=0.85,
        top_k=40,
        top_p=0.95,
    ),
    "excited": EmotionPreset(
        name="excited",
        description="Energetic, enthusiastic tone",
        temperature=1.0,
        top_k=75,
        top_p=1.0,
    ),
    "serious": EmotionPreset(
        name="serious",
        description="Formal, professional tone",
        temperature=0.75,
        top_k=30,
        top_p=0.9,
    ),
    "calm": EmotionPreset(
        name="calm",
        description="Relaxed, soothing tone",
        temperature=0.8,
        top_k=40,
        top_p=0.95,
    ),
    "angry": EmotionPreset(
        name="angry",
        description="Intense, frustrated tone",
        temperature=1.0,
        top_k=70,
        top_p=1.0,
    ),
    "whisper": EmotionPreset(
        name="whisper",
        description="Quiet, intimate tone",
        temperature=0.7,
        top_k=25,
        top_p=0.85,
    ),
    "narrator": EmotionPreset(
        name="narrator",
        description="Clear, engaging storytelling tone",
        temperature=0.85,
        top_k=45,
        top_p=0.95,
    ),
    "news": EmotionPreset(
        name="news",
        description="Clear, authoritative news anchor style",
        temperature=0.8,
        top_k=35,
        top_p=0.9,
    ),
    "conversational": EmotionPreset(
        name="conversational",
        description="Natural, casual conversation style",
        temperature=0.92,
        top_k=55,
        top_p=1.0,
    ),
}


def get_emotion(name: str) -> Optional[EmotionPreset]:
    """
    Get an emotion preset by name.

    Args:
        name: Emotion name

    Returns:
        EmotionPreset or None if not found
    """
    return EMOTION_PRESETS.get(name.lower())


def list_emotions() -> Dict[str, str]:
    """
    List available emotions with descriptions.

    Returns:
        Dictionary of emotion names to descriptions
    """
    return {name: preset.description for name, preset in EMOTION_PRESETS.items()}


def apply_emotion(
    base_kwargs: Dict[str, Any], emotion: str
) -> Dict[str, Any]:
    """
    Apply emotion preset to generation kwargs.

    Args:
        base_kwargs: Base generation parameters
        emotion: Emotion name to apply

    Returns:
        Updated generation parameters
    """
    preset = get_emotion(emotion)
    if preset is None:
        return base_kwargs

    result = base_kwargs.copy()
    result.update(preset.to_kwargs())
    return result


def create_custom_emotion(
    name: str,
    description: str,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
) -> EmotionPreset:
    """
    Create a custom emotion preset.

    Args:
        name: Emotion name
        description: Description of the emotion
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter

    Returns:
        EmotionPreset instance
    """
    preset = EmotionPreset(
        name=name,
        description=description,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return preset


def register_emotion(preset: EmotionPreset) -> None:
    """
    Register a custom emotion preset globally.

    Args:
        preset: EmotionPreset to register
    """
    EMOTION_PRESETS[preset.name.lower()] = preset
