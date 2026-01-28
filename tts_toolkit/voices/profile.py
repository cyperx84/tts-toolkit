"""Voice profile dataclass and utilities."""

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class VoiceMetadata:
    """Metadata about the voice characteristics."""

    gender: Optional[str] = None  # "male", "female", "neutral"
    age_range: Optional[str] = None  # "20-30", "30-40", etc.
    accent: Optional[str] = None  # "american", "british", "neutral"
    tone: Optional[str] = None  # "warm", "professional", "casual"
    use_cases: List[str] = field(default_factory=list)  # ["audiobook", "podcast"]


@dataclass
class GenerationParams:
    """Default generation parameters for this voice."""

    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    x_vector_only: bool = False


@dataclass
class VoiceProfile:
    """A voice profile for TTS generation with voice cloning."""

    # Required fields
    name: str
    reference_audio: str  # Path to reference audio file
    reference_text: str  # Transcript of reference audio

    # Optional identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Voice characteristics
    metadata: VoiceMetadata = field(default_factory=VoiceMetadata)

    # Generation defaults
    generation_params: GenerationParams = field(default_factory=GenerationParams)

    # Emotion presets
    emotion_presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default emotion presets if empty."""
        if not self.emotion_presets:
            self.emotion_presets = {
                "neutral": {},
                "happy": {"temperature": 0.95},
                "sad": {"temperature": 0.85},
                "excited": {"temperature": 1.0, "top_k": 75},
                "serious": {"temperature": 0.75, "top_k": 30},
                "calm": {"temperature": 0.8, "top_k": 40},
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceProfile":
        """Create from dictionary."""
        # Handle nested dataclasses
        if isinstance(data.get("metadata"), dict):
            data["metadata"] = VoiceMetadata(**data["metadata"])
        if isinstance(data.get("generation_params"), dict):
            data["generation_params"] = GenerationParams(**data["generation_params"])
        return cls(**data)

    def save(self, path: str) -> str:
        """
        Save profile to JSON file.

        Args:
            path: Path to save JSON file

        Returns:
            Path to saved file
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "VoiceProfile":
        """
        Load profile from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            VoiceProfile instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_generation_kwargs(self, emotion: Optional[str] = None) -> Dict[str, Any]:
        """
        Get generation kwargs for this voice, optionally with emotion.

        Args:
            emotion: Optional emotion preset to apply

        Returns:
            Dictionary of generation parameters
        """
        # Start with base params
        kwargs = {
            "temperature": self.generation_params.temperature,
            "top_k": self.generation_params.top_k,
            "top_p": self.generation_params.top_p,
        }

        # Apply emotion preset if specified
        if emotion and emotion in self.emotion_presets:
            kwargs.update(self.emotion_presets[emotion])

        return kwargs

    def validate(self) -> List[str]:
        """
        Validate the profile.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.name:
            errors.append("Profile name is required")

        if not self.reference_audio:
            errors.append("Reference audio path is required")
        elif not os.path.exists(self.reference_audio):
            errors.append(f"Reference audio file not found: {self.reference_audio}")

        if not self.reference_text:
            errors.append("Reference text is required")

        return errors

    def export(self, output_dir: str, include_audio: bool = True) -> str:
        """
        Export profile to a directory (for sharing).

        Args:
            output_dir: Directory to export to
            include_audio: Whether to include the reference audio file

        Returns:
            Path to exported directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Copy reference audio if requested
        if include_audio and os.path.exists(self.reference_audio):
            audio_filename = os.path.basename(self.reference_audio)
            dest_audio = os.path.join(output_dir, audio_filename)
            shutil.copy2(self.reference_audio, dest_audio)

            # Update path in exported profile
            exported_profile = VoiceProfile.from_dict(self.to_dict())
            exported_profile.reference_audio = audio_filename
        else:
            exported_profile = self

        # Save profile JSON
        profile_path = os.path.join(output_dir, "profile.json")
        exported_profile.save(profile_path)

        return output_dir

    @classmethod
    def import_from(cls, profile_dir: str) -> "VoiceProfile":
        """
        Import profile from exported directory.

        Args:
            profile_dir: Directory containing profile.json and audio

        Returns:
            VoiceProfile instance
        """
        profile_path = os.path.join(profile_dir, "profile.json")
        profile = cls.load(profile_path)

        # Resolve relative audio path
        if not os.path.isabs(profile.reference_audio):
            profile.reference_audio = os.path.join(
                profile_dir, profile.reference_audio
            )

        return profile
