"""Voice profile registry for managing voice library."""

import json
import os
import shutil
from typing import Dict, List, Optional

from .profile import VoiceProfile


class VoiceRegistry:
    """Manages a collection of voice profiles."""

    DEFAULT_REGISTRY_DIR = os.path.expanduser("~/.tts_toolkit/voices")

    def __init__(self, registry_dir: Optional[str] = None):
        """
        Initialize voice registry.

        Args:
            registry_dir: Directory to store voice profiles
        """
        self.registry_dir = registry_dir or self.DEFAULT_REGISTRY_DIR
        self._profiles: Dict[str, VoiceProfile] = {}
        self._ensure_registry()
        self._load_registry()

    def _ensure_registry(self) -> None:
        """Ensure registry directory exists."""
        os.makedirs(self.registry_dir, exist_ok=True)

        # Create registry index if it doesn't exist
        index_path = os.path.join(self.registry_dir, "registry.json")
        if not os.path.exists(index_path):
            with open(index_path, "w") as f:
                json.dump({"profiles": {}}, f)

    def _load_registry(self) -> None:
        """Load all profiles from registry."""
        index_path = os.path.join(self.registry_dir, "registry.json")

        if not os.path.exists(index_path):
            return

        with open(index_path, "r") as f:
            index = json.load(f)

        for name, profile_dir in index.get("profiles", {}).items():
            profile_path = os.path.join(profile_dir, "profile.json")
            if os.path.exists(profile_path):
                try:
                    profile = VoiceProfile.load(profile_path)
                    # Resolve relative paths
                    if not os.path.isabs(profile.reference_audio):
                        profile.reference_audio = os.path.join(
                            profile_dir, profile.reference_audio
                        )
                    self._profiles[name] = profile
                except Exception as e:
                    print(f"Warning: Failed to load profile '{name}': {e}")

    def _save_registry(self) -> None:
        """Save registry index."""
        index_path = os.path.join(self.registry_dir, "registry.json")

        index = {
            "profiles": {
                name: os.path.join(self.registry_dir, name)
                for name in self._profiles.keys()
            }
        }

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def create(
        self,
        name: str,
        reference_audio: str,
        reference_text: str,
        description: str = "",
        overwrite: bool = False,
        **kwargs,
    ) -> VoiceProfile:
        """
        Create a new voice profile.

        Args:
            name: Unique name for the profile
            reference_audio: Path to reference audio file
            reference_text: Transcript of the reference audio
            description: Optional description
            overwrite: Whether to overwrite existing profile
            **kwargs: Additional VoiceProfile parameters

        Returns:
            Created VoiceProfile
        """
        # Validate name
        if not name:
            raise ValueError("Profile name is required")

        if name in self._profiles and not overwrite:
            raise ValueError(f"Profile '{name}' already exists. Use overwrite=True to replace.")

        # Create profile directory
        profile_dir = os.path.join(self.registry_dir, name)
        os.makedirs(profile_dir, exist_ok=True)

        # Copy reference audio to profile directory
        audio_ext = os.path.splitext(reference_audio)[1]
        dest_audio = os.path.join(profile_dir, f"reference{audio_ext}")
        shutil.copy2(reference_audio, dest_audio)

        # Create profile
        profile = VoiceProfile(
            name=name,
            reference_audio=dest_audio,
            reference_text=reference_text,
            description=description,
            **kwargs,
        )

        # Validate
        errors = profile.validate()
        if errors:
            # Cleanup on validation failure
            shutil.rmtree(profile_dir, ignore_errors=True)
            raise ValueError(f"Profile validation failed: {', '.join(errors)}")

        # Save profile
        profile.save(os.path.join(profile_dir, "profile.json"))

        # Update registry
        self._profiles[name] = profile
        self._save_registry()

        return profile

    def get(self, name: str) -> VoiceProfile:
        """
        Get a voice profile by name.

        Args:
            name: Profile name

        Returns:
            VoiceProfile instance

        Raises:
            KeyError: If profile not found
        """
        if name not in self._profiles:
            raise KeyError(f"Profile '{name}' not found")
        return self._profiles[name]

    def list(self) -> List[str]:
        """
        List all profile names.

        Returns:
            List of profile names
        """
        return list(self._profiles.keys())

    def list_detailed(self) -> List[Dict]:
        """
        List all profiles with details.

        Returns:
            List of profile summary dictionaries
        """
        return [
            {
                "name": profile.name,
                "description": profile.description,
                "created_at": profile.created_at,
                "metadata": {
                    "gender": profile.metadata.gender,
                    "tone": profile.metadata.tone,
                    "use_cases": profile.metadata.use_cases,
                },
            }
            for profile in self._profiles.values()
        ]

    def delete(self, name: str) -> bool:
        """
        Delete a voice profile.

        Args:
            name: Profile name

        Returns:
            True if deleted, False if not found
        """
        if name not in self._profiles:
            return False

        # Remove profile directory
        profile_dir = os.path.join(self.registry_dir, name)
        if os.path.exists(profile_dir):
            shutil.rmtree(profile_dir)

        # Update registry
        del self._profiles[name]
        self._save_registry()

        return True

    def export(self, name: str, output_dir: str) -> str:
        """
        Export a profile for sharing.

        Args:
            name: Profile name
            output_dir: Directory to export to

        Returns:
            Path to exported directory
        """
        profile = self.get(name)
        return profile.export(output_dir)

    def import_profile(
        self, profile_dir: str, name: Optional[str] = None, overwrite: bool = False
    ) -> VoiceProfile:
        """
        Import a profile from an exported directory.

        Args:
            profile_dir: Directory containing profile.json and audio
            name: Optional new name for the profile
            overwrite: Whether to overwrite existing profile

        Returns:
            Imported VoiceProfile
        """
        profile = VoiceProfile.import_from(profile_dir)

        if name:
            profile.name = name

        if profile.name in self._profiles and not overwrite:
            raise ValueError(
                f"Profile '{profile.name}' already exists. Use overwrite=True or specify a new name."
            )

        # Create profile in registry
        return self.create(
            name=profile.name,
            reference_audio=profile.reference_audio,
            reference_text=profile.reference_text,
            description=profile.description,
            overwrite=overwrite,
            metadata=profile.metadata,
            generation_params=profile.generation_params,
            emotion_presets=profile.emotion_presets,
        )

    def search(
        self,
        gender: Optional[str] = None,
        tone: Optional[str] = None,
        use_case: Optional[str] = None,
    ) -> List[VoiceProfile]:
        """
        Search for profiles matching criteria.

        Args:
            gender: Filter by gender
            tone: Filter by tone
            use_case: Filter by use case

        Returns:
            List of matching profiles
        """
        results = []

        for profile in self._profiles.values():
            if gender and profile.metadata.gender != gender:
                continue
            if tone and profile.metadata.tone != tone:
                continue
            if use_case and use_case not in profile.metadata.use_cases:
                continue
            results.append(profile)

        return results

    def __contains__(self, name: str) -> bool:
        """Check if profile exists."""
        return name in self._profiles

    def __len__(self) -> int:
        """Get number of profiles."""
        return len(self._profiles)

    def __iter__(self):
        """Iterate over profile names."""
        return iter(self._profiles)
