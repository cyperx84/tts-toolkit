"""Configuration file support for TTS Toolkit.

Supports loading/saving configuration from YAML files:
- ~/.tts_toolkit/config.yml (global)
- .tts_toolkit.yml (project-local)

Example config file:
    backend: qwen
    device: cuda:0
    language: English
    temperature: 0.9

    voices:
      narrator: ~/.tts_toolkit/voices/narrator

    defaults:
      chunk_min: 100
      chunk_max: 300
      crossfade_ms: 75
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger("tts-toolkit")

# Config file locations (in order of priority)
CONFIG_LOCATIONS = [
    ".tts_toolkit.yml",           # Project local
    ".tts_toolkit.yaml",
    "tts_toolkit.yml",
    "tts_toolkit.yaml",
]

GLOBAL_CONFIG_DIR = Path.home() / ".tts_toolkit"
GLOBAL_CONFIG_FILE = GLOBAL_CONFIG_DIR / "config.yml"


@dataclass
class TTSConfig:
    """TTS Toolkit configuration."""

    # Backend settings
    backend: str = "qwen"
    device: str = "cpu"
    model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    # Generation settings
    language: str = "Auto"
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0

    # Chunking settings
    chunk_min: int = 100
    chunk_max: int = 300
    chunk_target: int = 200
    crossfade_ms: int = 75

    # Output settings
    output_format: str = "wav"
    sample_rate: int = 24000
    normalize: bool = True

    # Voice profiles
    default_voice: Optional[str] = None
    voices: Dict[str, str] = field(default_factory=dict)

    # Batch processing
    batch_workers: int = 1
    batch_timeout: int = 300

    # Memory limits
    max_text_chars: int = 100000
    warn_vram_threshold_mb: int = 1000

    # Metadata
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TTSConfig":
        """Create config from dictionary."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def merge(self, other: "TTSConfig") -> "TTSConfig":
        """Merge another config into this one (other takes priority)."""
        self_dict = self.to_dict()
        other_dict = other.to_dict()

        # Deep merge for nested dicts
        for key, value in other_dict.items():
            if value is not None:
                if isinstance(value, dict) and isinstance(self_dict.get(key), dict):
                    self_dict[key] = {**self_dict[key], **value}
                else:
                    self_dict[key] = value

        return TTSConfig.from_dict(self_dict)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed. Config files not supported.")
        return {}

    path = Path(path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data


def save_yaml(path: Union[str, Path], data: Dict[str, Any]) -> None:
    """Save data to YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for config files: pip install pyyaml")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def find_config_file() -> Optional[Path]:
    """Find the nearest config file."""
    # Check local config files
    for name in CONFIG_LOCATIONS:
        path = Path(name)
        if path.exists():
            return path

    # Check parent directories
    current = Path.cwd()
    for _ in range(5):  # Max 5 levels up
        for name in CONFIG_LOCATIONS:
            path = current / name
            if path.exists():
                return path
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    include_global: bool = True,
) -> TTSConfig:
    """Load configuration from files.

    Priority (highest to lowest):
    1. Explicitly provided config_path
    2. Local .tts_toolkit.yml
    3. Global ~/.tts_toolkit/config.yml
    4. Defaults

    Args:
        config_path: Explicit path to config file
        include_global: Whether to include global config

    Returns:
        Merged TTSConfig
    """
    config = TTSConfig()

    # Load global config
    if include_global and GLOBAL_CONFIG_FILE.exists():
        global_data = load_yaml(GLOBAL_CONFIG_FILE)
        if global_data:
            logger.debug(f"Loaded global config: {GLOBAL_CONFIG_FILE}")
            config = config.merge(TTSConfig.from_dict(global_data))

    # Load local config
    local_path = find_config_file()
    if local_path:
        local_data = load_yaml(local_path)
        if local_data:
            logger.debug(f"Loaded local config: {local_path}")
            config = config.merge(TTSConfig.from_dict(local_data))

    # Load explicit config
    if config_path:
        explicit_data = load_yaml(config_path)
        if explicit_data:
            logger.debug(f"Loaded explicit config: {config_path}")
            config = config.merge(TTSConfig.from_dict(explicit_data))

    return config


def save_config(
    config: TTSConfig,
    path: Optional[Union[str, Path]] = None,
    global_config: bool = False,
) -> Path:
    """Save configuration to file.

    Args:
        config: Configuration to save
        path: Explicit path (default: .tts_toolkit.yml or global)
        global_config: Save to global config instead of local

    Returns:
        Path to saved config file
    """
    if path:
        save_path = Path(path)
    elif global_config:
        save_path = GLOBAL_CONFIG_FILE
    else:
        save_path = Path(".tts_toolkit.yml")

    save_yaml(save_path, config.to_dict())
    logger.info(f"Saved config to: {save_path}")

    return save_path


def init_config(
    path: Optional[Union[str, Path]] = None,
    global_config: bool = False,
) -> Path:
    """Initialize a new config file with defaults.

    Args:
        path: Explicit path for config file
        global_config: Create global config instead of local

    Returns:
        Path to created config file
    """
    config = TTSConfig()
    return save_config(config, path, global_config)


def apply_config_to_args(args, config: TTSConfig):
    """Apply config values to argparse args (config is lower priority)."""
    # Only apply config values if args has default/None values
    defaults = {
        "backend": "qwen",
        "device": "cpu",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "language": "Auto",
        "temperature": 0.9,
    }

    for attr, default in defaults.items():
        if hasattr(args, attr):
            current = getattr(args, attr)
            config_val = getattr(config, attr, None)
            # If arg is at default and config has different value, use config
            if current == default and config_val is not None and config_val != default:
                setattr(args, attr, config_val)

    return args
