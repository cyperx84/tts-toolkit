"""Utility modules for TTS Toolkit.

Modules:
    - config: Configuration file support
    - memory: Memory and resource tracking
    - batch: Batch processing utilities
    - metadata: Audio metadata embedding
"""

from .config import (
    TTSConfig,
    load_config,
    save_config,
    init_config,
    apply_config_to_args,
)
from .memory import (
    MemoryInfo,
    GPUInfo,
    get_system_memory,
    get_gpu_memory,
    get_all_gpu_memory,
    estimate_model_memory,
    check_memory_available,
    warn_if_low_memory,
    log_memory_stats,
    clear_gpu_cache,
)
from .batch import (
    BatchJob,
    BatchResult,
    BatchSummary,
    BatchProcessor,
    create_jobs_from_directory,
    create_jobs_from_manifest,
)
from .metadata import (
    embed_metadata,
    embed_metadata_wav,
    embed_metadata_mp3,
    read_metadata,
    create_tts_metadata,
)

__all__ = [
    # Config
    "TTSConfig",
    "load_config",
    "save_config",
    "init_config",
    "apply_config_to_args",
    # Memory
    "MemoryInfo",
    "GPUInfo",
    "get_system_memory",
    "get_gpu_memory",
    "get_all_gpu_memory",
    "estimate_model_memory",
    "check_memory_available",
    "warn_if_low_memory",
    "log_memory_stats",
    "clear_gpu_cache",
    # Batch
    "BatchJob",
    "BatchResult",
    "BatchSummary",
    "BatchProcessor",
    "create_jobs_from_directory",
    "create_jobs_from_manifest",
    # Metadata
    "embed_metadata",
    "embed_metadata_wav",
    "embed_metadata_mp3",
    "read_metadata",
    "create_tts_metadata",
]
