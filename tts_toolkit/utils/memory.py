"""Memory and resource tracking utilities.

Provides functions to monitor system memory, GPU VRAM, and estimate
resource requirements for TTS operations.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger("tts-toolkit")


@dataclass
class MemoryInfo:
    """System memory information."""

    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float

    @property
    def available_gb(self) -> float:
        return self.available_mb / 1024

    def __str__(self) -> str:
        return f"Memory: {self.used_mb:.0f}MB used / {self.total_mb:.0f}MB total ({self.percent_used:.1f}%)"


@dataclass
class GPUInfo:
    """GPU memory information."""

    device_id: int
    name: str
    total_mb: float
    used_mb: float
    free_mb: float

    @property
    def percent_used(self) -> float:
        return (self.used_mb / self.total_mb) * 100 if self.total_mb > 0 else 0

    @property
    def free_gb(self) -> float:
        return self.free_mb / 1024

    def __str__(self) -> str:
        return f"GPU {self.device_id} ({self.name}): {self.used_mb:.0f}MB used / {self.total_mb:.0f}MB total ({self.percent_used:.1f}%)"


def get_system_memory() -> MemoryInfo:
    """Get system memory information.

    Returns:
        MemoryInfo with current memory stats
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return MemoryInfo(
            total_mb=mem.total / (1024 * 1024),
            available_mb=mem.available / (1024 * 1024),
            used_mb=mem.used / (1024 * 1024),
            percent_used=mem.percent,
        )
    except ImportError:
        # Fallback for systems without psutil
        return MemoryInfo(
            total_mb=0,
            available_mb=0,
            used_mb=0,
            percent_used=0,
        )


def get_gpu_memory(device_id: int = 0) -> Optional[GPUInfo]:
    """Get GPU memory information.

    Args:
        device_id: CUDA device index

    Returns:
        GPUInfo or None if GPU not available
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        if device_id >= torch.cuda.device_count():
            return None

        props = torch.cuda.get_device_properties(device_id)
        total = props.total_memory / (1024 * 1024)

        # Get current usage
        torch.cuda.set_device(device_id)
        allocated = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device_id) / (1024 * 1024)

        return GPUInfo(
            device_id=device_id,
            name=props.name,
            total_mb=total,
            used_mb=reserved,
            free_mb=total - reserved,
        )
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"Could not get GPU info: {e}")
        return None


def get_all_gpu_memory() -> Dict[int, GPUInfo]:
    """Get memory info for all available GPUs.

    Returns:
        Dictionary mapping device_id to GPUInfo
    """
    gpus = {}
    try:
        import torch
        if not torch.cuda.is_available():
            return gpus

        for i in range(torch.cuda.device_count()):
            info = get_gpu_memory(i)
            if info:
                gpus[i] = info
    except ImportError:
        pass

    return gpus


def estimate_model_memory(backend_name: str) -> int:
    """Estimate memory requirements for a backend (in MB).

    Args:
        backend_name: Name of the backend

    Returns:
        Estimated memory requirement in MB
    """
    # Approximate memory requirements based on model sizes
    estimates = {
        "qwen": 4000,        # ~4GB for Qwen3-TTS
        "chatterbox": 3000,  # ~3GB for Chatterbox
        "kokoro": 500,       # ~500MB for Kokoro (82M params)
        "bark": 6000,        # ~6GB for Bark
        "cosyvoice": 2500,   # ~2.5GB for CosyVoice2
        "coqui_xtts": 4000,  # ~4GB for XTTS v2
        "fish_speech": 0,    # API-based, no local memory
        "mock": 0,           # No memory needed
    }
    return estimates.get(backend_name, 2000)


def estimate_audio_memory(duration_sec: float, sample_rate: int = 24000) -> float:
    """Estimate memory for audio data (in MB).

    Args:
        duration_sec: Audio duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Estimated memory in MB
    """
    # float32 audio: 4 bytes per sample
    samples = duration_sec * sample_rate
    bytes_needed = samples * 4
    return bytes_needed / (1024 * 1024)


def estimate_text_memory(text_length: int) -> Tuple[float, float]:
    """Estimate processing memory for text.

    Args:
        text_length: Number of characters

    Returns:
        Tuple of (text_memory_mb, estimated_audio_memory_mb)
    """
    # Rough estimates
    # Text itself is small
    text_mb = text_length / (1024 * 1024)

    # Estimate audio output (assuming ~15 chars/second speech rate)
    duration_sec = text_length / 15
    audio_mb = estimate_audio_memory(duration_sec)

    return text_mb, audio_mb


def check_memory_available(
    required_mb: float,
    use_gpu: bool = True,
    device_id: int = 0,
) -> Tuple[bool, str]:
    """Check if enough memory is available.

    Args:
        required_mb: Required memory in MB
        use_gpu: Whether to check GPU memory
        device_id: GPU device to check

    Returns:
        Tuple of (is_available, message)
    """
    if use_gpu:
        gpu_info = get_gpu_memory(device_id)
        if gpu_info:
            if gpu_info.free_mb >= required_mb:
                return True, f"GPU has {gpu_info.free_mb:.0f}MB free (need {required_mb:.0f}MB)"
            else:
                return False, f"Insufficient GPU memory: {gpu_info.free_mb:.0f}MB free, need {required_mb:.0f}MB"

    # Fallback to system memory
    mem_info = get_system_memory()
    if mem_info.available_mb >= required_mb:
        return True, f"System has {mem_info.available_mb:.0f}MB free (need {required_mb:.0f}MB)"
    else:
        return False, f"Insufficient memory: {mem_info.available_mb:.0f}MB free, need {required_mb:.0f}MB"


def warn_if_low_memory(
    backend_name: str,
    text_length: int = 0,
    device: str = "cpu",
    threshold_mb: int = 1000,
) -> None:
    """Log warning if memory is low for the operation.

    Args:
        backend_name: Backend being used
        text_length: Length of text to process
        device: Device string (cpu, cuda:0, etc.)
        threshold_mb: Warning threshold in MB
    """
    model_mb = estimate_model_memory(backend_name)
    _, audio_mb = estimate_text_memory(text_length)
    required_mb = model_mb + audio_mb + 500  # Add buffer

    use_gpu = device.startswith("cuda")
    device_id = 0
    if use_gpu and ":" in device:
        try:
            device_id = int(device.split(":")[1])
        except ValueError:
            pass

    available, msg = check_memory_available(required_mb, use_gpu, device_id)

    if not available:
        logger.warning(f"Low memory warning: {msg}")
        logger.warning(f"Consider using a smaller model or processing in smaller chunks.")
    elif use_gpu:
        gpu_info = get_gpu_memory(device_id)
        if gpu_info and gpu_info.free_mb < threshold_mb:
            logger.warning(f"GPU memory is low: {gpu_info.free_mb:.0f}MB free")


def log_memory_stats(device: str = "cpu") -> None:
    """Log current memory statistics.

    Args:
        device: Device being used
    """
    mem_info = get_system_memory()
    logger.info(str(mem_info))

    if device.startswith("cuda"):
        device_id = 0
        if ":" in device:
            try:
                device_id = int(device.split(":")[1])
            except ValueError:
                pass

        gpu_info = get_gpu_memory(device_id)
        if gpu_info:
            logger.info(str(gpu_info))


def clear_gpu_cache() -> None:
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU cache")
    except ImportError:
        pass
